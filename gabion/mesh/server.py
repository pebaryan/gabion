from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List

from aiohttp import WSMsgType, web

from gabion.common.config import MeshConfig
from gabion.common.jobs import TrainingJob
from gabion.common.protocol import Message, RoundResultPayload, make_message
from gabion.mesh.aggregator import fedavg
from gabion.mesh.coordinator import RoundCoordinator

logger = logging.getLogger(__name__)


@dataclass
class WorkerSession:
    ws: web.WebSocketResponse
    last_heartbeat: float
    capabilities: Dict[str, str]
    joined_job_id: str | None = None


@dataclass
class JobRuntime:
    job: TrainingJob
    weights: List[float]
    coordinator: RoundCoordinator


class MeshServer:
    def __init__(self, config: MeshConfig, jobs: List[TrainingJob]) -> None:
        if not jobs:
            raise ValueError("at least one training job is required")

        self.config = config
        self._workers: Dict[str, WorkerSession] = {}
        self._workers_lock = asyncio.Lock()
        self._round_task: asyncio.Task[None] | None = None
        self._jobs: Dict[str, JobRuntime] = {
            job.job_id: JobRuntime(job=job, weights=list(job.initial_weights), coordinator=RoundCoordinator())
            for job in jobs
        }

        self.app = web.Application()
        self.app.add_routes(
            [
                web.get("/health", self.health_handler),
                web.get("/status", self.status_handler),
                web.get("/jobs", self.jobs_handler),
                web.get("/ws", self.ws_handler),
            ]
        )
        self.app.on_startup.append(self._on_startup)
        self.app.on_cleanup.append(self._on_cleanup)

    async def _on_startup(self, app: web.Application) -> None:
        self._round_task = asyncio.create_task(self._round_loop())

    async def _on_cleanup(self, app: web.Application) -> None:
        if self._round_task:
            self._round_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._round_task

    async def health_handler(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def jobs_handler(self, request: web.Request) -> web.Response:
        return web.json_response({"jobs": [runtime.job.to_dict() for runtime in self._jobs.values()]})

    async def status_handler(self, request: web.Request) -> web.Response:
        async with self._workers_lock:
            workers = list(self._workers.keys())
            joined_by_job: Dict[str, int] = {job_id: 0 for job_id in self._jobs.keys()}
            for session in self._workers.values():
                if session.joined_job_id:
                    joined_by_job[session.joined_job_id] = joined_by_job.get(session.joined_job_id, 0) + 1

        jobs = []
        for job_id, runtime in self._jobs.items():
            jobs.append(
                {
                    "job_id": job_id,
                    "name": runtime.job.name,
                    "joined_workers": joined_by_job.get(job_id, 0),
                    "weights_dim": len(runtime.weights),
                    "current_round": await runtime.coordinator.current_round_id(),
                }
            )

        return web.json_response(
            {
                "worker_count": len(workers),
                "workers": workers,
                "jobs": jobs,
            }
        )

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        worker_id: str | None = None
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                parsed = json.loads(msg.data)
                if not isinstance(parsed, dict):
                    continue

                message: Message = parsed
                msg_type = message.get("type")
                payload = message.get("payload", {})

                if msg_type == "register":
                    worker_id = str(payload.get("worker_id", ""))
                    if not worker_id:
                        await ws.send_json(make_message("error", {"reason": "missing_worker_id"}))
                        continue
                    capabilities = {
                        str(k): str(v) for k, v in dict(payload.get("capabilities", {})).items()
                    }
                    async with self._workers_lock:
                        self._workers[worker_id] = WorkerSession(
                            ws=ws,
                            last_heartbeat=time.time(),
                            capabilities=capabilities,
                        )
                    await ws.send_json(make_message("registered", {"worker_id": worker_id}))
                    logger.info("worker registered: %s", worker_id)

                elif msg_type == "list_jobs":
                    await ws.send_json(
                        make_message("job_list", {"jobs": [runtime.job.to_dict() for runtime in self._jobs.values()]})
                    )

                elif msg_type == "join_job" and worker_id:
                    job_id = str(payload.get("job_id", ""))
                    runtime = self._jobs.get(job_id)
                    if runtime is None:
                        await ws.send_json(
                            make_message("job_rejected", {"job_id": job_id, "reason": "unknown_job"})
                        )
                        continue

                    session = self._workers.get(worker_id)
                    trainer_backend = session.capabilities.get("trainer", "") if session else ""
                    if runtime.job.runtime == "tinygrad" and trainer_backend != "tinygrad":
                        await ws.send_json(
                            make_message(
                                "artifact_required",
                                {
                                    "job_id": job_id,
                                    "runtime": runtime.job.runtime,
                                    "artifact_uri": runtime.job.artifact_uri,
                                    "artifact_checksum": runtime.job.artifact_checksum,
                                },
                            )
                        )
                        await ws.send_json(
                            make_message(
                                "job_rejected",
                                {"job_id": job_id, "reason": "runtime_mismatch"},
                            )
                        )
                        continue

                    async with self._workers_lock:
                        joined = self._workers.get(worker_id)
                        if joined:
                            joined.joined_job_id = job_id
                    await ws.send_json(
                        make_message(
                            "job_joined",
                            {
                                "job_id": job_id,
                                "name": runtime.job.name,
                                "runtime": runtime.job.runtime,
                            },
                        )
                    )

                elif msg_type == "heartbeat" and worker_id:
                    async with self._workers_lock:
                        session = self._workers.get(worker_id)
                        if session:
                            session.last_heartbeat = time.time()

                elif msg_type == "round_result" and worker_id:
                    result: RoundResultPayload = {
                        "worker_id": worker_id,
                        "job_id": str(payload["job_id"]),
                        "round_id": int(payload["round_id"]),
                        "sample_count": int(payload["sample_count"]),
                        "weights": [float(v) for v in payload["weights"]],
                        "metrics": {
                            str(k): float(v) for k, v in dict(payload.get("metrics", {})).items()
                        },
                    }
                    job = self._jobs.get(result["job_id"])
                    if job is None:
                        await ws.send_json(make_message("error", {"reason": "unknown_job"}))
                        continue
                    accepted = await job.coordinator.submit_result(result)
                    if not accepted:
                        await ws.send_json(
                            make_message("error", {"reason": "stale_or_invalid_round_result"})
                        )
        finally:
            if worker_id:
                async with self._workers_lock:
                    self._workers.pop(worker_id, None)
                logger.info("worker disconnected: %s", worker_id)
        return ws

    async def _round_loop(self) -> None:
        for runtime in self._jobs.values():
            for round_id in range(1, runtime.job.max_rounds + 1):
                await self._wait_for_job_workers(runtime.job.job_id, runtime.job.min_quorum)
                await runtime.coordinator.start_round(round_id)

                workers = await self._workers_for_job(runtime.job.job_id)
                round_start = make_message(
                    "round_start",
                    {
                        "job_id": runtime.job.job_id,
                        "round_id": round_id,
                        "weights": runtime.weights,
                        "model_adapter": runtime.job.model_adapter,
                        "local_epochs": runtime.job.local_epochs,
                    },
                )
                participant_count = await self._send_to_workers(workers, round_start)
                logger.info(
                    "job=%s round=%s started with %s participants",
                    runtime.job.job_id,
                    round_id,
                    participant_count,
                )

                results = await runtime.coordinator.wait_for_quorum(
                    min_quorum=runtime.job.min_quorum,
                    timeout_s=self.config.round_timeout_s,
                )
                runtime.weights = fedavg(results, runtime.weights)

                losses = [result["metrics"].get("loss", 0.0) for result in results]
                mean_loss = sum(losses) / len(losses) if losses else 0.0
                await self._send_to_workers(
                    workers,
                    make_message(
                        "round_summary",
                        {
                            "job_id": runtime.job.job_id,
                            "round_id": round_id,
                            "participant_count": len(results),
                            "mean_loss": mean_loss,
                        },
                    ),
                )

    async def _wait_for_job_workers(self, job_id: str, min_quorum: int) -> None:
        while True:
            await self._evict_stale_workers()
            workers = await self._workers_for_job(job_id)
            if len(workers) >= min_quorum:
                return
            await asyncio.sleep(0.2)

    async def _workers_for_job(self, job_id: str) -> List[WorkerSession]:
        async with self._workers_lock:
            return [
                session
                for session in self._workers.values()
                if session.joined_job_id == job_id and not session.ws.closed
            ]

    async def _send_to_workers(self, sessions: List[WorkerSession], message: Message) -> int:
        sent = 0
        for session in sessions:
            if session.ws.closed:
                continue
            try:
                await session.ws.send_json(message)
                sent += 1
            except Exception:
                continue
        return sent

    async def _evict_stale_workers(self) -> None:
        cutoff = time.time() - self.config.heartbeat_timeout_s
        async with self._workers_lock:
            stale = [
                worker_id
                for worker_id, session in self._workers.items()
                if session.last_heartbeat < cutoff or session.ws.closed
            ]
            for worker_id in stale:
                self._workers.pop(worker_id, None)


def run_mesh(config: MeshConfig, jobs: List[TrainingJob]) -> None:
    server = MeshServer(config=config, jobs=jobs)
    web.run_app(server.app, host=config.host, port=config.port)
