from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import math
import os
import random
import struct
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

from aiohttp import WSMsgType, web

from gabion.common.config import MeshConfig
from gabion.common.jobs import TrainingJob
from gabion.common.protocol import Message, RoundResultPayload, make_message
from gabion.mesh.aggregator import fedavg
from gabion.mesh.coordinator import RoundCoordinator

logger = logging.getLogger(__name__)


def _decode_int8_delta(b64str: str, scale: float, original: List[float]) -> List[float]:
    """Decode int8 delta-compressed weights back to full float weights."""
    raw = base64.b64decode(b64str)
    n = len(raw)
    result = list(original[:n])
    for i in range(n):
        # Signed int8
        val = raw[i]
        if val >= 128:
            val -= 256
        result[i] += val * scale
    return result


def _decode_f16_base64(b64str: str, count: int = 0) -> List[float]:
    """Decode base64-encoded f16 weights to list of Python floats."""
    raw = base64.b64decode(b64str)
    n = count if count > 0 else len(raw) // 2
    # Unpack as unsigned 16-bit integers, then convert each to float32
    result = []
    for i in range(n):
        h = int.from_bytes(raw[i * 2 : i * 2 + 2], "little")
        sign = (h >> 15) & 1
        exp = (h >> 10) & 0x1F
        frac = h & 0x3FF
        if exp == 0x1F:
            val = float("inf") if frac == 0 else float("nan")
            if sign:
                val = -val
        elif exp == 0:
            # Denorm or zero
            val = ((-1) ** sign) * (frac / 1024.0) * (2 ** -14)
        else:
            val = ((-1) ** sign) * (1.0 + frac / 1024.0) * (2 ** (exp - 15))
        result.append(val)
    return result


def _encode_f16_base64(weights: List[float]) -> str:
    """Encode list of floats as base64-encoded f16 for compact transfer."""
    buf = bytearray(len(weights) * 2)
    for i, v in enumerate(weights):
        # Pack as f32 then convert to f16
        bits = struct.unpack("<I", struct.pack("<f", v))[0]
        sign = (bits >> 31) & 1
        exp = (bits >> 23) & 0xFF
        frac = bits & 0x7FFFFF
        if exp == 0xFF:
            h = (sign << 15) | 0x7C00 | (0x0200 if frac else 0)
        elif exp > 142:
            h = (sign << 15) | 0x7C00
        elif exp < 103:
            h = sign << 15
        elif exp < 113:
            m = (0x800000 | frac) >> (126 - exp)
            h = (sign << 15) | (m >> 13)
        else:
            h = (sign << 15) | ((exp - 112) << 10) | (frac >> 13)
        buf[i * 2] = h & 0xFF
        buf[i * 2 + 1] = (h >> 8) & 0xFF
    return base64.b64encode(bytes(buf)).decode("ascii")


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
    last_completed_round: int = 0
    round_history: List[Dict[str, Any]] = field(default_factory=list)
    eval_history: List[Dict[str, Any]] = field(default_factory=list)
    worker_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    eval_seed: int = 0
    eval_adapter: Any | None = None
    eval_template_params: List[Any] | None = None
    model_version: int = 0
    target_max_rounds: int = 0


class MeshServer:
    def __init__(self, config: MeshConfig, jobs: List[TrainingJob]) -> None:
        if not jobs:
            raise ValueError("at least one training job is required")

        self.config = config
        self._workers: Dict[str, WorkerSession] = {}
        self._workers_lock = asyncio.Lock()
        self._round_task: asyncio.Task[None] | None = None
        self._browser_shard_paths: List[Path] = []
        self._browser_shard_sizes: List[int] = []
        self._jobs: Dict[str, JobRuntime] = {
            job.job_id: JobRuntime(
            job=job,
            weights=list(job.initial_weights),
            coordinator=RoundCoordinator(),
            eval_seed=self._job_eval_seed(job.job_id),
            target_max_rounds=int(job.max_rounds),
        )
            for job in jobs
        }
        self._load_checkpoint()

        self.app = web.Application()
        self.app.add_routes(
            [
                web.get("/health", self.health_handler),
                web.get("/status", self.status_handler),
                web.get("/jobs", self.jobs_handler),
                web.get("/metrics", self.metrics_handler),
                web.get("/dashboard", self.dashboard_handler),
                web.get("/webgpu-worker", self.webgpu_worker_handler),
                web.get("/data/byte-batch", self.browser_byte_batch_handler),
                web.get("/assets/{name}", self.asset_handler),
                web.get("/assets/kernels/{name}", self.kernel_asset_handler),
                web.post("/jobs/{job_id}/max-rounds", self.set_max_rounds_handler),
                web.get("/ws", self.ws_handler),
            ]
        )
        self.app.on_startup.append(self._on_startup)
        self.app.on_cleanup.append(self._on_cleanup)

    def _job_eval_seed(self, job_id: str) -> int:
        return int((zlib.crc32(job_id.encode("utf-8")) + int(self.config.eval_seed)) & 0xFFFFFFFF)

    def _job_list_payload(self) -> List[Dict[str, Any]]:
        return [
            {
                **runtime.job.to_dict(),
                "max_rounds": runtime.target_max_rounds,
                "model_version": runtime.model_version,
                "last_completed_round": runtime.last_completed_round,
            }
            for runtime in self._jobs.values()
        ]

    def _set_job_max_rounds(self, job_id: str, max_rounds: int) -> int:
        runtime = self._jobs.get(job_id)
        if runtime is None:
            raise KeyError(job_id)
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        runtime.target_max_rounds = max(int(max_rounds), int(runtime.last_completed_round))
        return runtime.target_max_rounds

    async def _on_startup(self, app: web.Application) -> None:
        self._round_task = asyncio.create_task(self._round_loop())

    async def _on_cleanup(self, app: web.Application) -> None:
        if self._round_task:
            self._round_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._round_task
        self._save_checkpoint()

    async def health_handler(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def jobs_handler(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "jobs": self._job_list_payload()
            }
        )

    async def metrics_handler(self, request: web.Request) -> web.Response:
        return web.json_response(self._metrics_payload())

    async def dashboard_handler(self, request: web.Request) -> web.Response:
        return web.Response(text=self._dashboard_html(), content_type="text/html")

    async def set_max_rounds_handler(self, request: web.Request) -> web.Response:
        job_id = str(request.match_info.get("job_id", ""))
        if not job_id:
            return web.json_response({"error": "missing_job_id"}, status=400)
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        try:
            requested = int(dict(payload).get("max_rounds"))
        except Exception:
            return web.json_response({"error": "invalid_max_rounds"}, status=400)
        try:
            updated = self._set_job_max_rounds(job_id=job_id, max_rounds=requested)
        except KeyError:
            return web.json_response({"error": "unknown_job"}, status=404)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        self._save_checkpoint()
        runtime = self._jobs[job_id]
        return web.json_response(
            {
                "job_id": job_id,
                "max_rounds": updated,
                "last_completed_round": runtime.last_completed_round,
                "model_version": runtime.model_version,
            }
        )

    async def webgpu_worker_handler(self, request: web.Request) -> web.Response:
        return web.Response(text=self._webgpu_worker_html(), content_type="text/html")

    async def browser_byte_batch_handler(self, request: web.Request) -> web.Response:
        if not self.config.allow_browser_workers:
            return web.json_response({"error": "browser_workers_disabled"}, status=403)
        try:
            batch_size = max(1, min(8192, int(request.query.get("batch_size", "256"))))
        except Exception:
            batch_size = 256
        try:
            seed = int(request.query.get("seed", "0"))
        except Exception:
            seed = 0
        try:
            seq_len = int(request.query.get("seq_len", "0"))
        except Exception:
            seq_len = 0
        if seq_len > 0:
            payload = self._browser_sequence_batch(
                batch_size=batch_size, seq_len=seq_len, seed=seed, vocab_size=256
            )
        else:
            payload = self._browser_byte_batch(batch_size=batch_size, seed=seed, vocab_size=256)
        return web.json_response(payload)

    async def asset_handler(self, request: web.Request) -> web.StreamResponse:
        name = str(request.match_info.get("name", ""))
        allowed = {"tinygrad_v0.js", "webgpu_backend.js", "bbt_forward.js"}
        if name not in allowed:
            raise web.HTTPNotFound()
        asset_path = Path(__file__).resolve().parents[1] / "web" / name
        if not asset_path.exists():
            raise web.HTTPNotFound()
        return web.FileResponse(asset_path)

    async def kernel_asset_handler(self, request: web.Request) -> web.StreamResponse:
        name = str(request.match_info.get("name", ""))
        if Path(name).name != name or not name.endswith(".wgsl"):
            raise web.HTTPNotFound()
        kernels_dir = (Path(__file__).resolve().parents[1] / "web" / "kernels").resolve()
        asset_path = (kernels_dir / name).resolve()
        if asset_path.parent != kernels_dir or not asset_path.exists():
            raise web.HTTPNotFound()
        return web.FileResponse(asset_path, headers={"Content-Type": "text/plain"})

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
                    "last_completed_round": runtime.last_completed_round,
                    "max_rounds": runtime.target_max_rounds,
                    "model_version": runtime.model_version,
                    "latest_mean_loss": self._latest_mean_loss(runtime),
                    "latest_eval_loss": self._latest_eval_loss(runtime),
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
                        make_message(
                            "job_list",
                            {"jobs": self._job_list_payload()},
                        )
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
                    browser_proto_allowed = (
                        self.config.allow_browser_workers and trainer_backend == "browser-webgpu-proto"
                    )
                    if runtime.job.runtime == "tinygrad" and trainer_backend != "tinygrad" and not browser_proto_allowed:
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
                    worker_meta = {
                        str(k): str(v)
                        for k, v in dict(payload.get("worker_meta", {})).items()
                    }
                    # Decode weights: int8 delta, f16 base64, or plain JSON array
                    job_for_decode = self._jobs.get(str(payload.get("job_id", "")))
                    if payload.get("weights_format") == "int8_delta" and payload.get("weights_delta") and job_for_decode:
                        weights_list = _decode_int8_delta(
                            payload["weights_delta"],
                            float(payload.get("delta_scale", 0)),
                            job_for_decode.runtime.weights,
                        )
                    elif payload.get("weights_format") == "f16_base64" and payload.get("weights_f16"):
                        weights_list = _decode_f16_base64(payload["weights_f16"], int(payload.get("weights_count", 0)))
                    else:
                        weights_list = [float(v) for v in payload["weights"]]
                    result: RoundResultPayload = {
                        "worker_id": worker_id,
                        "job_id": str(payload["job_id"]),
                        "round_id": int(payload["round_id"]),
                        "round_token": str(payload.get("round_token", "")),
                        "model_version": int(payload.get("model_version", -1)),
                        "sample_count": int(payload["sample_count"]),
                        "weights": weights_list,
                        "metrics": {
                            str(k): float(v) for k, v in dict(payload.get("metrics", {})).items()
                        },
                    }
                    job = self._jobs.get(result["job_id"])
                    if job is None:
                        await ws.send_json(make_message("error", {"reason": "unknown_job"}))
                        continue
                    accepted, reason = await job.coordinator.submit_result(result)
                    if not accepted:
                        await ws.send_json(
                            make_message("error", {"reason": f"stale_or_invalid_round_result:{reason}"})
                        )
                        continue
                    # Track latest worker execution metadata for dashboard visibility.
                    loss = float(result["metrics"].get("loss", 0.0))
                    job.worker_stats[worker_id] = {
                        "worker_id": worker_id,
                        "round_id": int(result["round_id"]),
                        "loss": loss,
                        "trainer": self._workers.get(worker_id).capabilities.get("trainer", "") if self._workers.get(worker_id) else "",
                        "kind": self._workers.get(worker_id).capabilities.get("kind", "") if self._workers.get(worker_id) else "",
                        "mode": worker_meta.get("mode", ""),
                        "data_source": worker_meta.get("data_source", ""),
                        "updated_at": time.time(),
                    }
        finally:
            if worker_id:
                async with self._workers_lock:
                    self._workers.pop(worker_id, None)
                logger.info("worker disconnected: %s", worker_id)
        return ws

    async def _round_loop(self) -> None:
        while True:
            await self._evict_stale_workers()
            progressed = False
            for runtime in self._jobs.values():
                round_id = runtime.last_completed_round + 1
                if round_id > runtime.target_max_rounds:
                    continue
                workers = await self._workers_for_job(runtime.job.job_id)
                if len(workers) < runtime.job.min_quorum:
                    continue
                progressed = True
                round_token = f"{runtime.job.job_id}:{round_id}:{runtime.model_version}:{random.getrandbits(32):08x}"
                await runtime.coordinator.start_round(
                    round_id=round_id,
                    round_token=round_token,
                    model_version=runtime.model_version,
                )
                participant_count = await self._send_round_start(runtime, round_id, round_token, workers)
                logger.info(
                    "job=%s round=%s started with %s participants",
                    runtime.job.job_id,
                    round_id,
                    participant_count,
                )

                results = await runtime.coordinator.wait_for_quorum(
                    min_quorum=runtime.job.min_quorum,
                    timeout_s=self.config.round_timeout_s,
                    collect_until_timeout=self.config.async_participation,
                )
                if len(results) < runtime.job.min_quorum:
                    logger.warning(
                        "job=%s round=%s insufficient quorum (%s/%s), will retry",
                        runtime.job.job_id,
                        round_id,
                        len(results),
                        runtime.job.min_quorum,
                    )
                    await self._send_to_workers(
                        workers,
                        make_message(
                            "error",
                            {
                                "reason": "insufficient_quorum",
                                "job_id": runtime.job.job_id,
                                "round_id": round_id,
                                "received": len(results),
                                "required": runtime.job.min_quorum,
                            },
                        ),
                    )
                    continue
                runtime.weights = fedavg(results, runtime.weights)
                runtime.last_completed_round = round_id
                runtime.model_version += 1

                losses = [
                    float(result["metrics"].get("loss", 0.0))
                    for result in results
                    if math.isfinite(float(result["metrics"].get("loss", 0.0)))
                ]
                mean_loss = sum(losses) / len(losses) if losses else 0.0
                runtime.round_history.append(
                    {
                        "round_id": round_id,
                        "participant_count": len(results),
                        "mean_loss": mean_loss,
                        "timestamp": time.time(),
                    }
                )
                if len(runtime.round_history) > 500:
                    runtime.round_history = runtime.round_history[-500:]
                self._maybe_run_eval(runtime, round_id)
                await self._send_to_workers(
                    workers,
                    make_message(
                        "round_summary",
                        {
                            "job_id": runtime.job.job_id,
                            "round_id": round_id,
                            "participant_count": len(results),
                            "mean_loss": mean_loss,
                            "eval_loss": self._latest_eval_loss(runtime),
                        },
                    ),
                )
                if round_id % max(1, self.config.checkpoint_every_rounds) == 0:
                    self._save_checkpoint()
            if not progressed:
                await asyncio.sleep(0.2)

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

    async def _send_round_start(
        self, runtime: JobRuntime, round_id: int, round_token: str, sessions: List[WorkerSession]
    ) -> int:
        sent = 0
        for session in sessions:
            if session.ws.closed:
                continue
            try:
                work_scale = 1.0
                try:
                    work_scale = float(session.capabilities.get("work_scale", "1.0"))
                except Exception:
                    work_scale = 1.0
                work_scale = min(1.0, max(0.05, work_scale))
                round_start = make_message(
                    "round_start",
                    {
                        "job_id": runtime.job.job_id,
                        "round_id": round_id,
                        "round_token": round_token,
                        "model_version": runtime.model_version,
                        "weights": runtime.weights,
                        "model_adapter": runtime.job.model_adapter,
                        "local_epochs": runtime.job.local_epochs,
                        "work_scale": work_scale,
                        "learning_rate": runtime.job.learning_rate,
                        "optimizer": runtime.job.optimizer,
                        "grad_clip_norm": runtime.job.grad_clip_norm,
                        "warmup_steps": runtime.job.warmup_steps,
                        "adam_beta1": runtime.job.adam_beta1,
                        "adam_beta2": runtime.job.adam_beta2,
                    },
                )
                await session.ws.send_json(round_start)
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

    def _load_checkpoint(self) -> None:
        if not self.config.checkpoint_path:
            return
        ckpt_path = Path(self.config.checkpoint_path)
        if not ckpt_path.exists():
            return
        try:
            data = json.loads(ckpt_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("failed to read checkpoint %s: %s", ckpt_path, exc)
            return

        jobs = data.get("jobs", {})
        if not isinstance(jobs, dict):
            logger.warning("invalid checkpoint format in %s", ckpt_path)
            return

        restored = 0
        for job_id, entry in jobs.items():
            runtime = self._jobs.get(str(job_id))
            if runtime is None or not isinstance(entry, dict):
                continue
            weights = entry.get("weights")
            last_round = entry.get("last_completed_round", 0)
            model_version = entry.get("model_version", last_round)
            target_max_rounds = entry.get("target_max_rounds", runtime.job.max_rounds)
            if not isinstance(weights, list):
                continue
            if len(weights) != len(runtime.weights):
                logger.warning(
                    "checkpoint weights mismatch for job=%s (got %s, expected %s), skipping",
                    runtime.job.job_id,
                    len(weights),
                    len(runtime.weights),
                )
                continue
            try:
                runtime.weights = [float(v) for v in weights]
                runtime.last_completed_round = max(0, int(last_round))
                runtime.model_version = max(0, int(model_version))
                runtime.target_max_rounds = max(
                    int(runtime.last_completed_round),
                    int(target_max_rounds),
                )
                history = entry.get("round_history", [])
                if isinstance(history, list):
                    cleaned_history: List[Dict[str, Any]] = []
                    for point in history[-500:]:
                        if not isinstance(point, dict):
                            continue
                        try:
                            cleaned_history.append(
                                {
                                    "round_id": int(point.get("round_id", 0)),
                                    "participant_count": int(point.get("participant_count", 0)),
                                    "mean_loss": float(point.get("mean_loss", 0.0)),
                                    "timestamp": float(point.get("timestamp", 0.0)),
                                }
                            )
                        except Exception:
                            continue
                    runtime.round_history = cleaned_history
                eval_history = entry.get("eval_history", [])
                if isinstance(eval_history, list):
                    cleaned_eval: List[Dict[str, Any]] = []
                    for point in eval_history[-500:]:
                        if not isinstance(point, dict):
                            continue
                        try:
                            cleaned_eval.append(
                                {
                                    "round_id": int(point.get("round_id", 0)),
                                    "eval_loss": float(point.get("eval_loss", 0.0)),
                                    "sample_count": int(point.get("sample_count", 0)),
                                    "timestamp": float(point.get("timestamp", 0.0)),
                                }
                            )
                        except Exception:
                            continue
                    runtime.eval_history = cleaned_eval
                restored += 1
            except Exception:
                continue

        if restored:
            logger.info("restored checkpoint for %s job(s) from %s", restored, ckpt_path)

    def _save_checkpoint(self) -> None:
        if not self.config.checkpoint_path:
            return
        ckpt_path = Path(self.config.checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "saved_at": time.time(),
            "jobs": {
                job_id: {
                    "last_completed_round": runtime.last_completed_round,
                    "model_version": runtime.model_version,
                    "target_max_rounds": runtime.target_max_rounds,
                    "weights": runtime.weights,
                    "round_history": runtime.round_history,
                    "eval_history": runtime.eval_history,
                }
                for job_id, runtime in self._jobs.items()
            },
        }
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(ckpt_path.parent),
                prefix=f"{ckpt_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tf:
                json.dump(payload, tf)
                tf.flush()
                os.fsync(tf.fileno())
                tmp_path = Path(tf.name)
            os.replace(str(tmp_path), str(ckpt_path))
            tmp_path = None
        except Exception as exc:
            logger.warning("failed to save checkpoint %s: %s", ckpt_path, exc)
        finally:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    tmp_path.unlink(missing_ok=True)

    def _metrics_payload(self) -> Dict[str, Any]:
        jobs: List[Dict[str, Any]] = []
        for job_id, runtime in self._jobs.items():
            jobs.append(
                {
                    "job_id": job_id,
                    "name": runtime.job.name,
                    "max_rounds": runtime.target_max_rounds,
                    "last_completed_round": runtime.last_completed_round,
                    "model_version": runtime.model_version,
                    "latest_mean_loss": self._latest_mean_loss(runtime),
                    "latest_eval_loss": self._latest_eval_loss(runtime),
                    "recent_history": runtime.round_history[-200:],
                    "recent_eval_history": runtime.eval_history[-200:],
                    "worker_stats": list(runtime.worker_stats.values()),
                }
            )
        return {"generated_at": time.time(), "jobs": jobs}

    @staticmethod
    def _latest_mean_loss(runtime: JobRuntime) -> float | None:
        if not runtime.round_history:
            return None
        return float(runtime.round_history[-1]["mean_loss"])

    @staticmethod
    def _latest_eval_loss(runtime: JobRuntime) -> float | None:
        if not runtime.eval_history:
            return None
        return float(runtime.eval_history[-1]["eval_loss"])

    def _maybe_run_eval(self, runtime: JobRuntime, round_id: int) -> None:
        if self.config.eval_every_rounds <= 0:
            return
        if round_id % max(1, int(self.config.eval_every_rounds)) != 0:
            return
        try:
            from tinygrad import Tensor  # type: ignore

            from gabion.pebble.adapters import load_adapter, unflatten_to_tensors

            if runtime.eval_adapter is None:
                runtime.eval_adapter = load_adapter(runtime.job.model_adapter)
                runtime.eval_template_params = runtime.eval_adapter.init_params(seed=0)
            if runtime.eval_template_params is None:
                runtime.eval_template_params = runtime.eval_adapter.init_params(seed=0)

            params = unflatten_to_tensors(runtime.weights, runtime.eval_template_params, Tensor)
            for p in params:
                p.requires_grad = False

            x, y = runtime.eval_adapter.sample_batch(
                batch_size=max(1, int(self.config.eval_batch_size)),
                seed=runtime.eval_seed,
            )
            logits = runtime.eval_adapter.forward(params, x)
            loss = runtime.eval_adapter.loss(logits, y)
            eval_loss = float(loss.item())
            if not math.isfinite(eval_loss):
                return
            runtime.eval_history.append(
                {
                    "round_id": int(round_id),
                    "eval_loss": eval_loss,
                    "sample_count": max(1, int(self.config.eval_batch_size)),
                    "timestamp": time.time(),
                }
            )
            if len(runtime.eval_history) > 500:
                runtime.eval_history = runtime.eval_history[-500:]
        except Exception as exc:
            logger.warning("eval failed for job=%s round=%s: %s", runtime.job.job_id, round_id, exc)

    def _browser_byte_batch(self, batch_size: int, seed: int, vocab_size: int = 256) -> Dict[str, Any]:
        shards = self._load_browser_shards()
        if shards:
            rng = random.Random(seed)
            x: List[int] = []
            y: List[int] = []
            tries = batch_size * 10
            while len(x) < batch_size and tries > 0:
                tries -= 1
                i = rng.randint(0, len(self._browser_shard_paths) - 1)
                path = self._browser_shard_paths[i]
                size = self._browser_shard_sizes[i]
                if size < 2:
                    continue
                off = rng.randint(0, size - 2)
                with path.open("rb") as f:
                    f.seek(off)
                    b = f.read(2)
                if len(b) != 2:
                    continue
                x.append(int(b[0]) % vocab_size)
                y.append(int(b[1]) % vocab_size)
            if len(x) == batch_size:
                return {"source": "shards", "x": x, "y": y}

        # Deterministic synthetic fallback
        x = [int((seed + i * 17) % vocab_size) for i in range(batch_size)]
        y = [int((v + 1) % vocab_size) for v in x]
        return {"source": "synthetic", "x": x, "y": y}

    def _browser_sequence_batch(
        self, batch_size: int, seq_len: int, seed: int, vocab_size: int = 256
    ) -> Dict[str, Any]:
        """Return sequences of length seq_len+1 for next-byte prediction."""
        total_len = seq_len + 1
        shards = self._load_browser_shards()
        if shards:
            rng = random.Random(seed)
            sequences: List[List[int]] = []
            tries = batch_size * 10
            while len(sequences) < batch_size and tries > 0:
                tries -= 1
                i = rng.randint(0, len(self._browser_shard_paths) - 1)
                path = self._browser_shard_paths[i]
                size = self._browser_shard_sizes[i]
                if size < total_len:
                    continue
                off = rng.randint(0, size - total_len)
                with path.open("rb") as f:
                    f.seek(off)
                    chunk = f.read(total_len)
                if len(chunk) != total_len:
                    continue
                sequences.append([int(b) % vocab_size for b in chunk])
            if len(sequences) == batch_size:
                return {"source": "shards", "sequences": sequences, "seq_len": seq_len}

        # Synthetic fallback
        rng = random.Random(seed)
        sequences = []
        for _ in range(batch_size):
            seq = [rng.randint(0, vocab_size - 1) for _ in range(total_len)]
            sequences.append(seq)
        return {"source": "synthetic", "sequences": sequences, "seq_len": seq_len}

    def _load_browser_shards(self) -> bool:
        if self._browser_shard_paths:
            return True
        patterns = self._candidate_shard_globs()
        for pat in patterns:
            files = [Path(p) for p in sorted(glob(pat))]
            if not files:
                continue
            sizes = [os.path.getsize(p) for p in files]
            self._browser_shard_paths = files
            self._browser_shard_sizes = sizes
            logger.info("browser worker data source: %s (%s shards)", pat, len(files))
            return True
        return False

    @staticmethod
    def _candidate_shard_globs() -> List[str]:
        out: List[str] = []
        env_glob = os.environ.get("BBT_SHARD_GLOB")
        if env_glob:
            out.append(env_glob)
        repo_root = Path(__file__).resolve().parents[2]
        out.extend(
            [
                str(repo_root / ".deps" / "datasets" / "wikitext_103" / "shards" / "train" / "shard_*.bin"),
                str(repo_root.parent / "bbt" / "artifacts" / "datasets" / "wikitext_103" / "shards" / "train" / "shard_*.bin"),
                r"D:\code\bbt\artifacts\datasets\wikitext_103\shards\train\shard_*.bin",
            ]
        )
        return out

    @staticmethod
    def _dashboard_html() -> str:
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gabion Training Dashboard</title>
  <style>
    :root {
      --bg: #0b1220;
      --panel: #111a2b;
      --muted: #8da2c0;
      --text: #e7eefc;
      --ok: #3ecf8e;
      --line: #63a1ff;
      --border: #20314f;
    }
    body { margin: 0; font-family: ui-sans-serif, Segoe UI, Arial, sans-serif; background: var(--bg); color: var(--text); }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 18px; }
    .header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 14px; }
    .muted { color: var(--muted); font-size: 13px; }
    .job { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px; margin-bottom: 14px; }
    .row { display: flex; gap: 20px; flex-wrap: wrap; margin: 8px 0 10px; }
    .stat { min-width: 120px; }
    .label { color: var(--muted); font-size: 12px; margin-bottom: 2px; }
    .val { font-size: 18px; font-weight: 600; }
    canvas { width: 100%; height: 140px; background: #0b1527; border: 1px solid var(--border); border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--border); padding: 6px; text-align: left; }
    th { color: var(--muted); font-weight: 600; }
    .subhead { margin-top: 10px; font-size: 12px; color: var(--muted); }
    .ctl { display: flex; gap: 8px; align-items: center; margin: 8px 0; }
    .ctl input { background: #0b1527; color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; width: 110px; }
    .ctl button { background: #1f6feb; color: white; border: 0; border-radius: 6px; padding: 6px 10px; cursor: pointer; }
    .ctl button:disabled { opacity: 0.6; cursor: not-allowed; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h2>Gabion Training Dashboard</h2>
      <div class="muted" id="meta">loading...</div>
    </div>
    <div id="jobs"></div>
  </div>
<script src="/assets/tinygrad_v0.js"></script>
<script>
function fmtLoss(v) { return (v === null || Number.isNaN(v)) ? "n/a" : Number(v).toFixed(4); }
async function setMaxRounds(jobId, value) {
  const r = await fetch(`/jobs/${encodeURIComponent(jobId)}/max-rounds`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({max_rounds: value}),
  });
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(txt || `http_${r.status}`);
  }
  return r.json();
}
function drawLine(canvas, points, key, color) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width = canvas.clientWidth * devicePixelRatio;
  const h = canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  if (!points.length) return;
  const vals = points.map(p => p[key]).filter(v => Number.isFinite(v));
  if (!vals.length) return;
  const min = Math.min(...vals), max = Math.max(...vals);
  const range = Math.max(1e-9, max - min);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = (i / Math.max(1, points.length - 1)) * (canvas.clientWidth - 12) + 6;
    const y = canvas.clientHeight - (((p[key] - min) / range) * (canvas.clientHeight - 20) + 10);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function render(data) {
  const jobs = document.getElementById("jobs");
  const when = new Date(data.generated_at * 1000).toLocaleTimeString();
  document.getElementById("meta").textContent = "updated " + when;
  jobs.innerHTML = "";
  for (const job of data.jobs) {
    const card = document.createElement("div");
    card.className = "job";
    card.innerHTML = `
      <h3>${job.name} <span class="muted">(${job.job_id})</span></h3>
      <div class="row">
        <div class="stat"><div class="label">Target Max Rounds</div><div class="val">${job.max_rounds}</div></div>
        <div class="stat"><div class="label">Last Completed Round</div><div class="val">${job.last_completed_round}</div></div>
        <div class="stat"><div class="label">Current Mean Loss</div><div class="val">${fmtLoss(job.latest_mean_loss)}</div></div>
        <div class="stat"><div class="label">Held-out Eval Loss</div><div class="val">${fmtLoss(job.latest_eval_loss)}</div></div>
      </div>
      <div class="ctl">
        <span class="label">Set Max Rounds</span>
        <input class="max-rounds-input" type="number" min="1" step="1" value="${job.max_rounds}" />
        <button class="set-max-rounds-btn">Apply</button>
      </div>
      <canvas data-kind="train"></canvas>
      <table>
        <thead><tr><th>Round</th><th>Mean Loss</th><th>Participants</th></tr></thead>
        <tbody>
          ${job.recent_history.slice(-10).reverse().map(p => `<tr><td>${p.round_id}</td><td>${fmtLoss(p.mean_loss)}</td><td>${p.participant_count}</td></tr>`).join("")}
        </tbody>
      </table>
      <div class="subhead">Held-out Eval</div>
      <canvas data-kind="eval"></canvas>
      <table>
        <thead><tr><th>Round</th><th>Eval Loss</th><th>Batch</th></tr></thead>
        <tbody>
          ${(job.recent_eval_history || []).slice(-10).reverse().map(p => `<tr><td>${p.round_id}</td><td>${fmtLoss(p.eval_loss)}</td><td>${p.sample_count}</td></tr>`).join("")}
        </tbody>
      </table>
      <div class="subhead">Workers</div>
      <table>
        <thead><tr><th>Worker</th><th>Trainer</th><th>Mode</th><th>Data</th><th>Round</th><th>Loss</th></tr></thead>
        <tbody>
          ${(job.worker_stats || [])
            .slice()
            .sort((a, b) => (b.round_id || 0) - (a.round_id || 0))
            .map(w => `<tr><td>${w.worker_id}</td><td>${w.trainer || w.kind || "n/a"}</td><td>${w.mode || "n/a"}</td><td>${w.data_source || "n/a"}</td><td>${w.round_id || 0}</td><td>${fmtLoss(w.loss)}</td></tr>`)
            .join("")}
        </tbody>
      </table>
    `;
    const setBtn = card.querySelector(".set-max-rounds-btn");
    const maxInput = card.querySelector(".max-rounds-input");
    setBtn.addEventListener("click", async () => {
      const newVal = parseInt(maxInput.value || "0", 10);
      if (!Number.isFinite(newVal) || newVal < 1) return;
      setBtn.disabled = true;
      try {
        await setMaxRounds(job.job_id, newVal);
        await tick();
      } catch (e) {
        document.getElementById("meta").textContent = `failed to set max rounds for ${job.job_id}`;
      } finally {
        setBtn.disabled = false;
      }
    });
    jobs.appendChild(card);
    drawLine(card.querySelector("canvas[data-kind='train']"), job.recent_history || [], "mean_loss", "#63a1ff");
    drawLine(card.querySelector("canvas[data-kind='eval']"), job.recent_eval_history || [], "eval_loss", "#3ecf8e");
  }
}

async function tick() {
  try {
    const r = await fetch("/metrics", {cache: "no-store"});
    const data = await r.json();
    render(data);
  } catch (e) {
    document.getElementById("meta").textContent = "failed to load metrics";
  }
}
tick();
setInterval(tick, 2000);
</script>
</body>
</html>"""

    @staticmethod
    def _webgpu_worker_html() -> str:
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gabion WebGPU Worker</title>
  <style>
    :root {
      --bg: #0b0f1a;
      --panel: #121a2a;
      --text: #e8eeff;
      --muted: #8da0c6;
      --line: #253552;
      --ok: #38d39f;
      --warn: #ffcc66;
      --err: #ff7a7a;
    }
    body { margin: 0; font-family: ui-sans-serif, Segoe UI, Arial, sans-serif; background: var(--bg); color: var(--text); }
    .wrap { max-width: 920px; margin: 0 auto; padding: 16px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; margin-bottom: 12px; }
    .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    label { font-size: 12px; color: var(--muted); display: block; margin-bottom: 4px; }
    input { background: #0d1524; color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 8px; width: 250px; }
    button { background: #1f6feb; color: white; border: 0; border-radius: 6px; padding: 9px 12px; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .pill { font-size: 12px; border: 1px solid var(--line); border-radius: 999px; padding: 2px 8px; color: var(--muted); }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
    .err { color: var(--err); }
    pre { margin: 0; background: #0d1524; border: 1px solid var(--line); border-radius: 8px; padding: 10px; max-height: 380px; overflow: auto; font-size: 12px; line-height: 1.3; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h2>Gabion WebGPU Worker Prototype</h2>
      <div class="row" style="margin-bottom:8px">
        <span class="pill">requires mesh flag: --allow-browser-workers</span>
      </div>
      <div class="row">
        <div>
          <label>Worker ID</label>
          <input id="workerId" value="webgpu-browser-1" />
        </div>
        <div>
          <label>Job ID (optional)</label>
          <input id="jobId" value="" placeholder="bbt-job-v1" />
        </div>
        <div>
          <label>Local Epochs</label>
          <input id="localEpochs" value="1" />
        </div>
        <div>
          <label>Learning Rate</label>
          <input id="learningRate" value="0.0005" />
        </div>
        <div>
          <label>Grad Clip Norm</label>
          <input id="gradClipNorm" value="1.0" />
        </div>
        <div>
          <label>Warmup Steps</label>
          <input id="warmupSteps" value="10" />
        </div>
        <div>
          <label>Train Samples</label>
          <input id="trainSamples" value="2048" />
        </div>
        <div>
          <label>d_model</label>
          <input id="dModel" value="64" />
        </div>
        <div>
          <label>seq_len</label>
          <input id="seqLen" value="32" />
        </div>
        <div>
          <label>n_layers</label>
          <input id="nLayers" value="2" />
        </div>
        <div>
          <label>n_heads</label>
          <input id="nHeads" value="4" />
        </div>
        <div>
          <label>BitLinear (ternarize)</label>
          <input id="ternarize" type="checkbox" />
        </div>
        <div>
          <label>f16 weight transfer</label>
          <input id="useF16" type="checkbox" checked />
        </div>
        <div>
          <label>Debug GPU Sync</label>
          <input id="debugSync" type="checkbox" />
        </div>
        <div>
          <label>GPU Profiling</label>
          <input id="gpuProfile" type="checkbox" />
        </div>
        <button id="connectBtn">Connect</button>
        <button id="disconnectBtn" disabled>Disconnect</button>
        <span class="pill" id="statusPill">idle</span>
      </div>
      <div class="row" style="margin-top:8px">
        <span>WebGPU: <b id="webgpuState" class="warn">checking...</b></span>
        <span>Adapter: <b id="adapterName" class="muted">n/a</b></span>
      </div>
    </div>
    <div class="card">
      <pre id="log"></pre>
    </div>
  </div>
<script src="/assets/tinygrad_v0.js"></script>
<script src="/assets/webgpu_backend.js"></script>
<script src="/assets/bbt_forward.js"></script>
<script>
const logEl = document.getElementById("log");
const statusPill = document.getElementById("statusPill");
const connectBtn = document.getElementById("connectBtn");
const disconnectBtn = document.getElementById("disconnectBtn");
const workerIdEl = document.getElementById("workerId");
const jobIdEl = document.getElementById("jobId");
const localEpochsEl = document.getElementById("localEpochs");
const learningRateEl = document.getElementById("learningRate");
const gradClipNormEl = document.getElementById("gradClipNorm");
const warmupStepsEl = document.getElementById("warmupSteps");
const trainSamplesEl = document.getElementById("trainSamples");
const dModelEl = document.getElementById("dModel");
const seqLenEl = document.getElementById("seqLen");
const nLayersEl = document.getElementById("nLayers");
const nHeadsEl = document.getElementById("nHeads");
const ternarizeEl = document.getElementById("ternarize");
const useF16El = document.getElementById("useF16");
const debugSyncEl = document.getElementById("debugSync");
const gpuProfileEl = document.getElementById("gpuProfile");
const webgpuState = document.getElementById("webgpuState");
const adapterNameEl = document.getElementById("adapterName");

let ws = null;
let heartbeatTimer = null;
let gpuDevice = null;

function log(msg, cls = "") {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logEl.textContent += line + "\\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(text) {
  statusPill.textContent = text;
}

function makeMessage(type, payload) {
  return { type, payload };
}

function hashStr(s) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function() {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function targetForIndex(i, seedBase) {
  const x = Math.sin((i + 1) * 0.013 + seedBase * 0.0001);
  return 0.05 * x;
}

function surrogateTrain(weights, opts) {
  const n = weights.length;
  if (!n) return { weights, loss: 0.0 };
  const lr = opts.lr;
  const epochs = opts.epochs;
  const samples = Math.max(32, Math.min(opts.samples, n));
  const rng = mulberry32(opts.seed);

  let lossAcc = 0.0;
  let cnt = 0;
  for (let e = 0; e < epochs; e++) {
    for (let k = 0; k < samples; k++) {
      const idx = Math.floor(rng() * n);
      const t = targetForIndex(idx, opts.seedBase);
      const w = weights[idx];
      const grad = (w - t);
      weights[idx] = w - lr * grad;
      lossAcc += grad * grad;
      cnt += 1;
    }
  }
  const loss = cnt > 0 ? (lossAcc / cnt) : 0.0;
  return { weights, loss };
}

async function loadWGSLShaders() {
  const kernels = ["matmul", "elementwise", "reduce", "softmax", "rope", "batched_matmul", "fused_attention", "softmax_backward", "rope_backward", "batched_transpose", "rmsnorm_backward", "adam_update", "sgd_update", "cross_entropy_forward", "cross_entropy_backward", "embedding_forward", "silu_mul", "silu_mul_backward"];
  for (const name of kernels) {
    try {
      const r = await fetch(`/assets/kernels/${name}.wgsl`, { cache: "no-store" });
      if (r.ok) {
        const src = await r.text();
        WebGPUBackend.registerShader(name, src);
        log(`loaded kernel: ${name}.wgsl`);
      } else {
        log(`failed to load kernel ${name}.wgsl: ${r.status}`, "warn");
      }
    } catch (e) {
      log(`failed to fetch kernel ${name}.wgsl: ${e}`, "warn");
    }
  }
}

async function initWebGPU() {
  if (!navigator.gpu) {
    webgpuState.textContent = "not supported";
    webgpuState.className = "err";
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      webgpuState.textContent = "adapter unavailable";
      webgpuState.className = "err";
      return false;
    }
    const requiredFeatures = [];
    if (adapter.features.has("timestamp-query")) requiredFeatures.push("timestamp-query");
    gpuDevice = await adapter.requestDevice({ requiredFeatures });
    const info = adapter.info || {};
    adapterNameEl.textContent = info.description || info.vendor || "webgpu-adapter";

    // Initialize WebGPU backend and load WGSL shaders
    WebGPUBackend.init(gpuDevice);
    await loadWGSLShaders();

    webgpuState.textContent = "ready";
    webgpuState.className = "ok";
    log("WebGPU backend initialized with compute shaders");
    return true;
  } catch (e) {
    webgpuState.textContent = "init failed";
    webgpuState.className = "err";
    log(`WebGPU init error: ${e}`, "err");
    return false;
  }
}

async function webgpuTick() {
  if (!gpuDevice) return;
  const data = new Float32Array([1, 2, 3, 4]);
  const inBuf = gpuDevice.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(inBuf.getMappedRange()).set(data);
  inBuf.unmap();
  const outBuf = gpuDevice.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readBuf = gpuDevice.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const shader = gpuDevice.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read> inputData: array<f32>;
      @group(0) @binding(1) var<storage, read_write> outputData: array<f32>;
      @compute @workgroup_size(1)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let i = id.x;
        outputData[i] = inputData[i] * 1.0001;
      }
    `
  });
  const pipeline = gpuDevice.createComputePipeline({
    layout: "auto",
    compute: { module: shader, entryPoint: "main" },
  });
  const bindGroup = gpuDevice.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inBuf } },
      { binding: 1, resource: { buffer: outBuf } },
    ],
  });
  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(4);
  pass.end();
  encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, data.byteLength);
  gpuDevice.queue.submit([encoder.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();
  return Number(out[0]);
}

function wsUrl() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws`;
}

function send(type, payload) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify(makeMessage(type, payload)));
}

function startHeartbeat(workerId) {
  stopHeartbeat();
  heartbeatTimer = setInterval(() => {
    send("heartbeat", { worker_id: workerId });
  }, 5000);
}

function stopHeartbeat() {
  if (heartbeatTimer) clearInterval(heartbeatTimer);
  heartbeatTimer = null;
}

async function handleRoundStart(payload, workerId) {
  const roundId = Number(payload.round_id);
  const roundToken = String(payload.round_token || "");
  const modelVersion = Number(payload.model_version ?? -1);
  const jobId = String(payload.job_id);
  // Decode weights: support f16 base64 or plain JSON array
  let weights;
  if (payload.weights_format === "f16_base64" && payload.weights_f16) {
    weights = window.tinygradV0.f16Base64ToWeights(payload.weights_f16);
  } else {
    const weightsArr = (payload.weights || []).map(Number);
    weights = Float32Array.from(weightsArr);
  }
  const epochs = Math.max(1, parseInt(localEpochsEl.value || "1", 10) || 1);
  const lr = Math.max(1e-7, parseFloat(payload.learning_rate ?? learningRateEl.value ?? "0.0005") || 0.0005);
  const gradClipNorm = Math.max(0, parseFloat(payload.grad_clip_norm ?? gradClipNormEl.value ?? "1.0") || 0);
  const warmupSteps = Math.max(0, parseInt(payload.warmup_steps ?? warmupStepsEl.value ?? "10", 10) || 0);
  const optimizerName = String(payload.optimizer || "adam");
  const adamBeta1 = parseFloat(payload.adam_beta1 ?? 0.9) || 0.9;
  const adamBeta2 = parseFloat(payload.adam_beta2 ?? 0.999) || 0.999;
  const samples = Math.max(32, parseInt(trainSamplesEl.value || "2048", 10) || 2048);
  const dModel = Math.max(8, parseInt(dModelEl.value || "64", 10) || 64);
  const seqLen = Math.max(4, parseInt(seqLenEl.value || "32", 10) || 32);
  const nLayers = Math.max(1, parseInt(nLayersEl.value || "2", 10) || 2);
  const nHeads = Math.max(1, parseInt(nHeadsEl.value || "4", 10) || 4);
  const ternarize = !!ternarizeEl.checked;
  const debugSync = !!debugSyncEl.checked;
  const useF16 = !!useF16El.checked;
  const gpuProfile = !!gpuProfileEl.checked;
  const seedBase = hashStr(`${jobId}:${workerId}`);
  const trainSeed = (seedBase ^ (roundId * 2654435761)) >>> 0;

  let gpuProbe = 0.0;
  try {
    const v = await webgpuTick();
    if (Number.isFinite(v)) gpuProbe = v;
  } catch (e) {
    log(`round ${roundId}: webgpu tick failed: ${e}`);
  }

  let trained = null;
  let serverBatch = null;
  let serverSequences = null;

  // Try to fetch sequence batch for v1 transformer training
  try {
    const batchCount = Math.min(16, Math.max(2, Math.floor(samples / seqLen)));
    const r = await fetch(`/data/byte-batch?batch_size=${encodeURIComponent(batchCount)}&seed=${encodeURIComponent(trainSeed)}&seq_len=${encodeURIComponent(seqLen)}`, { cache: "no-store" });
    if (r.ok) {
      const j = await r.json();
      if (Array.isArray(j.sequences) && j.sequences.length > 0) {
        serverSequences = j.sequences;
        serverBatch = { source: String(j.source || "unknown") };
      } else if (Array.isArray(j.x) && Array.isArray(j.y) && j.x.length === j.y.length && j.x.length > 0) {
        serverBatch = {
          x: Int32Array.from(j.x.map(v => Number(v) | 0)),
          y: Int32Array.from(j.y.map(v => Number(v) | 0)),
          source: String(j.source || "unknown"),
        };
      }
    }
  } catch (e) {
    log(`round ${roundId}: failed to fetch batch: ${e}`);
  }

  if (window.tinygradV0) {
    // Try v1 full transformer first
    if (!trained && typeof window.tinygradV0.trainLocalV1 === "function") {
      try {
        trained = await window.tinygradV0.trainLocalV1(weights, {
          lr,
          epochs,
          batchSize: Math.min(16, Math.max(2, Math.floor(samples / seqLen))),
          seed: trainSeed,
          vocabSize: 256,
          dModel: dModel,
          seqLen: seqLen,
          nLayers: nLayers,
          nHeads: nHeads,
          dFF: dModel * 4,
          tieWeights: true,
          ternarize: ternarize,
          sequences: serverSequences,
          debugSync: debugSync,
          useF16: useF16,
          profile: gpuProfile,
          gradClipNorm: gradClipNorm,
          warmupSteps: warmupSteps,
          optimizer: optimizerName,
          adamBeta1: adamBeta1,
          adamBeta2: adamBeta2,
        });
        log(`round ${roundId}: v1 transformer training complete`);
      } catch (e) {
        log(`round ${roundId}: v1 transformer failed, trying v0: ${e}`);
      }
    }

    // Fall back to v0 (embedding-only)
    if (!trained) {
      const trainOpts = {
        lr,
        epochs,
        batchSize: Math.min(256, samples),
        seed: trainSeed,
        vocabSize: 256,
        dModel: dModel,
        batch: serverBatch && serverBatch.x ? { x: serverBatch.x, y: serverBatch.y } : null,
        debugSync: debugSync,
      };
      if (typeof window.tinygradV0.trainLocalV0Async === "function" && WebGPUBackend && WebGPUBackend.instance) {
        try {
          trained = await window.tinygradV0.trainLocalV0Async(weights, trainOpts);
        } catch (e) {
          log(`round ${roundId}: async GPU trainer failed, trying CPU: ${e}`);
        }
      }
      if (!trained && typeof window.tinygradV0.trainLocalV0 === "function") {
        try {
          trained = window.tinygradV0.trainLocalV0(weights, trainOpts);
        } catch (e) {
          log(`round ${roundId}: v0 trainer failed, fallback to surrogate: ${e}`);
        }
      }
    }
  }
  if (!trained) {
    const s = surrogateTrain(weights, {
      lr,
      epochs,
      samples,
      seed: trainSeed,
      seedBase,
    });
    trained = {
      updated: s.weights,
      loss: s.loss,
      sampleCount: epochs * Math.max(32, Math.min(samples, weights.length || samples)),
      mode: "surrogate-fallback",
    };
  }

  const src = serverBatch ? serverBatch.source : "synthetic";
  const probePenalty = Number.isFinite(gpuProbe) ? Math.abs(gpuProbe - 1.0) * 0.001 : 0.001;
  const loss = Number(trained.loss || 0) + probePenalty;
  const sampleCount = Number(trained.sampleCount || 32);
  // Build result payload — use f16 encoding if available
  const resultPayload = {
    job_id: jobId,
    round_id: roundId,
    round_token: roundToken,
    model_version: modelVersion,
    sample_count: sampleCount,
    metrics: { loss: loss },
    worker_meta: {
      mode: String(trained.mode || "unknown"),
      data_source: String(src),
    },
  };
  if (trained.weights_format === "int8_delta" && trained.weights_delta) {
    resultPayload.weights_delta = trained.weights_delta;
    resultPayload.delta_scale = trained.delta_scale;
    resultPayload.weights_format = "int8_delta";
    resultPayload.weights_count = trained.weights_count;
  } else if (trained.weights_format === "f16_base64" && trained.weights_f16) {
    resultPayload.weights_f16 = trained.weights_f16;
    resultPayload.weights_format = "f16_base64";
    resultPayload.weights_count = trained.weights_count;
  } else {
    resultPayload.weights = Array.from(trained.updated || weights);
  }
  send("round_result", resultPayload);
  log(`round ${roundId}: submitted update mode=${trained.mode || "unknown"} data=${src} n=${weights.length} epochs=${epochs} lr=${lr} loss=${loss.toFixed(6)}`);
}

async function connect() {
  const workerId = workerIdEl.value.trim() || "webgpu-browser-1";
  const preferredJobId = jobIdEl.value.trim();
  connectBtn.disabled = true;
  disconnectBtn.disabled = false;
  setStatus("connecting");
  const ok = await initWebGPU();
  if (!ok) {
    log("WebGPU unavailable. Worker can still connect but prototype expects WebGPU.", "warn");
  }

  ws = new WebSocket(wsUrl());
  ws.onopen = () => {
    setStatus("connected");
    log(`connected to ${wsUrl()}`);
    send("register", {
      worker_id: workerId,
      capabilities: { trainer: "browser-webgpu-proto", kind: "webgpu-browser" },
    });
    send("list_jobs", {});
    startHeartbeat(workerId);
  };

  ws.onmessage = async (ev) => {
    let msg = null;
    try { msg = JSON.parse(ev.data); } catch { return; }
    const type = msg.type;
    const payload = msg.payload || {};
    if (type === "registered") {
      log(`registered as ${payload.worker_id}`);
      return;
    }
    if (type === "job_list") {
      const jobs = payload.jobs || [];
      if (!jobs.length) {
        log("no jobs available");
        return;
      }
      let selected = jobs[0];
      if (preferredJobId) {
        const found = jobs.find(j => String(j.job_id) === preferredJobId);
        if (found) selected = found;
      }
      send("join_job", { job_id: selected.job_id });
      log(`joining job ${selected.job_id}`);
      return;
    }
    if (type === "job_joined") {
      log(`joined job ${payload.job_id}`);
      return;
    }
    if (type === "job_rejected") {
      log(`job rejected: ${payload.reason || "unknown"}`, "err");
      return;
    }
    if (type === "round_start") {
      await handleRoundStart(payload, workerId);
      return;
    }
    if (type === "round_summary") {
      log(`summary round=${payload.round_id} mean_loss=${Number(payload.mean_loss || 0).toFixed(4)} participants=${payload.participant_count}`);
      return;
    }
    if (type === "error") {
      log(`server error: ${payload.reason || "unknown"}`, "err");
      return;
    }
  };

  ws.onerror = (e) => {
    log("websocket error", "err");
  };

  ws.onclose = () => {
    stopHeartbeat();
    setStatus("closed");
    connectBtn.disabled = false;
    disconnectBtn.disabled = true;
    log("disconnected");
  };
}

function disconnect() {
  stopHeartbeat();
  if (ws) ws.close();
  ws = null;
  connectBtn.disabled = false;
  disconnectBtn.disabled = true;
  setStatus("idle");
}

connectBtn.addEventListener("click", connect);
disconnectBtn.addEventListener("click", disconnect);
setStatus("idle");
</script>
</body>
</html>"""


def run_mesh(config: MeshConfig, jobs: List[TrainingJob]) -> None:
    server = MeshServer(config=config, jobs=jobs)
    web.run_app(server.app, host=config.host, port=config.port)
