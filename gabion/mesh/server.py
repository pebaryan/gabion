from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

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
    last_completed_round: int = 0
    round_history: List[Dict[str, Any]] = field(default_factory=list)


class MeshServer:
    def __init__(self, config: MeshConfig, jobs: List[TrainingJob]) -> None:
        if not jobs:
            raise ValueError("at least one training job is required")

        self.config = config
        self._workers: Dict[str, WorkerSession] = {}
        self._workers_lock = asyncio.Lock()
        self._round_task: asyncio.Task[None] | None = None
        self._jobs: Dict[str, JobRuntime] = {
            job.job_id: JobRuntime(
                job=job,
                weights=list(job.initial_weights),
                coordinator=RoundCoordinator(),
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
        self._save_checkpoint()

    async def health_handler(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def jobs_handler(self, request: web.Request) -> web.Response:
        return web.json_response({"jobs": [runtime.job.to_dict() for runtime in self._jobs.values()]})

    async def metrics_handler(self, request: web.Request) -> web.Response:
        return web.json_response(self._metrics_payload())

    async def dashboard_handler(self, request: web.Request) -> web.Response:
        return web.Response(text=self._dashboard_html(), content_type="text/html")

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
                    "latest_mean_loss": self._latest_mean_loss(runtime),
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
            start_round = runtime.last_completed_round + 1
            if start_round > runtime.job.max_rounds:
                logger.info(
                    "job=%s already completed (%s/%s), skipping",
                    runtime.job.job_id,
                    runtime.last_completed_round,
                    runtime.job.max_rounds,
                )
                continue

            for round_id in range(start_round, runtime.job.max_rounds + 1):
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
                runtime.last_completed_round = round_id

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
                if round_id % max(1, self.config.checkpoint_every_rounds) == 0:
                    self._save_checkpoint()

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
                    "weights": runtime.weights,
                    "round_history": runtime.round_history,
                }
                for job_id, runtime in self._jobs.items()
            },
        }
        try:
            ckpt_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as exc:
            logger.warning("failed to save checkpoint %s: %s", ckpt_path, exc)

    def _metrics_payload(self) -> Dict[str, Any]:
        jobs: List[Dict[str, Any]] = []
        for job_id, runtime in self._jobs.items():
            jobs.append(
                {
                    "job_id": job_id,
                    "name": runtime.job.name,
                    "last_completed_round": runtime.last_completed_round,
                    "latest_mean_loss": self._latest_mean_loss(runtime),
                    "recent_history": runtime.round_history[-200:],
                }
            )
        return {"generated_at": time.time(), "jobs": jobs}

    @staticmethod
    def _latest_mean_loss(runtime: JobRuntime) -> float | None:
        if not runtime.round_history:
            return None
        return float(runtime.round_history[-1]["mean_loss"])

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
<script>
function fmtLoss(v) { return (v === null || Number.isNaN(v)) ? "n/a" : Number(v).toFixed(4); }
function drawLine(canvas, points) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width = canvas.clientWidth * devicePixelRatio;
  const h = canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  if (!points.length) return;
  const vals = points.map(p => p.mean_loss).filter(v => Number.isFinite(v));
  if (!vals.length) return;
  const min = Math.min(...vals), max = Math.max(...vals);
  const range = Math.max(1e-9, max - min);
  ctx.strokeStyle = "#63a1ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = (i / Math.max(1, points.length - 1)) * (canvas.clientWidth - 12) + 6;
    const y = canvas.clientHeight - (((p.mean_loss - min) / range) * (canvas.clientHeight - 20) + 10);
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
        <div class="stat"><div class="label">Last Completed Round</div><div class="val">${job.last_completed_round}</div></div>
        <div class="stat"><div class="label">Current Mean Loss</div><div class="val">${fmtLoss(job.latest_mean_loss)}</div></div>
      </div>
      <canvas></canvas>
      <table>
        <thead><tr><th>Round</th><th>Mean Loss</th><th>Participants</th></tr></thead>
        <tbody>
          ${job.recent_history.slice(-10).reverse().map(p => `<tr><td>${p.round_id}</td><td>${fmtLoss(p.mean_loss)}</td><td>${p.participant_count}</td></tr>`).join("")}
        </tbody>
      </table>
    `;
    jobs.appendChild(card);
    drawLine(card.querySelector("canvas"), job.recent_history);
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


def run_mesh(config: MeshConfig, jobs: List[TrainingJob]) -> None:
    server = MeshServer(config=config, jobs=jobs)
    web.run_app(server.app, host=config.host, port=config.port)
