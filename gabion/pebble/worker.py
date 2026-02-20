from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Dict, List

from aiohttp import ClientSession, WSMsgType

from gabion.common.config import PebbleConfig
from gabion.common.protocol import make_message
from gabion.pebble.adapters import load_adapter
from gabion.pebble.trainer import Trainer

logger = logging.getLogger(__name__)


class PebbleWorker:
    def __init__(self, config: PebbleConfig, trainer: Trainer) -> None:
        self.config = config
        self.trainer = trainer
        self._joined_job_id: str | None = None

    async def run(self) -> None:
        while True:
            try:
                async with ClientSession() as session:
                    async with session.ws_connect(self.config.mesh_ws_url) as ws:
                        await ws.send_json(
                            make_message(
                                "register",
                                {
                                    "worker_id": self.config.worker_id,
                                    "capabilities": {"trainer": self.trainer.backend},
                                },
                            )
                        )
                        await ws.send_json(make_message("list_jobs", {}))
                        logger.info("worker %s connected", self.config.worker_id)
                        hb_task = asyncio.create_task(self._heartbeat_loop(ws))
                        try:
                            async for msg in ws:
                                if msg.type != WSMsgType.TEXT:
                                    continue
                                payload = msg.json()
                                msg_type = payload.get("type")
                                data = payload.get("payload", {})
                                if msg_type == "job_list":
                                    await self._handle_job_list(ws, data)
                                elif msg_type == "job_joined":
                                    self._joined_job_id = str(data.get("job_id"))
                                    logger.info(
                                        "worker %s joined job %s",
                                        self.config.worker_id,
                                        self._joined_job_id,
                                    )
                                elif msg_type == "job_rejected":
                                    logger.warning(
                                        "worker %s rejected from job %s: %s",
                                        self.config.worker_id,
                                        data.get("job_id"),
                                        data.get("reason"),
                                    )
                                elif msg_type == "artifact_required":
                                    logger.info(
                                        "worker %s needs artifact %s checksum=%s for job %s",
                                        self.config.worker_id,
                                        data.get("artifact_uri"),
                                        data.get("artifact_checksum"),
                                        data.get("job_id"),
                                    )
                                elif msg_type == "round_start":
                                    await self._handle_round_start(ws, data)
                                elif msg_type == "round_summary":
                                    logger.info(
                                        "worker %s job=%s round=%s summary loss=%.4f",
                                        self.config.worker_id,
                                        data.get("job_id"),
                                        data.get("round_id"),
                                        float(data.get("mean_loss", 0.0)),
                                    )
                        finally:
                            hb_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await hb_task
            except Exception as exc:
                logger.warning("worker %s reconnecting after error: %s", self.config.worker_id, exc)
                await asyncio.sleep(1.0)

    async def _handle_job_list(self, ws, data: Dict[str, Any]) -> None:
        jobs = list(data.get("jobs", []))
        if not jobs:
            logger.warning("worker %s received empty job list", self.config.worker_id)
            return

        selected_job: Dict[str, Any] | None = None
        if self.config.preferred_job_id:
            for job in jobs:
                if str(job.get("job_id")) == self.config.preferred_job_id:
                    selected_job = job
                    break
        if selected_job is None:
            selected_job = jobs[0]

        runtime = str(selected_job.get("runtime", ""))
        adapter_ref = str(selected_job.get("model_adapter", ""))
        if runtime == "tinygrad" and self.trainer.backend != "tinygrad":
            logger.warning(
                "worker %s cannot join tinygrad job %s without tinygrad runtime",
                self.config.worker_id,
                selected_job.get("job_id"),
            )
            return
        if adapter_ref:
            try:
                load_adapter(adapter_ref)
            except Exception:
                logger.warning(
                    "worker %s cannot load adapter %s for job %s",
                    self.config.worker_id,
                    adapter_ref,
                    selected_job.get("job_id"),
                )
                return

        await ws.send_json(make_message("join_job", {"job_id": selected_job["job_id"]}))

    async def _handle_round_start(self, ws, data: Dict[str, Any]) -> None:
        job_id = str(data["job_id"])
        if self._joined_job_id and job_id != self._joined_job_id:
            return

        round_id = int(data["round_id"])
        weights = [float(v) for v in data["weights"]]
        local_epochs = int(data.get("local_epochs", 1))

        new_weights, sample_count, loss = self.trainer.train(
            weights=weights,
            local_epochs=local_epochs,
            job=data,
        )
        await ws.send_json(
            make_message(
                "round_result",
                {
                    "job_id": job_id,
                    "round_id": round_id,
                    "sample_count": sample_count,
                    "weights": new_weights,
                    "metrics": {"loss": loss},
                },
            )
        )

    async def _heartbeat_loop(self, ws) -> None:
        while True:
            await ws.send_json(make_message("heartbeat", {"worker_id": self.config.worker_id}))
            await asyncio.sleep(self.config.heartbeat_interval_s)
