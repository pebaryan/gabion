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
        self._gemma4 = None
        self._gemma4_tok = None
        import os
        self._shard_idx = int(os.environ.get("GEMMA4_SHARD", "0")) if os.environ.get("GEMMA4_NUM_SHARDS") else 0
        self._num_shards = int(os.environ.get("GEMMA4_NUM_SHARDS", "1"))
        # align mesh shard count with env; if pipeline enabled, num_shards=2
        if os.environ.get("GEMMA4_PIPELINE", "0") == "1":
            self._num_shards = 2

    def _get_gemma4(self):
        if self._gemma4 is not None:
            return self._gemma4, self._gemma4_tok
        # lazy load adapter + tokenizer on first infer, keeps E2B GGUF as-is
        try:
            from pathlib import Path
            import os

            gguf = os.environ.get("GEMMA4_GGUF", self.config.gemma4_gguf)
            from gabion.user_models.gemma4_text import Gemma4TextAdapter

            if self._num_shards > 1:
                self._gemma4 = Gemma4TextAdapter.from_gguf_shard(gguf, self._shard_idx, self._num_shards)
                shard_tag = f"{self._shard_idx}/{self._num_shards} layers {self._gemma4.layer_start}-{self._gemma4.layer_end-1}"
            else:
                self._gemma4 = Gemma4TextAdapter.from_gguf(gguf)
                shard_tag = "full"
            # tokenizer
            tok_path = Path(self.config.gemma4_tokenizer_path)
            if tok_path.exists():
                from tokenizers import Tokenizer

                self._gemma4_tok = Tokenizer.from_file(str(tok_path))
            else:
                from transformers import AutoTokenizer

                # Fallback only: the local tokenizer.json (config.gemma4_tokenizer_path)
                # is the primary path. trust_remote_code runs HF-supplied code on this
                # machine — pin a revision if you keep this fallback; consider dropping
                # it entirely once the local file is always provisioned.
                hf_tok = AutoTokenizer.from_pretrained("google/gemma-4-e2b-it", trust_remote_code=True)
                # wrap to mimic tokenizers API
                class _Wrap:
                    def decode(self, ids):
                        return hf_tok.decode(ids, skip_special_tokens=True)

                self._gemma4_tok = _Wrap()
            logger.info("pebble %s loaded gemma4 %s shard=%s", self.config.worker_id, gguf, shard_tag)
        except Exception as exc:
            logger.warning("pebble %s gemma4 load failed: %s", self.config.worker_id, exc)
            raise
        return self._gemma4, self._gemma4_tok

    async def _handle_infer_pipeline_request(self, ws, data):
        rid = str(data.get("request_id", ""))
        shard = int(data.get("shard", 0))
        # only handle if matches our shard
        if shard != self._shard_idx and self._num_shards > 1:
            return
        try:
            import base64, numpy as np, asyncio

            def _sync():
                adapter, _tok = self._get_gemma4()
                from tinygrad import Tensor
                from gabion.user_models.gemma4_text import _encode_hidden_f16, _decode_hidden_f16

                if shard == 0:
                    ids = list(data.get("ids", []))
                    inp = Tensor([ids[-2048:] if len(ids) > 2048 else ids])
                    hidden = adapter.forward_shard_ids_to_hidden(inp)
                    arr = hidden.numpy()
                    b64, shape = _encode_hidden_f16(arr)
                    return {"hidden_b64": b64, "hidden_shape": shape}
                else:
                    hidden_b64 = str(data.get("hidden_b64", ""))
                    hidden_shape = list(data.get("hidden_shape", []))
                    arr = _decode_hidden_f16(hidden_b64, hidden_shape)
                    hidden_t = Tensor(arr)
                    logits = adapter.forward_shard_hidden_to_logits(hidden_t)
                    last = logits.numpy()[0, -1]
                    half = last.astype(np.float16)
                    b64 = base64.b64encode(half.tobytes()).decode("ascii")
                    return {"logits_b64": b64, "logits_shape": [int(last.shape[0])]}

            res = await asyncio.to_thread(_sync)
            await ws.send_json(make_message("infer_pipeline_result", {"request_id": rid, **res}))
        except Exception as exc:
            logger.warning("pipeline shard %s failed %s: %s", shard, rid, exc)
            await ws.send_json(make_message("infer_pipeline_result", {"request_id": rid, "error": str(exc)}))

    async def _handle_infer_request(self, ws, data):
        rid = str(data.get("request_id", ""))
        prompts = list(data.get("prompts", []))
        max_tokens = int(data.get("max_tokens", 20))
        max_tokens = max(1, min(64, max_tokens))
        # run blocking gemma4 in thread pool so heartbeat loop stays alive (evict is 30s)
        def _sync_infer():
            adapter, tok = self._get_gemma4()
            from tinygrad import Tensor

            out = []
            for item in prompts:
                pidx = int(item.get("prompt_idx", 0))
                ids = list(item.get("ids", []))
                cur = list(ids)
                for _ in range(max_tokens):
                    inp = Tensor([cur[-2048:] if len(cur) > 2048 else cur])
                    logits = adapter.forward(inp)
                    last = logits.numpy()[0, -1]
                    import numpy as np

                    nxt = int(np.argmax(last))
                    if nxt in (1, 106, 0):
                        break
                    cur.append(nxt)
                    if len(cur) > 2048:
                        break
                gen_ids = cur[len(ids) :]
                try:
                    text = tok.decode(gen_ids)  # type: ignore
                except Exception:
                    text = ""
                out.append({"prompt_idx": pidx, "prompt": item.get("prompt", ""), "generated_ids": gen_ids, "text": text, "worker_id": self.config.worker_id})
            return out

        try:
            out = await asyncio.to_thread(_sync_infer)
            await ws.send_json(make_message("infer_result", {"request_id": rid, "results": out}))
        except Exception as exc:
            logger.warning("infer failed %s: %s", rid, exc)
            await ws.send_json(make_message("infer_result", {"request_id": rid, "results": [{"prompt_idx": p.get("prompt_idx", 0), "error": str(exc)} for p in prompts]}))

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
                                    "capabilities": {
                                        "trainer": self.trainer.backend,
                                        "work_scale": f"{self.config.work_scale:.4f}",
                                        "gemma4_shard": f"{self._shard_idx}/{self._num_shards}" if self._num_shards > 1 else "full",
                                        "gemma4_pipeline": "1" if self._num_shards > 1 else "0",
                                    },
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
                                elif msg_type == "infer_request":
                                    await self._handle_infer_request(ws, data)
                                elif msg_type == "infer_pipeline_request":
                                    await self._handle_infer_pipeline_request(ws, data)
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
        round_token = str(data.get("round_token", ""))
        model_version = int(data.get("model_version", -1))
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
                    "round_token": round_token,
                    "model_version": model_version,
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
