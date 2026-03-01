from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeshConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    round_timeout_s: float = 8.0
    local_epochs: int = 1
    max_rounds: int = 3
    min_quorum: int = 1
    heartbeat_timeout_s: float = 30.0
    checkpoint_path: str | None = None
    checkpoint_every_rounds: int = 1
    allow_browser_workers: bool = False
    async_participation: bool = False
    eval_every_rounds: int = 0
    eval_batch_size: int = 32
    eval_seed: int = 1337


@dataclass(frozen=True)
class PebbleConfig:
    worker_id: str
    mesh_ws_url: str = "ws://127.0.0.1:8765/ws"
    heartbeat_interval_s: float = 5.0
    preferred_job_id: str | None = None
    work_scale: float = 1.0
