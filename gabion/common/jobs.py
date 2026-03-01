from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class TrainingJob:
    job_id: str
    name: str
    description: str
    artifact_uri: str
    artifact_checksum: str
    runtime: str
    model_adapter: str
    local_epochs: int
    min_quorum: int
    max_rounds: int
    initial_weights: List[float]
    # Optimizer config
    learning_rate: float = 5e-4
    optimizer: str = "adam"            # "adam" or "sgd"
    grad_clip_norm: float = 1.0       # 0 = disabled
    warmup_steps: int = 10
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "description": self.description,
            "artifact_uri": self.artifact_uri,
            "artifact_checksum": self.artifact_checksum,
            "runtime": self.runtime,
            "model_adapter": self.model_adapter,
            "local_epochs": self.local_epochs,
            "min_quorum": self.min_quorum,
            "max_rounds": self.max_rounds,
            "weights_dim": len(self.initial_weights),
        }
