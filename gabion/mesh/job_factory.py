from __future__ import annotations

from gabion.common.jobs import TrainingJob
from gabion.pebble.adapters import flatten_tensors, load_adapter


def build_tinygrad_job(
    job_id: str,
    name: str,
    description: str,
    model_adapter: str,
    local_epochs: int,
    min_quorum: int,
    max_rounds: int,
    seed: int,
) -> TrainingJob:
    try:
        adapter = load_adapter(model_adapter)
        params = adapter.init_params(seed=seed)
        initial_weights = flatten_tensors(params)
    except Exception as exc:
        raise RuntimeError(
            f"failed to initialize model adapter '{model_adapter}'. Ensure tinygrad and adapter are available"
        ) from exc

    return TrainingJob(
        job_id=job_id,
        name=name,
        description=description,
        artifact_uri=f"mesh://bundles/{job_id}",
        artifact_checksum="sha256:dev",
        runtime="tinygrad",
        model_adapter=model_adapter,
        local_epochs=local_epochs,
        min_quorum=min_quorum,
        max_rounds=max_rounds,
        initial_weights=initial_weights,
    )
