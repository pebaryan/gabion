from __future__ import annotations

from pathlib import Path

from gabion.common.config import MeshConfig
from gabion.common.jobs import TrainingJob
from gabion.mesh.server import MeshServer


class _FakeLoss:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeParam:
    def __init__(self) -> None:
        self.requires_grad = True


class _FakeAdapter:
    def __init__(self) -> None:
        self.sample_calls = 0

    def init_params(self, seed: int):
        _ = seed
        return [_FakeParam()]

    def sample_batch(self, batch_size: int, seed: int):
        self.sample_calls += 1
        return [batch_size, seed], [0]

    def forward(self, params, x):
        _ = (params, x)
        return [0.0]

    def loss(self, logits, y):
        _ = (logits, y)
        return _FakeLoss(1.2345)


def _job() -> TrainingJob:
    return TrainingJob(
        job_id="job-1",
        name="Job 1",
        description="test",
        artifact_uri="mesh://bundles/job-1",
        artifact_checksum="sha256:test",
        runtime="tinygrad",
        model_adapter="gabion.user_models.linear:LinearAdapter",
        local_epochs=1,
        min_quorum=1,
        max_rounds=3,
        initial_weights=[0.0, 0.0],
    )


def test_eval_runs_on_configured_cadence(monkeypatch) -> None:
    server = MeshServer(
        config=MeshConfig(eval_every_rounds=2, eval_batch_size=7),
        jobs=[_job()],
    )
    runtime = server._jobs["job-1"]

    adapter = _FakeAdapter()
    load_calls = {"n": 0}

    def _fake_load_adapter(ref: str):
        _ = ref
        load_calls["n"] += 1
        return adapter

    def _fake_unflatten(flat, template, tensor_cls):
        _ = (flat, template, tensor_cls)
        return [_FakeParam()]

    monkeypatch.setattr("gabion.pebble.adapters.load_adapter", _fake_load_adapter)
    monkeypatch.setattr("gabion.pebble.adapters.unflatten_to_tensors", _fake_unflatten)

    class _FakeTensor:
        pass

    monkeypatch.setattr("tinygrad.Tensor", _FakeTensor)

    server._maybe_run_eval(runtime, round_id=1)
    assert runtime.eval_history == []

    server._maybe_run_eval(runtime, round_id=2)
    assert len(runtime.eval_history) == 1
    assert runtime.eval_history[0]["round_id"] == 2
    assert runtime.eval_history[0]["sample_count"] == 7
    assert abs(runtime.eval_history[0]["eval_loss"] - 1.2345) < 1e-9
    assert load_calls["n"] == 1

    server._maybe_run_eval(runtime, round_id=4)
    assert len(runtime.eval_history) == 2
    assert load_calls["n"] == 1
    assert adapter.sample_calls == 2


def test_checkpoint_roundtrip_includes_eval_history(tmp_path: Path) -> None:
    ckpt = tmp_path / "mesh-ckpt.json"

    server_a = MeshServer(
        config=MeshConfig(checkpoint_path=str(ckpt)),
        jobs=[_job()],
    )
    runtime_a = server_a._jobs["job-1"]
    runtime_a.eval_history = [
        {"round_id": 2, "eval_loss": 1.1, "sample_count": 32, "timestamp": 1000.0},
        {"round_id": 3, "eval_loss": 1.0, "sample_count": 32, "timestamp": 1010.0},
    ]
    server_a._save_checkpoint()
    assert not list(tmp_path.glob("mesh-ckpt.json.*.tmp"))

    server_b = MeshServer(
        config=MeshConfig(checkpoint_path=str(ckpt)),
        jobs=[_job()],
    )
    runtime_b = server_b._jobs["job-1"]
    assert len(runtime_b.eval_history) == 2
    assert runtime_b.eval_history[-1]["round_id"] == 3
    assert abs(runtime_b.eval_history[-1]["eval_loss"] - 1.0) < 1e-9


def test_set_job_max_rounds_updates_runtime() -> None:
    server = MeshServer(
        config=MeshConfig(),
        jobs=[_job()],
    )
    runtime = server._jobs["job-1"]
    runtime.last_completed_round = 9

    updated = server._set_job_max_rounds("job-1", 20)
    assert updated == 20
    assert runtime.target_max_rounds == 20

    updated = server._set_job_max_rounds("job-1", 5)
    assert updated == 9
    assert runtime.target_max_rounds == 9


def test_job_list_payload_uses_runtime_round_fields() -> None:
    server = MeshServer(
        config=MeshConfig(),
        jobs=[_job()],
    )
    runtime = server._jobs["job-1"]
    runtime.target_max_rounds = 123
    runtime.model_version = 11
    runtime.last_completed_round = 77

    jobs = server._job_list_payload()
    assert len(jobs) == 1
    entry = jobs[0]
    assert entry["job_id"] == "job-1"
    assert entry["max_rounds"] == 123
    assert entry["model_version"] == 11
    assert entry["last_completed_round"] == 77
