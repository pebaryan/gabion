from __future__ import annotations

import inspect

from gabion.pebble.autograd import parameter_grads, training_mode
from gabion.pebble.trainer import TinygradTrainer
from gabion.pebble.adapters import flatten_tensors, load_adapter


def test_package_does_not_set_requires_grad() -> None:
    from gabion.pebble import trainer
    from gabion.mesh import server

    assert "requires_grad" not in inspect.getsource(trainer)
    assert "requires_grad" not in inspect.getsource(server)


def test_parameter_grads_linear_no_requires_grad() -> None:
    from tinygrad import Tensor

    adapter = load_adapter("gabion.user_models.linear:LinearAdapter")
    params = adapter.init_params(seed=1)
    for p in params:
        if hasattr(p, "requires_grad"):
            p.requires_grad = False
    x, y = adapter.sample_batch(batch_size=8, seed=2)
    with training_mode():
        loss = adapter.loss(adapter.forward(params, x), y)
        grads = parameter_grads(loss, params)
    assert len(grads) == len(params)
    assert all(g.shape == p.shape for g, p in zip(grads, params))
    assert any(abs(float(g.numpy().sum())) > 0 for g in grads)
    _ = Tensor  # keep import used if type checkers complain


def test_tinygrad_trainer_sgd_updates_without_requires_grad() -> None:
    trainer = TinygradTrainer(sample_count=16, seed=1, learning_rate=0.05)
    adapter = load_adapter("gabion.user_models.linear:LinearAdapter")
    start = flatten_tensors(adapter.init_params(seed=1))
    updated, sample_count, loss = trainer.train(
        start,
        local_epochs=2,
        job={
            "model_adapter": "gabion.user_models.linear:LinearAdapter",
            "optimizer": "sgd",
            "grad_clip_norm": 0.0,
            "work_scale": 1.0,
        },
    )
    assert len(updated) == len(start)
    assert sample_count >= 8
    assert loss >= 0.0
    assert updated != start
