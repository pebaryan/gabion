from __future__ import annotations

import pytest
import shutil

from gabion.pebble.adapters import flatten_tensors, load_adapter
from gabion.pebble.trainer import TinygradTrainer


def _has_tinygrad() -> bool:
    try:
        import tinygrad  # type: ignore  # noqa: F401
        import numpy  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _has_clang() -> bool:
    return shutil.which("clang") is not None


@pytest.mark.skipif(not (_has_tinygrad() and _has_clang()), reason="tinygrad runtime unavailable")
def test_tinygrad_mnist_style_training_shape() -> None:
    trainer = TinygradTrainer(sample_count=32, seed=1)
    adapter = load_adapter("gabion.user_models.mnist_softmax:MnistSoftmaxAdapter")
    start = flatten_tensors(adapter.init_params(seed=1))

    updated, sample_count, loss = trainer.train(
        start,
        local_epochs=1,
        job={"model_adapter": "gabion.user_models.mnist_softmax:MnistSoftmaxAdapter"},
    )

    assert len(updated) == len(start)
    assert sample_count >= 32
    assert loss >= 0.0
