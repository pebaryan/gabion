from __future__ import annotations

import pytest

from gabion.pebble.trainer import TinygradTrainer


def _has_tinygrad() -> bool:
    try:
        import tinygrad  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_tinygrad(), reason="tinygrad not installed")
def test_tinygrad_mnist_style_training_shape() -> None:
    trainer = TinygradTrainer(sample_count=32, seed=1)
    dims = TinygradTrainer.MNIST_WEIGHT_DIM
    start = [0.0] * dims

    updated, sample_count, loss = trainer.train(start, local_epochs=1)

    assert len(updated) == dims
    assert sample_count >= 32
    assert loss >= 0.0
