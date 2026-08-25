from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator, List, Sequence


@contextmanager
def training_mode() -> Iterator[None]:
    """Enable tinygrad training across 0.12 (Tensor.train) and 0.14 (TRAINING)."""
    from tinygrad import Tensor  # type: ignore

    train = getattr(Tensor, "train", None)
    if callable(train):
        with train():
            yield
        return
    try:
        from tinygrad import Context  # type: ignore

        with Context(TRAINING=1):
            yield
    except Exception:
        with nullcontext():
            yield


def parameter_grads(loss: Any, params: Sequence[Any]) -> List[Any]:
    """Gradients of a scalar loss wrt params. Does not use Tensor.requires_grad."""
    if not params:
        return []
    try:
        return list(loss.gradient(*params, materialize_grads=True))
    except TypeError:
        return list(loss.gradient(*params))
