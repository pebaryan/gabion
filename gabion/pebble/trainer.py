from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple

from gabion.pebble.adapters import flatten_tensors, load_adapter, unflatten_to_tensors


class Trainer(Protocol):
    @property
    def backend(self) -> str:
        ...

    def train(
        self, weights: List[float], local_epochs: int, job: Dict[str, object] | None = None
    ) -> Tuple[List[float], int, float]:
        ...


@dataclass
class SyntheticTrainer:
    sample_count: int = 16
    learning_rate: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        self._target = [rng.uniform(-1.0, 1.0) for _ in range(8)]

    @property
    def backend(self) -> str:
        return "synthetic"

    def train(
        self, weights: List[float], local_epochs: int, job: Dict[str, object] | None = None
    ) -> Tuple[List[float], int, float]:
        current = list(weights)
        dims = min(len(current), len(self._target))
        for _ in range(max(1, local_epochs)):
            for i in range(dims):
                gradient = current[i] - self._target[i]
                current[i] -= self.learning_rate * gradient
        loss = sum((current[i] - self._target[i]) ** 2 for i in range(dims)) / max(1, dims)
        return current, self.sample_count, float(loss)


class TinygradTrainer(SyntheticTrainer):
    @property
    def backend(self) -> str:
        import tinygrad  # type: ignore  # noqa: F401
        return "tinygrad"

    def train(
        self, weights: List[float], local_epochs: int, job: Dict[str, object] | None = None
    ) -> Tuple[List[float], int, float]:
        from tinygrad.nn.optim import SGD  # type: ignore
        from tinygrad import Tensor  # type: ignore

        adapter_ref = "gabion.user_models.linear:LinearAdapter"
        if job is not None and isinstance(job.get("model_adapter"), str):
            adapter_ref = str(job["model_adapter"])

        adapter = load_adapter(adapter_ref)
        template_params = adapter.init_params(seed=self.seed)
        trainable_params = unflatten_to_tensors(weights, template_params, Tensor)
        for param in trainable_params:
            param.requires_grad = True

        opt = SGD(trainable_params, lr=self.learning_rate)
        epochs = max(1, local_epochs)
        batch_size = max(8, self.sample_count)
        loss_value = 0.0
        for epoch in range(epochs):
            x, y = adapter.sample_batch(batch_size=batch_size, seed=self.seed + epoch)
            opt.zero_grad()
            logits = adapter.forward(trainable_params, x)
            loss = adapter.loss(logits, y)
            loss.backward()
            opt.step()
            loss_value = float(loss.item())

        flat = flatten_tensors(trainable_params)
        return flat, batch_size, loss_value
