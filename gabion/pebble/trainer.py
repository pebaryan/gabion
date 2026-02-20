from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Protocol, Tuple


class Trainer(Protocol):
    @property
    def backend(self) -> str:
        ...

    def train(self, weights: List[float], local_epochs: int) -> Tuple[List[float], int, float]:
        ...


@dataclass
class SyntheticTrainer:
    sample_count: int = 16
    learning_rate: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        self._target = [rng.uniform(-1.0, 1.0) for _ in range(8)]
        self._dataset_x = [
            [rng.uniform(-1.0, 1.0) for _ in range(8)] for _ in range(self.sample_count)
        ]
        self._dataset_y = [
            sum(row[i] * self._target[i] for i in range(8)) + rng.uniform(-0.01, 0.01)
            for row in self._dataset_x
        ]

    @property
    def backend(self) -> str:
        return "synthetic"

    def train(self, weights: List[float], local_epochs: int) -> Tuple[List[float], int, float]:
        current = list(weights)
        dims = min(len(current), len(self._target))
        for _ in range(max(1, local_epochs)):
            for i in range(dims):
                gradient = current[i] - self._target[i]
                current[i] -= self.learning_rate * gradient
        loss = sum((current[i] - self._target[i]) ** 2 for i in range(dims)) / max(1, dims)
        return current, self.sample_count, float(loss)


class TinygradTrainer(SyntheticTrainer):
    MNIST_INPUT_DIM = 28 * 28
    MNIST_CLASSES = 10
    MNIST_WEIGHT_DIM = (MNIST_INPUT_DIM * MNIST_CLASSES) + MNIST_CLASSES

    @property
    def backend(self) -> str:
        try:
            import tinygrad  # type: ignore  # noqa: F401

            return "tinygrad"
        except Exception:
            return "synthetic"

    def train(self, weights: List[float], local_epochs: int) -> Tuple[List[float], int, float]:
        try:
            from tinygrad import Tensor  # type: ignore
            from tinygrad.nn.optim import SGD  # type: ignore
        except Exception:
            return super().train(weights, local_epochs)

        if len(weights) == self.MNIST_WEIGHT_DIM:
            return self._train_mnist_style(weights, local_epochs, Tensor, SGD)
        return self._train_linear(weights, local_epochs, Tensor, SGD)

    def _train_linear(self, weights, local_epochs, Tensor, SGD):
        dims = len(weights)
        x_rows = [row[:dims] for row in self._dataset_x]
        y_vals = self._dataset_y[:]

        x = Tensor(x_rows)
        y = Tensor(y_vals)
        w = Tensor(weights, requires_grad=True)
        opt = SGD([w], lr=self.learning_rate)
        epochs = max(1, local_epochs)
        loss_value = 0.0
        for _ in range(epochs):
            opt.zero_grad()
            prediction = x.matmul(w)
            loss = ((prediction - y) * (prediction - y)).mean()
            loss.backward()
            opt.step()
            loss_value = float(loss.item())

        return [float(v) for v in w.tolist()], self.sample_count, loss_value

    def _train_mnist_style(self, weights, local_epochs, Tensor, SGD):
        rng = random.Random(self.seed)
        sample_count = max(32, self.sample_count)
        x_rows = [
            [rng.random() for _ in range(self.MNIST_INPUT_DIM)] for _ in range(sample_count)
        ]
        y_labels = [rng.randrange(self.MNIST_CLASSES) for _ in range(sample_count)]

        w_flat = weights[: self.MNIST_INPUT_DIM * self.MNIST_CLASSES]
        b_vec = weights[self.MNIST_INPUT_DIM * self.MNIST_CLASSES :]
        w_rows = [
            w_flat[i * self.MNIST_CLASSES : (i + 1) * self.MNIST_CLASSES]
            for i in range(self.MNIST_INPUT_DIM)
        ]

        x = Tensor(x_rows)
        y = Tensor(y_labels)
        w = Tensor(w_rows, requires_grad=True)
        b = Tensor(b_vec, requires_grad=True)
        opt = SGD([w, b], lr=self.learning_rate)
        epochs = max(1, local_epochs)
        loss_value = 0.0
        for _ in range(epochs):
            opt.zero_grad()
            logits = x.matmul(w) + b
            loss = logits.sparse_categorical_crossentropy(y)
            loss.backward()
            opt.step()
            loss_value = float(loss.item())

        out_w_rows = w.tolist()
        out_b = b.tolist()
        flat_w = [float(v) for row in out_w_rows for v in row]
        flat_b = [float(v) for v in out_b]
        return flat_w + flat_b, sample_count, loss_value
