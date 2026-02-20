from __future__ import annotations


class LinearAdapter:
    def __init__(self, input_dim: int = 8) -> None:
        self.input_dim = input_dim

    def init_params(self, seed: int):
        from tinygrad import Tensor  # type: ignore

        Tensor.manual_seed(seed)
        w = Tensor.uniform(self.input_dim, 1) * 0.2 - 0.1
        b = Tensor.zeros(1)
        return [w, b]

    def sample_batch(self, batch_size: int, seed: int):
        from tinygrad import Tensor  # type: ignore

        Tensor.manual_seed(seed)
        x = Tensor.uniform(batch_size, self.input_dim)
        true_w = Tensor.arange(self.input_dim).reshape(self.input_dim, 1) / max(1, self.input_dim)
        y = x.matmul(true_w) + 0.1
        return x, y

    def forward(self, params, x):
        w, b = params
        return x.matmul(w) + b

    def loss(self, logits, y):
        return ((logits - y) * (logits - y)).mean()
