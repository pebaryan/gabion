from __future__ import annotations


class MnistSoftmaxAdapter:
    def __init__(self, input_dim: int = 28 * 28, num_classes: int = 10) -> None:
        self.input_dim = input_dim
        self.num_classes = num_classes

    def init_params(self, seed: int):
        from tinygrad import Tensor  # type: ignore

        Tensor.manual_seed(seed)
        w = Tensor.uniform(self.input_dim, self.num_classes) * 0.02 - 0.01
        b = Tensor.zeros(self.num_classes)
        return [w, b]

    def sample_batch(self, batch_size: int, seed: int):
        from tinygrad import Tensor  # type: ignore

        Tensor.manual_seed(seed)
        x = Tensor.uniform(batch_size, self.input_dim)
        y = Tensor.randint(batch_size, high=self.num_classes)
        return x, y

    def forward(self, params, x):
        w, b = params
        return x.matmul(w) + b

    def loss(self, logits, y):
        return logits.sparse_categorical_crossentropy(y)
