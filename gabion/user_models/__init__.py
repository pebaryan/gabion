"""User-defined models for gabion federated learning."""

from gabion.user_models.linear import LinearAdapter
from gabion.user_models.mnist_softmax import MnistSoftmaxAdapter
from gabion.user_models.bbt_transformer import BBTTransformerAdapter

__all__ = ["LinearAdapter", "MnistSoftmaxAdapter", "BBTTransformerAdapter"]