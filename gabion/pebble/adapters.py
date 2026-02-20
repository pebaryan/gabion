from __future__ import annotations

from importlib import import_module
from typing import Any, List, Protocol, Tuple


class TinygradModelAdapter(Protocol):
    def init_params(self, seed: int) -> List[Any]:
        ...

    def sample_batch(self, batch_size: int, seed: int) -> Tuple[Any, Any]:
        ...

    def forward(self, params: List[Any], x: Any) -> Any:
        ...

    def loss(self, logits: Any, y: Any) -> Any:
        ...


def load_adapter(adapter_ref: str) -> TinygradModelAdapter:
    if ":" not in adapter_ref:
        raise ValueError(f"adapter ref must be module:Class, got: {adapter_ref}")
    module_name, class_name = adapter_ref.split(":", 1)
    module = import_module(module_name)
    cls = getattr(module, class_name)
    return cls()


def flatten_tensors(params: List[Any]) -> List[float]:
    out: List[float] = []
    for tensor in params:
        values = tensor.tolist()
        if isinstance(values, list):
            _flatten_nested(values, out)
        else:
            out.append(float(values))
    return out


def _flatten_nested(values: list, out: List[float]) -> None:
    for value in values:
        if isinstance(value, list):
            _flatten_nested(value, out)
        else:
            out.append(float(value))


def unflatten_to_tensors(flat: List[float], template_params: List[Any], Tensor) -> List[Any]:
    cursor = 0
    rebuilt = []
    for template in template_params:
        shape = tuple(template.shape)
        size = 1
        for dim in shape:
            size *= int(dim)
        chunk = flat[cursor : cursor + size]
        cursor += size
        rebuilt.append(Tensor(chunk).reshape(shape).realize())
    if cursor != len(flat):
        raise ValueError("flat parameter vector does not match template parameter shapes")
    return rebuilt
