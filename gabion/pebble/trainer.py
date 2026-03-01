from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
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
    def __post_init__(self) -> None:
        super().__post_init__()
        # Adam optimizer state (persists across rounds for warmup, reset m/v each round)
        self._adam_m: Dict[int, object] = {}
        self._adam_v: Dict[int, object] = {}
        self._adam_t: int = 0

    @property
    def backend(self) -> str:
        import tinygrad  # type: ignore  # noqa: F401
        return "tinygrad"

    def train(
        self, weights: List[float], local_epochs: int, job: Dict[str, object] | None = None
    ) -> Tuple[List[float], int, float]:
        import numpy as np
        from tinygrad import Tensor  # type: ignore

        adapter_ref = "gabion.user_models.linear:LinearAdapter"
        if job is not None and isinstance(job.get("model_adapter"), str):
            adapter_ref = str(job["model_adapter"])

        adapter = load_adapter(adapter_ref)
        template_params = adapter.init_params(seed=self.seed)
        trainable_params = unflatten_to_tensors(weights, template_params, Tensor)
        for param in trainable_params:
            param.requires_grad = True

        work_scale = 1.0
        if job is not None:
            try:
                work_scale = float(job.get("work_scale", 1.0))
            except Exception:
                work_scale = 1.0
        work_scale = min(1.0, max(0.05, work_scale))
        round_id = 0
        if job is not None:
            try:
                round_id = max(0, int(job.get("round_id", 0)))
            except Exception:
                round_id = 0

        # Optimizer config from job
        lr = self.learning_rate
        optimizer = "adam"
        grad_clip_norm = 1.0
        warmup_steps = 10
        beta1, beta2 = 0.9, 0.999
        if job is not None:
            lr = float(job.get("learning_rate", lr))
            optimizer = str(job.get("optimizer", optimizer))
            grad_clip_norm = float(job.get("grad_clip_norm", grad_clip_norm))
            warmup_steps = int(job.get("warmup_steps", warmup_steps))
            beta1 = float(job.get("adam_beta1", beta1))
            beta2 = float(job.get("adam_beta2", beta2))

        # Reset Adam m/v each round (federated: fresh weights each round)
        self._adam_m.clear()
        self._adam_v.clear()

        epochs = max(1, int(round(max(1, local_epochs) * work_scale)))
        batch_size = max(8, int(round(max(8, self.sample_count) * work_scale)))
        loss_value = 0.0
        round_seed_base = self.seed + (round_id * 1_000_003)
        with Tensor.train():
            for epoch in range(epochs):
                x, y = adapter.sample_batch(batch_size=batch_size, seed=round_seed_base + epoch)
                for param in trainable_params:
                    param.grad = None
                logits = adapter.forward(trainable_params, x)
                loss = adapter.loss(logits, y)
                loss.backward()

                # Gradient clipping (global norm)
                if grad_clip_norm > 0:
                    total_norm_sq = 0.0
                    for param in trainable_params:
                        if param.grad is not None:
                            g = param.grad.numpy()
                            total_norm_sq += float((g * g).sum())
                    total_norm = math.sqrt(total_norm_sq)
                    if total_norm > grad_clip_norm:
                        clip_scale = grad_clip_norm / total_norm
                        for param in trainable_params:
                            if param.grad is not None:
                                param.grad = (param.grad * clip_scale).realize()

                if optimizer == "adam":
                    # Adam with bias correction + warmup
                    self._adam_t += 1
                    t = self._adam_t
                    eff_lr = lr * min(1.0, t / max(1, warmup_steps))
                    bc1 = 1 - beta1 ** t
                    bc2 = 1 - beta2 ** t
                    for idx, param in enumerate(trainable_params):
                        if param.grad is None:
                            continue
                        g = param.grad.numpy()
                        if idx not in self._adam_m:
                            self._adam_m[idx] = np.zeros_like(g)
                            self._adam_v[idx] = np.zeros_like(g)
                        m, v = self._adam_m[idx], self._adam_v[idx]
                        m[:] = beta1 * m + (1 - beta1) * g
                        v[:] = beta2 * v + (1 - beta2) * g * g
                        update = eff_lr * (m / bc1) / (np.sqrt(v / bc2) + 1e-8)
                        param.assign(Tensor(param.numpy() - update).realize())
                else:
                    # SGD fallback
                    for param in trainable_params:
                        if param.grad is not None:
                            param.assign((param - param.grad * lr).realize())

                loss_value = float(loss.item())

        flat = flatten_tensors(trainable_params)
        return flat, batch_size, loss_value

    def calibrate_work_scale(self, target_round_seconds: float = 1.0, steps: int = 2) -> float:
        """Estimate a per-worker work scale from a short local benchmark."""
        adapter_ref = "gabion.user_models.linear:LinearAdapter"
        adapter = load_adapter(adapter_ref)
        template_params = adapter.init_params(seed=self.seed)
        base_weights = flatten_tensors(template_params)

        # Warm-up to reduce first-iteration compile noise.
        self.train(
            weights=list(base_weights),
            local_epochs=1,
            job={"model_adapter": adapter_ref, "work_scale": 1.0},
        )

        timings: List[float] = []
        for _ in range(max(1, steps)):
            t0 = time.perf_counter()
            self.train(
                weights=list(base_weights),
                local_epochs=1,
                job={"model_adapter": adapter_ref, "work_scale": 1.0},
            )
            timings.append(time.perf_counter() - t0)

        timings.sort()
        median_s = timings[len(timings) // 2] if timings else 0.0
        if median_s <= 0.0:
            return 1.0
        scale = target_round_seconds / median_s
        return min(1.0, max(0.05, float(scale)))
