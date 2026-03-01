from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List

from gabion.common.protocol import RoundResultPayload


@dataclass
class RoundState:
    round_id: int
    round_token: str
    model_version: int
    started_at: float
    results: Dict[str, RoundResultPayload]


class RoundCoordinator:
    def __init__(self) -> None:
        self._round: RoundState | None = None
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def start_round(self, round_id: int, round_token: str, model_version: int) -> None:
        async with self._lock:
            self._round = RoundState(
                round_id=round_id,
                round_token=round_token,
                model_version=model_version,
                started_at=time.time(),
                results={},
            )
            self._event = asyncio.Event()

    async def submit_result(self, result: RoundResultPayload) -> tuple[bool, str]:
        async with self._lock:
            if self._round is None:
                return False, "no_active_round"
            if result["round_id"] != self._round.round_id:
                return False, "round_id_mismatch"
            round_token = str(result.get("round_token", ""))
            model_version = int(result.get("model_version", -1))
            if round_token != self._round.round_token:
                return False, "round_token_mismatch"
            if model_version != self._round.model_version:
                return False, "model_version_mismatch"
            self._round.results[result["worker_id"]] = result
            self._event.set()
            return True, "ok"

    async def wait_for_quorum(
        self, min_quorum: int, timeout_s: float, collect_until_timeout: bool = False
    ) -> List[RoundResultPayload]:
        deadline = time.monotonic() + timeout_s
        while True:
            async with self._lock:
                if self._round is None:
                    return []
                values = list(self._round.results.values())
                if len(values) >= min_quorum and not collect_until_timeout:
                    return values
                if collect_until_timeout and time.monotonic() >= deadline:
                    return values
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                async with self._lock:
                    return list(self._round.results.values()) if self._round else []
            try:
                await asyncio.wait_for(self._event.wait(), timeout=remaining)
            except TimeoutError:
                async with self._lock:
                    return list(self._round.results.values()) if self._round else []
            finally:
                self._event.clear()

    async def current_round_id(self) -> int | None:
        async with self._lock:
            return self._round.round_id if self._round else None
