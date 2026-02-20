from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List

from gabion.common.protocol import RoundResultPayload


@dataclass
class RoundState:
    round_id: int
    started_at: float
    results: Dict[str, RoundResultPayload]


class RoundCoordinator:
    def __init__(self) -> None:
        self._round: RoundState | None = None
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def start_round(self, round_id: int) -> None:
        async with self._lock:
            self._round = RoundState(round_id=round_id, started_at=time.time(), results={})
            self._event = asyncio.Event()

    async def submit_result(self, result: RoundResultPayload) -> bool:
        async with self._lock:
            if self._round is None or result["round_id"] != self._round.round_id:
                return False
            self._round.results[result["worker_id"]] = result
            self._event.set()
            return True

    async def wait_for_quorum(self, min_quorum: int, timeout_s: float) -> List[RoundResultPayload]:
        deadline = time.monotonic() + timeout_s
        while True:
            async with self._lock:
                if self._round is None:
                    return []
                values = list(self._round.results.values())
                if len(values) >= min_quorum:
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
