from __future__ import annotations

import asyncio

from gabion.mesh.coordinator import RoundCoordinator


def test_round_coordinator_rejects_stale_token_and_version() -> None:
    async def _run() -> None:
        c = RoundCoordinator()
        await c.start_round(round_id=3, round_token="tok-3", model_version=7)

        accepted, reason = await c.submit_result(
            {
                "worker_id": "w1",
                "job_id": "j1",
                "round_id": 3,
                "round_token": "tok-OLD",
                "model_version": 7,
                "sample_count": 8,
                "weights": [0.1, 0.2],
                "metrics": {"loss": 1.0},
            }
        )
        assert not accepted
        assert reason == "round_token_mismatch"

        accepted, reason = await c.submit_result(
            {
                "worker_id": "w1",
                "job_id": "j1",
                "round_id": 3,
                "round_token": "tok-3",
                "model_version": 6,
                "sample_count": 8,
                "weights": [0.1, 0.2],
                "metrics": {"loss": 1.0},
            }
        )
        assert not accepted
        assert reason == "model_version_mismatch"

        accepted, reason = await c.submit_result(
            {
                "worker_id": "w1",
                "job_id": "j1",
                "round_id": 3,
                "round_token": "tok-3",
                "model_version": 7,
                "sample_count": 8,
                "weights": [0.3, 0.4],
                "metrics": {"loss": 0.9},
            }
        )
        assert accepted
        assert reason == "ok"

    asyncio.run(_run())
