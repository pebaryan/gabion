from __future__ import annotations

import asyncio

from aiohttp import ClientSession, web

from gabion.common.config import MeshConfig
from gabion.common.jobs import TrainingJob
from gabion.common.protocol import make_message
from gabion.mesh.server import MeshServer


async def _fake_worker(worker_id: str, ws_url: str, seen_rounds: list[int]) -> None:
    async with ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            await ws.send_json(
                make_message(
                    "register",
                    {"worker_id": worker_id, "capabilities": {"trainer": "tinygrad"}},
                )
            )
            await ws.send_json(make_message("list_jobs", {}))

            joined_job: str | None = None
            while True:
                msg = await ws.receive_json(timeout=5)
                if msg["type"] == "job_list":
                    job_id = msg["payload"]["jobs"][0]["job_id"]
                    await ws.send_json(make_message("join_job", {"job_id": job_id}))
                elif msg["type"] == "job_joined":
                    joined_job = msg["payload"]["job_id"]
                elif msg["type"] == "round_start":
                    payload = msg["payload"]
                    if payload["job_id"] != joined_job:
                        continue
                    round_id = int(payload["round_id"])
                    seen_rounds.append(round_id)
                    base = [float(v) for v in payload["weights"]]
                    updated = [v + 0.01 for v in base]
                    await ws.send_json(
                        make_message(
                            "round_result",
                            {
                                "job_id": payload["job_id"],
                                "round_id": round_id,
                                "round_token": payload.get("round_token", ""),
                                "model_version": payload.get("model_version", 0),
                                "sample_count": 8,
                                "weights": updated,
                                "metrics": {"loss": 0.2},
                            },
                        )
                    )
                elif msg["type"] == "round_summary" and int(msg["payload"]["round_id"]) >= 2:
                    return


async def test_mesh_pebble_round_trip() -> None:
    server = MeshServer(
        config=MeshConfig(port=0, max_rounds=2, min_quorum=2, round_timeout_s=2.0),
        jobs=[
            TrainingJob(
                job_id="job-1",
                name="Job 1",
                description="test",
                artifact_uri="mesh://bundles/job-1",
                artifact_checksum="sha256:test",
                runtime="tinygrad",
                model_adapter="gabion.user_models.linear:LinearAdapter",
                local_epochs=1,
                min_quorum=2,
                max_rounds=2,
                initial_weights=[0.0] * 8,
            )
        ],
    )
    runner = web.AppRunner(server.app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()

    sockets = site._server.sockets  # type: ignore[attr-defined]
    port = sockets[0].getsockname()[1]
    ws_url = f"ws://127.0.0.1:{port}/ws"

    seen_a: list[int] = []
    seen_b: list[int] = []
    await asyncio.gather(
        _fake_worker("worker-a", ws_url, seen_a),
        _fake_worker("worker-b", ws_url, seen_b),
    )

    assert seen_a == [1, 2]
    assert seen_b == [1, 2]

    await runner.cleanup()
