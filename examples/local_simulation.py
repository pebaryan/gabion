from __future__ import annotations

import asyncio

from aiohttp import web

from gabion.common.config import MeshConfig, PebbleConfig
from gabion.common.jobs import TrainingJob
from gabion.common.logging import setup_logging
from gabion.mesh.server import MeshServer
from gabion.pebble.trainer import TinygradTrainer
from gabion.pebble.worker import PebbleWorker


async def main() -> None:
    setup_logging()

    initial_weights = [0.0] * 8
    mesh = MeshServer(
        config=MeshConfig(port=8770, max_rounds=3, min_quorum=2, round_timeout_s=5.0),
        jobs=[
            TrainingJob(
                job_id="tinygrad-linear-v1",
                name="Tinygrad Linear V1",
                description="Tinygrad linear model on local synthetic data",
                artifact_uri="mesh://bundles/tinygrad-linear-v1",
                artifact_checksum="sha256:dev",
                runtime="tinygrad",
                local_epochs=1,
                min_quorum=2,
                max_rounds=3,
                initial_weights=initial_weights,
            )
        ],
    )

    runner = web.AppRunner(mesh.app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=8770)
    await site.start()

    workers = [
        PebbleWorker(
            config=PebbleConfig(
                worker_id=f"pebble-{i}",
                mesh_ws_url="ws://127.0.0.1:8770/ws",
                preferred_job_id="tinygrad-linear-v1",
            ),
            trainer=TinygradTrainer(seed=i),
        )
        for i in (1, 2)
    ]
    tasks = [asyncio.create_task(worker.run()) for worker in workers]

    await asyncio.sleep(12)

    for task in tasks:
        task.cancel()
    await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
