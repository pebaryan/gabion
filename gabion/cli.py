from __future__ import annotations

import argparse
import asyncio
import random

from gabion.common.config import MeshConfig, PebbleConfig
from gabion.common.jobs import TrainingJob
from gabion.common.logging import setup_logging
from gabion.mesh.server import run_mesh
from gabion.pebble.trainer import TinygradTrainer
from gabion.pebble.worker import PebbleWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="gabion")
    sub = parser.add_subparsers(dest="command", required=True)

    mesh = sub.add_parser("mesh")
    mesh.add_argument("--host", default="127.0.0.1")
    mesh.add_argument("--port", type=int, default=8765)
    mesh.add_argument("--max-rounds", type=int, default=3)
    mesh.add_argument("--min-quorum", type=int, default=1)
    mesh.add_argument("--enable-mnist-job", action="store_true")

    pebble = sub.add_parser("pebble")
    pebble.add_argument("--id", required=True)
    pebble.add_argument("--mesh-ws-url", default="ws://127.0.0.1:8765/ws")
    pebble.add_argument("--job-id", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    if args.command == "mesh":
        config = MeshConfig(
            host=args.host,
            port=args.port,
            max_rounds=args.max_rounds,
            min_quorum=args.min_quorum,
        )
        random.seed(7)
        initial_weights = [random.uniform(-0.5, 0.5) for _ in range(8)]
        jobs = [
            TrainingJob(
                job_id="tinygrad-linear-v1",
                name="Tinygrad Linear V1",
                description="Tinygrad linear model on local synthetic data",
                artifact_uri="mesh://bundles/tinygrad-linear-v1",
                artifact_checksum="sha256:dev",
                runtime="tinygrad",
                local_epochs=config.local_epochs,
                min_quorum=config.min_quorum,
                max_rounds=config.max_rounds,
                initial_weights=initial_weights,
            )
        ]
        if args.enable_mnist_job:
            mnist_dim = (28 * 28 * 10) + 10
            mnist_weights = [random.uniform(-0.01, 0.01) for _ in range(mnist_dim)]
            jobs.append(
                TrainingJob(
                    job_id="tinygrad-mnist-v1",
                    name="Tinygrad MNIST V1",
                    description="Tinygrad MNIST-style softmax classifier",
                    artifact_uri="mesh://bundles/tinygrad-mnist-v1",
                    artifact_checksum="sha256:dev",
                    runtime="tinygrad",
                    local_epochs=config.local_epochs,
                    min_quorum=config.min_quorum,
                    max_rounds=config.max_rounds,
                    initial_weights=mnist_weights,
                )
            )
        run_mesh(config=config, jobs=jobs)
        return

    if args.command == "pebble":
        config = PebbleConfig(
            worker_id=args.id,
            mesh_ws_url=args.mesh_ws_url,
            preferred_job_id=args.job_id,
        )
        worker = PebbleWorker(config=config, trainer=TinygradTrainer(seed=hash(args.id) % 999))
        asyncio.run(worker.run())
        return

    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
