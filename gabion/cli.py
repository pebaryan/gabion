from __future__ import annotations

import argparse
import asyncio
import logging
import os
import zlib

from gabion.common.config import MeshConfig, PebbleConfig
from gabion.common.logging import setup_logging
from gabion.mesh.job_factory import build_tinygrad_job
from gabion.mesh.server import run_mesh
from gabion.pebble.trainer import TinygradTrainer
from gabion.pebble.worker import PebbleWorker

logger = logging.getLogger(__name__)


def apply_worker_device_flags(args: argparse.Namespace) -> None:
    backend_vars = ("CPU", "CUDA", "METAL", "CL", "WEBGPU")
    if args.device != "auto":
        for key in backend_vars:
            os.environ.pop(key, None)

    if args.device == "cpu":
        os.environ["CPU"] = "1"
    elif args.device == "cuda":
        os.environ["CUDA"] = "1"
    elif args.device == "metal":
        os.environ["METAL"] = "1"
    elif args.device == "cl":
        os.environ["CL"] = "1"
    elif args.device == "webgpu":
        os.environ["WEBGPU"] = "1"

    if args.visible_devices:
        os.environ["HCQ_VISIBLE_DEVICES"] = args.visible_devices
    if args.webgpu_backend:
        os.environ["WEBGPU_BACKEND"] = args.webgpu_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="gabion")
    sub = parser.add_subparsers(dest="command", required=True)

    mesh = sub.add_parser("mesh")
    mesh.add_argument("--host", default="127.0.0.1")
    mesh.add_argument("--port", type=int, default=8765)
    mesh.add_argument("--max-rounds", type=int, default=3)
    mesh.add_argument("--min-quorum", type=int, default=1)
    mesh.add_argument("--job-id", default="tinygrad-custom-v1")
    mesh.add_argument("--job-name", default="Tinygrad Custom V1")
    mesh.add_argument("--model-adapter", default="gabion.user_models.linear:LinearAdapter")
    mesh.add_argument("--enable-mnist-job", action="store_true")
    mesh.add_argument("--checkpoint-path", default=None)
    mesh.add_argument("--checkpoint-every-rounds", type=int, default=1)
    mesh.add_argument("--allow-browser-workers", action="store_true")
    mesh.add_argument("--async-participation", action="store_true")
    mesh.add_argument("--eval-every-rounds", type=int, default=0)
    mesh.add_argument("--eval-batch-size", type=int, default=32)
    mesh.add_argument("--eval-seed", type=int, default=1337)

    pebble = sub.add_parser("pebble")
    pebble.add_argument("--id", required=True)
    pebble.add_argument("--mesh-ws-url", default="ws://127.0.0.1:8765/ws")
    pebble.add_argument("--job-id", default=None)
    pebble.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "metal", "cl", "webgpu"],
    )
    pebble.add_argument("--visible-devices", default=None)
    pebble.add_argument("--webgpu-backend", default=None)
    pebble.add_argument("--work-scale", type=float, default=1.0)
    pebble.add_argument("--auto-work-scale", action="store_true")
    pebble.add_argument("--target-round-seconds", type=float, default=1.0)
    pebble.add_argument("--calibration-steps", type=int, default=2)

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
            checkpoint_path=args.checkpoint_path,
            checkpoint_every_rounds=max(1, int(args.checkpoint_every_rounds)),
            allow_browser_workers=bool(args.allow_browser_workers),
            async_participation=bool(args.async_participation),
            eval_every_rounds=max(0, int(args.eval_every_rounds)),
            eval_batch_size=max(1, int(args.eval_batch_size)),
            eval_seed=int(args.eval_seed),
        )
        jobs = [
            build_tinygrad_job(
                job_id=args.job_id,
                name=args.job_name,
                description=f"Tinygrad job from adapter {args.model_adapter}",
                model_adapter=args.model_adapter,
                local_epochs=config.local_epochs,
                min_quorum=config.min_quorum,
                max_rounds=config.max_rounds,
                seed=7,
            )
        ]
        if args.enable_mnist_job:
            jobs.append(
                build_tinygrad_job(
                    job_id="tinygrad-mnist-v1",
                    name="Tinygrad MNIST V1",
                    description="Tinygrad MNIST-style softmax classifier",
                    model_adapter="gabion.user_models.mnist_softmax:MnistSoftmaxAdapter",
                    local_epochs=config.local_epochs,
                    min_quorum=config.min_quorum,
                    max_rounds=config.max_rounds,
                    seed=11,
                )
            )
        run_mesh(config=config, jobs=jobs)
        return

    if args.command == "pebble":
        apply_worker_device_flags(args)
        # Use stable per-worker seed across restarts to keep training trajectories reproducible.
        stable_seed = zlib.crc32(args.id.encode("utf-8")) % 999
        # Use a conservative LR to keep transformer-style adapters numerically stable.
        trainer = TinygradTrainer(seed=stable_seed, learning_rate=0.001)
        try:
            backend = trainer.backend
        except Exception as exc:
            raise RuntimeError("tinygrad is required for worker runtime") from exc
        if backend != "tinygrad":
            raise RuntimeError("tinygrad is required for worker runtime")

        work_scale = max(0.05, float(args.work_scale))
        if args.auto_work_scale:
            try:
                auto_scale = trainer.calibrate_work_scale(
                    target_round_seconds=max(0.1, float(args.target_round_seconds)),
                    steps=max(1, int(args.calibration_steps)),
                )
                work_scale = float(auto_scale)
                logger.info(
                    "worker %s calibrated work_scale=%.3f (target_round_seconds=%.2f, steps=%s)",
                    args.id,
                    work_scale,
                    float(args.target_round_seconds),
                    int(args.calibration_steps),
                )
            except Exception as exc:
                logger.warning("auto work-scale calibration failed, using configured work-scale: %s", exc)

        config = PebbleConfig(
            worker_id=args.id,
            mesh_ws_url=args.mesh_ws_url,
            preferred_job_id=args.job_id,
            work_scale=work_scale,
        )
        worker = PebbleWorker(config=config, trainer=trainer)
        asyncio.run(worker.run())
        return

    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
