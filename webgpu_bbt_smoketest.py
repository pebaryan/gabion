from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print active tinygrad WebGPU adapter and run a 1-step BBT training smoke test."
    )
    p.add_argument("--webgpu-path", default=None, help="Path to libwebgpu_dawn.dll")
    p.add_argument(
        "--webgpu-backend",
        default=None,
        help="Example: WGPUBackendType_D3D12 or WGPUBackendType_Vulkan",
    )
    p.add_argument(
        "--visible-devices",
        default=None,
        help="Sets HCQ_VISIBLE_DEVICES. Note: this does not select WebGPU adapters in tinygrad today.",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--vocab-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--d-ff", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def configure_env(args: argparse.Namespace) -> None:
    for key in ("CPU", "CUDA", "METAL", "CL", "AMD", "NV", "WEBGPU"):
        os.environ.pop(key, None)
    os.environ["DEV"] = "WEBGPU"
    if args.webgpu_path:
        os.environ["WEBGPU_PATH"] = args.webgpu_path
    if args.webgpu_backend:
        os.environ["WEBGPU_BACKEND"] = args.webgpu_backend
    if args.visible_devices:
        os.environ["HCQ_VISIBLE_DEVICES"] = args.visible_devices


def print_adapter_info() -> None:
    from tinygrad.device import Device
    from tinygrad.runtime.autogen import webgpu
    from tinygrad.runtime.ops_webgpu import from_wgpu_str

    d = Device["WEBGPU"]
    info = webgpu.WGPUAdapterInfo()
    status = webgpu.wgpuDeviceGetAdapterInfo(d.device_res, info)
    print(f"default={Device.DEFAULT}")
    print(f"available={list(Device.get_available_devices())}")
    print(f"adapter_status={int(status)}")
    print(f"adapter={from_wgpu_str(info.device)}")
    print(f"vendor={from_wgpu_str(info.vendor)}")
    print(f"description={from_wgpu_str(info.description)}")
    print(f"backend_type={int(info.backendType)}")
    print(f"adapter_type={int(info.adapterType)}")
    webgpu.wgpuAdapterInfoFreeMembers(info)


def run_train_step(args: argparse.Namespace) -> float:
    from gabion.pebble.autograd import parameter_grads, training_mode
    from gabion.user_models.bbt_transformer import BBTTransformerAdapter

    adapter = BBTTransformerAdapter(
        input_dim=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        seq_len=args.seq_len,
        d_ff=args.d_ff,
    )

    params = adapter.init_params(seed=args.seed)
    x, y = adapter.sample_batch(batch_size=args.batch_size, seed=args.seed + 1)

    with training_mode():
        logits = adapter.forward(params, x, ternarize=False)
        loss = adapter.loss(logits, y)
        grads = parameter_grads(loss, params)
        for p, g in zip(params, grads):
            p.assign((p - g * 0.001).realize())
    return float(loss.item())


def main() -> int:
    args = parse_args()
    configure_env(args)
    print("=== WEBGPU BBT Smoke Test ===")
    print(f"DEV={os.getenv('DEV')}")
    print(f"WEBGPU_PATH={os.getenv('WEBGPU_PATH', '')}")
    print(f"WEBGPU_BACKEND={os.getenv('WEBGPU_BACKEND', '')}")
    print(f"HCQ_VISIBLE_DEVICES={os.getenv('HCQ_VISIBLE_DEVICES', '')}")
    try:
        print_adapter_info()
        loss = run_train_step(args)
    except Exception as e:
        print(f"status=FAIL error={type(e).__name__}: {e}")
        return 1
    print(f"loss={loss}")
    print("status=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
