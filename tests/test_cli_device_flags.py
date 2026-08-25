from __future__ import annotations

import argparse
import os

from gabion.cli import apply_worker_device_flags


def test_apply_worker_device_flags_sets_cuda_and_visible_devices() -> None:
    args = argparse.Namespace(device="cuda", visible_devices="1", webgpu_backend=None)
    apply_worker_device_flags(args)
    assert os.environ.get("DEV") == "CUDA"
    assert os.environ.get("CUDA") is None
    assert os.environ.get("HCQ_VISIBLE_DEVICES") == "1"


def test_apply_worker_device_flags_sets_webgpu_backend() -> None:
    args = argparse.Namespace(device="webgpu", visible_devices=None, webgpu_backend="WGPUBackendType_Vulkan")
    apply_worker_device_flags(args)
    assert os.environ.get("DEV") == "WEBGPU"
    assert os.environ.get("WEBGPU") is None
    assert os.environ.get("WEBGPU_BACKEND") == "WGPUBackendType_Vulkan"
