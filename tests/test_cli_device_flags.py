from __future__ import annotations

import argparse
import os

from gabion.cli import apply_worker_device_flags


def test_apply_worker_device_flags_sets_cuda_and_visible_devices() -> None:
    args = argparse.Namespace(device="cuda", visible_devices="1", webgpu_backend=None)
    apply_worker_device_flags(args)
    assert os.environ.get("CUDA") == "1"
    assert os.environ.get("HCQ_VISIBLE_DEVICES") == "1"


def test_apply_worker_device_flags_sets_webgpu_backend() -> None:
    args = argparse.Namespace(device="webgpu", visible_devices=None, webgpu_backend="WGPUBackendType_Vulkan")
    apply_worker_device_flags(args)
    assert os.environ.get("WEBGPU") == "1"
    assert os.environ.get("WEBGPU_BACKEND") == "WGPUBackendType_Vulkan"
