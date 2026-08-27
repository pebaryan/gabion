# gabion

A lightweight HTTP/WebSocket federated training mesh for tinygrad workers.

## 20-Second Overview

- Start one `mesh` process and multiple `pebble` workers.
- Workers can run mixed backends (for example CUDA + WebGPU) on different machines.
- Mesh coordinates rounds, aggregates updates, checkpoints state, and exposes live metrics/dashboard.
- You can resume training from checkpoint and change target max rounds from the dashboard without restart.
- Browser WebGPU worker prototype is included (`/webgpu-worker`) for lightweight participation.

## Custom model workflow

Users only implement a tinygrad model adapter. Gabion handles round orchestration, parameter transport, flatten/unflatten, and optimizer loop.

## Adapter interface

Create a class with this contract:

```python
class MyAdapter:
    def init_params(self, seed: int):
        ...
    def sample_batch(self, batch_size: int, seed: int):
        ...
    def forward(self, params, x):
        ...
    def loss(self, logits, y):
        ...
```

Reference it as `module.path:ClassName`.

Built-ins:
- `gabion.user_models.linear:LinearAdapter`
- Diffusion tiny (pixel DDPM + VAE 8x8→4x4 latent, `UNet`/`LatentUNet`/`VAE`, `DDPM`/`DDIM` samplers) verified vs tinygrad 0.14 in `tests/js_nn_optim_smoke`
- `gabion.user_models.mnist_softmax:MnistSoftmaxAdapter`
- `gabion.user_models.bbt_transformer:BBTTransformerAdapter`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Windows (`cmd.exe`):

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e .[dev]
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

`tinygrad` is a mandatory dependency and is installed via core package dependencies.

## Run custom model

Start mesh with your adapter:

```bash
gabion mesh \
  --host 127.0.0.1 \
  --port 8765 \
  --max-rounds 5 \
  --min-quorum 2 \
  --job-id my-job-v1 \
  --job-name "My Job" \
  --model-adapter my_models.my_adapter:MyAdapter
```

Optional mesh runtime flags:

- `--checkpoint-path <path>`: save/load job weights and last completed round
- `--checkpoint-every-rounds <n>`: checkpoint cadence (default `1`)
- Restarting mesh with the same `--checkpoint-path` resumes from the last completed round.
- `--async-participation`: keep each round open until timeout to include slower workers
- `--eval-every-rounds <n>`: run held-out evaluation every `n` rounds (`0` disables)
- `--eval-batch-size <n>`: held-out eval batch size (default `32`)
- `--eval-seed <n>`: fixed eval seed base for deterministic comparison across restarts
- Stale update protection is enabled: results must match the server-issued round token and model version.

Start workers and select the job:

```bash
gabion pebble --id pebble-1 --mesh-ws-url ws://127.0.0.1:8765/ws --job-id my-job-v1
gabion pebble --id pebble-2 --mesh-ws-url ws://127.0.0.1:8765/ws --job-id my-job-v1
```

Worker device flags (instead of manual env vars):

```bash
# CPU
gabion pebble --id w-cpu --job-id my-job-v1 --device cpu

# CUDA GPU 0
gabion pebble --id w-gpu0 --job-id my-job-v1 --device cuda --visible-devices 0

# WebGPU with explicit backend
gabion pebble --id w-webgpu --job-id my-job-v1 --device webgpu --webgpu-backend WGPUBackendType_Vulkan
```

Task chunking for smaller workers:

```bash
# Full-size worker
gabion pebble --id w-fast --job-id my-job-v1 --work-scale 1.0

# Smaller worker (about 25% local workload per round)
gabion pebble --id w-slow --job-id my-job-v1 --work-scale 0.25
```

Startup auto-calibration (optional):

```bash
gabion pebble --id w-auto --job-id my-job-v1 --auto-work-scale --target-round-seconds 1.0 --calibration-steps 2
```

Mixed GPU setup example (NVIDIA CUDA + AMD WebGPU):

```bash
gabion pebble --id w-nv --job-id my-job-v1 --device cuda --visible-devices 0
gabion pebble --id w-amd --job-id my-job-v1 --device webgpu --webgpu-backend WGPUBackendType_Vulkan
```

## Qwen3.8-27B CUDA inference

> **Perf warning (measured 2026-08-27):** the streaming layer-dequant path
> below runs at **~14s/token (0.07 tok/s)** — ~35× slower than the proven
> standalone runner (`D:/tmp/_q27_run.py`: fused IQ4_NL GEMV, persistent u8
> weights, GPU scan → **0.40s/tok**). Read `docs/qwen35-inference-status.md`
> before extending this path; porting the fused-kernel techniques in is the
> intended next step.
>
> **All speed/memory claims are measured with the protocol in
> `docs/qwen35-benchmark.md` via `tools/bench_qwen35_pipeline.py --verify`
> (single process, both shards, correctness gate + prefill/decode metrics).**
> Multi-agent optimization work must follow that protocol.

The Python pebble path supports the Qwen3.8 `qwen35` hybrid architecture and
keeps the GGUF quantized on the host while streaming one active layer at a
time to CUDA. Run two pebble processes, one per RTX 5060 Ti, with contiguous
model shards:

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
$env:NVRTC_PATH = "$env:CUDA_PATH\bin\nvrtc64_120_0.dll"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$gguf = 'D:\aimodels\Qwen3.8-27B-IQ4_NL.gguf'
$tok = 'D:\aimodels\hf\Qwen3.5-9B\tokenizer.json'

gabion mesh --host 127.0.0.1 --port 8765

gabion pebble --id qwen-gpu0 --mesh-ws-url ws://127.0.0.1:8765/ws `
  --device cuda --visible-devices 0 --model qwen35 --gguf $gguf `
  --tokenizer $tok --shard 0 --num-shards 2

gabion pebble --id qwen-gpu1 --mesh-ws-url ws://127.0.0.1:8765/ws `
  --device cuda --visible-devices 1 --model qwen35 --gguf $gguf `
  --tokenizer $tok --shard 1 --num-shards 2
```

Then call the greedy pipeline endpoint:

```powershell
curl.exe -X POST http://127.0.0.1:8765/infer -H 'Content-Type: application/json' `
  -d '{"model":"qwen35","prompt":"Explain model parallelism in one sentence.","max_tokens":32}'
```

`IQ4_NL` is the recommended local file for quality/speed. The uncensored
`IQ4_XS` file is also supported. The IQ3/UD files are smaller, but their
additional IQ1/IQ2/IQ3 formats are not implemented in this CUDA path. Keep
`QWEN35_MAX_CONTEXT` at or below the card's memory budget (the default is
2048). To measure steady-state shard time after tinygrad compilation, run
`python tools/bench_qwen35_cuda.py --shard 0
--visible-device 0` and the corresponding `--shard 1 --visible-device 1`.

List jobs:

```bash
curl http://127.0.0.1:8765/jobs
```

Status:

```bash
curl http://127.0.0.1:8765/status
```

Training dashboard:

```bash
http://127.0.0.1:8765/dashboard
```

Dashboard shows per-job train loss history, held-out eval loss history, and latest per-worker mode/data-source telemetry (including browser worker mode/data source).
You can also change a job's target max rounds directly from each dashboard card without restarting mesh.

Browser WebGPU worker prototype (localhost):

```bash
http://127.0.0.1:8765/webgpu-worker
```

Open this page in a Chromium-based browser, click `Connect`, and it will join a job over WebSocket and submit prototype round results using a minimal WebGPU compute pass.
Start mesh with `--allow-browser-workers` to enable this prototype path.
The browser worker now attempts a tinygrad-js-v0 local trainer (minimal autograd in JS, embedding + RMSNorm + tied projection) and falls back to surrogate training if v0 fails.
For shard-backed browser batches, set `BBT_SHARD_GLOB` in the mesh process environment.

Remote browser pebble via localtunnel:

```bash
# terminal 1: start mesh (browser workers enabled)
python -m gabion.cli mesh --host 127.0.0.1 --port 8766 --allow-browser-workers ...

# terminal 2: expose mesh over HTTPS tunnel
npx localtunnel --port 8766 --subdomain your-gabion-mesh
```

Then share this URL with remote web workers:

```text
https://your-gabion-mesh.loca.lt/webgpu-worker
```

Notes:
- WebGPU in browsers generally requires a secure context; localtunnel HTTPS satisfies this.
- Keep the `localtunnel` process running while remote workers are connected.
- If the selected subdomain is busy, remove `--subdomain` and use the random URL localtunnel prints.
- Do not expose mesh publicly without access controls. Management endpoints (for example changing target max rounds from dashboard) are writable by anyone who can reach the URL.

Raw metrics JSON:

```bash
curl http://127.0.0.1:8765/metrics
```

## MNIST example job

```bash
gabion mesh --enable-mnist-job
```

Then target `--job-id tinygrad-mnist-v1` from workers.

## Development

```bash
pytest -q
python examples/local_simulation.py
```
