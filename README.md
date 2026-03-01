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
