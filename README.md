# gabion

A lightweight HTTP/WebSocket federated training mesh for tinygrad workers.

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

## Install

```bash
python -m venv .venv
source .venv/bin/activate
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

List jobs:

```bash
curl http://127.0.0.1:8765/jobs
```

Status:

```bash
curl http://127.0.0.1:8765/status
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
