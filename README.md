# gabion

A lightweight, HTTP/WebSocket-based federated training mesh for tinygrad workers.

## Current model

Mesh defines one or more `TrainingJob`s. Workers connect, discover available jobs, choose one, preflight compatibility, and then join rounds for that job.

## What is implemented

- Mesh server with:
  - HTTP health/status endpoints
  - HTTP jobs endpoint (`/jobs`)
  - WebSocket registration + heartbeat
  - Job discovery (`list_jobs` / `job_list`)
  - Job join flow (`join_job` / `job_joined` / `job_rejected`)
  - Artifact/runtime requirement hints (`artifact_required`)
  - Job-scoped round orchestration with FedAvg
- Pebble worker with:
  - Reconnect + heartbeat loops
  - Job listing and selection (`--job-id` or first available)
  - Preflight runtime check for tinygrad jobs
  - Tinygrad training path (fallback only if tinygrad missing)

## Architecture

- `gabion/common/`: config, protocol, jobs, logging
- `gabion/mesh/`: aggregator, round coordinator, mesh server
- `gabion/pebble/`: trainer and worker runtime
- `examples/`: local simulation
- `tests/`: unit + integration

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,tinygrad]
```

## Run

Start mesh:

```bash
gabion mesh --host 127.0.0.1 --port 8765 --max-rounds 5 --min-quorum 2
```

Start mesh with MNIST-style tinygrad example job enabled:

```bash
gabion mesh --host 127.0.0.1 --port 8765 --max-rounds 5 --min-quorum 2 --enable-mnist-job
```

List jobs:

```bash
curl http://127.0.0.1:8765/jobs
```

Start workers:

```bash
gabion pebble --id pebble-1 --mesh-ws-url ws://127.0.0.1:8765/ws --job-id tinygrad-linear-v1
gabion pebble --id pebble-2 --mesh-ws-url ws://127.0.0.1:8765/ws --job-id tinygrad-linear-v1
```

Use MNIST job:

```bash
gabion pebble --id pebble-1 --mesh-ws-url ws://127.0.0.1:8765/ws --job-id tinygrad-mnist-v1
gabion pebble --id pebble-2 --mesh-ws-url ws://127.0.0.1:8765/ws --job-id tinygrad-mnist-v1
```

Status:

```bash
curl http://127.0.0.1:8765/status
```

## Development

Run tests:

```bash
pytest -q
```

Run local simulation:

```bash
python examples/local_simulation.py
```
