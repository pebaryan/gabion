# Bit/Byte Transformer (BBT)

A byte-level transformer adapter for **gabion** federated training, focused on next-byte language modeling.

## Overview

The `BBTTransformerAdapter` is a compact byte-level transformer with self-attention, trained with an autoregressive next-token objective. It is suited for:

- Byte-level language modeling
- domain adaptation with shard-based byte corpora
- backend validation across CUDA / WebGPU workers

## Architecture

```
Input (batch, seq_len)
    │
    ├─> Token Embedding (vocab_size → d_model)
    │
    ├─> + Positional Encoding (sin/cos)
    │
    ├─> Transformer Layer 1
    │   ├─> Multi-Head Self-Attention
    │   └─> Feed-Forward Network
    │
    ├─> Transformer Layer 2
    │   └─> ...
    │
    └─> Output Projection → logits (batch, seq_len, vocab)
        training uses next-token targets over seq_len-1 positions
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | 256 | Vocabulary size (byte-level) |
| `d_model` | 64 | Hidden dimension size |
| `n_heads` | 4 | Number of attention heads |
| `n_layers` | 2 | Number of transformer layers |
| `seq_len` | 32 | Maximum sequence length |
| `learning_rate` | 0.001 | Worker trainer learning rate (CLI/runtime), not adapter constructor |
| `vocab_size` | 256 | Vocabulary size for token embeddings |

## Usage with gabion

### Standalone Testing

```python
from gabion.user_models.bbt_transformer import BBTTransformerAdapter

# Create adapter
adapter = BBTTransformerAdapter(
    input_dim=256,
    d_model=64,
    n_heads=4,
    n_layers=2,
    seq_len=32,
)

# Initialize parameters
params = adapter.init_params(seed=42)

# Generate training batch (x: bytes, y: shifted next-byte targets)
x, y = adapter.sample_batch(batch_size=16, seed=42)

# Forward pass -> logits [B, T-1, vocab]
logits = adapter.forward(params, x)

# Compute loss
loss = adapter.loss(logits, y)
```

### Running with gabion mesh

Start the mesh server with BBT:

```cmd
python -m gabion.cli mesh ^
  --host 127.0.0.1 ^
  --port 8766 ^
  --max-rounds 200 ^
  --min-quorum 2 ^
  --job-id bbt-job-v1 ^
  --job-name "BitByte Transformer" ^
  --model-adapter gabion.user_models.bbt_transformer:BBTTransformerAdapter ^
  --checkpoint-path D:\code\gabion\artifacts\checkpoints\bbt-job-v1.json ^
  --checkpoint-every-rounds 1 ^
  --eval-every-rounds 10 ^
  --eval-batch-size 64
```

### Mixed GPU setup (CUDA + WebGPU/Vulkan)

Use 3 terminals.

Terminal 1 (`mesh`):

```cmd
cd /d D:\code\gabion
set CUDA=
set CPU=
set WEBGPU=
set CL=1
python -m gabion.cli mesh ^
  --host 127.0.0.1 ^
  --port 8766 ^
  --max-rounds 200 ^
  --min-quorum 2 ^
  --job-id bbt-job-v1 ^
  --job-name "BitByte Transformer" ^
  --model-adapter gabion.user_models.bbt_transformer:BBTTransformerAdapter ^
  --checkpoint-path D:\code\gabion\artifacts\checkpoints\bbt-job-v1.json ^
  --checkpoint-every-rounds 1 ^
  --eval-every-rounds 10 ^
  --eval-batch-size 64
```

Terminal 2 (`CUDA pebble`, NVIDIA):

```cmd
cd /d D:\code\gabion
set BBT_SHARD_GLOB=D:\code\bbt\artifacts\datasets\wikitext_103\shards\train\shard_*.bin
set CUDA=1
set CUDA_PATH=C:\Windows\System32\nvcuda.dll
set NVRTC_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvrtc64_120_0.dll
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;C:\Users\aryan\miniconda3\envs\p311\Lib\site-packages\torch\lib;%PATH%
python -m gabion.cli pebble ^
  --id bbt-nv ^
  --mesh-ws-url ws://127.0.0.1:8766/ws ^
  --job-id bbt-job-v1 ^
  --device cuda ^
  --visible-devices 0
```

Terminal 3 (`WebGPU pebble`, AMD via Vulkan):

```cmd
cd /d D:\code\gabion
set BBT_SHARD_GLOB=D:\code\bbt\artifacts\datasets\wikitext_103\shards\train\shard_*.bin
set WEBGPU_PATH=D:\code\gabion\.deps\webgpu\libwebgpu_dawn.dll
python -m gabion.cli pebble ^
  --id bbt-amd-wgpu ^
  --mesh-ws-url ws://127.0.0.1:8766/ws ^
  --job-id bbt-job-v1 ^
  --device webgpu ^
  --webgpu-backend WGPUBackendType_Vulkan
```

Dashboard and status:

```cmd
start http://127.0.0.1:8766/dashboard
curl http://127.0.0.1:8766/status
curl http://127.0.0.1:8766/metrics
```

### Custom configuration

For larger models, adjust hyperparameters:

```python
# Larger model: 128 dims, 8 heads, 4 layers, 64 seq
from gabion.user_models.bbt_transformer import BBTTransformerAdapter

adapter = BBTTransformerAdapter(
    input_dim=256,
    d_model=128,
    n_heads=8,
    n_layers=4,
    seq_len=64,
)
```

## Training Task

The default training task is byte-level next-token language modeling on WikiText:
- **Input**: Byte windows sampled from WikiText text
- **Target**: Next-byte targets (`x[:, 1:]`)
- **Loss**: Cross-entropy

Notes:
- By default, the adapter first tries byte shards (`shard_*.bin`) from:
  - `BBT_SHARD_GLOB` env var (if set)
  - `D:\code\bbt\artifacts\datasets\wikitext_103\shards\train\shard_*.bin` (if present)
- If no shards are found, it tries `datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")`.
- If `datasets` is unavailable, it falls back to synthetic random bytes.
- You can force local corpus usage via `wikitext_path` in `BBTTransformerAdapter(...)`.

Example (Windows cmd):

```cmd
set BBT_SHARD_GLOB=D:\code\bbt\artifacts\datasets\wikitext_103\shards\train\shard_*.bin
```

## Files

- `gabion/user_models/bbt_transformer.py` - BBT adapter used by gabion jobs
- `tests/test_bbt_transformer.py` - adapter tests
- `README.md` - project overview

## Requirements

- Python 3.8+
- tinygrad >= 0.9.2
- gabion (optional, for federated learning)

## License

MIT License
