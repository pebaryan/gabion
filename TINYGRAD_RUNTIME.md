# Tinygrad Runtime Notes (Windows, `p311`)

This note captures the runtime behavior and fixes we validated on this machine.

## Environment Snapshot

- OS: Windows 10 (`10.0.26200`)
- Conda env: `p311`
- tinygrad: `0.14.0` (in `C:\Users\aryan\miniconda3\envs\p311`)
- GPUs detected by `nvidia-smi`:
  - NVIDIA GeForce RTX 5060 Ti
  - NVIDIA GeForce RTX 3060

## Why `get_available_devices()` Showed Only `CL` and `CPU`

`CUDA` initially failed because tinygrad could not locate the CUDA driver library:

- Error: `failed to load library cuda: try setting CUDA_PATH?`
- Root cause in this env: dynamic lookup for `cuda` failed (`find_library("cuda")` returned `None`)

### CUDA Fix (Verified)

Set:

```cmd
set CUDA_PATH=C:\Windows\System32\nvcuda.dll
set DEV=CUDA
```

Then:

```cmd
python -c "from tinygrad.device import Device; print(Device.DEFAULT, list(Device.get_available_devices())); print(Device['CUDA'].device)"
```

Expected result includes `CUDA` and `Device['CUDA'] == CUDA`.

### CUDA Training Compiler Fix (NVRTC) (Verified)

`Device['CUDA']` can work while training still fails with:

- `No compiler for CUDA is available`
- inner error: `failed to load library nvrtc: try setting NVRTC_PATH?`

Fix by pointing tinygrad to NVRTC and ensuring companion DLLs are on `PATH`:

```cmd
set DEV=CUDA
set CUDA_PATH=C:\Windows\System32\nvcuda.dll
set NVRTC_PATH=C:\Users\aryan\miniconda3\envs\p311\Lib\site-packages\torch\lib\nvrtc64_120_0.dll
set PATH=C:\Users\aryan\miniconda3\envs\p311\Lib\site-packages\torch\lib;%PATH%
```

After this, BBT local training in `gabion` succeeded on CUDA.

## Why `WEBGPU` Initially Failed

`WEBGPU` failed because tinygrad could not load Dawn:

- Error: `failed to load library webgpu: try setting WEBGPU_PATH?`

tinygrad on Windows expects `libwebgpu_dawn.dll`.

### WEBGPU Fix (Verified)

1. Download Dawn DLL (example path used here):

```cmd
mkdir D:\code\gabion\.deps\webgpu
```

Download:

- `https://github.com/wpmed92/pydawn/releases/download/v0.3.0/libwebgpu_dawn.dll`
- Save as:
  - `D:\code\gabion\.deps\webgpu\libwebgpu_dawn.dll`

2. Set env vars:

```cmd
set WEBGPU_PATH=D:\code\gabion\.deps\webgpu\libwebgpu_dawn.dll
set DEV=WEBGPU
```

3. Verify:

```cmd
python -c "from tinygrad.device import Device; print(Device.DEFAULT, list(Device.get_available_devices())); print(Device['WEBGPU'].device)"
```

Expected result includes `WEBGPU` and `Device['WEBGPU'] == WEBGPU`.

## Important: Only One Default Backend Flag

If you set both `DEV=CUDA` and leftover `CUDA=1`/`WEBGPU=1` flags, tinygrad 0.14 errors (`CPU=1 is deprecated, use DEV=CPU instead`). Use `DEV=` only.

## Quick Backend Switch Commands

### Use CUDA

```cmd
set DEV=CUDA
set CUDA_PATH=C:\Windows\System32\nvcuda.dll
```

### Use WEBGPU

```cmd
set DEV=WEBGPU
set WEBGPU_PATH=D:\code\gabion\.deps\webgpu\libwebgpu_dawn.dll
```

## Optional: Persist Variables in Conda Env

```cmd
conda env config vars set CUDA_PATH=C:\Windows\System32\nvcuda.dll
conda env config vars set DEV=CUDA
conda env config vars unset CUDA
conda env config vars unset WEBGPU
conda env config vars unset WEBGPU_PATH
conda deactivate
conda activate p311
```

For WEBGPU, set `DEV=WEBGPU` and `WEBGPU_PATH` instead of `DEV=CUDA`.

## Runtime Probe Snippet

```python
from tinygrad.device import Device, ALL_DEVICES

for d in ALL_DEVICES:
    try:
        print(f"{d} OK -> {Device[d].device}")
    except Exception as e:
        print(f"{d} FAIL -> {type(e).__name__}: {e}")
```
