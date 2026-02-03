# PyTorch Static Peak Memory Analyzer (TorchScript IR) - Desktop Computer version

A work-in-progress tool to estimate inference peak GPU memory (parameters + activations)
from TorchScript IR, and validate it against PyTorch CUDA allocator measurements.

> Goal: given a model + input shape, predict the peak CUDA memory you’ll see at deployment.

## What’s in this repo

- **Python (runtime measurement):** runs a forward pass and reports CUDA allocator peaks.
- **C++ (static estimator):** loads TorchScript, annotates input shapes, runs shape propagation,
  and simulates liveness to estimate peak activation bytes.

## Terminology (PyTorch CUDA memory)

- **allocated**: bytes occupied by *live tensors*.
- **reserved**: bytes managed by PyTorch’s caching allocator (allocated + cached blocks).
- **peak delta**: `peak_allocated - baseline_allocated` (roughly activations + temporaries tracked by PyTorch).

Note: `reserved` can exceed `allocated` and can look higher in tools like `nvidia-smi` due to caching.

## Quickstart

1) Export a traced TorchScript model:
```bash
python 1_ExportingIR/3_export_resnet_traced.py --model resnet50 --batch 1 --h 224 --w 224 --out_dir data
```

2) Measure runtime peaks:
```bash
python 1_ExportingIR/4_measure_runtime_peak.py --model resnet50 --batch 1 --h 224 --w 224
```

3) Build the C++ analyzer:
```bash
cd 2_EstimateMem && mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" ..
cmake --build . -j
```

4) Run the estimator:
```bash
./torch_ir_mem
```

## Benchmarks (batch=1, 224×224)

### Runtime peak (PyTorch CUDA allocator)

| Model | Baseline allocated (MB) | Peak allocated (MB) | Peak delta (MB) | Peak reserved (MB) |
|---|---:|---:|---:|---:|
| alexnet | 234.03 | 242.22 | 8.19 | 246.00 |
| mobilenet_v3_small | 10.38 | 18.51 | 8.14 | 28.00 |
| resnet50 | 98.30 | 109.37 | 11.07 | 142.00 |
| resnet101 | 171.23 | 181.98 | 10.75 | 196.00 |
| vgg16 | 529.62 | 566.51 | 36.89 | 578.00 |

### Current static estimation output (C++ analyzer)

| Model | Parameter bytes (MB) | Peak activations (MB) | Total peak (MB) |
|---|---:|---:|---:|
| alexnet | 233.081207 | 233.081207 | 466.162415 |
| mobilenet_v3_small | 9.746689 | 9.746429 | 19.493118 |
| resnet50 | 97.695381 | 97.692047 | 195.387428 |
| resnet101 | 170.344208 | 170.343414 | 340.687622 |
| vgg16 | 527.792145 | 527.792145 | 1055.584290 |

## Notes

- The runtime **peak delta** is the best proxy here for “activations + temporaries” tracked by PyTorch.
- `reserved` is allocator policy (caching / fragmentation) and is expected to be >= `allocated`.
- The static activation numbers above are **not correct yet** (they’re far larger than runtime delta).
  Next steps are to separate constants/weights from activations, fix free-after-use timing, and add alias handling.

## Future Direction

- Make it work for different deep learning models other than computer vision models
- Add alias analysis to avoid double-counting views/in-place.
- Add a `torch.export` path for more robust shape/dtype metadata (TorchScript is deprecated in recent PyTorch).

## References

- PyTorch CUDA semantics (allocated vs reserved): https://docs.pytorch.org/docs/stable/notes/cuda.html
- `torch.cuda.memory.max_memory_reserved`: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html
- `torch.cuda.memory.reset_peak_memory_stats`: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.reset_peak_memory_stats.html
- TorchScript deprecation notice: https://docs.pytorch.org/docs/stable/jit.html
- `torch.jit.freeze` (inlines parameters as constants): https://docs.pytorch.org/docs/stable/generated/torch.jit.freeze.html

