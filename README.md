# PyTorch Static Peak GPU Memory Analyzer (TorchScript IR) — Desktop Edition

A **work-in-progress** tool to estimate **inference-time peak GPU memory** (weights + activations)
*statically* from a TorchScript IR graph, and to validate the estimate against **PyTorch CUDA allocator**
measurements from a real forward pass.

> **What this project demonstrates:** reading and analyzing TorchScript IR, running shape propagation,
> simulating tensor liveness, and building a runtime validator that reports allocator peaks.

---

## What’s in this repo

- **`1_ExportingIR/` (Python):**
  - exports a traced/scripted TorchScript model for a given model + input shape
  - measures **CUDA allocator** memory stats (baseline / peak / delta, allocated vs reserved)
- **`2_EstimateMem/` (C++):**
  - loads the TorchScript model
  - annotates input shapes + runs shape propagation
  - simulates liveness to estimate **peak live bytes**

---

## Terminology (PyTorch CUDA memory)

PyTorch exposes two commonly confused notions of “GPU memory”:

- **allocated**: bytes currently held by *live tensors* (what you conceptually “use”)
- **reserved**: bytes owned by PyTorch’s caching allocator (allocated + cached blocks)

For this project, two derived terms matter:

- **baseline allocated**: `torch.cuda.memory_allocated()` right before the forward pass  
  (for inference, this is typically close to **model weights** already on GPU)
- **peak delta (allocated)**: `max_memory_allocated - baseline_allocated`  
  (best proxy for **activations + temporaries** tracked by PyTorch during forward)

> Note: `reserved` can be substantially larger than `allocated` due to caching/fragmentation, and
> process-level tools like `nvidia-smi` tend to look closer to **reserved** than **allocated**.

All memory numbers below are reported in **MiB** (2^20 bytes) to match PyTorch CUDA memory APIs.

---

## Test environment (desktop)

- **CPU**: AMD Ryzen 9 9950X
- **GPU**: NVIDIA GeForce RTX 4080 SUPER
- **CUDA**: 13.1
- **PyTorch**: 2.10.0

---

## Quickstart

0) Create virtual environments per folder (each folder has its own `requirements.txt`)

1) Export a TorchScript model:
```bash
python 1_ExportingIR/1_exportIr.py --model resnet50 --batch 1 --h 224 --w 224 --out_dir data
```

2) Measure runtime peaks (allocator stats):
```bash
python 1_ExportingIR/2_testPre.py --model resnet50 --batch 1 --h 224 --w 224
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

---

## Benchmarks (batch=1, 224×224)

### Runtime peak (PyTorch CUDA allocator)

Interpretation tips:
- **Baseline allocated** ≈ weights already resident on GPU
- **Peak delta (allocated)** ≈ activations + temporaries during forward
- **Peak reserved** reflects allocator policy and may exceed peak allocated

| Model | Baseline allocated (MiB) | Peak allocated (MiB) | Peak delta (MiB) | Peak reserved (MiB) |
|---|---:|---:|---:|---:|
| alexnet | 234.03 | 242.22 | 8.19 | 246.00 |
| mobilenet_v3_small | 10.38 | 18.51 | 8.14 | 28.00 |
| resnet50 | 98.30 | 109.37 | 11.07 | 142.00 |
| resnet101 | 171.23 | 181.98 | 10.75 | 196.00 |
| vgg16 | 529.62 | 566.51 | 36.89 | 578.00 |

---

## Static estimator status (C++ analyzer)

### What it currently outputs

The analyzer prints three numbers:

- **Parameter bytes**: bytes attributable to model weights
- **Peak activations**: peak live bytes from the liveness simulation
- **Total peak**: `params + peak_activations`

Current output (same models / shape):

| Model | Parameter bytes (MiB) | Peak activations (MiB) | Total peak (MiB) |
|---|---:|---:|---:|
| alexnet | 233.081207 | 233.081207 | 466.162415 |
| mobilenet_v3_small | 9.746689 | 9.746429 | 19.493118 |
| resnet50 | 97.695381 | 97.692047 | 195.387428 |
| resnet101 | 170.344208 | 170.343414 | 340.687622 |
| vgg16 | 527.792145 | 527.792145 | 1055.584290 |

### Why static and runtime don’t match yet

For inference, the most meaningful runtime proxy for “activation peak” is:

- **runtime activation peak ≈ peak delta (allocated)**

**Known root cause (current):**  
When TorchScript graphs include weights as constants/attributes (especially after `torch.jit.freeze()`),
the liveness simulation is treating some of those weight tensors as if they were activations.  
If weights are also counted separately as “Parameter bytes”, this can lead to an **effective 2× weights**
effect in the reported totals.

## Notes

- `reserved` is allocator policy (caching / fragmentation) and is expected to be >= `allocated`.
- The static activation numbers above are **not correct yet**. Next steps are to separate constants/weights from activations and add alias handling.

---

## Future direction

- Broaden model coverage beyond common CV architectures.
- Add an exporter based on `torch.export` (TorchScript is deprecated in recent PyTorch).
- Investigate a `torch.compile` / AOTAutograd path for more robust graphs and metadata.
- Add a memory timeline visualization (per-node live set).

---

## References

- PyTorch CUDA semantics (allocated vs reserved): https://docs.pytorch.org/docs/stable/notes/cuda.html
- `torch.cuda.memory.max_memory_reserved`: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html
- `torch.cuda.memory.reset_peak_memory_stats`: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.reset_peak_memory_stats.html
- TorchScript deprecation notice: https://docs.pytorch.org/docs/stable/jit.html
- `torch.jit.freeze` (inlines parameters as constants): https://docs.pytorch.org/docs/stable/generated/torch.jit.freeze.html
