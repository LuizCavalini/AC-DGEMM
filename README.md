# AC-DGEMM — Matrix Multiplication Optimizations

**UFRJ — Escola Politécnica — Computer Architecture**

Implementation and benchmarking of DGEMM (Double-precision General Matrix Multiply) optimizations following the *Going Faster* sections of Chapters 1–6 of *Computer Organization and Design: RISC-V Edition* (Patterson & Hennessy, 2nd ed.), with an extension to PyTorch CPU and GPU.

## Results Summary

| Chapter | Version | GFLOP/s | Speedup vs Python | n |
|---------|---------|---------|-------------------|---|
| 1 | Python (baseline) | 0.040 | 1× | 128 |
| 2 | C (`-O3`) | 2.46 | ~62× | 128 |
| 3 | AVX2 SIMD | 11.15 | ~279× | 128 |
| 4 | AVX2 + Loop Unrolling | 21.70 | ~543× | 128 |
| 5 | Cache Blocking (`bs=64`) | 15.22 | — | 1024 |
| 6 | OpenMP (8 threads) | 85.36 | ~2134× | 1024 |
| 7 | PyTorch CPU (MKL) | 206.86 | ~5171× | 2048 |
| 7 | GPU compute (cuBLAS) | 108.15 | ~2704× | 2048 |
| 7 | GPU total (w/ PCIe) | 88.19 | ~2205× | 2048 |

## Repository Structure

```
AC-DGEMM/
├── src/
│   ├── chapter_1.py        # Python baseline
│   ├── chapter_2.c         # C baseline (-O3)
│   ├── chapter_3.c         # AVX2 SIMD intrinsics
│   ├── chapter_4.c         # AVX2 + loop unrolling ×4
│   ├── chapter_5.c         # Cache blocking (tiling)
│   ├── chapter_6.c         # OpenMP multi-core
│   └── pytorch.py          # PyTorch CPU (MKL) + GPU (cuBLAS)
└── README.md
```

## Platform

| Component | Details |
|-----------|---------|
| OS | Linux (Ubuntu/Debian) |
| Compiler | GCC with `-O3 -mavx2 -mfma -fopenmp` |
| CPU | Intel TigerLake — 4 cores / 8 threads (HT) |
| Cache | L1d: 48 KB/core — L2: 5 MB — L3: 8 MB |
| GPU | NVIDIA GeForce RTX 3050 Mobile (4 GB VRAM) |
| CUDA | 13.2 |
| PyTorch | 2.5.1 |

## How to Run

### C versions (Chapters 2–6)

```bash
# Chapter 2 — C baseline
gcc -O3 -o chapter_2 src/chapter_2.c && ./chapter_2

# Chapter 3 — AVX2
gcc -O3 -mavx2 -mfma -o chapter_3 src/chapter_3.c && ./chapter_3

# Chapter 4 — AVX2 + unrolling
gcc -O3 -mavx2 -mfma -o chapter_4 src/chapter_4.c && ./chapter_4

# Chapter 5 — Cache blocking
gcc -O3 -mavx2 -mfma -o chapter_5 src/chapter_5.c && ./chapter_5

# Chapter 6 — OpenMP
gcc -O3 -mavx2 -mfma -fopenmp -o chapter_6 src/chapter_6.c && ./chapter_6
```

### Python (Chapter 1)

```bash
python3 src/chapter_1.py
```

### PyTorch CPU/GPU (Chapter 7)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
python3 src/pytorch.py
```

## Benchmark Methodology

Each measurement is the **average of 5 independent runs** with the minimum and maximum discarded, eliminating OS scheduling outliers. Timing uses `clock_gettime(CLOCK_MONOTONIC)` for C versions and `omp_get_wtime()` for OpenMP (wall-clock time, essential for parallel measurements). PyTorch GPU timings use `torch.cuda.Event` for microsecond precision on the device side.

## Key Techniques

- **Chapter 3 — SIMD/AVX2:** 256-bit YMM registers holding 4 `double`s; `_mm256_fmadd_pd` fuses multiply-add into a single instruction
- **Chapter 4 — ILP + Unrolling:** 4 independent accumulators hide the ~4-cycle FMA latency, keeping all execution units busy
- **Chapter 5 — Cache Blocking:** `BLOCKSIZE=64` keeps 3 blocks (A+B+C) within the 5 MB L2 cache, stabilizing throughput across all matrix sizes
- **Chapter 6 — OpenMP:** single `#pragma omp parallel for schedule(dynamic)` distributes outer loop iterations across 8 threads with no data races
- **Chapter 7 — PyTorch:** MKL uses AVX-512 and production-grade scheduling, reaching 206.86 GFLOP/s — 2.4× above our manual OpenMP

## Authors

Luiz Cavalini, Eduardo Viana, Henrique Kezen, Rafael Maurício

## References

- Patterson, D. A. & Hennessy, J. L. *Computer Organization and Design: RISC-V Edition*, 2nd ed., Elsevier, 2021.
- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- OpenMP API Specification v5.2: https://www.openmp.org/spec-html/5.2/
