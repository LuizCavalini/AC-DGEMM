# Going Faster: PyTorch CPU and GPU DGEMM — Chapter 7 (extended)
# Patterson & Hennessy, Computer Organization and Design
#
# Compares PyTorch's optimized DGEMM against our manual implementations.
# PyTorch CPU uses MKL/OpenBLAS internally.
# PyTorch GPU uses cuBLAS via CUDA.
#
# Run: python3 src/pytorch.py [n] [reps]

import sys
import os
import csv
import time
import torch


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_gflops(n: int, elapsed_ms: float) -> float:
    """Returns throughput in GFLOP/s."""
    return (2.0 * n ** 3) / (elapsed_ms * 1e6)


def save_result(chapter: int, version: str, n: int,
                elapsed_ms: float, gflops: float) -> None:
    """Appends one row to results/benchmark.csv."""
    os.makedirs("results", exist_ok=True)
    csv_path = "results/benchmark.csv"
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["capitulo", "versao", "n", "tempo_ms", "gflops"])
        w.writerow([chapter, version, n, f"{elapsed_ms:.2f}", f"{gflops:.4f}"])


# ── Benchmark functions ───────────────────────────────────────────────────────

def benchmark_cpu(n: int, reps: int) -> float:
    """
    Runs torch.mm on CPU.
    PyTorch CPU uses MKL or OpenBLAS internally — highly optimized BLAS.
    Returns average elapsed time in ms (drops min and max).
    """
    a = torch.rand(n, n, dtype=torch.float64)
    b = torch.rand(n, n, dtype=torch.float64)

    # Warmup — ensures MKL thread pool is initialized before timing.
    _ = torch.mm(a, b)

    times = []
    for r in range(reps):
        t0 = time.monotonic()
        c = torch.mm(a, b)
        elapsed = (time.monotonic() - t0) * 1000.0
        times.append(elapsed)
        print(f"  [CPU] run {r+1}/{reps}: {elapsed:.2f} ms")

    if reps >= 3:
        times.remove(max(times))
        times.remove(min(times))

    return sum(times) / len(times)


def benchmark_gpu(n: int, reps: int) -> tuple[float, float]:
    """
    Runs torch.mm on GPU (cuBLAS).
    Returns (avg_compute_ms, avg_total_ms) where:
      - avg_compute_ms: GPU computation only (excludes data transfer)
      - avg_total_ms:   full pipeline (CPU→GPU transfer + compute + GPU→CPU)
    Uses torch.cuda.Event for accurate GPU timing.
    """
    a_cpu = torch.rand(n, n, dtype=torch.float64)
    b_cpu = torch.rand(n, n, dtype=torch.float64)

    # Warmup
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()
    _ = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()

    compute_times = []
    total_times   = []

    for r in range(reps):
        # ── Total time: includes CPU→GPU transfer + compute + GPU→CPU ──
        t0_total = time.monotonic()

        a_gpu = a_cpu.cuda()          # transfer A to VRAM
        b_gpu = b_cpu.cuda()          # transfer B to VRAM

        # ── Compute time only: GPU-side events ──
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)

        start_event.record()
        c_gpu = torch.mm(a_gpu, b_gpu)
        end_event.record()

        torch.cuda.synchronize()       # wait for GPU to finish
        compute_ms = start_event.elapsed_time(end_event)

        c_cpu = c_gpu.cpu()            # transfer result back to RAM
        total_ms = (time.monotonic() - t0_total) * 1000.0

        compute_times.append(compute_ms)
        total_times.append(total_ms)
        print(f"  [GPU] run {r+1}/{reps}: compute={compute_ms:.2f} ms  "
              f"total={total_ms:.2f} ms")

    def avg_drop(lst):
        if len(lst) >= 3:
            lst = sorted(lst)[1:-1]
        return sum(lst) / len(lst)

    return avg_drop(compute_times), avg_drop(total_times)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    n    = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"\nPyTorch DGEMM — n={n}, {reps} repetitions")
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024 ** 3
        print(f"GPU             : {props.name} ({vram_gb:.1f} GB VRAM)")

    # ── CPU benchmark ──────────────────────────────────────────────────────
    print("\n--- PyTorch CPU (MKL/OpenBLAS) ---")
    cpu_ms     = benchmark_cpu(n, reps)
    cpu_gflops = compute_gflops(n, cpu_ms)
    print(f"Average: {cpu_ms:.2f} ms  |  {cpu_gflops:.4f} GFLOP/s")
    save_result(7, "pytorch_cpu", n, cpu_ms, cpu_gflops)

    # ── GPU benchmark ──────────────────────────────────────────────────────
    if torch.cuda.is_available():
        print("\n--- PyTorch GPU (cuBLAS) ---")
        gpu_compute_ms, gpu_total_ms = benchmark_gpu(n, reps)

        gpu_compute_gflops = compute_gflops(n, gpu_compute_ms)
        gpu_total_gflops   = compute_gflops(n, gpu_total_ms)

        print(f"Compute only : {gpu_compute_ms:.2f} ms  |  "
              f"{gpu_compute_gflops:.4f} GFLOP/s")
        print(f"Total (w/ transfer): {gpu_total_ms:.2f} ms  |  "
              f"{gpu_total_gflops:.4f} GFLOP/s")

        save_result(7, "pytorch_gpu_compute", n, gpu_compute_ms, gpu_compute_gflops)
        save_result(7, "pytorch_gpu_total",   n, gpu_total_ms,   gpu_total_gflops)
    else:
        print("\nGPU not available, skipping GPU benchmark.")

    print("\nResults saved to results/benchmark.csv")


if __name__ == "__main__":
    main()
