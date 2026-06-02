// Going Faster: Instruction-Level Parallelism and Matrix Multiply — Chapter 4
// Figure 4.82 — Patterson & Hennessy, Computer Organization and Design
//
// Combines AVX2 intrinsics (Chapter 3) with loop unrolling to expose
// independent operations to the superscalar out-of-order processor.
//
// Compile: gcc -O3 -mavx2 -mfma -o bin/chapter_4 src/chapter_4.c -lm

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const int kDefaultReps  = 5;
static const int kDefaultN     = 256;
static const int kSimdWidth    = 4;   // AVX2: 4 doubles per YMM register
static const int kUnrollFactor = 4;   // process 4 columns of j per iteration
static const int kAlignment    = 32;  // bytes, required by _mm256_load_pd

// Returns current time in milliseconds.
static double get_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Returns throughput in GFLOP/s given matrix size and elapsed time.
static double compute_gflops(int n, double elapsed_ms) {
  return (2.0 * n * n * n) / (elapsed_ms * 1e6);
}

// Allocates n doubles aligned to kAlignment bytes.
static double *alloc_aligned(int n) {
  double *p = NULL;
  if (posix_memalign((void **)&p, kAlignment, n * sizeof(double)) != 0) {
    fprintf(stderr, "alloc_aligned: allocation failed\n");
    exit(EXIT_FAILURE);
  }
  memset(p, 0, n * sizeof(double));
  return p;
}

// Fills matrix m (size n*n) with deterministic values.
static void init_matrix(int n, double *m) {
  for (int i = 0; i < n * n; i++) {
    m[i] = (i + 1) * 0.001;
  }
}

// DGEMM with AVX2 + loop unrolling — Figure 4.82.
//
// Loop j is unrolled by kUnrollFactor: each iteration processes 4 columns
// simultaneously using 4 independent YMM accumulators (c0..c3).
//
// Why this helps: FMA has ~4 cycle latency. With one accumulator the CPU
// stalls waiting for the previous result. With 4 independent accumulators
// the CPU keeps all FMA units busy across the latency window.
static void dgemm(int n, double *a, double *b, double *c) {
  for (int i = 0; i < n; i += kSimdWidth) {
    for (int j = 0; j < n; j += kUnrollFactor) {
      // Load 4 independent accumulators — one per column of j.
      __m256d c0 = _mm256_load_pd(c + i + (j + 0) * n);
      __m256d c1 = _mm256_load_pd(c + i + (j + 1) * n);
      __m256d c2 = _mm256_load_pd(c + i + (j + 2) * n);
      __m256d c3 = _mm256_load_pd(c + i + (j + 3) * n);

      for (int k = 0; k < n; k++) {
        // Load 4 rows of A[i..i+3][k] — shared across all 4 columns.
        __m256d a_vec = _mm256_load_pd(a + i + k * n);

        // Each column uses a different scalar from B, broadcast to 4 slots.
        // All 4 FMAs below are independent — the CPU executes them in parallel.
        c0 = _mm256_fmadd_pd(a_vec, _mm256_broadcast_sd(b + k + (j + 0) * n), c0);
        c1 = _mm256_fmadd_pd(a_vec, _mm256_broadcast_sd(b + k + (j + 1) * n), c1);
        c2 = _mm256_fmadd_pd(a_vec, _mm256_broadcast_sd(b + k + (j + 2) * n), c2);
        c3 = _mm256_fmadd_pd(a_vec, _mm256_broadcast_sd(b + k + (j + 3) * n), c3);
      }

      // Store all 4 accumulators back to C.
      _mm256_store_pd(c + i + (j + 0) * n, c0);
      _mm256_store_pd(c + i + (j + 1) * n, c1);
      _mm256_store_pd(c + i + (j + 2) * n, c2);
      _mm256_store_pd(c + i + (j + 3) * n, c3);
    }
  }
}

// Runs dgemm `reps` times, drops min and max, returns average time in ms.
static double run_benchmark(int n, int reps) {
  double *a = alloc_aligned(n * n);
  double *b = alloc_aligned(n * n);
  double *c = alloc_aligned(n * n);

  init_matrix(n, a);
  init_matrix(n, b);

  double t_min = 1e18;
  double t_max = -1e18;
  double sum   = 0.0;

  for (int r = 0; r < reps; r++) {
    memset(c, 0, n * n * sizeof(double));

    double t0 = get_time_ms();
    dgemm(n, a, b, c);
    double dt = get_time_ms() - t0;

    printf("  run %d/%d: %.2f ms\n", r + 1, reps, dt);

    if (dt < t_min) t_min = dt;
    if (dt > t_max) t_max = dt;
    sum += dt;
  }

  free(a);
  free(b);
  free(c);

  return (reps >= 3) ? (sum - t_min - t_max) / (reps - 2) : sum / reps;
}

// Appends one result row to results/benchmark.csv.
static void save_result(int n, double elapsed_ms, double gflops) {
  FILE *fp = fopen("results/benchmark.csv", "a");
  if (!fp) {
    perror("fopen");
    return;
  }
  fprintf(fp, "4,avx2_unroll4,%d,%.2f,%.4f\n", n, elapsed_ms, gflops);
  fclose(fp);
}

int main(int argc, char *argv[]) {
  int n    = (argc > 1) ? atoi(argv[1]) : kDefaultN;
  int reps = (argc > 2) ? atoi(argv[2]) : kDefaultReps;

  // n must be a multiple of both kSimdWidth and kUnrollFactor.
  int step = kSimdWidth * kUnrollFactor;
  if (n % step != 0) {
    int adjusted = n + (step - n % step);
    printf("Note: n adjusted from %d to %d (must be multiple of %d)\n",
           n, adjusted, step);
    n = adjusted;
  }

  printf("\nDGEMM AVX2 + Unroll x%d — n=%d, %d repetitions\n",
         kUnrollFactor, n, reps);
  printf("----------------------------------------\n");

  double avg_ms = run_benchmark(n, reps);
  double gflops = compute_gflops(n, avg_ms);

  printf("----------------------------------------\n");
  printf("Average: %.2f ms  |  %.4f GFLOP/s\n", avg_ms, gflops);

  save_result(n, avg_ms, gflops);
  printf("Result saved to results/benchmark.csv\n");

  return 0;
}
