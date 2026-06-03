// Going Faster: Subword Parallelism and Matrix Multiply — Chapter 3
// Figure 3.19 — Patterson & Hennessy, Computer Organization and Design
//
// Uses AVX2 intrinsics to process 4 doubles per cycle instead of 1.
//
// Compile: gcc -O3 -mavx2 -mfma -o bin/chapter_3 src/chapter_3.c -lm

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const int kDefaultReps  = 5;
static const int kDefaultN     = 256;
static const int kSimdWidth    = 4;  // AVX2: 256 bits / 64 bits per double
static const int kAlignment    = 32; // AVX2 loads require 32-byte alignment

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
// _mm256_load_pd requires addresses that are multiples of 32 bytes.
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

// DGEMM with AVX2 intrinsics — Figure 3.19.
//
// Key differences from Chapter 2:
//   - Outer loop advances i by 4 (SIMD width) instead of 1.
//   - _mm256_load_pd   : loads 4 doubles from A into one YMM register.
//   - _mm256_broadcast_sd: replicates one element of B across 4 slots.
//   - _mm256_fmadd_pd  : computes c += a*b as a single fused instruction.
//   - _mm256_store_pd  : stores 4 results back to C.
static void dgemm(int n, double *a, double *b, double *c) {
  for (int i = 0; i < n; i += kSimdWidth) {
    for (int j = 0; j < n; j++) {
      // Load 4 elements of C[i..i+3][j] into one YMM register.
      __m256d c0 = _mm256_load_pd(c + i + j * n);

      for (int k = 0; k < n; k++) {
        // Load 4 elements of A[i..i+3][k].
        __m256d a_vec = _mm256_load_pd(a + i + k * n);

        // Broadcast B[k][j] — same scalar replicated in all 4 slots.
        __m256d b_vec = _mm256_broadcast_sd(b + k + j * n);

        // c0 = a_vec * b_vec + c0  (fused multiply-add, one instruction).
        c0 = _mm256_fmadd_pd(a_vec, b_vec, c0);
      }

      // Store 4 results back to C[i..i+3][j].
      _mm256_store_pd(c + i + j * n, c0);
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
  fprintf(fp, "3,avx2,%d,%.2f,%.4f\n", n, elapsed_ms, gflops);
  fclose(fp);
}

int main(int argc, char *argv[]) {
  int n    = (argc > 1) ? atoi(argv[1]) : kDefaultN;
  int reps = (argc > 2) ? atoi(argv[2]) : kDefaultReps;

  // n must be a multiple of kSimdWidth for the i += 4 loop to be correct.
  if (n % kSimdWidth != 0) {
    int adjusted = n + (kSimdWidth - n % kSimdWidth);
    printf("Note: n adjusted from %d to %d (must be multiple of %d)\n",
           n, adjusted, kSimdWidth);
    n = adjusted;
  }

  printf("\nDGEMM AVX2 — n=%d, %d repetitions\n", n, reps);
  printf("----------------------------------------\n");

  double avg_ms = run_benchmark(n, reps);
  double gflops = compute_gflops(n, avg_ms);

  printf("----------------------------------------\n");
  printf("Average: %.2f ms  |  %.4f GFLOP/s\n", avg_ms, gflops);

  save_result(n, avg_ms, gflops);
  printf("Result saved to results/benchmark.csv\n");

  return 0;
}