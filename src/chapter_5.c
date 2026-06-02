// Going Faster: Cache Blocking and Matrix Multiply — Chapter 5
// Figure 5.21 — Patterson & Hennessy, Computer Organization and Design
//
// Divides matrices into BLOCKSIZE x BLOCKSIZE submatrices that fit in L1/L2
// cache, dramatically reducing cache misses for large matrices.
//
// Compile: gcc -O3 -mavx2 -mfma -o bin/chapter_5 src/chapter_5.c -lm

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const int kDefaultReps    = 5;
static const int kDefaultN       = 512;
static const int kDefaultBlock   = 64;  // tunable — see run_benchmark_sweep()
static const int kSimdWidth      = 4;   // AVX2: 4 doubles per YMM register
static const int kUnrollFactor   = 4;   // columns unrolled per j iteration
static const int kAlignment      = 32;  // bytes, required by _mm256_load_pd

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

// Computes one BLOCKSIZE x BLOCKSIZE submatrix of C — Figure 5.21 do_block.
// Applies AVX2 + unroll from Chapter 4 within each block.
// si, sj, sk: starting indices of the submatrices in A, B, C.
static void do_block(int n, int block_size, int si, int sj, int sk,
                     double *a, double *b, double *c) {
  for (int i = si; i < si + block_size; i += kSimdWidth) {
    for (int j = sj; j < sj + block_size; j += kUnrollFactor) {
      __m256d c0 = _mm256_load_pd(c + i + (j + 0) * n);
      __m256d c1 = _mm256_load_pd(c + i + (j + 1) * n);
      __m256d c2 = _mm256_load_pd(c + i + (j + 2) * n);
      __m256d c3 = _mm256_load_pd(c + i + (j + 3) * n);

      for (int k = sk; k < sk + block_size; k++) {
        __m256d a_vec = _mm256_load_pd(a + i + k * n);
        c0 = _mm256_fmadd_pd(a_vec, _mm256_broadcast_sd(b + k + (j + 0) * n), c0);
        c1 = _mm256_fmadd_pd(a_vec, _mm256_broadcast_sd(b + k + (j + 1) * n), c1);
        c2 = _mm256_fmadd_pd(a_vec, _mm256_broadcast_sd(b + k + (j + 2) * n), c2);
        c3 = _mm256_fmadd_pd(a_vec, _mm256_broadcast_sd(b + k + (j + 3) * n), c3);
      }

      _mm256_store_pd(c + i + (j + 0) * n, c0);
      _mm256_store_pd(c + i + (j + 1) * n, c1);
      _mm256_store_pd(c + i + (j + 2) * n, c2);
      _mm256_store_pd(c + i + (j + 3) * n, c3);
    }
  }
}

// DGEMM with cache blocking — Figure 5.21.
// Iterates over blocks of size block_size x block_size,
// calling do_block for each (si, sj, sk) triplet.
static void dgemm(int n, int block_size, double *a, double *b, double *c) {
  for (int sj = 0; sj < n; sj += block_size) {
    for (int sk = 0; sk < n; sk += block_size) {
      for (int si = 0; si < n; si += block_size) {
        do_block(n, block_size, si, sj, sk, a, b, c);
      }
    }
  }
}

// Runs dgemm `reps` times, drops min and max, returns average time in ms.
static double run_benchmark(int n, int block_size, int reps) {
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
    dgemm(n, block_size, a, b, c);
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
static void save_result(int n, int block_size, double elapsed_ms, double gflops) {
  FILE *fp = fopen("results/benchmark.csv", "a");
  if (!fp) {
    perror("fopen");
    return;
  }
  fprintf(fp, "5,blocked_bs%d,%d,%.2f,%.4f\n", block_size, n, elapsed_ms, gflops);
  fclose(fp);
}

int main(int argc, char *argv[]) {
  int n          = (argc > 1) ? atoi(argv[1]) : kDefaultN;
  int block_size = (argc > 2) ? atoi(argv[2]) : kDefaultBlock;
  int reps       = (argc > 3) ? atoi(argv[3]) : kDefaultReps;

  // n must be a multiple of block_size, which must be a multiple of
  // kSimdWidth * kUnrollFactor (= 16).
  int step = kSimdWidth * kUnrollFactor;
  if (block_size % step != 0) {
    fprintf(stderr, "Error: block_size must be a multiple of %d\n", step);
    return EXIT_FAILURE;
  }
  if (n % block_size != 0) {
    int adjusted = n + (block_size - n % block_size);
    printf("Note: n adjusted from %d to %d (must be multiple of %d)\n",
           n, adjusted, block_size);
    n = adjusted;
  }

  printf("\nDGEMM Cache Blocking — n=%d, block=%d, %d repetitions\n",
         n, block_size, reps);
  printf("----------------------------------------\n");

  double avg_ms = run_benchmark(n, block_size, reps);
  double gflops = compute_gflops(n, avg_ms);

  printf("----------------------------------------\n");
  printf("Average: %.2f ms  |  %.4f GFLOP/s\n", avg_ms, gflops);

  save_result(n, block_size, avg_ms, gflops);
  printf("Result saved to results/benchmark.csv\n");

  return 0;
}
