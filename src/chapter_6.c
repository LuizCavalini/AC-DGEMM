// Going Faster: Multiple Processors and Matrix Multiply — Chapter 6
// Patterson & Hennessy, Computer Organization and Design
//
// Adds OpenMP parallelism to the cache-blocked DGEMM from Chapter 5.
// A single #pragma divides the outer loop across all available cores.
//
// Compile: gcc -O3 -mavx2 -mfma -fopenmp -o bin/chapter_6 src/chapter_6.c -lm

#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const int kDefaultReps   = 5;
static const int kDefaultN      = 512;
static const int kDefaultBlock  = 64;
static const int kSimdWidth     = 4;
static const int kUnrollFactor  = 4;
static const int kAlignment     = 32;

// Returns current time in milliseconds using OpenMP timer.
// omp_get_wtime() is more accurate than clock_gettime for parallel code
// because it measures wall-clock time, not per-thread CPU time.
static double get_time_ms(void) {
  return omp_get_wtime() * 1000.0;
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

// Computes one block of C — identical to Chapter 5.
// Each thread calls this independently on its own submatrix region,
// so no synchronization is needed inside do_block.
static void do_block(int n, int bs, int si, int sj, int sk,
                     double *a, double *b, double *c) {
  for (int i = si; i < si + bs; i += kSimdWidth) {
    for (int j = sj; j < sj + bs; j += kUnrollFactor) {
      __m256d c0 = _mm256_load_pd(c + i + (j + 0) * n);
      __m256d c1 = _mm256_load_pd(c + i + (j + 1) * n);
      __m256d c2 = _mm256_load_pd(c + i + (j + 2) * n);
      __m256d c3 = _mm256_load_pd(c + i + (j + 3) * n);

      for (int k = sk; k < sk + bs; k++) {
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

// DGEMM with cache blocking + OpenMP parallelism.
//
// The only change from Chapter 5: #pragma omp parallel for on the outer loop.
// OpenMP splits iterations of sj across all available threads automatically.
// si and sk loops run independently per thread — no race conditions because
// each (si, sj, sk) triplet writes to a unique region of C.
static void dgemm(int n, int bs, double *a, double *b, double *c) {
  #pragma omp parallel for schedule(dynamic)
  for (int sj = 0; sj < n; sj += bs) {
    for (int sk = 0; sk < n; sk += bs) {
      for (int si = 0; si < n; si += bs) {
        do_block(n, bs, si, sj, sk, a, b, c);
      }
    }
  }
}

// Runs dgemm `reps` times, drops min and max, returns average time in ms.
static double run_benchmark(int n, int bs, int reps) {
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
    dgemm(n, bs, a, b, c);
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
static void save_result(int n, int bs, int threads,
                        double elapsed_ms, double gflops) {
  FILE *fp = fopen("results/benchmark.csv", "a");
  if (!fp) {
    perror("fopen");
    return;
  }
  fprintf(fp, "6,openmp_t%d_bs%d,%d,%.2f,%.4f\n",
          threads, bs, n, elapsed_ms, gflops);
  fclose(fp);
}

int main(int argc, char *argv[]) {
  int n    = (argc > 1) ? atoi(argv[1]) : kDefaultN;
  int bs   = (argc > 2) ? atoi(argv[2]) : kDefaultBlock;
  int reps = (argc > 3) ? atoi(argv[3]) : kDefaultReps;

  int step = kSimdWidth * kUnrollFactor;
  if (bs % step != 0) {
    fprintf(stderr, "Error: block_size must be a multiple of %d\n", step);
    return EXIT_FAILURE;
  }
  if (n % bs != 0) {
    int adjusted = n + (bs - n % bs);
    printf("Note: n adjusted from %d to %d (must be multiple of %d)\n",
           n, adjusted, bs);
    n = adjusted;
  }

  int num_threads = omp_get_max_threads();
  printf("\nDGEMM OpenMP — n=%d, block=%d, threads=%d, %d repetitions\n",
         n, bs, num_threads, reps);
  printf("----------------------------------------\n");

  double avg_ms = run_benchmark(n, bs, reps);
  double gflops = compute_gflops(n, avg_ms);

  printf("----------------------------------------\n");
  printf("Average: %.2f ms  |  %.4f GFLOP/s\n", avg_ms, gflops);
  printf("Threads used: %d\n", num_threads);

  save_result(n, bs, num_threads, avg_ms, gflops);
  printf("Result saved to results/benchmark.csv\n");

  return 0;
}
