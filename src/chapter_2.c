// Going Faster: Matrix Multiply in C — Chapter 2
// Figure 2.43 — Patterson & Hennessy, Computer Organization and Design
//
// Compile: gcc -O3 -o bin/chapter_2 src/chapter_2.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Number of benchmark repetitions (drops min and max).
static const int kDefaultReps = 5;
static const int kDefaultN    = 256;

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

// Fills matrix M (size n*n) with deterministic values.
static void init_matrix(int n, double *m) {
  for (int i = 0; i < n * n; i++) {
    m[i] = (i + 1) * 0.001;
  }
}

// Double-precision General Matrix Multiply — Figure 2.43.
// C[i][j] += A[i][k] * B[k][j] for all i, j, k.
// Matrices stored as 1D arrays in column-major order.
static void dgemm(int n, double *a, double *b, double *c) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double cij = c[i + j * n];
      for (int k = 0; k < n; k++) {
        cij += a[i + k * n] * b[k + j * n];
      }
      c[i + j * n] = cij;
    }
  }
}

// Runs dgemm `reps` times, drops min and max, returns average time in ms.
static double run_benchmark(int n, int reps) {
  double *a = malloc(n * n * sizeof(double));
  double *b = malloc(n * n * sizeof(double));
  double *c = calloc(n * n, sizeof(double));

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
  fprintf(fp, "2,c_baseline,%d,%.2f,%.4f\n", n, elapsed_ms, gflops);
  fclose(fp);
}

int main(int argc, char *argv[]) {
  int n    = (argc > 1) ? atoi(argv[1]) : kDefaultN;
  int reps = (argc > 2) ? atoi(argv[2]) : kDefaultReps;

  printf("\nDGEMM C baseline — n=%d, %d repetitions\n", n, reps);
  printf("----------------------------------------\n");

  double avg_ms = run_benchmark(n, reps);
  double gflops = compute_gflops(n, avg_ms);

  printf("----------------------------------------\n");
  printf("Average: %.2f ms  |  %.4f GFLOP/s\n", avg_ms, gflops);

  save_result(n, avg_ms, gflops);
  printf("Result saved to results/benchmark.csv\n");

  return 0;
}