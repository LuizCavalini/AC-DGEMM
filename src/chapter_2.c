/* Going Faster: Matrix Multiply in C — Chapter 2
 * Figure 2.43 — Patterson & Hennessy
 *
 * Compile: gcc -O3 -o bin/chapter_2 src/chapter_2.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

/* -----------------------------------------------------------
 * Figure 2.43 — exatamente como no livro
 *
 * As matrizes são vetores 1D em column-major order:
 * elemento [i][j] fica na posição i + j*n
 * ----------------------------------------------------------- */
void dgemm(int n, double *A, double *B, double *C)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            double cij = C[i + j*n];
            for (int k = 0; k < n; k++)
                cij += A[i + k*n] * B[k + j*n];
            C[i + j*n] = cij;
        }
}

/* -----------------------------------------------------------
 * Utilitários de benchmark
 * ----------------------------------------------------------- */
double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

double gflops(int n, double ms)
{
    return (2.0 * n * n * n) / (ms * 1e6);
}

void init_matrix(int n, double *M)
{
    for (int i = 0; i < n * n; i++)
        M[i] = (i + 1) * 0.001;
}

/* -----------------------------------------------------------
 * Benchmark: N repetições, descarta min e max, retorna média
 * ----------------------------------------------------------- */
double benchmark(int n, int reps)
{
    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = calloc(n * n, sizeof(double));

    init_matrix(n, A);
    init_matrix(n, B);

    double *tempos = malloc(reps * sizeof(double));
    double  tmin   =  1e18;
    double  tmax   = -1e18;
    double  soma   =  0.0;

    for (int r = 0; r < reps; r++)
    {
        memset(C, 0, n * n * sizeof(double));

        double t0    = get_time_ms();
        dgemm(n, A, B, C);
        tempos[r]    = get_time_ms() - t0;

        printf("  run %d/%d: %.1f ms\n", r + 1, reps, tempos[r]);

        if (tempos[r] < tmin) tmin = tempos[r];
        if (tempos[r] > tmax) tmax = tempos[r];
        soma += tempos[r];
    }

    double media = (reps >= 3) ? (soma - tmin - tmax) / (reps - 2)
                               :  soma / reps;

    free(A); free(B); free(C); free(tempos);
    return media;
}

/* -----------------------------------------------------------
 * Salva resultado no CSV
 * ----------------------------------------------------------- */
void save_result(int n, double ms, double gf)
{
    FILE *fp = fopen("results/benchmark.csv", "a");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "2,c_baseline,%d,%.1f,%.4f\n", n, ms, gf);
    fclose(fp);
}

int main(int argc, char *argv[])
{
    int n    = (argc > 1) ? atoi(argv[1]) : 256;
    int reps = (argc > 2) ? atoi(argv[2]) : 5;

    printf("\nDGEMM C baseline — n=%d, %d repetitions\n", n, reps);
    printf("----------------------------------------\n");

    double media_ms = benchmark(n, reps);
    double gf       = gflops(n, media_ms);

    printf("----------------------------------------\n");
    printf("Average: %.1f ms  |  %.4f GFLOP/s\n", media_ms, gf);

    save_result(n, media_ms, gf);
    printf("Result saved to results/benchmark.csv\n");

    return 0;
}