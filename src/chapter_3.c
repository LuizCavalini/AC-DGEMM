/* Going Faster: Subword Parallelism — Chapter 3
 * Figure 3.19 — Patterson & Hennessy
 *
 * Compile: gcc -O3 -mavx2 -mfma -o bin/chapter_3 src/chapter_3.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>   /* intrínsecos AVX2 */

/* -----------------------------------------------------------
 * Figura 3.19 — DGEMM com intrínsecos AVX2
 *
 * Diferenças em relação ao Cap. 2:
 *  1. Loop i avança de 4 em 4 (largura do registrador YMM)
 *  2. Carregamos 4 doubles de A de uma vez com _mm256_load_pd
 *  3. Replicamos 1 elemento de B em 4 slots com _mm256_broadcast_sd
 *  4. Usamos FMA: c += a*b numa única instrução
 * ----------------------------------------------------------- */
void dgemm(int n, double *A, double *B, double *C)
{
    for (int i = 0; i < n; i += 4)          /* passo 4: processa 4 linhas por vez */
        for (int j = 0; j < n; j++)
        {
            /* carrega 4 elementos de C[i..i+3][j] no registrador YMM */
            __m256d c0 = _mm256_load_pd(C + i + j*n);

            for (int k = 0; k < n; k++)
            {
                /* carrega 4 elementos de A[i..i+3][k] */
                __m256d a = _mm256_load_pd(A + i + k*n);

                /* replica B[k][j] nos 4 slots do registrador */
                __m256d b = _mm256_broadcast_sd(B + k + j*n);

                /* c0 = a*b + c0  (FMA — uma instrução só) */
                c0 = _mm256_fmadd_pd(a, b, c0);
            }

            /* armazena os 4 resultados de volta em C */
            _mm256_store_pd(C + i + j*n, c0);
        }
}

/* -----------------------------------------------------------
 * Alocação alinhada — OBRIGATÓRIA para _mm256_load_pd
 * Os endereços precisam ser múltiplos de 32 bytes
 * ----------------------------------------------------------- */
double *alloc_aligned(int n)
{
    double *p = NULL;
    if (posix_memalign((void **)&p, 32, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc_aligned failed\n");
        exit(1);
    }
    memset(p, 0, n * sizeof(double));
    return p;
}

/* -----------------------------------------------------------
 * Utilitários (mesmos do Cap. 2)
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

double benchmark(int n, int reps)
{
    double *A = alloc_aligned(n * n);
    double *B = alloc_aligned(n * n);
    double *C = alloc_aligned(n * n);

    init_matrix(n, A);
    init_matrix(n, B);

    double tmin = 1e18, tmax = -1e18, soma = 0.0;

    for (int r = 0; r < reps; r++)
    {
        memset(C, 0, n * n * sizeof(double));

        double t0  = get_time_ms();
        dgemm(n, A, B, C);
        double dt  = get_time_ms() - t0;

        printf("  run %d/%d: %.2f ms\n", r + 1, reps, dt);

        if (dt < tmin) tmin = dt;
        if (dt > tmax) tmax = dt;
        soma += dt;
    }

    free(A); free(B); free(C);

    return (reps >= 3) ? (soma - tmin - tmax) / (reps - 2)
                       :  soma / reps;
}

void save_result(int n, double ms, double gf)
{
    FILE *fp = fopen("results/benchmark.csv", "a");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "3,avx2,%d,%.2f,%.4f\n", n, ms, gf);
    fclose(fp);
}

int main(int argc, char *argv[])
{
    int n    = (argc > 1) ? atoi(argv[1]) : 256;
    int reps = (argc > 2) ? atoi(argv[2]) : 5;

    /* n precisa ser múltiplo de 4 para o loop i += 4 funcionar */
    if (n % 4 != 0) {
        printf("Aviso: n ajustado de %d para %d (multiplo de 4)\n", n, n + (4 - n%4));
        n = n + (4 - n % 4);
    }

    printf("\nDGEMM AVX2 — n=%d, %d repetitions\n", n, reps);
    printf("----------------------------------------\n");

    double media_ms = benchmark(n, reps);
    double gf       = gflops(n, media_ms);

    printf("----------------------------------------\n");
    printf("Average: %.2f ms  |  %.4f GFLOP/s\n", media_ms, gf);

    save_result(n, media_ms, gf);
    printf("Result saved to results/benchmark.csv\n");

    return 0;
}
