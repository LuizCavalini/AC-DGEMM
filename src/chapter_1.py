import time
import sys
import os
import csv

def dgemm_python(n, A, B, C):
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

def make_matrix(n, fill=0.0):
    return [[fill] * n for _ in range(n)]

def init_matrix(n):
    M = make_matrix(n)
    for i in range(n):
        for j in range(n):
            M[i][j] = (i * n + j + 1) * 0.001
    return M

def gflops(n, elapsed_s):
    return (2.0 * n**3) / (elapsed_s * 1e9)

def benchmark(n, repeticoes=5):
    """Roda o DGEMM várias vezes e retorna a média sem outliers."""
    tempos = []

    for i in range(repeticoes):
        # Reinicializa as matrizes a cada rodada para medir justo
        A = init_matrix(n)
        B = init_matrix(n)
        C = make_matrix(n)

        t0 = time.monotonic()
        dgemm_python(n, A, B, C)
        tempos.append(time.monotonic() - t0)

        print(f"  rodada {i+1}/{repeticoes}: {tempos[-1]*1000:.1f} ms")

    # Remove o maior e o menor antes de calcular a média
    if repeticoes >= 3:
        tempos.remove(max(tempos))
        tempos.remove(min(tempos))

    media = sum(tempos) / len(tempos)
    return media

def salvar_resultado(capitulo, versao, n, tempo_ms, gf):
    os.makedirs("results", exist_ok=True)
    csv_path = "results/benchmark.csv"
    escrever_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if escrever_header:
            w.writerow(["capitulo", "versao", "n", "tempo_ms", "gflops"])
        w.writerow([capitulo, versao, n, f"{tempo_ms:.1f}", f"{gf:.4f}"])

def main():
    n           = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    repeticoes  = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"\nDGEMM Python puro — n={n}, {repeticoes} repetições")
    print("-" * 40)

    media_s  = benchmark(n, repeticoes)
    tempo_ms = media_s * 1000
    gf       = gflops(n, media_s)

    print("-" * 40)
    print(f"Média: {tempo_ms:.1f} ms  |  {gf:.4f} GFLOP/s")

    salvar_resultado(1, "python", n, tempo_ms, gf)
    print(f"Resultado salvo em results/benchmark.csv")

if __name__ == "__main__":
    main()