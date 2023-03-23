#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 3
#define N 3
#define K 3

int A[M][K] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
int B[K][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
int C[M][N];

int num_threads = 2;

int main() {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < K; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}
