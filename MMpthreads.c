#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define M 3
#define N 3
#define K 3

int A[M][K] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
int B[K][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
int C[M][N];

int num_threads = 2;

void* matrix_multiplication(void* arg) {
    int thread_id = (int) arg;
    int start = (thread_id * M) / num_threads;
    int end = ((thread_id + 1) * M) / num_threads;

    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < K; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    pthread_t threads[num_threads];
    int thread_id[num_threads];

    for (int i = 0; i < num_threads; i++) {
        thread_id[i] = i;
        pthread_create(&threads[i], NULL, matrix_multiplication, &thread_id[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}
