#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define M 1024
#define N 1024
#define K 1024
#define TILE_WIDTH 32

__global__ void matrixMul(float *a, float *b, float *c, int m, int n, int k)
{
    __shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_b[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float sum = 0.0;
    for (int i = 0; i < (k - 1) / TILE_WIDTH + 1; i++) {
        if (row < m && i * TILE_WIDTH + tx < k) {
            s_a[ty][tx] = a[row * k + i * TILE_WIDTH + tx];
        } else {
            s_a[ty][tx] = 0.0;
        }
        if (col < n && i * TILE_WIDTH + ty < k) {
            s_b[ty][tx] = b[(i * TILE_WIDTH + ty) * n + col];
        } else {
            s_b[ty][tx] = 0.0;
        }
        __syncthreads();
        for (int j = 0; j < TILE_WIDTH; j++) {
            sum += s_a[ty][j] * s_b[j][tx];
        }
        __syncthreads();
    }
    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

int main()
{
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    int size = M * K * sizeof(float);
    cudaMalloc((void **)&d_a, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    size = K * N * sizeof(float);
    cudaMalloc((void **)&d_b, size);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    size = M * N * sizeof(float);
    cudaMalloc((void **)&d_c, size);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N - 1) / dimBlock.x + 1, (M - 1) / dimBlock.y + 1);
    matrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
