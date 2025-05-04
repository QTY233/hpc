// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#define MAX_P_SIZE 32

namespace {
__global__ void diag_floyed(int n, int p, int p_size, int *graph) {
    int i = p * p_size + threadIdx.y;
    int j = p * p_size + threadIdx.x;
    if (i >= n || j >= n) return;

    __shared__ int shared_row[MAX_P_SIZE];
    __shared__ int shared_col[MAX_P_SIZE];

    for (int k = p * p_size; k < (p + 1) * p_size; ++k) {
        if (k >= n) break;
        if (threadIdx.x == 0) shared_col[threadIdx.y] = graph[i * n + k];
        if (threadIdx.y == 0) shared_row[threadIdx.x] = graph[k * n + j];
        __syncthreads();

        int d_ik = shared_row[threadIdx.y];
        int d_kj = shared_col[threadIdx.x];
        int &d_ij = graph[i * n + j];
        if (d_ik + d_kj < d_ij)
            d_ij = d_ik + d_kj;
        __syncthreads();
    }
}
__global__ void cross_floyed_col(int n, int p, int p_size, int *graph) {
    if (blockIdx.x == p) return;
    int i = blockIdx.x * p_size + threadIdx.y;
    int j = p * p_size + threadIdx.x;
    if (i >= n || j >= n) return;

    __shared__ int shared_col[MAX_P_SIZE];

    for (int k = p * p_size; k < (p + 1) * p_size; ++k) {
        if (k >= n) break;

        if (threadIdx.x == 0) shared_col[threadIdx.y] = graph[i * n + k];
        __syncthreads();

        int d_ik = graph[i * n + k];
        int d_kj = shared_col[threadIdx.x];
        int &d_ij = graph[i * n + j];
        if (d_ik + d_kj < d_ij)
            d_ij = d_ik + d_kj;
        __syncthreads();
    }
}

__global__ void cross_floyed_row(int n, int p, int p_size, int *graph) {
    if (blockIdx.x == p) return;
    int i = p * p_size + threadIdx.y;
    int j = blockIdx.x * p_size + threadIdx.x;
    if (i >= n || j >= n) return;

    __shared__ int shared_row[MAX_P_SIZE];

    for (int k = p * p_size; k < (p + 1) * p_size; ++k) {
        if (k >= n) break;

        if (threadIdx.x == 0) shared_row[threadIdx.y] = graph[i * n + k];
        __syncthreads();

        int d_ik = shared_row[threadIdx.y];
        int d_kj = graph[k * n + j];
        int &d_ij = graph[i * n + j];
        if (d_ik + d_kj < d_ij)
            d_ij = d_ik + d_kj;
        __syncthreads();
    }
}

__global__ void others_floyed(int n, int p, int p_size, int *graph) {
    if (blockIdx.y == p || blockIdx.x == p) return;
    int i = blockIdx.y * p_size + threadIdx.y;
    int j = blockIdx.x * p_size + threadIdx.x;
    if (i >= n || j >= n) return;

    __shared__ int shared_row[MAX_P_SIZE];
    __shared__ int shared_col[MAX_P_SIZE];

    for (int k = p * p_size; k < (p + 1) * p_size; ++k) {
        if (k >= n) break;

        shared_row[threadIdx.x] = graph[i * n + k];
        shared_col[threadIdx.y] = graph[k * n + j];
        __syncthreads();

        int d_ik = shared_row[threadIdx.x];
        int d_kj = shared_col[threadIdx.y];
        int &d_ij = graph[i * n + j];
        if (d_ik + d_kj < d_ij)
            d_ij = d_ik + d_kj;

        __syncthreads();
    }
}

}

void apsp(int n, /* device */ int *graph) {
    int blk_size = (n - 1) / MAX_P_SIZE + 1;
    for (int p = 0; p * MAX_P_SIZE < n; p++) {
        dim3 thr(MAX_P_SIZE, MAX_P_SIZE);
        diag_floyed<<<1, thr>>>(n, p, MAX_P_SIZE, graph);
        cudaDeviceSynchronize();
        cross_floyed_col<<<blk_size, thr>>>(n, p, MAX_P_SIZE, graph);
        cudaDeviceSynchronize();
        cross_floyed_row<<<blk_size, thr>>>(n, p, MAX_P_SIZE, graph);
        cudaDeviceSynchronize();
        dim3 blk(blk_size, blk_size);
        others_floyed<<<blk, thr>>>(n, p, MAX_P_SIZE, graph);
        cudaDeviceSynchronize();
    }
}

/*
for (int p=0;p*b<n;p++){
    for (int k = p * b; k < (p + 1) * b; k++)
        for (int i = p * b; i < (p + 1) * b; i++)
            for (int j = p * b; j < (p + 1) * b; j++)
                D[i][j] = min(D[i][j], D[i][k] + D[k][j]);

    for (int k = p * b; k < (p + 1) * b; k++) {
        for (int i = p * b; i < (p + 1) * b; i++) {
            for (int j = 0; j < p * b; j++)
                D[i][j] = min(D[i][j], D[i][k] + D[k][j]);
            for (int j = (p + 1) * b; j < n; j++)
                D[i][j] = min(D[i][j], D[i][k] + D[k][j]);
        }
        for (int j = p * b; j < (p + 1) * b; j++) {
            for (int i = 0; i < p * b; i++)
                D[i][j] = min(D[i][j], D[i][k] + D[k][j]);
            for (int i = (p + 1) * b; i < n; i++)
                D[i][j] = min(D[i][j], D[i][k] + D[k][j]);
        }
    }

    for (int k = p * b; k < (p + 1) * b; k++)
        for (int i, j ∈ [0, p * b) ∪ [(p + 1) * b, n))
            D[i][j] = min(D[i][j], D[i][k] + D[k][j]);
}
*/