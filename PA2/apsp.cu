// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#define MAX_P_SIZE 32

namespace {
__global__ void diag_floyed(int n, int p, int p_size, int *graph) {
    int i = p * p_size + threadIdx.y;
    int j = p * p_size + threadIdx.x;

    __shared__ int shared_int[MAX_P_SIZE][MAX_P_SIZE + 1];
    if (i < n && j < n) shared_int[threadIdx.y][threadIdx.x] = graph[i * n + j];
    else shared_int[threadIdx.y][threadIdx.x] = INT_MAX;
    __syncthreads();

    for (int k = p * p_size, k_off = 0; k < (p + 1) * p_size; ++k, ++k_off) {
        if (i < n && j < n && k < n) 
            shared_int[threadIdx.y][threadIdx.x] = min(shared_int[threadIdx.y][threadIdx.x], shared_int[k_off][threadIdx.x] + shared_int[threadIdx.y][k_off]);
        __syncthreads();
    }
    if (i < n && j < n) {
        graph[i * n + j] = shared_int[threadIdx.y][threadIdx.x];
    }
    __syncthreads();
}

__global__ void cross_floyed(int n, int p, int p_size, int *graph) {
    if (blockIdx.x == p) return;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    bool is_row = (blockIdx.y == 1);
    int i = is_row ? p * p_size + ty : blockIdx.x * p_size + ty;
    int j = is_row ? blockIdx.x * p_size + tx : p * p_size + tx;

    __shared__ int shared_pivot_row[MAX_P_SIZE][MAX_P_SIZE + 1];
    __shared__ int shared_pivot_col[MAX_P_SIZE][MAX_P_SIZE + 1];

    if ((p * p_size + ty) < n && j < n)
        shared_pivot_col[tx][ty] = graph[(p * p_size + ty) * n + j];
    else
        shared_pivot_col[tx][ty] = INT_MAX;
    if (i < n && (p * p_size + tx) < n)
        shared_pivot_row[ty][tx] = graph[i * n + p * p_size + tx];
    else
        shared_pivot_row[ty][tx] = INT_MAX;

    __syncthreads();

    if (i < n && j < n) {
        int val = graph[i * n + j];

        for (int k = 0; k < p_size; ++k) if (p * p_size + k < n) {
            int temp;
            if (is_row)
                temp = shared_pivot_row[ty][k] + shared_pivot_col[tx][k];
            else
                temp = shared_pivot_row[ty][k] + shared_pivot_col[tx][k];
            val = min(val, temp);
        }

        graph[i * n + j] = val;
    }
}

// __global__ void cross_floyed_col(int n, int p, int p_size, int *graph) {
//     if (blockIdx.x == p) return;
//     int i = blockIdx.x * p_size + threadIdx.y;
//     int j = p * p_size + threadIdx.x;
//     if (i >= n || j >= n) return;

//     __shared__ int shared_col[MAX_P_SIZE];

//     for (int k = p * p_size; k < (p + 1) * p_size; ++k) {
//         if (k >= n) break;

//         if (threadIdx.y == 0) shared_col[threadIdx.x] = graph[k * n + j];
//         __syncthreads();

//         int d_ik = graph[i * n + k];
//         int d_kj = shared_col[threadIdx.x];
//         int &d_ij = graph[i * n + j];
//         if (d_ik + d_kj < d_ij)
//             d_ij = d_ik + d_kj;
//         __syncthreads();
//     }
// }

// __global__ void cross_floyed_row(int n, int p, int p_size, int *graph) {
//     if (blockIdx.x == p) return;
//     int i = p * p_size + threadIdx.y;
//     int j = blockIdx.x * p_size + threadIdx.x;
//     if (i >= n || j >= n) return;

//     __shared__ int shared_row[MAX_P_SIZE];

//     for (int k = p * p_size; k < (p + 1) * p_size; ++k) {
//         if (k >= n) break;

//         if (threadIdx.x == 0) shared_row[threadIdx.y] = graph[i * n + k];
//         __syncthreads();

//         int d_ik = shared_row[threadIdx.y];
//         int d_kj = graph[k * n + j];
//         int &d_ij = graph[i * n + j];
//         if (d_ik + d_kj < d_ij)
//             d_ij = d_ik + d_kj;
//         __syncthreads();
//     }
// }

__global__ void others_floyed(int n, int p, int p_size, int *graph) {
    if (blockIdx.y == p || blockIdx.x == p) return;
    int i = blockIdx.y * p_size + threadIdx.y;
    int j = blockIdx.x * p_size + threadIdx.x;
    if (i >= n || j >= n) return;

    __shared__ int shared_row[MAX_P_SIZE];
    __shared__ int shared_col[MAX_P_SIZE];

    for (int k = p * p_size; k < (p + 1) * p_size; ++k) {
        if (k >= n) break;

        shared_row[threadIdx.y] = graph[i * n + k];
        shared_col[threadIdx.x] = graph[k * n + j];
        __syncthreads();

        int d_ik = shared_row[threadIdx.y];
        int d_kj = shared_col[threadIdx.x];
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
        // cross_floyed_col<<<blk_size, thr>>>(n, p, MAX_P_SIZE, graph);
        // cross_floyed_row<<<blk_size, thr>>>(n, p, MAX_P_SIZE, graph);
        // cudaDeviceSynchronize();
        dim3 blk1(blk_size, 2);
        cross_floyed<<<blk1, thr>>>(n, p, MAX_P_SIZE, graph);
        cudaDeviceSynchronize();
        dim3 blk(blk_size, blk_size);
        others_floyed<<<blk, thr>>>(n, p, MAX_P_SIZE, graph);
        cudaDeviceSynchronize();
    }
}

// namespace {

// __global__ void kernel(int n, int k, int *graph) {
//     auto i = blockIdx.y * blockDim.y + threadIdx.y;
//     auto j = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n && j < n && i != k && j != k) {
//         graph[i * n + j] = min(graph[i * n + j], graph[i * n + k] + graph[k * n + j]);
//     }
// }

// }

// void apsp(int n, /* device */ int *graph) {
//     for (int k = 0; k < n; k++) {
//         dim3 thr(32, 32);
//         dim3 blk((n - 1) / 32 + 1, (n - 1) / 32 + 1);
//         kernel<<<blk, thr>>>(n, k, graph);
//     }
// }
    
    