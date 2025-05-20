// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#define MAX_P_SIZE (64)
#define Max_p_SIZE (4)

#define INF (400000)

namespace {
__global__ void diag_floyed(int n, int p, int *graph) {
    int tx = threadIdx.x * Max_p_SIZE;
    int ty = threadIdx.y * Max_p_SIZE;
    int i = p * MAX_P_SIZE + ty;
    int j = p * MAX_P_SIZE + tx;

    __shared__ int shared_int[MAX_P_SIZE][MAX_P_SIZE];
    register int local[Max_p_SIZE][Max_p_SIZE];
    #pragma unroll
    for (int k = 0; k < Max_p_SIZE; ++k) {
        #pragma unroll
        for (int l = 0; l < Max_p_SIZE; ++l) {
            if (k + i < n && l + j < n) 
                local[k][l] = graph[(k + i) * n + (l + j)];
            else
                local[k][l] = INF;
        }
    }
    #pragma unroll
    for (int k = 0; k < Max_p_SIZE; ++k) {
        #pragma unroll
        for (int l = 0; l < Max_p_SIZE; ++l) 
            shared_int[ty + k][tx + l] = local[k][l];
    }__syncthreads();
    
    register int need_ik[Max_p_SIZE], need_kj[Max_p_SIZE];
    for (int k_off = 0; k_off < MAX_P_SIZE; ++k_off) {
        if (p * MAX_P_SIZE + k_off < n) {
            #pragma unroll
            for (int l = 0; l < Max_p_SIZE; ++l) 
                need_ik[l] = shared_int[ty + l][k_off];
            #pragma unroll
            for (int l = 0; l < Max_p_SIZE; ++l) 
                need_kj[l] = shared_int[k_off][tx + l];
            
            #pragma unroll
            for (int l = 0; l < Max_p_SIZE; ++l) {
                #pragma unroll
                for (int m = 0; m < Max_p_SIZE; ++m) // if (i + l < n && j + m < n) 
                    local[l][m] = min(local[l][m], need_ik[l] + need_kj[m]);
            }
            #pragma unroll
            for (int l = 0; l < Max_p_SIZE; ++l) {
                #pragma unroll
                for (int m = 0; m < Max_p_SIZE; ++m) 
                    shared_int[ty + l][tx + m] = local[l][m];
            }
        } __syncthreads();
    }
    #pragma unroll
    for (int k = 0; k < Max_p_SIZE; ++k) {
        #pragma unroll
        for (int l = 0; l < Max_p_SIZE; ++l) 
            if (k + i < n && l + j < n) 
                graph[(k + i) * n + (l + j)] = local[k][l];
    }
}

__global__ void cross_floyed(int n, int p, int *graph, int *corss_row, int *corss_col) {
    if (blockIdx.x == p) return;
    int tx = threadIdx.x * Max_p_SIZE;
    int ty = threadIdx.y * Max_p_SIZE;
    bool is_row = (blockIdx.y == 1);
    int i = is_row ? p * MAX_P_SIZE + ty : blockIdx.x * MAX_P_SIZE + ty;
    int j = is_row ? blockIdx.x * MAX_P_SIZE + tx : p * MAX_P_SIZE + tx;

    __shared__ int shared_pivot_row[MAX_P_SIZE][MAX_P_SIZE];
    __shared__ int shared_pivot_col[MAX_P_SIZE][MAX_P_SIZE];
    register int local[Max_p_SIZE][Max_p_SIZE];
    #pragma unroll
    for (int k = 0; k < Max_p_SIZE; ++k) {
        #pragma unroll
        for (int l = 0; l < Max_p_SIZE; ++l) {
            int tmp = p * MAX_P_SIZE + ty + k;
            if (tmp < n && j + l < n) shared_pivot_col[ty + k][tx + l] = graph[tmp * n + j + l];
            else shared_pivot_col[ty + k][tx + l] = INF;
        }
    }
    #pragma unroll
    for (int k = 0; k < Max_p_SIZE; ++k) {
        #pragma unroll
        for (int l = 0; l < Max_p_SIZE; ++l) {
            int tmp = p * MAX_P_SIZE + tx + l;
            if (i + k < n && tmp < n) shared_pivot_row[ty + k][tx + l] = graph[(i + k) * n + tmp];
            else shared_pivot_row[ty + k][tx + l] = INF;
        }
    }
    #pragma unroll
    for (int k = 0; k < Max_p_SIZE; ++k) {
        #pragma unroll
        for (int l = 0; l < Max_p_SIZE; ++l) {
            local[k][l] = is_row ? shared_pivot_col[ty + k][tx + l] : local[k][l] = shared_pivot_row[ty + k][tx + l];
        }
    }
    __syncthreads();
    
    register int need_ik[Max_p_SIZE], need_kj[Max_p_SIZE];
    for (int k = 0; k < MAX_P_SIZE; ++k) {
        if (p * MAX_P_SIZE + k >= n) break; 
        for (int l = 0; l < Max_p_SIZE; ++l) {
            need_ik[l] = shared_pivot_row[ty + l][k];
        }
        
        for (int l = 0; l < Max_p_SIZE; ++l) {
            need_kj[l] = shared_pivot_col[k][tx + l];
        }
        for (int l = 0; l < Max_p_SIZE; ++l) 
            for (int m = 0; m < Max_p_SIZE; ++m) // if (i + l < n && j + m < n) {
                local[l][m] = min(local[l][m], need_ik[l] + need_kj[m]);
            // }
    }

    for (int k = 0; k < Max_p_SIZE; ++k) 
        for (int l = 0; l < Max_p_SIZE; ++l) 
            if (k + i < n && l + j < n) 
                graph[(k + i) * n + (l + j)] = local[k][l];
                
    for (int k = 0; k < Max_p_SIZE; ++k) 
        for (int l = 0; l < Max_p_SIZE; ++l) {
            if (is_row) corss_col[(ty + k) * n + (j + l)] = local[k][l];
            else corss_row[(i + k) * MAX_P_SIZE + (tx + l)] = local[k][l];
        }
}

__global__ void others_floyed(int n, int p, int *graph, int *corss_row, int *corss_col) {
    if (blockIdx.y == p || blockIdx.x == p) return;

    int tx = threadIdx.x * Max_p_SIZE;
    int ty = threadIdx.y * Max_p_SIZE;

    int i = blockIdx.y * MAX_P_SIZE + ty;
    int j = blockIdx.x * MAX_P_SIZE + tx;

    __shared__ int shared_row[MAX_P_SIZE][MAX_P_SIZE];
    __shared__ int shared_col[MAX_P_SIZE][MAX_P_SIZE];

    register int local[Max_p_SIZE][Max_p_SIZE];
    #pragma unroll
    for (int k = 0; k < Max_p_SIZE; ++k) {
        #pragma unroll
        for (int l = 0; l < Max_p_SIZE; ++l) {
            // shared_row[ty + k][tx + l] = corss_row[(i + k) * MAX_P_SIZE + (tx + l)];
            if (i + k < n && (p * MAX_P_SIZE + tx + l) < n)
                shared_row[ty + k][tx + l] = graph[(i + k) * n + p * MAX_P_SIZE + tx + l];
            else
                shared_row[ty + k][tx + l] = INF;
        }
    }
    #pragma unroll
    for (int k = 0; k < Max_p_SIZE; ++k) {
        #pragma unroll
        for (int l = 0; l < Max_p_SIZE; ++l) {
            // shared_col[ty + k][tx + l] = corss_col[(ty + k) * n + (j + l)];
            if ((p * MAX_P_SIZE + ty) + k < n && j + l < n)
                shared_col[ty + k][tx + l] = graph[(p * MAX_P_SIZE + ty + k) * n + j + l];
            else
                shared_col[ty + k][tx + l] = INF;
        }
    }
    for (int k = 0; k < Max_p_SIZE; ++k) 
        for (int l = 0; l < Max_p_SIZE; ++l) 
            local[k][l] = (i + k < n && j + l < n) ? graph[(i + k) * n + j + l] : INF;
        
    
    __syncthreads();

    register int need_ik[Max_p_SIZE], need_kj[Max_p_SIZE];
    for (int k_off = 0; k_off < MAX_P_SIZE; ++k_off) {
        if (p * MAX_P_SIZE + k_off >= n) break;
        #pragma unroll
        for (int l = 0; l < Max_p_SIZE; ++l) {
            need_ik[l] = shared_row[ty + l][k_off];
        }
        #pragma unroll
        for (int l = 0; l < Max_p_SIZE; ++l) {
            need_kj[l] = shared_col[k_off][tx + l];
        }
        #pragma unroll
        for (int l = 0; l < Max_p_SIZE; ++l) {
            #pragma unroll
            for (int m = 0; m < Max_p_SIZE; ++m) // if (i + l < n && j + m < n) {
                local[l][m] = min(local[l][m], need_ik[l] + need_kj[m]);
        }
            // }
    }

    for (int k = 0; k < Max_p_SIZE; ++k) 
        for (int l = 0; l < Max_p_SIZE; ++l) 
            if (k + i < n && l + j < n) 
                graph[(k + i) * n + (l + j)] = local[k][l];
}
}

void apsp(int n, /* device */ int *graph) {
    int blk_size = (n - 1) / MAX_P_SIZE + 1;
    int *corss_row, *corss_col;
    cudaMalloc(&corss_row, blk_size * MAX_P_SIZE * MAX_P_SIZE * sizeof(int));
    cudaMalloc(&corss_col, blk_size * MAX_P_SIZE * MAX_P_SIZE * sizeof(int));
    for (int p = 0; p * MAX_P_SIZE < n; p++) {
        dim3 thr_new(MAX_P_SIZE / Max_p_SIZE, MAX_P_SIZE / Max_p_SIZE);
        // dim3 thr(MAX_P_SIZE, MAX_P_SIZE);
        diag_floyed<<<1, thr_new>>>(n, p, graph);
        cudaDeviceSynchronize();
        dim3 blk1(blk_size, 2);
        cross_floyed<<<blk1, thr_new>>>(n, p, graph, corss_row, corss_col);
        cudaDeviceSynchronize();
        dim3 blk(blk_size, blk_size);
        others_floyed<<<blk, thr_new>>>(n, p, graph, corss_row, corss_col);
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
    
    