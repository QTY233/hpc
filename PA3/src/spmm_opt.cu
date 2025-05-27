#include "spmm_opt.h"

#define B(i, j) vin[(i) * INFEATURE + (j)]
#define C(i, j) vout[(i) * INFEATURE + (j)]

#define shared_size_32 128
#define shared_size_256 512

__global__ void spmm_kernel_placeholder_32(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    int i = blockIdx.x, j = threadIdx.x;
    int begin = ptr[i], end = ptr[i + 1];
    if (end <= begin) return;
    float res = 0;
    __shared__ float sval[shared_size_32];
    __shared__ int sidx[shared_size_32];
    for (int p = begin; p < end; p+= shared_size_32) {
        __syncthreads();
        int p_r = min(p + shared_size_32, end);
        for (int q = 0; q < 4; ++q) {
            if (p + j * 4 + q < p_r) {
                sval[j * 4 + q] = val[p + j * 4 + q];
                sidx[j * 4 + q] = idx[p + j * 4 + q];
            }
        }
        __syncthreads();
        for (int q = p; q < p_r; ++q) {
            res += sval[q - p] * B(sidx[q - p], j);
        }
    }
    C(i, j) = res;
}

__global__ void spmm_kernel_placeholder_256(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    int i = blockIdx.x, j = threadIdx.x;
    int begin = ptr[i], end = ptr[i + 1];
    if (end <= begin) return;
    float res = 0;
    __shared__ float sval[shared_size_256];
    __shared__ int sidx[shared_size_256];
    for (int p = begin; p < end; p+= shared_size_256) {
        __syncthreads();
        int p_r = min(p + shared_size_256, end);
        for (int q = 0; q < 2; ++q) {
            if (p + j * 2 + q < p_r) {
                sval[j * 2 + q] = val[p + j * 2 + q];
                sidx[j * 2 + q] = idx[p + j * 2 + q];
            }
        }
        __syncthreads();
        for (int q = p; q < p_r; ++q) {
            res += sval[q - p] * B(sidx[q - p], j);
        }
    }
    C(i, j) = res;
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    // TODO: your code
    grid.x = num_v;     // M
    block.x = feat_in;  // K
}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: your code
    feat_in == 32 ? spmm_kernel_placeholder_32<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in)
        : spmm_kernel_placeholder_256<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}

#undef B
#undef C