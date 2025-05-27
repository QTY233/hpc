#include "spmm_opt.h"

#define B(i, j) vin[(i) * INFEATURE + (j)]
#define C(i, j) vout[(i) * INFEATURE + (j)]

__global__ void spmm_kernel_placeholder(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    int i = blockIdx.x, j = threadIdx.x;
    int begin = ptr[i], end = ptr[i + 1];
    if (end <= begin) return;
    float res = 0;
    for (int p = begin; p < end; p++) {
        res += val[p] * B(idx[p], j);
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
    spmm_kernel_placeholder<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}

#undef B
#undef C