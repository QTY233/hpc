#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

void Worker::sort() {
    // TODO: implement the odd-even sort algorithm here
    float* temp_data = new float[block_len];
    float* sorted_data = new float[block_len << 1];
    for (size_t step = 0; step < nprocs; ++step) {
        if (step == 0) std::sort(data, data + block_len);
        if (step & 1) {
            if (rank & 1) {
                MPI_Sendrecv(data, block_len, MPI_FLOAT, rank + 1, 0,
                    temp_data, block_len, MPI_FLOAT, rank + 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (temp_data[0] < data[block_len - 1]) {
                    size_t i = 0, j = 0, k = 0;
                    while (i < block_len && j < block_len) {
                        if (data[i] < temp_data[j]) sorted_data[k++] = data[i++];
                        else sorted_data[k++] = temp_data[j++];
                    }
                    while (i < block_len) sorted_data[k++] = data[i++];
                    while (j < block_len) sorted_data[k++] = temp_data[j++];
                    std::copy(sorted_data, sorted_data + block_len, data);
                }
            } else {
                if (!rank) continue;
                MPI_Sendrecv(data, block_len, MPI_FLOAT, rank - 1, 0,
                    temp_data, block_len, MPI_FLOAT, rank - 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (data[0] < temp_data[block_len - 1]) {
                    size_t i = 0, j = 0, k = 0;
                    while (i < block_len && j < block_len) {
                        if (data[i] < temp_data[j]) sorted_data[k++] = data[i++];
                        else sorted_data[k++] = temp_data[j++];
                    }
                    while (i < block_len) sorted_data[k++] = data[i++];
                    while (j < block_len) sorted_data[k++] = temp_data[j++];
                    std::copy(sorted_data + block_len, sorted_data + (block_len << 1), data);
                }
            }
        } else {
            if (rank & 1) {
                MPI_Sendrecv(data, block_len, MPI_FLOAT, rank - 1, 0,
                    temp_data, block_len, MPI_FLOAT, rank - 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (data[0] < temp_data[block_len - 1]) {
                    size_t i = 0, j = 0, k = 0;
                    while (i < block_len && j < block_len) {
                        if (data[i] < temp_data[j]) sorted_data[k++] = data[i++];
                        else sorted_data[k++] = temp_data[j++];
                    }
                    while (i < block_len) sorted_data[k++] = data[i++];
                    while (j < block_len) sorted_data[k++] = temp_data[j++];
                    std::copy(sorted_data + block_len, sorted_data + (block_len << 1), data);
                }
            } else {
                if (last_rank) continue;
                MPI_Sendrecv(data, block_len, MPI_FLOAT, rank + 1, 0,
                    temp_data, block_len, MPI_FLOAT, rank + 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (temp_data[0] < data[block_len - 1]) {
                    size_t i = 0, j = 0, k = 0;
                    while (i < block_len && j < block_len) {
                        if (data[i] < temp_data[j]) sorted_data[k++] = data[i++];
                        else sorted_data[k++] = temp_data[j++];
                    }
                    while (i < block_len) sorted_data[k++] = data[i++];
                    while (j < block_len) sorted_data[k++] = temp_data[j++];
                    std::copy(sorted_data, sorted_data + block_len, data);
                }
            }
        }
    }
    delete[] temp_data;
    delete[] sorted_data;
}
/*private:
    int nprocs, rank;
    size_t n, block_len;
    float *data;
    bool last_rank, out_of_range;*/