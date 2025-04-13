#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

inline int floatToInt(float x) {
    int i;
    std::memcpy(&i, &x, sizeof(float));
    return i >= 0 ? i : ~i;
}

float IntToFloat(int ordered) {
    int i = (ordered >= 0) ? ordered : ~ordered;
    float f;
    std::memcpy(&f, &i, sizeof(float));
    return f;
}

void Worker::sort() {
    // TODO: implement the odd-even sort algorithm here
    int* data_int = new int[block_len];
    int* temp_data = new int[block_len];
    int* sorted_data = new int[block_len << 1];
    for (size_t i = 0; i < block_len; ++i) data_int[i] = floatToInt(data[i]);
        
    for (int step = 0; step < nprocs; ++step) {
        if (step == 0)  {
            std::sort(data_int, data_int + block_len);
        }

        if (step & 1) {
            if (rank & 1) {
                if (last_rank) continue;
                MPI_Sendrecv(data_int, block_len, MPI_INT, rank + 1, 0,
                    temp_data, block_len, MPI_INT, rank + 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                if (!rank) continue;
                MPI_Sendrecv(data_int, block_len, MPI_INT, rank - 1, 0,
                    temp_data, block_len, MPI_INT, rank - 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            if (rank & 1) {
                MPI_Sendrecv(data_int, block_len, MPI_INT, rank - 1, 0,
                    temp_data, block_len, MPI_INT, rank - 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                if (last_rank) continue;
                MPI_Sendrecv(data_int, block_len, MPI_INT, rank + 1, 0,
                    temp_data, block_len, MPI_INT, rank + 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            }
        }

        if ((rank + step) & 1) {
            if (data_int[0] < temp_data[block_len - 1]) {
                size_t i = 0, j = 0, k = 0;
                while (i < block_len && j < block_len) {
                    if (data_int[i] < temp_data[j]) sorted_data[k++] = data_int[i++];
                    else sorted_data[k++] = temp_data[j++];
                }
                while (i < block_len) sorted_data[k++] = data_int[i++];
                while (j < block_len) sorted_data[k++] = temp_data[j++];
                std::copy(sorted_data + block_len, sorted_data + (block_len << 1), data_int);
            }
        } else {
            if (temp_data[0] < data_int[block_len - 1]) {
                size_t i = 0, j = 0, k = 0;
                while (i < block_len && j < block_len) {
                    if (data_int[i] < temp_data[j]) sorted_data[k++] = data_int[i++];
                    else sorted_data[k++] = temp_data[j++];
                }
                while (i < block_len) sorted_data[k++] = data_int[i++];
                while (j < block_len) sorted_data[k++] = temp_data[j++];
                std::copy(sorted_data, sorted_data + block_len, data_int);
            }
        }
    }
    for (size_t i = 0; i < block_len; ++i) data[i] = IntToFloat(data_int[i]);
    delete[] data_int;
    delete[] temp_data;
    delete[] sorted_data;
}
/*private:
    int nprocs, rank;
    size_t n, block_len;
    float *data;
    bool last_rank, out_of_range;*/