#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <cstring>

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
    if (out_of_range) return;
    int max_block_len = n / nprocs + 1;
    std::cerr << n << " " << nprocs << " " << block_len << " " << max_block_len << std::endl;
    int* data_int = new int[max_block_len];
    int* temp_data = new int[max_block_len];
    int* sorted_data = new int[max_block_len << 1];
    for (size_t i = 0; i < block_len; ++i) data_int[i] = floatToInt(data[i]);
    std::sort(data_int, data_int + block_len);

    for (int step = 0; step < nprocs; ++step) {
        int send_num = 1, receive_num;
        if ((rank + step) & 1) {
            if (!rank) continue;
            MPI_Sendrecv(data_int, 1, MPI_INT, rank - 1, 2,
                temp_data + block_len - 1, 1, MPI_INT, rank - 1, 2,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (data_int[0] >= temp_data[block_len - 1]) continue;
            int l = 1, r = block_len, mid;
            while (l <= r) {
                mid = (l + r) >> 1;
                if (data_int[mid - 1] < temp_data[block_len - 1]) {
                    l = mid + 1;
                    send_num = mid;
                } else r = mid - 1;
            }
            MPI_Sendrecv(&send_num, 1, MPI_INT, rank - 1, 1,
                &receive_num, 1, MPI_INT, rank - 1, 1,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            if (last_rank) continue;
            MPI_Sendrecv(data_int + block_len - 1, 1, MPI_INT, rank + 1, 2,
                temp_data, 1, MPI_INT, rank + 1, 2,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (temp_data[0] >= data_int[block_len - 1]) continue;
            int l = 1, r = block_len, mid;
            while (l <= r) {
                mid = (l + r) >> 1;
                if (temp_data[0] < data_int[mid - 1]) {
                    r = mid - 1;
                    send_num = block_len - mid + 1;
                } else l = mid + 1;
            }
            MPI_Sendrecv(&send_num, 1, MPI_INT, rank + 1, 1,
                &receive_num, 1, MPI_INT, rank + 1, 1,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank == 6) std::cerr << "change num" << std::endl;

        if (step & 1) {
            if (rank & 1) {
                MPI_Sendrecv(data_int + block_len - send_num, send_num, MPI_INT, rank + 1, 0,
                    temp_data, receive_num, MPI_INT, rank + 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Sendrecv(data_int, send_num, MPI_INT, rank - 1, 0,
                    temp_data + block_len - receive_num, receive_num, MPI_INT, rank - 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            if (rank & 1) {
                MPI_Sendrecv(data_int, send_num, MPI_INT, rank - 1, 0,
                    temp_data + block_len - receive_num, receive_num, MPI_INT, rank - 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Sendrecv(data_int + block_len - send_num, send_num, MPI_INT, rank + 1, 0,
                    temp_data, receive_num, MPI_INT, rank + 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        if (rank == 6) std::cerr << "change data" << std::endl;

        if ((rank + step) & 1) {
            size_t i = 0, j = block_len - receive_num, k = 0;
            while (i < (size_t)send_num && j < block_len) {
                if (data_int[i] < temp_data[j]) sorted_data[k++] = data_int[i++];
                else sorted_data[k++] = temp_data[j++];
            }
            while (i < (size_t)send_num) sorted_data[k++] = data_int[i++];
            while (j < block_len) sorted_data[k++] = temp_data[j++];
            std::copy(sorted_data + receive_num, sorted_data + receive_num + send_num, data_int);
        } else {
            size_t i = block_len - send_num, j = 0, k = 0;
            while (i < block_len && j < (size_t)receive_num) {
                if (data_int[i] < temp_data[j]) sorted_data[k++] = data_int[i++];
                else sorted_data[k++] = temp_data[j++];
            }
            while (i < block_len) sorted_data[k++] = data_int[i++];
            while (j < (size_t)receive_num) sorted_data[k++] = temp_data[j++];
            std::copy(sorted_data, sorted_data + send_num, data_int + block_len - send_num);
        }
        if (rank == 6) std::cerr << "sort" << std::endl;
    }
    for (size_t i = 0; i < block_len; ++i) data[i] = IntToFloat(data_int[i]);
    delete[] data_int;
    if (rank == 6) std::cerr << "data_int" << std::endl;
    delete[] temp_data;
    if (rank == 6) std::cerr << "temp_data" << std::endl;
    delete[] sorted_data;
    if (rank == 6) std::cerr << "sorted_data" << std::endl;
}
/*private:
    int nprocs, rank;
    size_t n, block_len;
    float *data;
    bool last_rank, out_of_range;*/