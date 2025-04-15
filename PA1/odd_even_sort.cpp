#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <cstring>

#include "worker.h"

unsigned float_to_uint(float f) {
    unsigned u;
    memcpy(&u, &f, sizeof(float));
    return (u & 0x80000000) ? ~u : (u ^ 0x80000000);
}

float uint_to_float(unsigned u) {
    u = (u & 0x80000000) ? (u ^ 0x80000000) : ~u;
    float f;
    memcpy(&f, &u, sizeof(float));
    return f;
}

void Worker::sort() {
    // TODO: implement the odd-even sort algorithm here
    if (out_of_range) return;
    int max_block_len = n / nprocs + (n % nprocs > 0 ? 1 : 0);
    // std::cerr << "rank" << rank << " max_block_len: " << max_block_len << std::endl;
    unsigned* data_int = new unsigned[max_block_len];
    unsigned* temp_data = new unsigned[max_block_len];
    unsigned* sorted_data = new unsigned[max_block_len << 1];

    for (size_t i = 0; i < block_len; ++i) data_int[i] = float_to_uint(data[i]);
    size_t count[256];

    for (int pass = 0; pass < 4; ++pass) {
        memset(count, 0, sizeof(count));
        for (size_t i = 0; i < block_len; ++i) 
            count[(data_int[i] >> (pass * 8)) & 0xFF]++;
        for (int i = 1; i < 256; ++i) 
            count[i] += count[i - 1];
        for (ssize_t i = block_len - 1; i >= 0; --i) 
            temp_data[--count[(data_int[i] >> (pass * 8)) & 0xFF]] = data_int[i];
        memcpy(data_int, temp_data, block_len * sizeof(unsigned int));
    }

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

            MPI_Sendrecv(data_int, send_num, MPI_INT, rank - 1, 0,
                    temp_data, receive_num, MPI_INT, rank - 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            size_t i = 0, j = 0, k = 0;
            while (i < (size_t)send_num && j < (size_t)receive_num) {
                if (data_int[i] < temp_data[j]) sorted_data[k++] = data_int[i++];
                else sorted_data[k++] = temp_data[j++];
            }
            while (i < (size_t)send_num) sorted_data[k++] = data_int[i++];
            while (j < (size_t)receive_num) sorted_data[k++] = temp_data[j++];
            std::memcpy(data_int, sorted_data + receive_num, send_num * sizeof(unsigned));
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

            MPI_Sendrecv(data_int + block_len - send_num, send_num, MPI_INT, rank + 1, 0,
                    temp_data, receive_num, MPI_INT, rank + 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            size_t i = block_len - send_num, j = 0, k = 0;
            while (i < block_len && j < (size_t)receive_num) {
                if (data_int[i] < temp_data[j]) sorted_data[k++] = data_int[i++];
                else sorted_data[k++] = temp_data[j++];
            }
            while (i < block_len) sorted_data[k++] = data_int[i++];
            while (j < (size_t)receive_num) sorted_data[k++] = temp_data[j++];
            std::memcpy(data_int + (block_len - send_num), sorted_data, send_num * sizeof(unsigned));
        }
    }
    for (size_t i = 0; i < block_len; ++i) data[i] = uint_to_float(data_int[i]);
    delete[] data_int;
    delete[] temp_data;
    delete[] sorted_data;
}
/*private:
    int nprocs, rank;
    size_t n, block_len;
    float *data;
    bool last_rank, out_of_range;*/