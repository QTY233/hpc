#include <chrono>
#include <iostream>
#include <mpi.h>
#include <time.h>
#include <cstring>
#include <cmath>
#include <algorithm>

#define EPS 1e-5

namespace ch = std::chrono;

void Ring_Allreduce(void* sendbuf, void* recvbuf, int n, MPI_Comm comm, int comm_sz, int my_rank)
{
    int nxt = (my_rank + 1) % comm_sz;
    int pre = (my_rank - 1 + comm_sz) % comm_sz;
    int step = n / comm_sz;
    float* temp_buf = new float[step+10]; 

    std::memcpy(recvbuf, sendbuf, n * sizeof(float));
    // std::cerr << "step is " << step << "n is " << n << std::endl;

    for (int i = 0; i < comm_sz - 1; ++i)
    {
        int send_offset = ((my_rank - i + comm_sz) % comm_sz) * step;
        int recv_offset = ((my_rank - i - 1 + comm_sz) % comm_sz) * step;
        // std::cerr << "my_rank is " << my_rank << " i is " << i << " send_offset is " << send_offset << " recv_offset is " << recv_offset << std::endl;

        MPI_Request send_req, recv_req;
        MPI_Isend((char*)recvbuf + send_offset * sizeof(float), step, MPI_FLOAT, nxt, 0, comm, &send_req);
        MPI_Irecv(temp_buf, step, MPI_FLOAT, pre, 0, comm, &recv_req);
        
        MPI_Wait(&send_req, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

        for (int j = 0; j < step; ++j)
            ((float*)recvbuf)[recv_offset + j] += temp_buf[j];
    }

    for (int i = 0; i < comm_sz - 1; ++i)
    {
        int send_offset = ((my_rank - i + 1 + comm_sz) % comm_sz) * step;
        int recv_offset = ((my_rank - i + comm_sz) % comm_sz) * step;
        // std::cerr << "In cycle2: my_rank is " << my_rank << " i is " << i << " send_offset is " << send_offset << " recv_offset is " << recv_offset << std::endl;

        MPI_Request send_req, recv_req;
        MPI_Isend((char*)recvbuf + send_offset * sizeof(float), step, MPI_FLOAT, nxt, 0, comm, &send_req);
        MPI_Irecv((char*)recvbuf + recv_offset * sizeof(float), step, MPI_FLOAT, pre, 0, comm, &recv_req);

        MPI_Wait(&send_req, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
    }

    delete[] temp_buf;
}

// reduce + bcast
void Naive_Allreduce(void* sendbuf, void* recvbuf, int n, MPI_Comm comm, int comm_sz, int my_rank)
{
    MPI_Reduce(sendbuf, recvbuf, n, MPI_FLOAT, MPI_SUM, 0, comm);
    MPI_Bcast(recvbuf, n, MPI_FLOAT, 0, comm);
}

int main(int argc, char *argv[])
{
    int ITER = atoi(argv[1]);
    int n = atoi(argv[2]);
    float* mpi_sendbuf = new float[n];
    float* mpi_recvbuf = new float[n];
    float* naive_sendbuf = new float[n];
    float* naive_recvbuf = new float[n];
    float* ring_sendbuf = new float[n];
    float* ring_recvbuf = new float[n];

    MPI_Init(nullptr, nullptr);
    int comm_sz;
    int my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    srand(time(NULL) + my_rank);
    for (int i = 0; i < n; ++i)
        mpi_sendbuf[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    memcpy(naive_sendbuf, mpi_sendbuf, n * sizeof(float));
    memcpy(ring_sendbuf, mpi_sendbuf, n * sizeof(float));

    //warmup and check
    MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    // std::cerr << "check1" << std::endl;
    Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
    // std::cerr << "check2" << std::endl;
    Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
    // std::cerr << "check3" << std::endl;
    bool correct = true;
    for (int i = 0; i < n; ++i)
        if (abs(mpi_recvbuf[i] - ring_recvbuf[i]) > EPS)
        {
            // std::cerr << "wrong i is " << i << " mpi is " << mpi_recvbuf[i] << " ring is " << ring_recvbuf[i] << std::endl;
            correct = false;
            break;
        }

    if (correct)
    {
        auto beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        auto end = ch::high_resolution_clock::now();
        double mpi_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms

        beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
        end = ch::high_resolution_clock::now();
        double naive_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms

        beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
        end = ch::high_resolution_clock::now();
        double ring_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms
        
        if (my_rank == 0)
        {
            std::cout << "Correct." << std::endl;
            std::cout << "MPI_Allreduce:   " << mpi_dur << " ms." << std::endl;
            std::cout << "Naive_Allreduce: " << naive_dur << " ms." << std::endl;
            std::cout << "Ring_Allreduce:  " << ring_dur << " ms." << std::endl;
        }
    }
    else
        if (my_rank == 0)
            std::cout << "Wrong!" << std::endl;

    delete[] mpi_sendbuf;
    delete[] mpi_recvbuf;
    delete[] naive_sendbuf;
    delete[] naive_recvbuf;
    delete[] ring_sendbuf;
    delete[] ring_recvbuf;
    MPI_Finalize();
    return 0;
}
