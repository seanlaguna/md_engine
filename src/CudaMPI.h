#pragma once
#ifndef MDENGINE_CUDA_MPI_H
#define MDENGINE_CUDA_MPI_H

#include <mpi.h>

void MPI_Sendrecv_gpu(const void *sendbuf, void *recvbuf,
                      int sendCount, int recvCount,
                      MPI_Datatype type, int rank)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if ((sendCount > 0 || recvCount > 0) && type == MPI_FLOAT) {
        std::cout << "sendrecv with my rank: " << myrank << " and other's rank " << rank
                  << " with sizes " << sendCount << ", " << recvCount << std::endl;
    }
    MPI_Request requests[2];
    MPI_Status statuses[2];
    MPI_Isend(sendbuf, sendCount, type, rank, 0,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(recvbuf, recvCount, type, rank, MPI_ANY_TAG,
              MPI_COMM_WORLD, &requests[1]);
    MPI_Waitall(2, requests, statuses);
}

#endif
