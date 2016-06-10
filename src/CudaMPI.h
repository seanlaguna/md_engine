#pragma once
#ifndef CUDA_MPI_LOL_H
#define CUDA_MPI_LOL_H

#include <mpi.h>

void MPI_Sendrecv_gpu(const void *sendbuf, void *recvbuf, 
        			  int sendCount, int recvCount, 
                      MPI_Datatype type, int rank)
{
    MPI_Request requests[2];
    MPI_Status statuses[2];
    MPI_Isend(sendbuf, sendCount, type, rank, 0,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(recvbuf, recvCount, type, rank, MPI_ANY_TAG,
              MPI_COMM_WORLD, &requests[1]);
    MPI_Waitall(2, requests, statuses);
}

#endif
