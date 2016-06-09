void MPI_Sendrecv_gpu(const void *sendbuf, void *recvbuf, 
        			  int count, MPI_Datatype type, int rank)
{
    MPI_Request requests[2];
    MPI_Status statuses[2];
    MPI_Isend(sendbuf, count, type, mpi_rank, 0,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(recvbuf, count, type, mpi_rank, MPI_ANY_TAG,
              MPI_COMM_WORLD, &requests[1]);
    MPI_Waitall(2, requests, statuses);
}

