#pragma once
#ifndef GLOBAL_DEFS_H
#define GLOBAL_DEFS_H

#include <boost/shared_ptr.hpp>

#define DEBUG

#include "cuda_runtime.h"
#include "Logging.h"

typedef double num;

#define SHARED(X) boost::shared_ptr<X>

template <typename T>
using b_shared_ptr = boost::shared_ptr<T>;

//some files get grumpy if this is within the if.  doesn't hurt to have it declared multiple times

#ifdef DEBUG
    // SAFECALL is designed for kernel calls
    #define SAFECALL(com) do { \
        CUT_CHECK_ERROR("before call " #com); \
        {com;} \
        CUT_CHECK_ERROR("after call " #com); \
    } while (false)

    // Check if error occured
    #define CUT_CHECK_ERROR(errmsg) do { \
        cudaDeviceSynchronize(); \
        cudaError_t err = cudaGetLastError(); \
        mdAssert(cudaSuccess == err, "Error " errmsg ": (%d): %s", (int)err, \
                                   cudaGetErrorString(err)); \
    } while(false)
#else
    #define SAFECALL(com) com
    #define CUT_CHECK_ERROR(err) do { } while (false)
#endif

// CUCHECK is designed for CUDA API calls
#define CUCHECK(com) do { \
    cudaError_t err = com; \
    mdAssert(cudaSuccess == err, "Error (%d) in CUDA call: %s", (int)err, \
                                cudaGetErrorString(err)); \
} while(false)

#define GETIDX() (blockIdx.x*blockDim.x + threadIdx.x)
#define PERLINE 65536
#define XIDX(x, SIZE) (x % (PERLINE / SIZE))
#define YIDX(y, SIZE) (y / (PERLINE / SIZE))
#define PERBLOCK 256
#define NBLOCK(x) ((int) (ceil(x / (float) PERBLOCK)))

//#define LINEARIDX(idx, ns) (ns.z*ns.y*idx.x + ns.z*idx.y + idx.z)

#endif
