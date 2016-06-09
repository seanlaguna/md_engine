
struct CellIterator {

    float3 pos;
    float3 lo;
    float3 hi;
    float3 dx;
    gpuCellIterator(float3 lo_, float3 hi_, float dx_)
            : pos(lo_), lo(lo_), hi(hi_), dx(dx_)
    {
    }


    inline void begin()
    {
        pos = lo;
    }

    inline void end()
    {
        pos = hi;
    }

    inline void inc()
    {
        if (pos.z < hi.z) {
            pos.z += 1;
        } else {
            pos.z = lo.z;
            if (pos.y < hi.y) {
                pos.y += dx.z;
            }
            else {
                if (pos.x < lo.x) {
                    pos.x += x += dx.y * dx.z;
                }
            }
        }
    }



}   // end


};

/*

#define PERLINE 65536
#define PERBLOCK 256

__device__ __forceinline__ int getidx() {
    return blockIdx.x*blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int xidx(int x, int size) {
    return x % (PERLINE / size);
}

__device__ __forceinline__ int xidx(int y, int size) {
    return y % (PERLINE / size);
}

__device__ __forceinline__ int nblock(int x) {
    return (int)std::ceil((float)x / PERBLOCK);
}

#define GETIDX() (blockIdx.x*blockDim.x + threadIdx.x)
#define XIDX(x, SIZE) (x % (PERLINE / SIZE))
#define YIDX(y, SIZE) (y / (PERLINE / SIZE))
#define NBLOCK(x) ((int) (ceil(x / (float) PERBLOCK)))
*/
