
class GPUGridSliceIterator
{
public:
    GridSliceIterator(const int3 &lo_, const int3 &hi_,
                      const int3 &ns_, const int3 &ds_)
            : pos(lo_), lo(lo_), hi(hi_), ns(ns_), ds(ds_)
    { }

    GridSliceIterator(const GridGPU &grid)
            : pos({0, 0, 0}), lo({0, 0, 0}), hi(grid.ns),
              ns(grid.ns), ds(grid.ds)
    { }

    GridSliceIterator(const int3 &lo_, const int3 &hi_,
                      const GridGPU &grid)
            : pos(lo_), lo(lo_), hi(hi_), ns(grid.ns), ds(grid.ds)
    { }

    inline __host__ __device__ GridSliceIterator begin() const
    {
        auto result(*this);
        result.pos = lo;
        return result;
    }
    inline __host__ __device__ GridSliceIterator end() const
    {
        auto result(*this);
        result.pos = hi;
        return result;
    }

    inline __host__ __device__ size_t operator*() const
    {
        return linearidx(pos, ns);
    }

    inline __host__ __device__ GridSliceIterator &operator++()
    {
        if (++pos.z >= hi.z) {
            pos.z = lo.z;
            if (pos.y += ds.z >= hi.y) {
                pos.y = lo.y;
                pos.x += ds.y * ds.z;
            }
        }
        return *this;
    }
    inline __host__ __device__ GridSliceIterator operator++(int)
    {
        auto result(*this);
        ++(*this);
        return result;
    }
    inline __host__ __device__ GridSliceIterator &operator+=(const size_t n)
    {
        for (size_t i = 0; i < n; ++i) { ++(*this); }
        return *this;
    }

private:
    int3 pos;
    const int3 lo;
    const int3 hi;
    const int3 ns,
    const int3 ds;

};  // end

class CellIterator
{
    friend class GridSliceIterator;
public:
    CellIterator(const int3 &lo_, const int3 &hi_,
                 const int3 &ns_, const int3 &ds_, const uint *cellOffs)
            : i(prod(lo_, ns_)), j(0),
              GridSlice(lo_, hi_, ns_, ds_), cellOffs(cellOffs_), cellCount(0)
    { }

    CellIterator(const int3 &lo_, const int3 &hi_, const GridGPU &grid)
            : i(prod(lo_, ns_)), j(0),
              GridSlice(lo_, hi_, grid),
              cellOffs(grid.perCellArray.getDevData()), cellCount(0)
    { }

    inline __host__ __device__ CellIterator begin() const
    {
        auto result(*this);
        result.i = prod(gridSlice.lo, gridSlice.ns);
        result.j = 0;
        result.gridSlice = gridSlice.begin();
        result.cellCount = 0;
        return result;
    }

    inline __host __device__ CellIterator end() const
    {
        auto result(*this);
        result.i = prod(gridSlice.hi);
        result.j = cellOffs[i];
        result.gridSlice = gridSlice.end();
        result.cellCount = prod(gridSlice.hi) - prod(gridSlice.lo);
        return result;
    }

    inline __host__ __device__ uint32_t operator*() const       { return j; }
    inline __host__ __device__ uint16_t getCellCount() const    { return cellCount; }

    inline __host__ __device__ CellIterator &operator++()
    {
        if (++j >= cellOffs[i+1]) {
            i = *(++gridSlice);
            j = cellOffs[i];
            ++cellCount;
        }
    }
    inline __host__ __device__ CellIterator operator++(int)
    {
        auto result(*this);
        ++(*this);
        return result;
    }
    inline __host__ __device__ GridSliceIterator &operator+=(const size_t n)
    {
        for (size_t m = 0; m < n; ++m) { ++(*this); }
        return *this;
    }

private:
    uint32_t i;
    uint32_t j;
    GridSliceIterator gridSlice;
    const uint32_t *const cellOffs;
    uint16_t cellCount;

};

class GhostIterator
{
public:
    // calculate lo and hi based on base and dir
    GhostIterator(float3 base, OOBDir dir, const int3 &ns, const int3 &ds,
                  const uint16_t *cellOffs)
        : cells(
    {
    }
    // full set of passthroughs to CellIterator

private:
    CellIterator cells;
    // we will use these to copy directly into the ghost vectors, to -> from
    uint32_t idxGPUData;
    uint32_t idxGhost;
};

class AdjacentIterator
{
    // iterator over adjacent cells in good order
    // handle ghosts appropriately; may need periodicity and some other things
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
