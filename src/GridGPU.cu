#include "GridGPU.h"

#include "State.h"
#include "helpers.h"
#include "Bond.h"
#include "BoundsGPU.h"
#include "list_macro.h"
#include "Mod.h"
#include "Fix.h"
#include "cutils_func.h"
#include "cutils_math.h"


/* GridGPU members */

void GridGPU::initArrays() {
    perCellArray =
        GPUArrayGlobal<uint32_t>(prod(ns) + 1);
    perAtomArray =
        GPUArrayGlobal<uint16_t>(state->atoms.size()+1);
    // also cumulative sum, tracking cumul. sum of max per block
    perBlockArray =
        GPUArrayGlobal<uint32_t>(NBLOCK(state->atoms.size()) + 1);
    // not +1 on this one, isn't cumul sum
    perBlockArray_maxNeighborsInBlock =
        GPUArrayDeviceGlobal<uint16_t>(NBLOCK(state->atoms.size()));
    xsLastBuild =
        GPUArrayDeviceGlobal<float4>(state->atoms.size());

    // in prepare for run, you make GPU grid _after_ copying xs to device
    buildFlag = GPUArrayGlobal<int>(1);
    buildFlag.d_data.memset(0);
}


void GridGPU::initStream() {
    //std::cout << "initializing stream" << std::endl;
    //streamCreated = true;
    //CUCHECK(cudaStreamCreate(&rebuildCheckStream));
}


GridGPU::GridGPU() {
    streamCreated = false;
    //initStream();
}

GridGPU::GridGPU(State *state_, float3 ds_, float3 dsOrig_,
                 float3 os_, int3 ns_, float maxRCut_)
  : state(state_), ds(ds_), dsOrig(dsOrig_), os(os_), ns(ns_),
    neighCutoffMax(maxRCut_ + state->padding) {

    streamCreated = false;
    initArrays();
    initStream();
    handleExclusions();
    numChecksSinceLastBuild = 0;
}

GridGPU::GridGPU(State *state_, float dx_, float dy_, float dz_)
  : state(state_) {

    streamCreated = false;
    Vector trace = state->bounds.trace;  // EEHHHHH SHOULD CHANGE TO BOUNDSGPU, but it doesn't really matter because you initialize them at the same time.  FOR NOW
    Vector attemptDDim = Vector(dx_, dy_, dz_);
    VectorInt nGrid = trace / attemptDDim;  // so rounding to bigger grid
    Vector actualDDim = trace / nGrid;

    // making grid that is exactly size of box.  This way can compute offsets
    // easily from Grid that doesn't have to deal with higher-level stuff like
    // bounds
    ns = nGrid.asInt3();
    ds = actualDDim.asFloat3();
    os = state->boundsGPU.lo;
    if (state->is2d) {
        ns.z = 1;
        ds.z = 1;
        assert(os.z == -.5);
    }

    dsOrig = actualDDim.asFloat3();
    initArrays();
    initStream();
    handleExclusions();
    numChecksSinceLastBuild = 0;
}

GridGPU::~GridGPU() {
    if (streamCreated) {
        CUCHECK(cudaStreamDestroy(rebuildCheckStream));
    }
}

void GridGPU::copyPositionsAsync() {

    state->gpd.xs.d_data[state->gpd.activeIdx()].copyToDeviceArray((void *) xsLastBuild.data());//, rebuildCheckStream);

}


/* grid kernels */

// overkill, but this is telling the function to be inlined in all cases
inline __host__ __forceinline__ __device__
int linearidx(int3 idx, int3 ns) __attribute__((always_inline))
{
    return (ns.z*ns.y)*idx.x + (ns.z)*idx.y + idx.z;
}

__global__ void periodicWrap(float4 *xs, int nAtoms, BoundsGPU bounds) {

    int idx = GETIDX();
    if (idx < nAtoms) {

        float4 pos = xs[idx];

        float id = pos.w;
        float3 trace = bounds.trace();
        float3 diffFromLo = make_float3(pos) - bounds.lo;
        float3 imgs = floorf(diffFromLo / trace); //are unskewed at this point
        float3 pos_orig = make_float3(pos);
        pos -= make_float4(trace * imgs * bounds.periodic);
        pos.w = id;
        //if (not(pos.x==orig.x and pos.y==orig.y and pos.z==orig.z)) { //sigh
        if (imgs.x != 0 or imgs.y != 0 or imgs.z != 0) {
            xs[idx] = pos;
        }
    }

}


__global__ void countNumInGridCells(float4 *xs, int nAtoms,
                                    uint32_t *counts, uint16_t *atomIdxs,
                                    float3 os, float3 ds, int3 ns) {

    int idx = GETIDX();
    if (idx < nAtoms) {
        //printf("idx %d\n", idx);
        int3 sqrIdx = make_int3((make_float3(xs[idx]) - os) / ds);
        int sqrLinIdx = linearidx(sqrIdx, ns);
        //printf("lin is %d\n", sqrLinIdx);
        uint16_t myPlaceInGrid = atomicAdd(counts + sqrLinIdx, 1); //atomicAdd returns old value
        //printf("grid is %d\n", myPlaceInGrid);
        //printf("myPlaceInGrid %d\n", myPlaceInGrid);
        atomIdxs[idx] = myPlaceInGrid;
        //okay - atoms seem to be getting assigned the right idx in grid
    }

}


/*
__global__ void printNeighbors(int *neighborlistBounds, cudaTextureObject_t neighbors,
                               int nAtoms) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        int begin = neighborlistBounds[idx];
        int end = neighborlistBounds[idx+1];
        for (int i=begin; i<end; i++) {
            int xIdx = XIDX(i);
            int yIdx = YIDX(i);
            int x = tex2D<int>(neighbors, xIdx, yIdx);
            printf("idx %d has neighbor of idx %d\n", idx, x);
        }
    }
}
*/


template <typename T>
__device__ void copyToOtherSurf(cudaSurfaceObject_t from, cudaSurfaceObject_t to,
                                int idx_init, int idx_final)
{
    int xIdx, yIdx, xAddr;
    xIdx = XIDX(idx_init, sizeof(T));
    yIdx = YIDX(idx_init, sizeof(T));
    xAddr = xIdx * sizeof(T);
    T val = surf2Dread<T>(from, xAddr, yIdx);
    xIdx = XIDX(idx_final, sizeof(T));
    yIdx = YIDX(idx_final, sizeof(T));
    xAddr = xIdx * sizeof(T);
    surf2Dwrite(val, to, xAddr, yIdx);
}

template <typename T>
__device__ void copyToOtherList(T *from, T *to, int idx_init, int idx_final)
{
    to[idx_final] = from[idx_init];
}

__global__ void sortPerAtomArrays(
                    float4 *xsFrom,     float4 *xsTo,
                    float4 *vsFrom,     float4 *vsTo,
                    float4 *fsFrom,     float4 *fsTo,
                    float4 *fsLastFrom, float4 *fsLastTo,
                    uint *idsFrom, uint *idsTo,
                    float *qsFrom, float *qsTo,
                    cudaSurfaceObject_t idToIdx,
                    uint32_t *gridCellArrayIdxs, uint16_t *idxInGridCell, int nAtoms,
                    float3 os, float3 ds, int3 ns)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 posWhole = xsFrom[idx];
        float3 pos = make_float3(posWhole);
        uint id = idsFrom[idx];
        int3 sqrIdx = make_int3((pos - os) / ds);
        int sqrLinIdx = linearidx(sqrIdx, ns);
        int sortedIdx = gridCellArrayIdxs[sqrLinIdx] + idxInGridCell[idx];
        //printf("I MOVE FROM %d TO %d, id is %d , MY POS IS %f %f %f\n", idx, sortedIdx, id, pos.x, pos.y, pos.z);

        //okay, now have all data needed to do copies
        copyToOtherList<float4>(xsFrom, xsTo, idx, sortedIdx);
        copyToOtherList<uint>(idsFrom, idsTo, idx, sortedIdx);
        copyToOtherList<float4>(vsFrom, vsTo, idx, sortedIdx);
        copyToOtherList<float4>(fsFrom, fsTo, idx, sortedIdx);
        copyToOtherList<float4>(fsLastFrom, fsLastTo, idx, sortedIdx);
        copyToOtherList<float>(qsFrom, qsTo, idx, sortedIdx);

        int xAddrId = XIDX(id, sizeof(int)) * sizeof(int);
        int yIdxId = YIDX(id, sizeof(int));

        surf2Dwrite(sortedIdx, idToIdx, xAddrId, yIdxId);

    }
    //annnnd copied!
}


/*! modifies myCount to be the number of neighbors in this cell */
__device__ void checkCell(float3 pos, uint myId, float4 *xs, uint *ids,
                          uint32_t *gridCellArrayIdxs, int squareIdx,
                          float3 loop, float neighCutSqr, int &myCount) {

    uint32_t idxMin = gridCellArrayIdxs[squareIdx];
    uint32_t idxMax = gridCellArrayIdxs[squareIdx+1];
    for (int i=idxMin; i<idxMax; i++) {
        float3 otherPos = make_float3(xs[i]);
        float3 distVec = otherPos + loop - pos;
        if (ids[i] != myId && dot(distVec, distVec) < neighCutSqr) {
            myCount++;
        }
    }
}

__global__ void countNumNeighbors(float4 *xs, int nAtoms, uint *ids,
                                  uint16_t *neighborCounts, uint32_t *gridCellArrayIdxs,
                                  float3 os, float3 ds, int3 ns,
                                  float3 periodic, float3 trace, float neighCutSqr) {

    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 posWhole = xs[idx];
        float3 pos = make_float3(posWhole);
        uint myId = ids[idx];
        int3 sqrIdx = make_int3((pos - os) / ds);

        int xIdx, yIdx, zIdx;
        int xIdxLoop, yIdxLoop, zIdxLoop;
        float3 offset = make_float3(0, 0, 0);
        int myCount = 0;
        for (xIdx=sqrIdx.x-1; xIdx<=sqrIdx.x+1; xIdx++) {
            offset.x = -floorf((float) xIdx / ns.x);
            xIdxLoop = xIdx + ns.x * offset.x;
            if (periodic.x || (!periodic.x && xIdxLoop == xIdx)) {

                for (yIdx=sqrIdx.y-1; yIdx<=sqrIdx.y+1; yIdx++) {
                    offset.y = -floorf((float) yIdx / ns.y);
                    yIdxLoop = yIdx + ns.y * offset.y;
                    if (periodic.y || (!periodic.y && yIdxLoop == yIdx)) {

                        for (zIdx=sqrIdx.z-1; zIdx<=sqrIdx.z+1; zIdx++) {
                            offset.z = -floorf((float) zIdx / ns.z);
                            zIdxLoop = zIdx + ns.z * offset.z;
                            if (periodic.z || (!periodic.z && zIdxLoop == zIdx)) {
                                int3 sqrIdxOther = make_int3(xIdxLoop, yIdxLoop, zIdxLoop);
                                int sqrIdxOtherLin = linearidx(sqrIdxOther, ns);
                                float3 loop = (-offset) * trace;
                                // updates myCount for this cell
                                checkCell(pos, myId, xs, ids,
                                          gridCellArrayIdxs, sqrIdxOtherLin,
                                          loop, neighCutSqr, myCount);
                                //note sign switch on offset!

                            } // endif periodic.z
                        } // endfor zIdx

                    } // endif periodic.y
                } // endfor yIdx

            } //endif periodic.x
        } // endfor xIdx
        neighborCounts[idx] = myCount;
    }
}


__device__ uint addExclusion(uint otherId, uint *exclusionIds_shr,
                             int idxLo, int idxHi) {

    uint exclMask = EXCL_MASK;
    for (int i=idxLo; i<idxHi; i++) {
        if ((exclusionIds_shr[i] & exclMask) == otherId) {
            return exclusionIds_shr[i] & (~exclMask);
        }
    }
    return 0;
}

__device__ int assignFromCell(float3 pos, int idx, uint myId, float4 *xs, uint *ids,
                              uint32_t *gridCellArrayIdxs, int squareIdx,
                              float3 offset, float3 trace, float neighCutSqr,
                              int currentNeighborIdx, uint *neighborlist,
                              uint *exclusionIds_shr, int exclIdxLo_shr, int exclIdxHi_shr,
                              int warpSize) {

    uint idxMin = gridCellArrayIdxs[squareIdx];
    uint idxMax = gridCellArrayIdxs[squareIdx+1];
    for (uint i=idxMin; i<idxMax; i++) {
        float3 otherPos = make_float3(xs[i]);
        float3 distVec = otherPos + (offset * trace) - pos;
        uint otherId = ids[i];

        if (myId != otherId && dot(distVec, distVec) < neighCutSqr/* &&
            !(isExcluded(otherId, exclusions, numExclusions, maxExclusions))*/) {
            uint exclusionTag = addExclusion(otherId, exclusionIds_shr, exclIdxLo_shr, exclIdxHi_shr);
            // if (myId==16) {
            //     printf("my id is 16 and my threadIdx is %d\n\n\n\n\n", threadIdx.x);
            // }
            neighborlist[currentNeighborIdx] = (i | exclusionTag);
            currentNeighborIdx += warpSize;
        }

    }

    return currentNeighborIdx;
}

__global__ void assignNeighbors(float4 *xs, int nAtoms, uint *ids,
                                uint32_t *gridCellArrayIdxs, uint32_t *cumulSumMaxPerBlock,
                                float3 os, float3 ds, int3 ns,
                                float3 periodic, float3 trace, float neighCutSqr,
                                uint *neighborlist, int warpSize,
                                int *exclusionIndexes, uint *exclusionIds, int maxExclusionsPerAtom) {

    // extern __shared__ int exclusions_shr[];
    extern __shared__ uint exclusionIds_shr[];

    /*
    int tidLo = blockIdx.x * blockDim.x;
    int tidHi = min((blockIdx.x+1) * blockDim.x, nAtoms) - 1;
    int idLo = *(int *) &tex2D<float4>(xs, XIDX(tidLo, sizeof(float4)), YIDX(tidLo, sizeof(float4))).w;
    int idHi = *(int *) &tex2D<float4>(xs, XIDX(tidHi, sizeof(float4)), YIDX(tidHi, sizeof(float4))).w;
    int copyLo = exclusionIndexes[idLo];
    int copyHi = exclusionIndexes[idHi+1];

    copyToShared<uint>(exclusionIds + copyLo, exclusionIds_shr, copyHi - copyLo);
    __syncthreads();
    */

    // so the exclusions that this contiguous block of atoms needs are scattered
    // around the exclusionIndexes list because they're sorted by id.  Need to
    // copy it into shared.  Each thread has to copy from diff block b/c
    // scattered
    int idx = GETIDX();
    float4 posWhole;
    int myId;
    int exclIdxLo_shr, exclIdxHi_shr, numExclusions;
    exclIdxLo_shr = threadIdx.x * maxExclusionsPerAtom;
    if (idx < nAtoms) {
        posWhole = xs[idx];
        myId = ids[idx];
        int exclIdxLo = exclusionIndexes[myId];
        int exclIdxHi = exclusionIndexes[myId+1];
        numExclusions = exclIdxHi - exclIdxLo;
        exclIdxHi_shr = exclIdxLo_shr + numExclusions;
        for (int i=exclIdxLo; i<exclIdxHi; i++) {
            uint exclusion = exclusionIds[i];
            exclusionIds_shr[maxExclusionsPerAtom*threadIdx.x + i - exclIdxLo] = exclusion;
            //printf("I am thread %d and I am copying %u from global %d to shared %d\n",
            //threadIdx.x, exclusion, i, maxExclusionsPerAtom*threadIdx.x+i-exclIdxLo);
        }
    }
    //okay, now we have exclusions copied into shared
    __syncthreads();

    //int cumulSumUpToMe = cumulSumMaxPerBlock[blockIdx.x];
    //int maxNeighInMyBlock = cumulSumMaxPerBlock[blockIdx.x+1] - cumulSumUpToMe;
    //int myWarp = threadIdx.x / warpSize;
    //int myIdxInWarp = threadIdx.x % warpSize;
    //okay, then just start here and space by warpSize;
    //YOU JUST NEED TO UPDATE HOW WE CHECK EXCLUSIONS (IDXS IN SHEARED)
    if (idx < nAtoms) {
        //printf("threadid %d idx %x has lo, hi of %d, %d\n", threadIdx.x, idx, exclIdxLo_shr, exclIdxHi_shr);
        int currentNeighborIdx = baseNeighlistIdx(cumulSumMaxPerBlock, warpSize);
        float3 pos = make_float3(posWhole);
        int3 sqrIdx = make_int3((pos - os) / ds);
        int xIdx, yIdx, zIdx;
        int xIdxLoop, yIdxLoop, zIdxLoop;
        float3 offset = make_float3(0, 0, 0);

        for (xIdx=sqrIdx.x-1; xIdx<=sqrIdx.x+1; xIdx++) {
            offset.x = -floorf((float) xIdx / ns.x);
            xIdxLoop = xIdx + ns.x * offset.x;
            if (periodic.x || (!periodic.x && xIdxLoop == xIdx)) {

                for (yIdx=sqrIdx.y-1; yIdx<=sqrIdx.y+1; yIdx++) {
                    offset.y = -floorf((float) yIdx / ns.y);
                    yIdxLoop = yIdx + ns.y * offset.y;
                    if (periodic.y || (!periodic.y && yIdxLoop == yIdx)) {

                        for (zIdx=sqrIdx.z-1; zIdx<=sqrIdx.z+1; zIdx++) {
                            offset.z = -floorf((float) zIdx / ns.z);
                            zIdxLoop = zIdx + ns.z * offset.z;
                            if (periodic.z || (!periodic.z && zIdxLoop == zIdx)) {

                                int3 sqrIdxOther = make_int3(xIdxLoop, yIdxLoop, zIdxLoop);
                                int sqrIdxOtherLin = linearidx(sqrIdxOther, ns);
                                //__device__ int assignFromCell(
                                //      float3 pos, int idx, uint myId, float4 *xs, uint *ids, int *gridCellArrayIdxs,
                                //      int squareIdx, float3 offset, float3 trace, float neighCutSqr,
                                //      int currentNeighborIdx, cudaSurfaceObject_t neighborlist,
                                //      uint *exclusionIds_shr, int exclIdxLo_shr, int exclIdxHi_shr,
                                //      int warpSize)
                                currentNeighborIdx = assignFromCell(
                                        pos, idx, myId, xs, ids, gridCellArrayIdxs,
                                        sqrIdxOtherLin, -offset, trace, neighCutSqr,
                                        currentNeighborIdx, neighborlist,
                                        exclusionIds_shr, exclIdxLo_shr, exclIdxHi_shr,
                                        warpSize);

                            } // endif periodic.z
                        } // endfor zIdx

                    } // endif periodic.y
                } // endfor yIdx

            } // endif periodic.x
        } // endfor xIdx

    } // endif idx < natoms
}


void setPerBlockCounts(std::vector<uint16_t> &neighborCounts, std::vector<uint32_t> &numNeighborsInBlocks) {
    numNeighborsInBlocks[0] = 0;
    for (int i=0; i<numNeighborsInBlocks.size()-1; i++) {
        uint16_t maxNeigh = 0;
        int maxIdx = std::fmin(neighborCounts.size()-1, (i+1)*PERBLOCK);
        for (int j=i*PERBLOCK; j<maxIdx; j++) {
            uint16_t numNeigh = neighborCounts[j];
            //std::cout << "summing at idx " << j << ", it has " << numNeigh << std::endl;
            maxNeigh = std::fmax(numNeigh, maxNeigh);
        }
        // cumulative sum of # in block
        numNeighborsInBlocks[i+1] = numNeighborsInBlocks[i] + maxNeigh;
    }

}


__global__ void setBuildFlag(float4 *xsA, float4 *xsB, int nAtoms, BoundsGPU boundsGPU,
                             float paddingSqr, int *buildFlag, int numChecksSinceBuild, int warpSize) {

    int idx = GETIDX();
    extern __shared__ short flags_shr[];
    if (idx < nAtoms) {
        float3 distVector = boundsGPU.minImage(make_float3(xsA[idx] - xsB[idx]));
        float lenSqr = lengthSqr(distVector);
        float maxMoveRatio = std::fminf(
                        0.95,
                        (numChecksSinceBuild+1) / (float)(numChecksSinceBuild+2));
        float maxMoveSqr = paddingSqr * maxMoveRatio * maxMoveRatio;
        // printf("moved %f\n", sqrtf(lenSqr));
        // printf("max move is %f\n", maxMoveSqr);
        flags_shr[threadIdx.x] = (short) (lenSqr > maxMoveSqr);
    } else {
        flags_shr[threadIdx.x] = 0;
    }
    __syncthreads();
    //just took from parallel reduction in cutils_func
    reduceByN<short>(flags_shr, blockDim.x, warpSize);
    if (threadIdx.x == 0 and flags_shr[0] != 0) {
        buildFlag[0] = 1;
    }

}


__global__ void computeMaxNumNeighPerBlock(int nAtoms, uint16_t *neighborCounts,
                                           uint16_t *maxNeighInBlock, int warpSize) {

    int idx = GETIDX();
    extern __shared__ uint16_t counts_shr[];
    if (idx < nAtoms) {
        uint16_t count = neighborCounts[idx];
        counts_shr[threadIdx.x] = count;
    } else {
        counts_shr[threadIdx.x] = 0;
    }
    __syncthreads();
    maxByN<uint16_t>(counts_shr, blockDim.x, warpSize);
    if (threadIdx.x == 0) {
        maxNeighInBlock[blockIdx.x] = counts_shr[0];
    }

}


__global__ void setCumulativeSumPerBlock(int numBlocks, uint32_t *perBlockArray,
                                         uint16_t *maxNeighborsInBlock)
{
    int idx = GETIDX();
    // doing this in simplest way possible, can optimize later if problem
    if (idx < numBlocks+1) {
        uint32_t sum = 0;
        for (int i=0; i<idx; i++) {
            sum += maxNeighborsInBlock[i];
        }
        perBlockArray[idx] = sum;
    }
}


__global__ void countTransferAtoms(float4 *xs, float4 *xsMoved,
                                   float4 *vs, float4 *vsMoved,
                                   float4 *fs, float4 *fsMoved,
                                   uint *ids, uint *idsMoved,
                                   float *qs, float4 *qsMoved,
                                   uint16_t *szMoved)
{
    int idx = GETIDX();
    OOBDirs dir = boundsLocalGPU.oobInDir(make_float3(xs[idx]));
    if (dir != OOBDirs.MMM) {
        // atomicAdd returns old value
        atomicAdd(szMoved[dir], 1);
    }
}

__global__ void copySendAtoms(float4 *xs, float4 *xsMoved,
                              float4 *vs, float4 *vsMoved,
                              float4 *fs, float4 *fsMoved,
                              uint *ids, uint *idsMoved,
                              float *qs, float4 *qsMoved,
                              uint16_t *idxsMoved, BoundsGPU boundsLocalGPU)
{
    int idx = GETIDX();
    OOBDirs dir = boundsLocalGPU.oobInDir(make_float3(xs[idx]));
    int idxMoved = 0;
    if (dir != OOBDirs.MMM) {
        int idxM = atomicAdd(idxMoved, 1);
        copyToOtherList<float4>(xs, xsMoved(send), idx, idxM);
        copyToOtherList<float4>(vs, vsMoved(send), idx, idxM);
        copyToOtherList<float4>(fs, fsMoved(send), idx, idxM);
        copyToOtherList<uint>(ids, idsMoved(send), idx, idxM);
        copyToOtherList<float>(qs, qsMoved(send), idx, idxM);
        idxsMoved[idxM] = idx;
        // TODO: when sorting happens, this will be at the end
        xs[idx].x = boundsLocalGPU.sides[0].x;
        xs[idx].y = boundsLocalGPU.sides[1].y;
        xs[idx].z = boundsLocalGPU.sides[2].z;
    }
}

__global__ void copyRecvAtoms(float4 *xs, float4 *xsMoved,
                              float4 *vs, float4 *vsMoved,
                              float4 *fs, float4 *fsMoved,
                              uint *ids, uint *idsMoved,
                              float *qs, float4 *qsMoved,
                              uint16_t *idxsMoved)
{
    int idx = GETIDX();
    copyToOtherList<float4>(xsMoved(recv), xs, idx, idxsMoved[idx]);
    copyToOtherList<float4>(vsMoved(recv), vs, idx, idxsMoved[idx]);
    copyToOtherList<float4>(fsMoved(recv), fs, idx, idxsMoved[idx]);
    copyToOtherList<uint>(idsMoved(recv), ids, idx, idxsMoved[idx]);
    copyToOtherList<float>(qsMoved(recv), qs, idx, idxsMoved[idx]);
}


void GridGPU::periodicBoundaryConditions(float neighCut, bool forceBuild) {

    DeviceManager &devManager = state->devManager;
    int warpSize = devManager.prop.warpSize;
    // TODO: remove sorting option.  Must sort every time if using mpi, and
    // also I think building without sorting isn't even working right now
    if (neighCut == -1) {
        neighCut = neighCutoffMax;
    }

    Vector nsV = Vector(make_float3(ns));
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    BoundsGPU bounds = state->boundsGPU;

    // DO ASYNC COPY TO xsLastBuild
    // FINISH FUTURE WHICH SETS REBUILD FLAG BY NOW PLEASE
    // CUCHECK(cudaStreamSynchronize(rebuildCheckStream));
    // multigpu: needs to rebuild if any proc needs to rebuild
    setBuildFlag<<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK * sizeof(short)>>>(
                state->gpd.xs(activeIdx), xsLastBuild.data(), nAtoms, bounds,
                state->padding*state->padding, buildFlag.d_data.data(),
                numChecksSinceLastBuild, warpSize);
    buildFlag.dataToHost();
    cudaDeviceSynchronize();

    // std::cout << "I AM BUILDING" << std::endl;
    if (buildFlag.h_data[0] or forceBuild) {

        float3 ds_orig = ds;
        float3 os_orig = os;

        // as defined in Vector.h
        // PAIN AND NUMERICAL ERROR AWAIT ALL THOSE WHO ALTER THE FOLLOWING
        // TWO LINES
        ds += make_float3(EPSILON, EPSILON, EPSILON);
        os -= make_float3(EPSILON, EPSILON, EPSILON);

        BoundsGPU boundsUnskewed = bounds.unskewed();
        if (bounds.sides[0].y or bounds.sides[1].x) {
            Mod::unskewAtoms<<<NBLOCK(nAtoms), PERBLOCK>>>(
                        state->gpd.xs(activeIdx), nAtoms,
                        bounds.sides[0], bounds.sides[1], bounds.lo);
        }
        periodicWrap<<<NBLOCK(nAtoms), PERBLOCK>>>(
                    state->gpd.xs(activeIdx), nAtoms, boundsUnskewed);

        // get how many atoms move in each direction
        
        auto szMoved = GPUArrayPair(partition.adjacentRanks.size());
        std::fill(szMoved.data(), szMoved.data() + szMoved.size(), 0);
        countTransferAtoms<<<NBLOCK(nAtoms), PERBLOCK>>>(
                    state->gpd.xs(activeIdx), state->gpd.xsMoved(send),
                    state->gpd.vs(activeIdx), state->gpd.vsMoved(send),
                    state->gpd.fs(activeIdx), state->gpd.fsMoved(send),
                    state->gpd.ids(activeIdx), state->gpd.idsMoved(send),
                    state->gpd.qs(activeIdx), state->gpd.qsMoved(send),
                    szMoved(send));

        // sendrecv sizes
        for (auto dir : OOBDirList) {
            int transferIdx = dirToTransferIdx[dir] * maxMoved;
            MPI_sendrecv_gpu(state->gpd.szMoved(send)[transferIdx],
                             state->gpd.szMoved(recv)[transferIdx],
                             szMoved(send)[transferIdx].size(),
                             szMoved(recv)[transferIdx].size(),
                             MPI_FLOAT, dirToRank(dir));
        }

        // get max moved in any direction
        int szMaxSend = *std::max_element(szMoved(send).data(),
                                          zMoved(send).data() + szMoved.size());
        int szMaxRecv = *std::max_element(szMoved(recv).data(),
                                          zMoved(recv).data() + szMoved.size())
        int szMaxTransfer = std::max(szMaxSend, szMaxRecv);
        
        // get size of lists to send, recv, and max of both
        int szTotalSend = std::accumulate(szMoved(send).data().begin(),
                                          szMoved(send).data().end(), 0);
        int szTotalRecv = std::accumulate(szMoved(recv).data().begin(),
                                          szMoved(recv).data().end(), 0);
        szTotalMax = std::max(szTotalSend, szTotalRecv);

        nAtoms += (szTotalRecv - szTotalSend);
        // reallocate atoms if necessary
        if(nAtoms > state->gpd.xs.size()) {
            auto xsTemp = GPUArrayDeviceGlobal(nAtoms);
            xsTemp.set(state->gpd.xs(activeIdx), state->gpd.xs.size());
            state->gpd.xs.d_data[activeIdx] = std::move(xsTemp);
            state->gpd.xs.d_data[!activeIdx] = GpuArrayDeviceGlobal(nAtoms);
            state->gpd.xs.h_data = std::vector<float4>(nAtoms);
            auto vsTemp = GPUArrayDeviceGlobal(nAtoms);
            vsTemp.set(state->gpd.vs(activeIdx), state->gpd.vs.size());
            state->gpd.vs.d_data[activeIdx] = std::move(vsTemp);
            state->gpd.vs.d_data[!activeIdx] = GpuArrayDeviceGlobal(nAtoms);
            state->gpd.vs.h_data = std::vector<float4>(nAtoms);
            auto fsTemp = GPUArrayDeviceGlobal(nAtoms);
            fsTemp.set(state->gpd.fs(activeIdx), state->gpd.fs.size());
            state->gpd.fs.d_data[activeIdx] = std::move(fsTemp);
            state->gpd.fs.d_data[!activeIdx] = GpuArrayDeviceGlobal(nAtoms);
            state->gpd.fs.h_data = std::vector<float4>(nAtoms);
            auto idsTemp = GPUArrayDeviceGlobal(nAtoms);
            idsTemp.set(state->gpd.ids(activeIdx), state->gpd.ids.size());
            state->gpd.ids.d_data[activeIdx] = std::move(idsTemp);
            state->gpd.ids.d_data[!activeIdx] = GpuArrayDeviceGlobal(nAtoms);
            state->gpd.ids.h_data = std::vector<uint>(nAtoms);
            auto qsTemp = GPUArrayDeviceGlobal(nAtoms);
            qsTemp.set(state->gpd.qs(activeIdx), state->gpd.qs.size());
            state->gpd.qs.d_data[activeIdx] = std::move(qsTemp);
            state->gpd.qs.d_data[!activeIdx] = GpuArrayDeviceGlobal(nAtoms);
            state->gpd.qs.h_data = std::vector<float>(nAtoms);
        }

        // reallocate moved if necessary; for now, send and recv will be same size
        auto idxsMoved = GPUArrayPair<uint>(szTotalMax);
        if (    szMaxTransfer * 26 > xsMoved.size() ||
                szMaxTransfer * 26 < xsMoved.size() / 2) {
            xsMoved = GPUArrayPair<float4>(szMaxTransfer*26*1.5);
            vsMoved = GPUArrayPair<float4>(szMaxTransfer*26*1.5);
            fsMoved = GPUArrayPair<float4>(szMaxTransfer*26*1.5);
            idsMoved = GPUArrayPair<uint>(szMaxTransfer*26*1.5);
            qsMoved = GPUArrayPair<float>(szMaxTransfer*26*1.5);
        }

        // copy atoms to moved arrays, and MPI them
        copySendAtoms<<<NBLOCK(nAtoms), PERBLOCK>>>(
                    state->gpd.xs(activeIdx), state->gpd.xsMoved(send),
                    state->gpd.vs(activeIdx), state->gpd.vsMoved(send),
                    state->gpd.fs(activeIdx), state->gpd.fsMoved(send),
                    state->gpd.ids(activeIdx), state->gpd.idsMoved(send),
                    state->gpd.qs(activeIdx), state->gpd.qsMoved(send),
                    idxsMoved, state->boundsLocalGPU);
        int end = nAtoms;
        for (int i = szTotalSend; i < szTotalRecv; ++i) {
            idxsMoved[i] = end++;
        }
        for (auto dir : OOBDirList) {
            int transferIdx = dirToTransferIdx(dir) * maxMoved;
            int rankOther = dirToRank(dir);
            int szSend = szMoved(send)[dir];
            int szRecv = szMoved(recv)[dir];
            int szSendf4 = szMoved(send)[dir] * sizeof(float4)/sizeof(float)
            int szRecvf4 = szMoved(recv)[dir] * sizeof(float4)/sizeof(float)
            MPI_sendrecv_gpu(state->gpd.xsMoved(send)[transferIdx],
                             state->gpd.xsMoved(recv)[transferIdx],
                             szSendf4, szRecvf4, MPI_FLOAT, rankOther);
            MPI_sendrecv_gpu(state->gpd.vsMoved(send)[transferIdx],
                             state->gpd.vsMoved(recv)[transferIdx],
                             szSendf4, szRecvf4, MPI_FLOAT, rankOther);
            MPI_sendrecv_gpu(state->gpd.fsMoved(send)[transferIdx],
                             state->gpd.fsMoved(recv)[transferIdx],
                             szSendf4, szRecvf4, MPI_FLOAT, rankOther);
            MPI_sendrecv_gpu(state->gpd.idsMoved(send)[transferIdx],
                             state->gpd.idsMoved(recv)[transferIdx],
                             szSend, szRecv, MPI_UINT, rankOther);
            MPI_sendrecv_gpu(state->gpd.qsMoved(send)[transferIdx],
                             state->gpd.qsMoved(recv)[transferIdx],
                             szSend, szRecv, MPI_FLOAT, rankOther);
        }

        // copy atoms to moved arrays, and MPI-send them
        copyRecvAtoms<<<NBLOCK(idxsMoved.size()), PERBLOCK>>>(
                    state->gpd.xs(activeIdx), state->gpd.xsMoved,
                    state->gpd.vs(activeIdx), state->gpd.vsMoved,
                    state->gpd.fs(activeIdx), state->gpd.fsMoved,
                    state->gpd.fsLast(activeIdx), state->gpd.fsLastMoved,
                    state->gpd.ids(activeIdx), state->gpd.idsMoved,
                    state->gpd.qs(activeIdx), state->gpd.qsMoved,
                    idxsMoved);

        // TODO: map between dirs and ranks
        // TODO: fix idsToIdxs

        // TODO: ghosts
        // 1. count ghosts atoms/cell, store in array
        //    (len: cells/dim * cells/dim * cells/dim * cells/dim,
        //          or 1, for point (1*1*1), line (n*1*1), plane (n*n*1)
        // 2. sendrecv ghost counts per cell
        // 3. resize in same manner as above
        // 4. sendrecv ghosts into buffers, either cell-wise or w/ intermediate
        //    copy
        // 5. figure out where the atoms are accessed, and change interface so
        //    that the neighbors can be found etc

        // increase number of grid cells if necessary
        int numGridCells = prod(ns);
        if (numGridCells + 1 != perCellArray.size()) {
            perCellArray = GPUArrayGlobal<uint32_t>(numGridCells + 1);
        }

        perCellArray.d_data.memset(0);
        perAtomArray.d_data.memset(0);
        countNumInGridCells<<<NBLOCK(nAtoms), PERBLOCK>>>(
                    state->gpd.xs(activeIdx), nAtoms,
                    perCellArray.d_data.data(), perAtomArray.d_data.data(),
                    os, ds, ns
        );
        perCellArray.dataToHost();
        cudaDeviceSynchronize();

        uint32_t *gridCellCounts_h = perCellArray.h_data.data();
        //repurposing this as starting indexes for each grid square
        cumulativeSum(gridCellCounts_h, perCellArray.size());
        perCellArray.dataToDevice();
        int gridIdx;

        //sort atoms by position, matching grid ordering
        sortPerAtomArrays<<<NBLOCK(nAtoms), PERBLOCK>>>(
                    state->gpd.xs(activeIdx), state->gpd.xs(!activeIdx),
                    state->gpd.vs(activeIdx), state->gpd.vs(!activeIdx),
                    state->gpd.fs(activeIdx), state->gpd.fs(!activeIdx),
                    state->gpd.fsLast(activeIdx), state->gpd.fsLast(!activeIdx),
                    state->gpd.ids(activeIdx), state->gpd.ids(!activeIdx),
                    state->gpd.qs(activeIdx), state->gpd.qs(!activeIdx),
                    state->gpd.idToIdxs.getSurf(),
                    perCellArray.d_data.data(), perAtomArray.d_data.data(),
                    nAtoms, os, ds, ns
        );
        activeIdx = state->gpd.switchIdx();
        gridIdx = activeIdx;

        float3 trace = boundsUnskewed.trace();

        perAtomArray.d_data.memset(0);
        /* multigpu:
         *     call this for ghosts too; everything after this has to be done on
         *     ghosts too
         */
        countNumNeighbors<<<NBLOCK(nAtoms), PERBLOCK>>>(
                    state->gpd.xs(gridIdx), nAtoms, state->gpd.ids(gridIdx),
                    perAtomArray.d_data.data(), perCellArray.d_data.data(),
                    os, ds, ns, bounds.periodic, trace, neighCut*neighCut);
        //, state->gpd.nlistExclusionIdxs.getTex(), state->gpd.nlistExclusions.getTex(),
        //state->maxExclusions);

        computeMaxNumNeighPerBlock<<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK*sizeof(uint16_t)>>>(
                    nAtoms, perAtomArray.d_data.data(),
                    perBlockArray_maxNeighborsInBlock.data(), warpSize);

        int numBlocks = perBlockArray_maxNeighborsInBlock.size();
        setCumulativeSumPerBlock<<<NBLOCK(numBlocks+1), PERBLOCK>>>(
                    numBlocks, perBlockArray.d_data.data(),
                    perBlockArray_maxNeighborsInBlock.data());
        uint32_t cumulSumPerBlock;
        perBlockArray.d_data.get(&cumulSumPerBlock, numBlocks, 1);
        cudaDeviceSynchronize();

        //perAtomArray.dataToHost();
        //cudaDeviceSynchronize();
        //setPerBlockCounts(perAtomArray.h_data, perBlockArray.h_data);  // okay, now this is the start index (+1 is end index) of each atom's neighbors
        //perBlockArray.dataToDevice();

        //int totalNumNeighbors = perBlockArray.h_data.back() * PERBLOCK;
        int totalNumNeighbors = cumulSumPerBlock * PERBLOCK;
        //std::cout << "TOTAL NUM IS " << totalNumNeighbors << std::endl;
        if (totalNumNeighbors > neighborlist.size()) {
            neighborlist = GPUArrayDeviceGlobal<uint>(totalNumNeighbors*1.5);
        } else if (totalNumNeighbors < neighborlist.size() * 0.5) {
            //neighborlist = GPUArrayDeviceGlobal<uint>(totalNumNeighbors*0.8);//REALLY??
            neighborlist = GPUArrayDeviceGlobal<uint>(totalNumNeighbors*1.5);//REALLY??
        }
        /*
        SAFECALL(assignNeighbors<<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK*maxExclusionsPerAtom*sizeof(uint)>>>(
                state->gpd.xs(gridIdx), nAtoms, state->gpd.idToIdxs.getTex(), state->gpd.ids(gridIdx),
                perBlockArray.d_data.data(), os, ds, ns,
                bounds.periodic, trace, neighCut*neighCut,
                neighborlist.data(), warpSize,
                exclusionIndexes.data(), exclusionIds.size(), maxExclusionsPerAtom
        ));//, state->gpd.nlistExclusionIdxs.getTex(), state->gpd.nlistExclusions.getTex(), state->maxExclusions);
        */
        assignNeighbors<<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK*maxExclusionsPerAtom*sizeof(uint)>>>(
                state->gpd.xs(gridIdx), nAtoms, state->gpd.ids(gridIdx),
                perCellArray.d_data.data(), perBlockArray.d_data.data(), os, ds, ns,
                bounds.periodic, trace, neighCut*neighCut, neighborlist.data(), warpSize,
                exclusionIndexes.data(), exclusionIds.data(), maxExclusionsPerAtom
        );//, state->gpd.nlistExclusionIdxs.getTex(), state->gpd.nlistExclusions.getTex(), state->maxExclusions);

        // printNeighbors<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
        //      perAtomArray.ptr, neighborlist.tex, state->atoms.size());
        /*
        //int *neighCounts = perAtomArray.get((int *) NULL); // Warning: usage changed
        cudaDeviceSynchronize();
        printNeighborCounts(neighCounts, state->atoms.size());
        free(neighCounts);
        */
        if (bounds.sides[0].y or bounds.sides[1].x) {
            Mod::skewAtomsFromZero<<<NBLOCK(nAtoms), PERBLOCK>>>(
                    state->gpd.xs(activeIdx), nAtoms,
                    bounds.sides[0], bounds.sides[1], bounds.lo);
        }
        ds = ds_orig;
        os = os_orig;
        //verifyNeighborlists(neighCut);

        numChecksSinceLastBuild = 0;
        copyPositionsAsync();
    } else {
        numChecksSinceLastBuild++;
    }

    buildFlag.d_data.memset(0);

}


bool GridGPU::verifyNeighborlists(float neighCut) {
    std::cout << "going to verify" << std::endl;
    uint *nlist = (uint *) malloc(neighborlist.size()*sizeof(uint));
    neighborlist.get(nlist);
    float cutSqr = neighCut * neighCut;
    perAtomArray.dataToHost();
    uint16_t *neighCounts = perAtomArray.h_data.data();
    state->gpd.xs.dataToHost();
    state->gpd.ids.dataToHost();
    perBlockArray.dataToHost();
    cudaDeviceSynchronize();

    // std::cout << "Neighborlist" << std::endl;
    // for (int i=0; i<neighborlist.size(); i++) {
    //     std::cout << "idx " << i << " " << nlist[i] << std::endl;
    // }
    // std::cout << "end neighborlist" << std::endl;

    std::vector<float4> xs = state->gpd.xs.h_data;
    std::vector<uint> ids = state->gpd.ids.h_data;
    // std::cout << "ids" << std::endl;
    // for (int i=0; i<ids.size(); i++) {
    //     std::cout << ids[i] << std::endl;
    // }
    state->gpd.xs.dataToHost(!state->gpd.xs.activeIdx);
    cudaDeviceSynchronize();
    std::vector<float4> sortedXs = state->gpd.xs.h_data;

    // int gpuId = *(int *)&sortedXs[TESTIDX].w;
    // int cpuIdx = gpuId;

    std::vector<std::vector<int> > cpu_neighbors;
    for (int i=0; i<xs.size(); i++) {
        std::vector<int> atom_neighbors;
        float3 self = make_float3(xs[i]);
        for (int j=0; j<xs.size(); j++) {
            if (i!=j) {
                float4 otherWhole = xs[j];
                float3 minImage = state->boundsGPU.minImage(self - make_float3(otherWhole));
                if (lengthSqr(minImage) < cutSqr) {
                    uint otherId = ids[j];
                    atom_neighbors.push_back(otherId);
                }
            }
        }
        sort(atom_neighbors.begin(), atom_neighbors.end());
        cpu_neighbors.push_back(atom_neighbors);
    }
    // std::cout << "cpu dist is " << sqrt(lengthSqr(state->boundsGPU.minImage(xs[0]-xs[1])))  << std::endl;

    int warpSize = state->devManager.prop.warpSize;
    for (int i=0; i<xs.size(); i++) {
        int blockIdx = i / PERBLOCK;
        int warpIdx = (i - blockIdx * PERBLOCK) / warpSize;
        int idxInWarp = i - blockIdx * PERBLOCK - warpIdx * warpSize;
        int cumSumUpToMyBlock = perBlockArray.h_data[blockIdx];
        int perAtomMyWarp = perBlockArray.h_data[blockIdx+1] - cumSumUpToMyBlock;
        int baseIdx = PERBLOCK * perBlockArray.h_data[blockIdx] + perAtomMyWarp * warpSize * warpIdx + idxInWarp;

        //std::cout << "i is " << i << " blockIdx is " << blockIdx << " warp idx is " << warpIdx << " and idx in that warp is " << idxInWarp << " resulting base idx is " << baseIdx << std::endl;
        //std::cout << "id is " << ids[i] << std::endl;
        std::vector<int> neighIds;
        // std::cout << "begin end " << neighIdxs[i] << " " << neighIdxs[i+1] << std::endl;
        for (int j=0; j<neighCounts[i]; j++) {
            int nIdx = baseIdx + j*warpSize;
            // std::cout << "looking at neighborlist index " << nIdx << std::endl;
            // std::cout << "idx " << nlist[nIdx] << std::endl;
            float4 atom = xs[nlist[nIdx]];
            uint id = ids[nlist[nIdx]];
            // std::cout << "id is " << id << std::endl;
            neighIds.push_back(id);
        }

        sort(neighIds.begin(), neighIds.end());
        if (neighIds != cpu_neighbors[i]) {
            std::cout << "problem at idx " << i << " id " << ids[i] << std::endl;
            std::cout << "cpu " << cpu_neighbors[i].size() << " gpu " << neighIds.size() << std::endl;
            std::cout << "cpu neighbor ids" << std::endl;
            for (int x : cpu_neighbors[i]) {
                std::cout << x << " ";
            }
            std::cout << std::endl;
            std::cout << "gpu neighbor ids" << std::endl;
            for (int x : neighIds) {
                std::cout << x << " ";
            }
            std::cout << std::endl;
            break;
        }
    }

    free(nlist);
    std::cout << "end verification" << std::endl;
    return true;
}


bool GridGPU::checkSorting(int gridIdx, int *gridIdxs,
                           GPUArrayDeviceGlobal<int> &gridIdxsDev) {

    int numGridIdxs = prod(ns);
    std::vector<int> activeIds = LISTMAPREF(Atom, int, atom, state->atoms, atom.id);
    std::vector<int> gpuIds;

    gpuIds.reserve(activeIds.size());
    state->gpd.xs.dataToHost(gridIdx);
    cudaDeviceSynchronize();
    std::vector<float4> &xs = state->gpd.xs.h_data;
    bool correct = true;
    for (int i=0; i<numGridIdxs; i++) {
        int gridLo = gridIdxs[i];
        int gridHi = gridIdxs[i+1];
        // std::cout << "hi for " << i << " is " << gridHi << std::endl;
        for (int atomIdx=gridLo; atomIdx<gridHi; atomIdx++) {
            float4 posWhole = xs[atomIdx];
            float3 pos = make_float3(posWhole);
            int id = *(int *) &posWhole.w;
            gpuIds.push_back(id);

            int3 sqr = make_int3((pos - os) / ds);
            int linear = linearidx(sqr, ns);
            if (linear != i) {
                correct = false;
            }
        }
    }

    sort(activeIds.begin(), activeIds.end());
    sort(gpuIds.begin(), gpuIds.end());
    std::cout << activeIds.size() << " " << gpuIds.size() << std::endl;
    if (activeIds != gpuIds) {
        correct = false;
        std::cout << "different ids!  Serious problem!" << std::endl;
        assert(activeIds.size() == gpuIds.size());
    }

    return correct;
}


void GridGPU::handleExclusions() {

    const ExclusionList exclList = generateExclusionList(4);
    std::vector<int> idxs;
    std::vector<uint> excludedById;
    excludedById.reserve(state->maxIdExisting+1);

    auto fillToId = [&] (int id) {  // paired list is indexed by id.  Some ids could be missing, so need to fill in empty values
        while (idxs.size() <= id) {
            idxs.push_back(excludedById.size());
        }
    };

    uint exclusionTags[3] = {(uint) 1 << 30, (uint) 2 << 30, (uint) 3 << 30};
    maxExclusionsPerAtom = 0;
    for (auto it = exclList.begin(); it!=exclList.end(); it++) {  // is ordered map, so it sorted by ascending id
        int id = it->first;
        // std::cout << "id is " << id << std::endl;
        const std::vector< std::set<int> > &atomExclusions = it->second;
        fillToId(id);
        // std::cout << "filled" << std::endl;
        // for (int id : idxs) {
        //     std::cout << id << std::endl;
        // }
        for (int i=0; i<atomExclusions.size(); i++) {
            const std::set<int> &idsAtLevel = atomExclusions[i];
            for (auto itId=idsAtLevel.begin(); itId!=idsAtLevel.end(); itId++) {
                uint id = *itId;
                id |= exclusionTags[i];
                excludedById.push_back(id);
            }
        }
        idxs.push_back(excludedById.size());
        maxExclusionsPerAtom = std::fmax(maxExclusionsPerAtom, idxs.back() - idxs[idxs.size()-2]);
    }

    // std::cout << "max excl per atom is " << maxExclusionsPerAtom << std::endl;
    exclusionIndexes = GPUArrayDeviceGlobal<int>(idxs.size());
    exclusionIndexes.set(idxs.data());
    exclusionIds = GPUArrayDeviceGlobal<uint>(excludedById.size());
    exclusionIds.set(excludedById.data());
    //atoms is sorted by id.  list of ids may be sparse, so need to make sure
    //there's enough shared memory for PERBLOCK _atoms_, not just PERBLOCK ids
    //(when calling assign exclusions kernel)

    //for test output
    /*
    std::cout << "index ptrs " << std::endl;
    for (int id : idxs) {
        std::cout << id << std::endl;
    }
    std::cout << "end" << std::endl;
    for (int i=0; i<idxs.size()-1; i++) {
        for (int exclIdx=idxs[i]; exclIdx < idxs[i+1]; exclIdx++) {
            uint excl = excludedById[exclIdx];
            uint filter = (uint) 3 << 30;
            std::cout << filter << std::endl;
            uint dist = (excl & filter) >> 30;
            uint id = excl & (~filter);
            std::cout << "id " << i << " excludes " << id << " with dist " << dist << std::endl;
        }
    }
    */
}

bool GridGPU::closerThan(const ExclusionList &exclude,
                         int atomid, int otherid, int16_t depthi) {
    bool closerThan = false;
    // because we want to check lower depths
    --depthi;
    while (depthi >= 0) {
        const std::set<int> &closer = exclude.at(atomid)[depthi];
        closerThan |= (closer.find(otherid) != closer.end());
        --depthi;
    }
    // atoms are closer to themselves than any other depth away
    closerThan |= (atomid == otherid);
    return closerThan;
}

// allows us to extract any type of Bond from a BondVariant
class bondDowncast : public boost::static_visitor<const Bond &> {
    const BondVariant &_bv;
    public:
        bondDowncast(BondVariant &bv) : _bv(bv) {}
        template <typename T>
        const Bond &operator()(const T &b) const {
            return boost::get<T>(_bv);
        }
};

GridGPU::ExclusionList GridGPU::generateExclusionList(const int16_t maxDepth) {

    ExclusionList exclude;
    // not called depth because it's really the depth index, which is one
    // smaller than the depth
    int16_t depthi = 0;

    // computes adjacent bonds (depth -> 1, depthi -> 0)
    std::vector<std::vector<BondVariant> *> allBonds;
    for (Fix *f : state->fixes) {
        std::vector<BondVariant> *fixBonds = f->getBonds();
        if (fixBonds != nullptr) {
            allBonds.push_back(fixBonds);
        }
    }
    for (Atom atom : state->atoms) {
        exclude[atom.id].push_back(std::set<int>());
    }

    // typedef std::map<int, std::vector<std::set<int>>> ExclusionList;
    for (std::vector<BondVariant> *fixBonds : allBonds) {
        for (BondVariant &bondVariant : *fixBonds) {
            // boost variant magic that takes any BondVariant and turns it into a Bond
            const Bond &bond = boost::apply_visitor(bondDowncast(bondVariant), bondVariant);
            // atoms in the same bond are 1 away from each other
            exclude[bond.ids[0]][depthi].insert(bond.ids[1]);
            exclude[bond.ids[1]][depthi].insert(bond.ids[0]);
        }
    }
    depthi++;

    // compute the rest
    while (depthi < maxDepth) {
        for (Atom atom : state->atoms) {
            // for every atom at the previous depth away
            exclude[atom.id].push_back(std::set<int>());
            for (int extendFrom : exclude[atom.id][depthi-1]) {
                // extend to all atoms bonded with it
                exclude[atom.id][depthi].insert(
                  exclude[extendFrom][0].begin(), exclude[extendFrom][0].end());
            }
            // remove all atoms that are already excluded to a lower degree
            // TODO: may be a more efficient way
            for (auto it = exclude[atom.id][depthi].begin();
                 it != exclude[atom.id][depthi].end(); /*blank*/ ) {
                if (closerThan(exclude, atom.id, *it, depthi)) {
                   exclude[atom.id][depthi].erase(it++);
                } else {
                    ++it;
                }
            }
        }
        depthi++;
    }
    return exclude;
}

