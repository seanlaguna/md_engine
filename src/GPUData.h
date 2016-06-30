#pragma once
#ifndef GPUDATA_H
#define GPUDATA_H

#include <map>

#include "GPUArrayDeviceGlobal.h"
#include "GPUArrayGlobal.h"
#include "GPUArrayPair.h"
#include "GPUArrayTex.h"
#include "PartitionData.h"
#include "Virial.h"

class GPUData
{
public:
    GPUData()
      : idToIdxs(cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned))
    {   }

    unsigned int activeIdx() {
        return xs.activeIdx;
    }
    unsigned int switchIdx() {
        /*! \todo Find a better way to keep track of all data objects */
        xs.switchIdx();
        vs.switchIdx();
        fs.switchIdx();
        ids.switchIdx();
        return qs.switchIdx();
    }

    int nAtoms;

    /* types (ints) are bit cast into the w value of xs.  Cast as int pls */
    GPUArrayPair<float4> xs;
    /* mass is stored in w value of vs.  ALWAYS do arithmetic as float3s, or
     * you will mess up id or mass */
    GPUArrayPair<float4> vs;
    /* groupTags (uints) are bit cast into the w value of fs */
    GPUArrayPair<float4> fs;
    GPUArrayPair<uint> ids;
    GPUArrayPair<float> qs;
    GPUArrayTex<int> idToIdxs;
    GPUArrayGlobal<Virial> virials;

    GPUArrayGlobal<float4> xsBuffer;
    GPUArrayGlobal<float4> vsBuffer;
    GPUArrayGlobal<float4> fsBuffer;
    GPUArrayGlobal<uint> idsBuffer;

    /* for transfer between GPUs */
    PartitionData partition;

    GPUArrayPair<float4> xsMoved;
    GPUArrayPair<float4> vsMoved;
    GPUArrayPair<float4> fsMoved;
    GPUArrayPair<uint> idsMoved;
    GPUArrayPair<float> qsMoved;

    GPUArrayPair<float4> xsGhost;
    GPUArrayPair<float4> vsGhost;
    GPUArrayPair<float4> fsGhost;
    GPUArrayPair<uint> idsGhost;
    GPUArrayPair<float> qsGhost;

    /* for data collection.  If we re-use per-particle arrays, we can't do
     * async kernels to do per-group sums.  Would use less memory though */
    GPUArrayGlobal<float> perParticleEng;

    std::vector<int> idToIdxsOnCopy;

    // OMG REMEMBER TO ADD EACH NEW ARRAY TO THE ACTIVE DATA LIST IN INTEGRATOR OR PAIN AWAITS

};

#endif
