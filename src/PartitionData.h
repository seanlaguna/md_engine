#pragma once
#ifndef PARTITION_DATA_H
#define PARTITION_DATA_H

#include "BoundsGPU.h"

#include <mpi.h>

const OOBDir OOBDirList[26] = {
    LDI, MDI, RDI,
    LMI, MMI, RMI,
    LUI, MUI, RUI,

    LDM, MDM, RDM,
    LMM,      RMM,
    LUM, MUM, RUM,

    LDO, MDO, RDO,
    LMO, MMO, RMO,
    LUO, MUO, RUO
};

class PartitionData
{
public:
    PartitionData()
            : nDims(3)
    { }
    PartitionData(bool is2d, bool periodic_in[3], BoundsGPU boundsLocalGPU_in)
            : nDims(is2d? 2 : 3), boundsLocalGPU(boundsLocalGPU_in)
    {
        int nRanks;
        MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
        // for now
        dimSizes[0] = nRanks;
        for (int i = 1; i < nDims; ++i) {
            dimSizes[i] = 1;
            periodic[i] = periodic_in[i];
        }
        MPI_Cart_create(MPI_COMM_WORLD, nDims, dimSizes, periodic, 0, &comm);
        fillAdjInfo();
        adjSize = adjRanks.size();
    }

    // returns index in list of adjacent directions
    // returns size of adjacint directions if not adjacent
    __host__ __device__ int getAdjIdx(OOBDir dir) {
        int i = 0;
        for (; i < adjDirs.size(); ++i) {
            if (dir == adjDirs[i]) { break; }
        }
        return i;
    }

    // returns MPI rank for a given dir
    // returns size of adjacint directions if not adjacent
    __host__ __device__ int dirToRank(OOBDir dir) {
        int i = getAdjIdx(dir);
        if (i < adjDirs.size()) {
            return adjDirs[i];
        } else {
            return i;
        }
    }

public:
    // todo: make GPU arrays
    std::vector<int> adjRanks;
    std::vector<int> adjDirs;
    uint16_t adjSize;
    BoundsGPU boundsLocalGPU; //!< Bounds on the GPU

private:
    __host__ __device__ OOBDir offsToDir(int offs[3]) {
        uint bases[3] = { 0, 2, 4 };
        uint8_t dir = 0;
        for (int i = 0; i < nDims; ++i) {
            if (offs[i] >= 0) {
                dir &= (1 << ((uint)offs[i] + bases[i]));
            }
            if (nDims == 2) { dir &= 16; }
        }
        return static_cast<OOBDir>(dir);
    }

    void fillAdjInfo()
    {
        int rank;
        MPI_Comm_rank(comm, &rank);
        int coords[nDims];
        MPI_Cart_coords(comm, rank, nDims, coords);

        int coordsAdj[nDims];
        int rankAdj;
        for (int offx : { -1, 0, 1 }) {
             coordsAdj[0] = coords[0] + offx;
             for (int offy : { -1, 0, 1 }) {
                  coordsAdj[1] = coords[1] + offy;
                  for (int offz : { -1, 0, 1 }) {
                       if (offx == offy == offz == 0) { continue; }
                       if (nDims == 2 && offz != 0) { continue; }
                       if (nDims == 3) {
                            coordsAdj[2] = coords[2] + offz;
                       }
                       bool skip = false;
                       for (int i = 0; i < nDims; ++i) {
                           if ((!periodic[i]) ||
                               (coordsAdj[i] < 0 || coordsAdj[i] >= dimSizes[i])) {
                               skip = true;
                               break;
                           }
                       }
                       if (skip) { continue; }
                       MPI_Cart_rank(comm, coordsAdj, &rankAdj);
                       adjRanks.push_back(rankAdj);
                       int offs[3] = { offx, offy, offz };
                       adjDirs.push_back(offsToDir(offs));
                  }  // end offz
             }  // end offy
        }  // end offx
        std::sort(adjRanks.begin(), adjRanks.end());
        std::sort(adjDirs.begin(), adjDirs.end());
    }

private:
    int nDims;

    // x, y, z outer -> inner, just like rest of code
    int dimSizes[3];
    int periodic[3];

    MPI_Comm comm;

};

#endif
