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
            : boundsLocalGPU(boundsLocalGPU_in),
              nDims(is2d? 2 : 3)
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
    }

    // returns index in list of adjacent directions
    // returns size of adjacint directions if not adjacent
    __host__ __device__ int getAdjIdx(OOBDir dir) {
        int i = 0;
        for (; i < adjSize; ++i) {
            if (dir == adjDirsRaw[i]) { break; }
        }
        return i;
    }

    // returns MPI rank for a given dir
    // returns size of adjacint directions if not adjacent
    __host__ __device__ int dirToRank(OOBDir dir) {
        int i = getAdjIdx(dir);
        if (i < adjSize) {
            return adjRanksRaw[i];
        } else {
            return i;
        }
    }

public:
    // todo: make GPU arrays
    GPUArrayGlobal<int> adjRanks;
    int *adjRanksRaw;
    GPUArrayGlobal<OOBDir> adjDirs;
    OOBDir *adjDirsRaw;
    uint16_t adjSize;
    BoundsGPU boundsLocalGPU; //!< Bounds on the GPU

private:
    __host__ __device__ OOBDir offsToDir(int offs[3]) {
        uint bases[3] = { 0, 2, 4 };
        uint8_t dir = 0;
        for (int i = 0; i < nDims; ++i) {
            if (offs[i] >= 0) {
                dir |= (1 << (static_cast<uint>(offs[i]) + bases[i]));
            }
        }
        if (nDims == 2) { dir |= 16; }
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
        std::vector<int> adjRanksCpu;
        std::vector<OOBDir> adjDirsCpu;
        for (int offx : { -1, 0, 1 }) {
             coordsAdj[0] = coords[0] + offx;
             for (int offy : { -1, 0, 1 }) {
                  coordsAdj[1] = coords[1] + offy;
                  for (int offz : { -1, 0, 1 }) {
                      // our boundaries; no need to check
                      if (offx == 0 && offy == 0 && offz == 0) { continue; }
                      if (nDims == 2 &&
                          ((offx == 0 && offy == 0) || offz != 0)) { continue; }
                      if (nDims == 3) {
                          coordsAdj[2] = coords[2] + offz;
                      }
                      bool skip = false;
                      for (int i = 0; i < nDims; ++i) {
                          if ((!periodic[i]) &&
                              (coordsAdj[i] < 0 || coordsAdj[i] >= dimSizes[i])) {
                              skip = true;
                              break;
                          }
                      }
                      // not periodic and not in bounds
                      if (skip) { continue; }
                      MPI_Cart_rank(comm, coordsAdj, &rankAdj);
                      // we are adjacent to ourselves
                      if (rank == rankAdj) { continue; }
                      adjRanksCpu.push_back(rankAdj);
                      int offs[3] = { offx, offy, offz };
                      std::cout << "adding rank/dir pair for offset: " << offs[0] << "," << offs[1] << "," << offs[2] << ": "
                                << rankAdj << " & " << static_cast<int>(offsToDir(offs)) << std::endl;
                      adjDirsCpu.push_back(offsToDir(offs));
                  }  // end offz
             }  // end offy
        }  // end offx
        std::sort(adjRanksCpu.begin(), adjRanksCpu.end());
        adjRanks = adjRanksCpu;
        adjRanks.dataToDevice();
        adjRanksRaw = adjRanks.getDevData();
        std::sort(adjDirsCpu.begin(), adjDirsCpu.end());
        adjDirs = adjDirsCpu;
        adjDirs.dataToDevice();
        adjDirsRaw = adjDirs.getDevData();
        adjSize = adjRanks.size();
        std::cout << "rank " << rank << " has ";
        for (int i = 0; i < adjSize; ++i) {
            std::cout << adjRanksCpu[i] << " rank, " << static_cast<int>(adjDirsCpu[i]) << " dir " << std::endl;
        }
        std::cout << std::endl;
    }

private:
    int nDims;

    // x, y, z outer -> inner, just like rest of code
    int dimSizes[3];
    int periodic[3];

    MPI_Comm comm;

};

#endif
