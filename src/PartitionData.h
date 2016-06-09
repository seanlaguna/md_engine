#pragma once
#ifndef PARTITION_DATA_H
#define PARTITION_DATA_H

enum OOBDir {
    LDI = 0,  MDI = 1,  RDI = 2,
    LMI = 4,  MMI = 5,  RMI = 6,
    LUI = 8,  MUI = 9,  RUI = 10,

    LDM = 16, MDM = 17, RDM = 18,
    LMM = 20, MMM = 21, RMM = 22,
    LUM = 24, MUM = 25, RUM = 26,

    LDO = 32, MDO = 33, RDO = 34,
    LMO = 36, MMO = 37, RMO = 38,
    LUO = 40, MUO = 41, RUO = 42
}
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
}

class PartitionData
{
public:
    PartitionData(bool is2d, bool periodic_in[3])
            : nDims(is2d? 2 : 3)
    {
        int nRanks;
        MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
        // for now
        dimSizes[0] = nRanks;
        for (int i = 1; i < nDims; ++i) {
            dimSizes[i] = 1;
        }
        if (nDims == 2) {
            periodic = { periodic_in[0], periodic_in[1] };
        } else {
            periodic = periodic_in;
        }
        MPI_Cart_create(MPI_COMM_WORLD, nDims, dimSizes, periodic, 0, &comm);
        adjacentRanks = getAdjacentRanks();
    }

    // returns index in list of adjacent directions
    // returns size of adjacint directions if not adjacent
    __host__ __device__ int getAdjIdx(OOBDir dir) {
        for (int i = 0; i < adjDirs.size(); ++i) {
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

private:
    OOBDir offsToDir(int offs[3]) {
        uint bases[3] = { 0, 2, 4 };
        OOBDir dir = 0;
        for (int i = 0; i < nDims; ++i) {
            if (offs[i] >= 0) {
                dir &= (1 << ((uint)offs[i] + bases[i]));
            }
            if (nDims == 2) { dir &= 16; }
        }
        return dir;
    }

    std::vector<int> getAdjacentRanks()
    {
        std::vector<int> adjacentRanks;
        std::vector<OOBDir> adjacentDirs;
        
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
                       if (nRanks == 2 && offz != 0) { continue; }
                       if (nRanks == 3) { 
                            coordsAdj[2] = coords[2] + offz;
                       }
                       bool skip = false;
                       for (int i = 0; i < nDims; ++i) {
                           if ((!periodic[i]) ||
                               (coordAdj[i] < 0 || coordAdj[i] >= dimSizes[i])) {
                               skip = true;
                               break;
                           }
                       }
                       if (skip) { continue; }
                       MPI_Cart_rank(comm, coordsAdj, &rankAdj);
                       adjacentRanks.push_back(rankAdj);
                       adjacentDirs.push_back(offsToDir({ offx, offy, offz }));
                  }  // end offz
             }  // end offy
        }  // end offx
        std::sort(adjacentRanks);
        return adjacentRanks;
    }

private:
    const int nDims;

    // x, y, z outer -> inner, just like rest of code
    int dimSizes[nDims];
    bool periodic[nDims];

    MPI_Comm comm;

}

#endif
