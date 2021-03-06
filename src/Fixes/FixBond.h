#pragma once 
#ifndef FIXBOND_H
#define FIXBOND_H

#include <array>
#include <unordered_map>

#include <boost/python/list.hpp>

#include "Fix.h"
#include "Bond.h"
#include "State.h"
#include "helpers.h"  // cumulative sum
#include "TypedItemHolder.h"


template <class SRC, class DEST>
int copyBondsToGPU(std::vector<Atom> &atoms, 
                   std::vector<BondVariant> &src, std::vector<int> &idxFromIdCache,
                   GPUArrayDeviceGlobal<DEST> *dest, GPUArrayDeviceGlobal<int> *destIdxs) {

    std::vector<int> idxs(atoms.size()+1, 0);  // started out being used as counts
    std::vector<int> numAddedPerAtom(atoms.size(), 0);

    // so I can arbitrarily order.  I choose to do it by the the way atoms happen to be sorted currently.  Could be improved.
    for (BondVariant &sVar : src) {
        SRC &s = boost::get<SRC>(sVar);
        for (int i=0; i<2; i++) {
            idxs[idxFromIdCache[s.ids[i]]] ++;
        }
    }
    cumulativeSum(idxs.data(), atoms.size()+1);  
    std::vector<DEST> destHost(idxs.back());
    for (BondVariant &sv : src) {
        SRC &s = boost::get<SRC>(sv);
        std::array<int, 2> atomIds = s.ids;
        std::array<int, 2> atomIndexes;
        for (int i=0; i<2; i++) {
            atomIndexes[i] = idxFromIdCache[atomIds[i]];
        }
        for (int i=0; i<2; i++) {
            DEST a;
            a.myId = atomIds[i];
            a.idOther = atomIds[!i];
            a.takeParameters(s);
            destHost[idxs[atomIndexes[i]] + numAddedPerAtom[atomIndexes[i]]] = a;
            numAddedPerAtom[atomIndexes[i]]++;
        }
    }
    *dest = GPUArrayDeviceGlobal<DEST>(destHost.size());
    dest->set(destHost.data());
    *destIdxs = GPUArrayDeviceGlobal<int>(idxs.size());
    destIdxs->set(idxs.data());

    //getting max # bonds per block
    int maxPerBlock = 0;
    for (uint32_t i=0; i<atoms.size(); i+=PERBLOCK) {
        maxPerBlock = std::fmax(maxPerBlock, idxs[std::fmin(i+PERBLOCK+1, idxs.size()-1)] - idxs[i]);
    }
    return maxPerBlock;

}


template <class CPUMember, class GPUMember>
class FixBond : public Fix, public TypedItemHolder {
    public:
        GPUArrayDeviceGlobal<GPUMember> bondsGPU;
        GPUArrayDeviceGlobal<int> bondIdxs;
        std::vector<BondVariant> bonds;
        boost::python::list pyBonds;
        int maxBondsPerBlock;
        std::unordered_map<int, CPUMember> forcerTypes;
        
        FixBond(SHARED(State) state_, std::string handle_, std::string groupHandle_, std::string type_,
                bool forceSingle_, int applyEvery_)
            : Fix(state_, handle_, groupHandle_, type_, forceSingle_, applyEvery_) {
            maxBondsPerBlock = 0;
        }


        void setForcerType(int n, CPUMember &forcer) {
            if (n<0) {
                std::cout << "Tried to set bonded potential for invalid type " << n << std::endl;
                assert(n>=0);
            }
            forcerTypes[n] = forcer;
        }


        bool prepareForRun() {
            std::vector<Atom> &atoms = state->atoms;
            refreshAtoms();
            for (BondVariant &bondVar: bonds) { //applying types to individual elements
                CPUMember &bond = boost::get<CPUMember>(bondVar);
                if (bond.type != -1) {
                    auto it = forcerTypes.find(bond.type);
                    if (it == forcerTypes.end()) {
                        std::cout << "Invalid bonded potential type " << bond.type << std::endl;
                        assert(it != forcerTypes.end());
                    }
                    bond.takeParameters(it->second); 
                }
            }
            maxBondsPerBlock = copyBondsToGPU<CPUMember, GPUMember>(
                    atoms, bonds, state->idxFromIdCache, &bondsGPU, &bondIdxs);

            return true;

        }
        
        
        std::vector<int> getTypeIds() {
            std::vector<int> ids;
            for (auto it=forcerTypes.begin(); it!=forcerTypes.end(); it++) {
                ids.push_back(it->first);
            }
            return ids;
        }


};

#endif
