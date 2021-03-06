#pragma once
#ifndef FIXBONDHARMONIC_H
#define FIXBONDHARMONIC_H

#include "Bond.h"
#include "FixBond.h"
#include "VariantPyListInterface.h"

void export_FixBondHarmonic();

class FixBondHarmonic : public FixBond<BondHarmonic, BondHarmonicGPU> {

public:
    VariantPyListInterface<BondVariant, BondHarmonic> pyListInterface;
    //int maxBondsPerBlock;
    //DataSet *eng;
    //DataSet *press;

    FixBondHarmonic(boost::shared_ptr<State> state_, std::string handle);

    ~FixBondHarmonic(){};

    void compute(bool);
    std::string restartChunk(std::string format);

    // HEY - NEED TO IMPLEMENT REFRESHATOMS
    // consider that if you do so, max bonds per block could change
    //bool refreshAtoms();

    void createBond(Atom *, Atom *, double, double, int);  // by ids
    void setBondTypeCoefs(int, double, double);

    const BondHarmonic getBond(size_t i) {
        return boost::get<BondHarmonic>(bonds[i]);
    }
    virtual std::vector<BondVariant> *getBonds() {
        return &bonds;
    }

    //std::vector<pair<int, std::vector<int> > > neighborlistExclusions();

};

#endif
