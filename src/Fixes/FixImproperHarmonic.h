#pragma once
#ifndef FIXIMPROPERHARMONIC_H
#define FIXIMPROPERHARMONIC_H

#include "FixPotentialMultiAtom.h"
#include "Improper.h"
#include "VariantPyListInterface.h"

void export_FixImproperHarmonic();
class FixImproperHarmonic: public FixPotentialMultiAtom<ImproperVariant, ImproperHarmonic, ImproperHarmonicGPU, 4> {

public:
    VariantPyListInterface<ImproperVariant, ImproperHarmonic> pyListInterface;
    //DataSet *eng;
    //DataSet *press;

    FixImproperHarmonic(SHARED(State) state_, std::string handle);

    void compute(bool);
    std::string restartChunk(std::string format);

    void createImproper(Atom *, Atom *, Atom *, Atom *, double, double, int);
    void setImproperTypeCoefs(int, double, double);

    //std::vector<pair<int, std::vector<int> > > neighborlistExclusions();

};

#endif
