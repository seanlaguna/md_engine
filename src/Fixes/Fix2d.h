#pragma once
#ifndef FIX2D_H
#define FIX2D_H

#include "Fix.h"

class State;

void export_Fix2d();
class Fix2d : public Fix {

private:
    const std::string _2dType = "2d";

public:
    Fix2d(boost::shared_ptr<State> state_, std::string handle_, int applyEvery_)
      : Fix(state_, handle_, "all", _2dType, true, applyEvery_, 999)
    {   }

    void compute(bool);

};


#endif
