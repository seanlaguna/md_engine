#include "IntegratorVerlet.h"

#include <chrono>

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "Logging.h"
#include "State.h"

namespace py = boost::python;

__global__ void preForce_cu(int nAtoms, float4 *xs, float4 *vs, float4 *fs,
                            float dt)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocity by a half timestep
        float4 vel = vs[idx];
        float invmass = vel.w;

        float4 force = fs[idx];

        float3 dv = 0.5f * dt * invmass * make_float3(force);
        vel += dv;
        vs[idx] = vel;

        // Update position by a full timestep
        float4 pos = xs[idx];

        float3 dx = dt*make_float3(vel);
        pos += dx;
        xs[idx] = pos;

        // Set forces to zero before force calculation
        fs[idx] = make_float4(0.0f, 0.0f, 0.0f, force.w);
    }
}

__global__ void postForce_cu(int nAtoms, float4 *vs, float4 *fs, float dt)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocities by a halftimestep
        float4 vel = vs[idx];
        float invmass = vel.w;

        float4 force = fs[idx];

        float3 dv = 0.5f * dt * invmass * make_float3(force);
        vel += dv;
        vs[idx] = vel;
    }
}

IntegratorVerlet::IntegratorVerlet(State *statePtr)
    : Integrator(statePtr)
{

}
//so now each thread is responsibe for NPERTHREAD pieces of data
/*
template <class K, class T, int NPERTHREAD>
__global__ void NAME (K *dest, T *src, int n, int warpSize) {
    extern __shared__ K tmp[];
    const int copyBaseIdx = blockDim.x*blockIdx.x * NPERTHREAD + threadIdx.x;
    //printf("idx %d gets base %d\n", GETIDX(), copyBaseIdx);
    const int copyIncrement = blockDim.x;
    for (int i=0; i<NPERTHREAD; i++) {
        int step = i * copyIncrement;

        if (copyBaseIdx + step < n) {
            tmp[threadIdx.x + step] = length(src[copyBaseIdx + step]);
           // printf("copyBase getting idx %d got %f\n", copyBaseIdx, tmp[step + threadIdx.x]);
        } else {
            tmp[threadIdx.x + step] = 0;
        }
    }
    int curLookahead = NPERTHREAD;
    int numLookaheadSteps = log2f(blockDim.x-1);
    const int sumBaseIdx = threadIdx.x * NPERTHREAD;
    __syncthreads();
    for (int i=sumBaseIdx+1; i<sumBaseIdx + NPERTHREAD; i++) {
        tmp[sumBaseIdx] += tmp[i];

    }
  //  printf("idx %d summed to %f lookahead is %d sumBase is %d\n", GETIDX(), tmp[sumBaseIdx], curLookahead, sumBaseIdx);
    for (int i=0; i<=numLookaheadSteps; i++) {
        if (! (sumBaseIdx % (curLookahead*2))) {
     //       printf("thread base %d fetching from %d lookahead %d\n", sumBaseIdx, sumBaseIdx + curLookahead, curLookahead);
            tmp[sumBaseIdx] += tmp[sumBaseIdx + curLookahead];
        }
        if (curLookahead >= (NPERTHREAD * warpSize)) {
            __syncthreads();
        }
        curLookahead *= 2;
    }
    if (threadIdx.x == 0) {
        atomicAdd(dest, tmp[0]);
    }
}
*/
void IntegratorVerlet::run(int numTurns)
{
    basicPreRunChecks();

    //! \todo Call basicPreRunChecks() in basicPrepare()
    basicPrepare(numTurns);

    int periodicInterval = state->periodicInterval;

    auto start = std::chrono::high_resolution_clock::now();
    bool computeVirials = state->computeVirials;
    for (int i=0; i<numTurns; ++i) {
        if (state->turn % periodicInterval == 0) {
            state->gridGPU.periodicBoundaryConditions();
        }
        // Prepare for timestep
        //! \todo Should asyncOperations() and doDataCollection() go into
        //!       Integrator::stepInit()? Same for periodicBoundayConditions()
        asyncOperations();
        doDataCollection();

        stepInit(computeVirials);

        // Perform first half of velocity-Verlet step
        preForce();

        // Recalculate forces
        force(computeVirials);

        // Perform second half of velocity-Verlet step
        postForce();

        stepFinal();

        //! \todo The following parts could also be moved into stepFinal
        state->turn++;
        if (state->verbose && (i+1 == numTurns || state->turn % state->shoutEvery == 0)) {
            mdMessage("Turn %d %.2f percent done.\n", (int)state->turn, 100.0*(i+1)/numTurns);
        }
    }

    //! \todo These parts could be moved to basicFinish()
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    mdMessage("runtime %f\n%e particle timesteps per second\n",
              duration.count(), state->atoms.size()*numTurns / duration.count());

    basicFinish();
}

void IntegratorVerlet::preForce()
{
    uint activeIdx = state->gpd.activeIdx();
    preForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.xs.getDevData(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            state->dt);
}

void IntegratorVerlet::postForce()
{
    uint activeIdx = state->gpd.activeIdx();
    postForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            state->dt);
    std::cout << "atom 0 has pos: ("
              << state->gpd.xs.h_data[0].x << ","
              << state->gpd.xs.h_data[0].y << ","
              << state->gpd.xs.h_data[0].z << "); velocity: ("
              << state->gpd.vs.h_data[0].x << ","
              << state->gpd.vs.h_data[0].y << ","
              << state->gpd.vs.h_data[0].z << "); force: ("
              << state->gpd.fs.h_data[0].x << ","
              << state->gpd.fs.h_data[0].y << ","
              << state->gpd.fs.h_data[0].z << ")" << std::endl;
}

void export_IntegratorVerlet()
{
    py::class_<IntegratorVerlet,
               boost::shared_ptr<IntegratorVerlet>,
               py::bases<Integrator>,
               boost::noncopyable>
    (
        "IntegratorVerlet",
        py::init<State *>()
    )
    .def("run", &IntegratorVerlet::run)
    ;
}
