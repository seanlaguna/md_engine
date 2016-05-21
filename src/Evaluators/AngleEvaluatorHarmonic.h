#pragma once
#ifndef EVALUATOR_ANGLE_HARMONIC
#define EVALUATOR_ANGLE_HARMONIC

#include "cutils_math.h"
#include "Angle.h"
#define EPSILON 0.00001f
class AngleEvaluatorHarmonic {
public:

    //evaluator.force(theta, angleType, s, distSqrs, directors, invDotProd);
    inline __device__ float3 force(AngleHarmonicType angleType, float theta, float s, float c, float distSqrs[2], float3 directors[2], float invDistProd, int myIdxInAngle) {
        float dTheta = theta - angleType.thetaEq;
        //   printf("current %f theta eq %f idx %d, type %d\n", acosf(c), angleType.thetaEq, myIdxInAngle, type);
        

        float forceConst = angleType.k * dTheta;
        float a = -2.0f * forceConst * s;
        float a11 = a*c/distSqrs[0];
        float a12 = -a*invDistProd;
        float a22 = a*c/distSqrs[1];
        //   printf("forceConst %f a %f s %f dists %f %f %f\n", forceConst, a, s, a11, a12, a22);
        //printf("hey %f, eq %f\n", theta, angleType.thetaEq);
        //printf("directors %f %f %f .. %f %f %f\n", directors[0].x, directors[0].y, directors[0].z,directors[1].x, directors[1].y, directors[1].z);
        //printf("a a11 a12 a22 %f %f %f %f\n", a, a11, a12, a22);
        if (myIdxInAngle==0) {
            return ((directors[0] * a11) + (directors[1] * a12)) * 0.5;
        } else if (myIdxInAngle==1) {
            return ((directors[0] * a11) + (directors[1] * a12) + (directors[1] * a22) + (directors[0] * a12)) * -0.5; 
        } else {
            return ((directors[1] * a22) + (directors[0] * a12)) * 0.5;
        }


    }

};

#endif

