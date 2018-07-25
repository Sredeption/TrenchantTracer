//
// Created by issac on 18-7-22.
//

#ifndef TRENCHANTTRACER_HIT_H
#define TRENCHANTTRACER_HIT_H


#include <host_defines.h>
#include <math/LinearMath.h>

class Hit {
public:
    int index;
    float distance;
    Vec3f noraml;

    __host__ __device__ Hit();

    __host__ __device__ Hit(const int &index, const float &distance);

    __host__ __device__ Hit(const Hit &hit);
};


#endif //TRENCHANTTRACER_HIT_H
