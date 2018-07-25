//
// Created by issac on 18-7-22.
//

#include "geometry/Hit.h"

__host__ __device__ Hit::Hit() :
        index(-1), distance(-1) {

}

__host__ __device__ Hit::Hit(const int &index, const float &distance) :
        index(index), distance(distance) {

}

__host__ __device__ Hit::Hit(const Hit &hit) :
        index(hit.index), distance(hit.distance) {

}
