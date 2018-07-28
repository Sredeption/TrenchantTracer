#ifndef TRENCHANTTRACER_HIT_H
#define TRENCHANTTRACER_HIT_H


#include <math/LinearMath.h>

class Hit {
public:
    int index;
    float distance;
    Vec3f normal;
    Vec3f n; //normalized normal
    Vec3f nl; // correctly oriented normal
    Vec3f point; // intersection point

    __host__ __device__ Hit() :
            index(-1), distance(-1), normal(), n(), nl() {
    }

    __host__ __device__ Hit(const int &index, const float &distance) :
            index(index), distance(distance), normal(), n(), nl() {
    }

    __host__ __device__ Hit(const Hit &hit) :
            index(hit.index), distance(hit.distance), normal(hit.normal), n(hit.n), nl(hit.nl) {
    }

};

#endif //TRENCHANTTRACER_HIT_H
