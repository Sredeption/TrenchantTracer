#ifndef TRENCHANTTRACER_HIT_H
#define TRENCHANTTRACER_HIT_H


#include <math/LinearMath.h>

class Hit {
public:
    int index;
    int matIndex;
    float distance;
    Vec3f normal;
    Vec3f n; //normalized normal
    Vec3f nl; // correctly oriented normal
    Vec3f point; // intersection point

    __host__ __device__ Hit() :
            index(-1), matIndex(-1), distance(-1), normal(), n(), nl() {
    }

    __host__ __device__ Hit(const int &index, const float &distance) :
            index(index), matIndex(-1), distance(distance), normal(), n(), nl() {
    }

    __host__ __device__ Hit(const Hit &hit) :
            index(hit.index), matIndex(hit.matIndex), distance(hit.distance), normal(hit.normal),
            n(hit.n), nl(hit.nl), point(hit.point) {
    }

    inline __host__ __device__ bool operator<(const Hit &that) const {
        if (this->distance == that.distance)
            return false;
        if (that.distance == -1)
            return true;
        return this->distance < that.distance;
    }

    inline __host__ __device__ bool operator>(const Hit &that) const {
        if (this->distance == that.distance)
            return false;
        if (this->distance == -1)
            return true;
        return this->distance > that.distance;
    }
};

#endif //TRENCHANTTRACER_HIT_H
