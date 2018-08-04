#ifndef TRENCHANTTRACER_RAY_H
#define TRENCHANTTRACER_RAY_H


#include <math/LinearMath.h>


class BVHCompact;

class Ray {
public:
    Vec3f origin;
    Vec3f direction;
    float tMin;
    float tMax;


    __host__ __device__ Ray(const Vec3f &origin, const Vec3f &direction, float tMin, float tMax) :
            origin(origin), direction(direction), tMin(tMin), tMax(tMax) {
    }

    __host__ __device__ Ray(const Ray &ray) :
            origin(ray.origin), direction(ray.direction), tMin(ray.tMin), tMax(ray.tMax) {

    }

};


#endif //TRENCHANTTRACER_RAY_H
