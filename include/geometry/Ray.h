#ifndef TRENCHANTTRACER_RAY_H
#define TRENCHANTTRACER_RAY_H

#include <cuda_runtime.h>

#include <geometry/Hit.h>
#include <math/LinearMath.h>
#include <bvh/BVHCompact.h>

#define STACK_SIZE  64  // Size of the traversal stack in local memory.
#define EntrypointSentinel 0x76543210

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
    __device__ Hit intersect(const BVHCompact *bvh, bool needClosestHit);

};


#endif //TRENCHANTTRACER_RAY_H
