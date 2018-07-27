#ifndef TRENCHANTTRACER_RAY_H
#define TRENCHANTTRACER_RAY_H

#include <host_defines.h>

#include <geometry/Hit.h>
#include <math/LinearMath.h>
#include <bvh/BVHCompact.h>

#define STACK_SIZE  64  // Size of the traversal stack in local memory.
#define EntrypointSentinel 0x76543210

#define F32_MIN          (1.175494351e-38f)
#define F32_MAX          (3.402823466e+38f)

class BVHCompact;

class Ray {
public:
    Vec3f origin;
    Vec3f direction;
    float tMin;
    float tMax;


    __host__ __device__ Ray(const Vec3f &origin, const Vec3f &direction, float tMin, float tMax);

    __host__ __device__ Ray(const Ray &ray);

    __device__ Hit intersect(const BVHCompact *bvh, bool needClosestHit);

    __device__ Vec3f intersect(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2);

};


#endif //TRENCHANTTRACER_RAY_H
