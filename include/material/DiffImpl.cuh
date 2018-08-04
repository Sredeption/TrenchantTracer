#ifndef TRENCHANTTRACER_DIFFIMPL_H
#define TRENCHANTTRACER_DIFFIMPL_H

#include <material/Diff.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <geometry/Ray.h>

__device__ __inline__ Ray diffSample(Diff *diff, curandState *randState, const Ray &ray, const Hit &hit, Vec3f &mask) {
    Ray nextRay = ray;// ray of next path segment
    // pick two random numbers
    float phi = 2 * M_PI * curand_uniform(randState);
    float r2 = curand_uniform(randState);
    float r2s = sqrtf(r2);

    // compute orthonormal coordinate frame uvw with hitpoint as origin
    Vec3f w = hit.nl;
    w.normalize();
    Vec3f u = cross((fabs(w.x) > .1 ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)), w);
    u.normalize();
    Vec3f v = cross(w, u);

    // compute cosine weighted random ray direction on hemisphere
    nextRay.direction = u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2);
    nextRay.direction.normalize();

    // offset origin next path segment to prevent self intersection
    nextRay.origin = hit.point + hit.nl * 0.001f; // scene size dependent

    // multiply mask with colour of object
    mask *= diff->diffuseColor;
    return nextRay;
}

#endif //TRENCHANTTRACER_DIFFIMPL_H
