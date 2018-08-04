#ifndef TRENCHANTTRACER_SPECIMPL_H
#define TRENCHANTTRACER_SPECIMPL_H

#include <material/Spec.h>

#include <cuda_runtime.h>

#include <curand_kernel.h>
#include <geometry/Ray.h>

__device__ __inline__ Ray specSample(Spec *spec, curandState *randState, const Ray &ray, const Hit &hit, Vec3f &mask) {
    Ray nextRay = ray;
    // ray of next path segment
    // compute reflected ray direction according to Snell's law
    nextRay.direction = ray.direction - hit.n * dot(hit.n, ray.direction) * 2.0f;
    nextRay.direction.normalize();

    // offset origin next path segment to prevent self intersection
    nextRay.origin = hit.point + hit.nl * 0.001f;

    // multiply mask with colour of object
    mask *= spec->color;
    return nextRay;
}

#endif //TRENCHANTTRACER_SPECIMPL_H
