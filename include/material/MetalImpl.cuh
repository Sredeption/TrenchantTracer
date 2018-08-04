#ifndef TRENCHANTTRACER_METALIMPL_H
#define TRENCHANTTRACER_METALIMPL_H

#include <material/Metal.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <geometry/Ray.h>

// Phong metal material from "Realistic Ray Tracing", P. Shirley
__device__ __inline__ Ray metalSample(Metal *metal, curandState *randState, const Ray &ray, const Hit &hit, Vec3f &mask) {
    Ray nextRay = ray;// ray of next path segment
    // compute random perturbation of ideal reflection vector
    // the higher the phong exponent, the closer the perturbed vector is to the ideal reflection direction
    float phi = 2 * M_PI * curand_uniform(randState);
    float r2 = curand_uniform(randState);
    float phongExponent = 30;
    float cosTheta = powf(1 - r2, 1.0f / (phongExponent + 1));
    float sinTheta = sqrtf(1 - cosTheta * cosTheta);

    // create orthonormal basis uvw around reflection vector with hitpoint as origin
    // w is ray direction for ideal reflection
    Vec3f w = ray.direction - hit.n * 2.0f * dot(hit.n, ray.direction);
    w.normalize();
    Vec3f u = cross((fabs(w.x) > .1 ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)), w);
    u.normalize();
    Vec3f v = cross(w, u); // v is already normalised because w and u are normalised

    // compute cosine weighted random ray direction on hemisphere
    nextRay.direction = u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta;
    nextRay.direction.normalize();

    // offset origin next path segment to prevent self intersection
    nextRay.origin = hit.point + hit.nl * 0.0001f;  // scene size dependent

    // multiply mask with colour of object
    mask *= metal->color;
    return nextRay;
}

#endif //TRENCHANTTRACER_METALIMPL_H
