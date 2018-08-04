#ifndef TRENCHANTTRACER_PLANEIMPL_H
#define TRENCHANTTRACER_PLANEIMPL_H

#include <geometry/Plane.h>
#include <cuda_runtime.h>

#include <geometry/Ray.h>
#include <geometry/Hit.h>

__device__ __forceinline__ Hit planeIntersect(const Plane *plane, const Ray &ray) {
    // Möller–Trumbore intersection algorithm

    Hit hit;
    Vec3f e1, e2;
    Vec3f P, Q, T;
    float det, inv_det, u, v;
    float t;

    hit.distance = ray.tMax;
    // Find vectors for two edges sharing p0
    e1 = plane->p1 - plane->p0;
    e2 = plane->p2 - plane->p0;
    // Begin calculating determinant - also used to calculate u parameter
    P = cross(ray.direction, e2);
    // if determinant is near zero, ray lies in plane of triangle
    det = dot(e1, P);
    // NOT CULLING
    if (det > -0.00001 && det < 0.00001)
        return hit;
    inv_det = 1.f / det;

    // calculate distance from p0 to ray origin
    T = ray.origin - plane->p0;
    Q = cross(T, e1);

    // Calculate u parameter and test bound
    u = dot(T, P) * inv_det;
    v = dot(ray.direction, Q) * inv_det;

    // The intersection lies outside of the plane
    if (u < 0.f || u > 1.f || v < 0.f || v > 1.f)
        return hit;

    t = dot(e2, Q) * inv_det;
    if (t > 0.00001) { //ray intersection
        hit.distance = t;
        hit.normal = cross(e1, e2);
    }

    return hit;
}

#endif //TRENCHANTTRACER_PLANEIMPL_H
