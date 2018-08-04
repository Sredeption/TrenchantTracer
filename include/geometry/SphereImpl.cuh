#ifndef TRENCHANTTRACER_SPHEREIMPL_H
#define TRENCHANTTRACER_SPHEREIMPL_H

#include <geometry/Sphere.h>

#include <cuda_runtime.h>

#include <geometry/Ray.h>

__device__ __forceinline__ Hit sphereIntersect(const Sphere *sphere, const Ray &ray) {
    Hit hit;
    hit.distance = ray.tMax;

    Vec3f op = sphere->position - ray.origin;
    float t, epsilon = 0.01f;
    float b = dot(op, ray.direction);
    float disc = b * b - dot(op, op) + sphere->radius * sphere->radius; // discriminant of quadratic formula
    if (disc >= 0) {
        disc = sqrtf(disc);
        float distance = (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0.0f);
        if (0.01f < distance && distance < ray.tMax) {
            hit.distance = distance;
            hit.point = ray.origin + ray.direction * hit.distance; // intersection point
            hit.normal = Vec3f(hit.point.x - sphere->position.x,
                               hit.point.y - sphere->position.y,
                               hit.point.z - sphere->position.z); // normal
        }
    }

    return hit;
}

#endif //TRENCHANTTRACER_SPHEREIMPL_H
