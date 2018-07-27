#include <geometry/Ray.h>


__host__ __device__ Ray::Ray(const Vec3f &origin, const Vec3f &direction, float tMin, float tMax)
        : origin(origin), direction(direction), tMin(tMin), tMax(tMax) {
}

__host__ __device__ Ray::Ray(const Ray &ray)
        : origin(ray.origin), direction(ray.direction), tMin(ray.tMin), tMax(ray.tMax) {
}


