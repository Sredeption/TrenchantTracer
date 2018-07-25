#include <geometry/Ray.h>


Ray::Ray(const Vec3f &origin, const Vec3f &direction, float tMin, float tMax) {
    this->origin = Vec4f(origin.x, origin.y, origin.z, tMin);
    this->direction = Vec4f(direction.x, direction.y, direction.z, tMax);
}

Ray::Ray(const Vec4f &origin, const Vec4f &direction) : origin(origin), direction(direction) {
}

Ray::Ray(const Ray &ray) : origin(ray.origin), direction(ray.direction) {
}
