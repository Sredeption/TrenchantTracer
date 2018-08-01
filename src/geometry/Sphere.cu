#include <geometry/Sphere.h>

const std::string Sphere::TYPE = "Sphere";

Sphere::Sphere() : Geometry(SPHERE) {
    radius = 2.5;
    position = Vec3f(-6, 0.5, 0);
}

Sphere::Sphere(const nlohmann::json &geometry) : Sphere() {
    radius = geometry["radius"];
    position = jsonToVec(geometry["position"]);
}

U32 Sphere::size() const {
    return sizeof(Sphere);
}

__device__ Hit Sphere::intersect(const Ray &ray) const {
    Hit hit;
    hit.distance = ray.tMax;

    Vec3f op = position - ray.origin;
    float t, epsilon = 0.01f;
    float b = dot(op, ray.direction);
    float disc = b * b - dot(op, op) + radius * radius; // discriminant of quadratic formula
    if (disc >= 0) {
        disc = sqrtf(disc);
        float distance = (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0.0f);
        if (0.01f < distance && distance < ray.tMax) {
            hit.distance = distance;
            hit.point = ray.origin + ray.direction * hit.distance; // intersection point
            hit.normal = Vec3f(hit.point.x - position.x, hit.point.y - position.y, hit.point.z - position.z); // normal
        }
    }

    return hit;
}
