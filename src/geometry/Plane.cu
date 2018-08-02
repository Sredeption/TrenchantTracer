#include <geometry/Plane.h>

const std::string Plane::TYPE = "Plane";

__host__ __device__ Plane::Plane() : Geometry(PLANE) {

}

__host__ Plane::Plane(const nlohmann::json &geometry) : Plane() {
    p0 = jsonToVec(geometry["p0"]);
    p1 = jsonToVec(geometry["p1"]);
    p2 = jsonToVec(geometry["p2"]);
}

__host__ U32 Plane::size() const {
    return sizeof(Plane);
}

__device__ Hit Plane::intersect(const Ray &ray) const {
    // Möller–Trumbore intersection algorithm

    Hit hit;
    Vec3f e1, e2;
    Vec3f P, Q, T;
    float det, inv_det, u, v;
    float t;

    hit.distance = ray.tMax;
    // Find vectors for two edges sharing p0
    e1 = p1 - p0;
    e2 = p2 - p0;
    // Begin calculating determinant - also used to calculate u parameter
    P = cross(ray.direction, e2);
    // if determinant is near zero, ray lies in plane of triangle
    det = dot(e1, P);
    // NOT CULLING
    if (det > -0.00001 && det < 0.00001)
        return hit;
    inv_det = 1.f / det;

    // calculate distance from p0 to ray origin
    T = ray.origin - p0;
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
