#include <bvh/AABB.h>

AABB::AABB() : minBound(FW_F32_MAX, FW_F32_MAX, FW_F32_MAX), maxBound(-FW_F32_MAX, -FW_F32_MAX, -FW_F32_MAX) {

}

AABB::AABB(const Vec3f &mn, const Vec3f &mx) : minBound(mn), maxBound(mx) {

}

void AABB::grow(const Vec3f &vertex) {
    minBound = min3f(minBound, vertex);
    maxBound = max3f(maxBound, vertex);
}

void AABB::grow(const AABB &aabb) {
    grow(aabb.minBound);
    grow(aabb.maxBound);
}

void AABB::intersect(const AABB &aabb) {
    minBound = max3f(minBound, aabb.minBound);
    maxBound = min3f(maxBound, aabb.maxBound);
}

float AABB::volume() const {
    if (!valid()) return 0.0f;
    return (maxBound.x - minBound.x) * (maxBound.y - minBound.y) * (maxBound.z - minBound.z);
}

Vec3f AABB::midPoint() const {
    return (minBound + maxBound) * 0.5f;
}

float AABB::area() const {
    if (!valid()) return 0.0f;
    Vec3f d = maxBound - minBound;
    return (d.x * d.y + d.y * d.z + d.z * d.x) * 2.0f;
}

bool AABB::valid() const {
    return minBound.x <= maxBound.x && minBound.y <= maxBound.y && minBound.z <= maxBound.z;
}

const Vec3f &AABB::min() const {
    return minBound;
}

const Vec3f &AABB::max() const {
    return maxBound;
}

Vec3f &AABB::min() {
    return minBound;
}

Vec3f &AABB::max() {
    return maxBound;
}

AABB AABB::operator+(const AABB &aabb) const {
    AABB u(*this);
    u.grow(aabb);
    return u;
}
