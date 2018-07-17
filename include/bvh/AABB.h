#ifndef TRENCHANTTRACER_AABB_H
#define TRENCHANTTRACER_AABB_H


#include <math/LinearMath.h>

// Axis Aligned Bounding Box
class AABB {
private:
    Vec3f minBound; // AABB min bound
    Vec3f maxBound; // AABB max bound
public:
    AABB();

    AABB(const Vec3f &mn, const Vec3f &mx);

    // grows bounds to include 3d point pt
    void grow(const Vec3f &vertex);

    void grow(const AABB &aabb);

    // box formed by intersection of 2 AABB boxes
    void intersect(const AABB &aabb);

    // volume = AABB side along X-axis * side along Y * side along Z
    float volume() const;

    float area() const;

    bool valid() const;

    // AABB centroid or midpoint
    Vec3f midPoint() const;

    const Vec3f &min() const;

    const Vec3f &max() const;

    Vec3f &min();

    Vec3f &max();

    AABB operator+(const AABB &aabb) const;

};


#endif //TRENCHANTTRACER_AABB_H
