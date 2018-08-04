#ifndef TRENCHANTTRACER_PLANE_H
#define TRENCHANTTRACER_PLANE_H

#include <string>

#include <json.hpp>
#include <math/LinearMath.h>
#include <geometry/Geometry.h>

class Plane : public Geometry {
public:
    static const std::string TYPE;
    Vec3f p0, p1, p2;

    __host__ __device__ Plane();

    __host__ Plane(const nlohmann::json &geometry);

    __host__ U32 size() const override;

};


#endif //TRENCHANTTRACER_PLANE_H
