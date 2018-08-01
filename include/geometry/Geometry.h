#ifndef TRENCHANTTRACER_GEOMETRY_H
#define TRENCHANTTRACER_GEOMETRY_H

#include <string>

#include <json.hpp>

#include <math/LinearMath.h>

enum GeometryType : U8 {
    SPHERE, CUBE, MESH, PLANE
};  // geometry types

class Geometry {
public:
    static const std::string TYPE;

    GeometryType type;

    __host__ __device__ explicit Geometry(GeometryType type);

    __host__ virtual U32 size() const = 0;

    static Vec3f jsonToVec(const nlohmann::json &j);
};


#endif //TRENCHANTTRACER_GEOMETRY_H
