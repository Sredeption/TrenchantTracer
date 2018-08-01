#include <geometry/Geometry.h>

const std::string Geometry::TYPE = "type";

Geometry::Geometry(GeometryType type) :
        type(type) {

}

Vec3f Geometry::jsonToVec(const nlohmann::json &j) {
    return Vec3f(j[0], j[1], j[2]);
}
