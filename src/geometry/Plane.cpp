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

