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

