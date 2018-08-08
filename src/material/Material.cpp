#include <material/Material.h>
#include <geometry/Geometry.h>

const std::string Material::TYPE = "type";
const std::string Material::EMISSION = "emission";

__host__ Material::Material(MaterialType type, const nlohmann::json &material) :
        type(type) {
    if (material.find(EMISSION) == material.end())
        emission = Vec3f(0, 0, 0);
    else
        emission = Geometry::jsonToVec(material[EMISSION]);
}

Vec3f Material::jsonToColor(const nlohmann::json &j) {
    return Vec3f(j[0], j[1], j[2]);
}
