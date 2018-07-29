#include <material/Material.h>

const std::string Material::TYPE = "type";

__host__ __device__ Material::Material(MaterialType type) :
        type(type) {
}

Vec3f Material::jsonToColor(const nlohmann::json &j) {
    return Vec3f(j[0], j[1], j[2]);
}
