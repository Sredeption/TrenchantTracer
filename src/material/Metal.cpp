#include <material/Metal.h>

const std::string Metal::TYPE = "Metal";

__host__ Metal::Metal(const nlohmann::json &material) : Material(METAL, material) {
    color = jsonToColor(material["color"]);
}

U32 Metal::size() const {
    return sizeof(Metal);
}
