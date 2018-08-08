#include <material/Coat.h>

const std::string Coat::TYPE = "Coat";

__host__ Coat::Coat(const nlohmann::json &material) : Material(COAT, material) {
    specularColor = jsonToColor(material["specularColor"]);
    diffuseColor = jsonToColor(material["diffuseColor"]);
}

__host__ U32 Coat::size() const {
    return sizeof(Coat);
}

