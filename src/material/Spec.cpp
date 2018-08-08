#include <material/Spec.h>

const std::string Spec::TYPE = "Spec";

__host__ Spec::Spec(const nlohmann::json &material) : Material(SPEC, material) {
    color = jsonToColor(material["color"]);
}

__host__ U32 Spec::size() const {
    return sizeof(Spec);
}

