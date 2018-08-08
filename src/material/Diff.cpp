#include <material/Diff.h>

const std::string Diff::TYPE = "Diff";

__host__ Diff::Diff(const nlohmann::json &material) : Material(DIFF,material) {
    diffuseColor = jsonToColor(material["diffuseColor"]);
}

__host__ U32 Diff::size() const {
    return sizeof(Diff);
}

