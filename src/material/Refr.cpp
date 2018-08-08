#include <material/Refr.h>

const std::string Refr::TYPE = "Refr";

__host__ Refr::Refr(const nlohmann::json &material) : Material(REFR, material) {

}

__host__ U32 Refr::size() const {
    return sizeof(Refr);
}

