#include <material/Material.h>

const std::string Material::TYPE = "type";

__host__ __device__ Material::Material(MaterialType type) :
        type(type) {

}
