#ifndef TRENCHANTTRACER_COAT_H
#define TRENCHANTTRACER_COAT_H


#include <json.hpp>

#include <material/Material.h>

class Ray;


// COAT material based on https://github.com/peterkutz/GPUPathTracer
class Coat : public Material {
public:
    static const std::string TYPE;
    Vec3f specularColor;
    Vec3f diffuseColor;

    __host__ explicit Coat();

    __host__ explicit Coat(const nlohmann::json &material);

    __host__ U32 size() const override;

};

#endif //TRENCHANTTRACER_COAT_H
