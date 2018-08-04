#ifndef TRENCHANTTRACER_SPEC_H
#define TRENCHANTTRACER_SPEC_H

#include <curand_kernel.h>

#include <json.hpp>

#include <material/Material.h>
#include <geometry/Ray.h>

class Spec : public Material {
public:
    static const std::string TYPE;

    Vec3f color;

    __host__ explicit Spec();

    __host__ explicit Spec(const nlohmann::json &material);

    __host__ U32 size() const override;

};


#endif //TRENCHANTTRACER_SPEC_H
