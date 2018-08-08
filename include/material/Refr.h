#ifndef TRENCHANTTRACER_REFR_H
#define TRENCHANTTRACER_REFR_H

#include <curand_kernel.h>

#include <material/Material.h>
#include <geometry/Ray.h>

class Refr : public Material {

public:
    static const std::string TYPE;

    __host__ explicit Refr(const nlohmann::json &material);

    __host__ U32 size() const override;

};


#endif //TRENCHANTTRACER_REFR_H
