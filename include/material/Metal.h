//
// Created by issac on 18-7-29.
//

#ifndef TRENCHANTTRACER_METAL_H
#define TRENCHANTTRACER_METAL_H

#include <curand_kernel.h>

#include <json.hpp>

#include <material/Material.h>
#include <geometry/Ray.h>

class Metal : public Material {
public:
    static const std::string TYPE;

    Vec3f color;

    __host__ explicit Metal(const nlohmann::json &material);

    __host__ U32 size() const override;

};


#endif //TRENCHANTTRACER_METAL_H
