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
private:
    Vec3f color;
public:

    static const std::string TYPE;

    __host__ __device__ explicit Metal();

    __host__ explicit Metal(const nlohmann::json &material);

    __host__ U32 size() const override;

    __device__ Ray sample(curandState *randState, const Ray &ray, const Hit &hit, Vec3f &mask);
};


#endif //TRENCHANTTRACER_METAL_H
