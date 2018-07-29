#ifndef TRENCHANTTRACER_DIFF_H
#define TRENCHANTTRACER_DIFF_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <geometry/Ray.h>
#include <geometry/Hit.h>
#include <math/LinearMath.h>
#include "Material.h"

// diffuse material, based on smallpt by Kevin Beason
class Diff : public Material {
    Vec3f diffuseColor;
public:
    static const std::string TYPE;

    __host__ __device__ Diff();

    __host__ explicit Diff(const nlohmann::json &material);

    __host__ U32 size() const override;

    __device__ Ray sample(curandState *randState, const Ray &ray, const Hit &hit, Vec3f &mask);
};

#endif //TRENCHANTTRACER_DIFF_H
