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
public:
    static const std::string TYPE;

    Vec3f diffuseColor;

    __host__ Diff();

    __host__ explicit Diff(const nlohmann::json &material);

    __host__ U32 size() const override;

};

#endif //TRENCHANTTRACER_DIFF_H
