#ifndef TRENCHANTTRACER_REFR_H
#define TRENCHANTTRACER_REFR_H

#include <curand_kernel.h>

#include <material/Material.h>
#include <geometry/Ray.h>

class Refr : public Material {

public:
    static const std::string TYPE;

    __host__ __device__ Refr();

    __host__ explicit Refr(const nlohmann::json &material);

    __host__ U32 size() const override;

    __device__ Ray sample(curandState *randState, const Ray &ray, const Hit &hit, Vec3f &mask);
};


#endif //TRENCHANTTRACER_REFR_H
