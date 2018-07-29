#ifndef TRENCHANTTRACER_SPEC_H
#define TRENCHANTTRACER_SPEC_H

#include <curand_kernel.h>

#include <json.hpp>

#include <material/Material.h>
#include <geometry/Ray.h>

class Spec : public Material {
private:
    Vec3f color;
public:
    static const std::string TYPE;

    __host__ __device__ explicit Spec();

    __host__ explicit Spec(const nlohmann::json &material);

    __host__ U32 size() const override;

    __device__ Ray sample(curandState *randState, const Ray &ray, const Hit &hit, Vec3f &mask);
};


#endif //TRENCHANTTRACER_SPEC_H
