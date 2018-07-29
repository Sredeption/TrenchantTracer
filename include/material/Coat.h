#ifndef TRENCHANTTRACER_COAT_H
#define TRENCHANTTRACER_COAT_H


#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <json.hpp>

#include <geometry/Ray.h>
#include <geometry/Hit.h>
#include <math/LinearMath.h>
#include <material/Material.h>

class Ray;


// COAT material based on https://github.com/peterkutz/GPUPathTracer
class Coat : public Material {
private:
    Vec3f specularColor;
    Vec3f diffuseColor;
public:
    static const std::string TYPE;

    __host__ __device__ explicit Coat();

    __host__ explicit Coat(const nlohmann::json &material);

    __host__ U32 size() const override;

    __device__ Ray sample(curandState *randState, const Ray &ray, const Hit &hit, Vec3f &mask);

};

#endif //TRENCHANTTRACER_COAT_H
