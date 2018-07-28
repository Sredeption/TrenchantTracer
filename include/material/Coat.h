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
// randomly select diffuse or specular reflection
// looks okay-ish but inaccurate (no Fresnel calculation yet)
class Coat : public Material {
public:
    static const std::string TYPE;

    Vec3f specularColor;
    Vec3f diffuseColor;

    __host__ __device__ explicit Coat();

    __host__ explicit Coat(const nlohmann::json &material);

    __device__ Ray sample(curandState *randState, const Ray &ray, const Hit &hit, Vec3f &mask);

};

#endif //TRENCHANTTRACER_COAT_H
