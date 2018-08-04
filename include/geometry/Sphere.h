#ifndef TRENCHANTTRACER_SPHERE_H
#define TRENCHANTTRACER_SPHERE_H


#include <geometry/Geometry.h>
#include <geometry/Ray.h>

class Sphere : public Geometry {
public:

    static const std::string TYPE;

    float radius;
    Vec3f position;

    __host__ __device__ Sphere();

    __host__ Sphere(const nlohmann::json &geometry);

    __host__ U32 size() const override;


};


#endif //TRENCHANTTRACER_SPHERE_H
