#ifndef TRENCHANTTRACER_DIFF_H
#define TRENCHANTTRACER_DIFF_H


#include <material/Material.h>
#include <math/LinearMath.h>

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
