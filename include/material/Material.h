#ifndef TRENCHANTTRACER_MATERIAL_H
#define TRENCHANTTRACER_MATERIAL_H


#include <string>

#include <json.hpp>

#include <math/LinearMath.h>

enum MaterialType : U8 {
    DIFF, METAL, SPEC, REFR, COAT
};  // material types

class Material {
public:
    static const std::string TYPE;
    static const std::string EMISSION;

    U32 index;
    MaterialType type;
    Vec3f emission;

    __host__ explicit Material(MaterialType type, const nlohmann::json &material);

    __host__ virtual U32 size() const = 0;

    static Vec3f jsonToColor(const nlohmann::json &j);
};


#endif //TRENCHANTTRACER_MATERIAL_H
