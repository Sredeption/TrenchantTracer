#ifndef TRENCHANTTRACER_MATERIAL_H
#define TRENCHANTTRACER_MATERIAL_H

#include <math/LinearMath.h>
#include <string>

enum MaterialType : U8 {
    DIFF, METAL, SPEC, REFR, COAT
};  // material types

class Material {
public:
    static const std::string TYPE;

    U32 index;
    MaterialType type;

    __host__ __device__ explicit Material(MaterialType type);
};


#endif //TRENCHANTTRACER_MATERIAL_H
