#ifndef TRENCHANTTRACER_MATERIALCOMPACT_H
#define TRENCHANTTRACER_MATERIALCOMPACT_H


#include <core/Scene.h>

class Scene;

class MaterialCompact {
public:
    Material **cpuMaterials;
    Material **materials; //device memory
    U32 *materialLength;

    U32 materialsSize;

    __host__ explicit MaterialCompact(Scene *scene);

    __host__ explicit MaterialCompact(FILE *matFile);

    __host__ ~MaterialCompact();

    __host__ void save(const std::string &fileName);
};


#endif //TRENCHANTTRACER_MATERIALCOMPACT_H
