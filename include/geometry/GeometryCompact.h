#ifndef TRENCHANTTRACER_GEOMETRYCOMPACT_H
#define TRENCHANTTRACER_GEOMETRYCOMPACT_H


#include <core/Scene.h>
#include <geometry/Geometry.h>
#include <geometry/GeometryUnion.h>

class GeometryCompact {
private:
    __host__ void createTexture();
public:
    GeometryUnion *geometries; // device memory
    int1 *matIndices; // device memory
    U32 geometriesSize;
    cudaTextureObject_t matIndicesTexture;

    __host__ explicit GeometryCompact(Scene *scene);

    __host__ explicit GeometryCompact(FILE *geoFile);

    __host__ ~GeometryCompact();

    __host__ void save(const std::string &fileName);
};


#endif //TRENCHANTTRACER_GEOMETRYCOMPACT_H
