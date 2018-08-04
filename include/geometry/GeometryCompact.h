#ifndef TRENCHANTTRACER_GEOMETRYCOMPACT_H
#define TRENCHANTTRACER_GEOMETRYCOMPACT_H


#include <core/Scene.h>
#include <geometry/Geometry.h>

class GeometryCompact {
public:
    Geometry **cpuGeometries;
    Geometry **geometries; // device memory
    int1 *matIndices; // device memory
    U32 *geometryLength;

    U32 geometriesSize;

    __host__ explicit GeometryCompact(Scene *scene);

    __host__ explicit GeometryCompact(FILE *geoFile);

    __host__ ~GeometryCompact();

    __host__ void save(const std::string &fileName);
};


#endif //TRENCHANTTRACER_GEOMETRYCOMPACT_H
