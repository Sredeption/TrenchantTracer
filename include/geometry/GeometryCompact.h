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

    ~GeometryCompact();
};


#endif //TRENCHANTTRACER_GEOMETRYCOMPACT_H
