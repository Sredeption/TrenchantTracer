#ifndef TRENCHANTTRACER_GEOMETRYUNION_H
#define TRENCHANTTRACER_GEOMETRYUNION_H

#include <geometry/Plane.h>
#include <geometry/Sphere.h>

union GeometryUnion {
    Plane plane;
    Sphere sphere;
};

#endif //TRENCHANTTRACER_GEOMETRYUNION_H
