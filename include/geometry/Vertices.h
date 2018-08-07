#ifndef TRENCHANTTRACER_VERTICES_H
#define TRENCHANTTRACER_VERTICES_H


#include <geometry/Transform.h>
#include <util/Array.h>

class Vertices {
private:
    Array<Vec3f> vertex;

public:
    Vertices();

    void add(Vec3f &vertex);

    Array<Vec3f> &getVertex();

    void apply(const Transform &transform);

};


#endif //TRENCHANTTRACER_VERTICES_H
