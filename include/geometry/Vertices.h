#ifndef TRENCHANTTRACER_VERTICES_H
#define TRENCHANTTRACER_VERTICES_H


#include <util/Array.h>

class Vertices {
private:
    Array<Vec3f> vertex;

public:
    Vertices();

    void add(Vec3f &vertex);

    Array<Vec3f> &getVertex();

};


#endif //TRENCHANTTRACER_VERTICES_H
