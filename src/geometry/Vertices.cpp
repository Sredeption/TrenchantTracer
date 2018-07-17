
#include <geometry/Vertices.h>

Vertices::Vertices() = default;

void Vertices::add(Vec3f &v) {
    this->vertex.add(v);
}

Array<Vec3f> &Vertices::getVertex() {
    return this->vertex;
}

