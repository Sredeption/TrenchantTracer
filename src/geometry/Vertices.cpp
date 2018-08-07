#include <geometry/Vertices.h>

Vertices::Vertices() = default;

void Vertices::add(Vec3f &v) {
    this->vertex.add(v);
}

Array<Vec3f> &Vertices::getVertex() {
    return this->vertex;
}

void Vertices::apply(const Transform &transform) {
    for (int i = 0; i < vertex.getSize(); i++) {
        vertex[i] = transform.apply(vertex[i]);
    }
}

