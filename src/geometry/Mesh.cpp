#include <geometry/Mesh.h>

void Mesh::addVertex(const Vec3i &vertex) {
    vertexIndices.add(vertex);
}

void Mesh::addTexture(const Vec3i &texture) {
    textureIndices.add(texture);
}

void Mesh::addNormal(const Vec3i &normal) {
    normalIndices.add(normal);
}

bool Mesh::empty() {
    return vertexIndices.getSize() == 0;
}

int Mesh::size() {
    return vertexIndices.getSize();
}

Array<Vec3i> &Mesh::getVertexIndices() {
    return this->vertexIndices;
}

Array<Vec3i> &Mesh::getTextureIndices() {
    return this->textureIndices;
}

Array<Vec3i> &Mesh::getNormalIndices() {
    return this->normalIndices;
}

