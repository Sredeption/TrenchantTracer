#include <geometry/Mesh.h>

const std::string Mesh::TYPE = "Mesh";

Mesh::Mesh() : Geometry(MESH) {
}

U32 Mesh::size() const {
    return sizeof(Mesh);
}

void Mesh::addVertex(const Vec3i &vertex) {
    vertexIndices.add(vertex);
}

void Mesh::addTexture(const Vec3i &texture) {
    textureIndices.add(texture);
}

void Mesh::addNormal(const Vec3i &normal) {
    normalIndices.add(normal);
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

