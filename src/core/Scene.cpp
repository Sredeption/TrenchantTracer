#include <core/Scene.h>

Scene::Scene() = default;

Scene::~Scene() = default;

void Scene::add(Object *object) {
    int base = this->vertices.getSize();
    Vertices *vertices = object->getVertices();
    this->vertices.add(vertices->getVertex());

    for (Group *group:object->getGroups()) {
        Array<Vec3i> vertexIndices = group->getMesh()->getVertexIndices();
        for (int i = 0; i < vertexIndices.getSize(); i++) {
            this->vertexIndices.add(vertexIndices[i] + base);
        }
    }
}

void Scene::add(HDRImage *hdrEnv) {
    this->hdrEnv = hdrEnv;
}

int Scene::getNumTriangles() const {
    return vertexIndices.getSize();
}

const Vec3i *Scene::getTrianglePtr(int idx) {
    FW_ASSERT(idx >= 0 && idx <= m_numTris);
    return vertexIndices.getPtr() + idx;
}

const Vec3i &Scene::getTriangle(int idx) {
    FW_ASSERT(idx < m_numTris);
    return *getTrianglePtr(idx);
}

int Scene::getNumVertices() const {
    return vertices.getSize();
}

const Vec3f *Scene::getVertexPtr(int idx) {
    FW_ASSERT(idx >= 0 && idx <= m_numVerts);
    return (const Vec3f *) vertices.getPtr() + idx;
}

const Vec3f &Scene::getVertex(int idx) {
    FW_ASSERT(idx < m_numVerts);
    return *getVertexPtr(idx);
}

const HDRImage *Scene::getHDREnv() {
    return hdrEnv;
}
