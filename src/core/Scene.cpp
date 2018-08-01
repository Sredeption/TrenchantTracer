#include <core/Scene.h>

Scene::Scene() = default;

Scene::~Scene() = default;

void Scene::add(Object *object) {
    int vertexBase = this->vertices.getSize();
    Vertices *vertices = object->getVertices();
    this->vertices.add(vertices->getVertex());
    add(object->getMaterialPool());

    for (Group *group:object->getGroups()) {
        Array<Vec3i> vertexIndices = static_cast<Mesh *>(group->getGeometry())->getVertexIndices();
        for (int i = 0; i < vertexIndices.getSize(); i++) {
            this->vertexIndices.add(vertexIndices[i] + vertexBase);
            this->materialIndices.add(group->getMaterial()->index);
        }
    }
}

void Scene::add(Group *group) {
    this->geometries.add(group);
}

void Scene::add(MaterialPool *pool) {
    Array<Material *> objectMaterials = pool->all();
    auto materialBase = (U32) this->materials.getSize();
    for (int i = 0; i < objectMaterials.getSize(); i++) {
        objectMaterials[i]->index = materialBase + i;
        this->materials.add(objectMaterials[i]);
    }
}

int Scene::getTrianglesNum() const {
    return vertexIndices.getSize();
}

const Vec3i *Scene::getTrianglePtr(int idx) {
    if (0 <= idx && idx <= vertexIndices.getSize())
        return vertexIndices.getPtr() + idx;
    else
        throw std::runtime_error("vertexIndices out of bound");
}

const Vec3i &Scene::getTriangle(int idx) {
    return *getTrianglePtr(idx);
}

int Scene::getVerticesNum() const {
    return vertices.getSize();
}

const Vec3f *Scene::getVertexPtr(int idx) {
    if (0 <= idx && idx <= vertices.getSize())
        return (const Vec3f *) vertices.getPtr() + idx;
    else
        throw std::runtime_error("vertices out of bound");
}

const Vec3f &Scene::getVertex(int idx) {
    return *getVertexPtr(idx);
}

const U32 *Scene::getMatIndexPtr(int idx) {
    if (0 <= idx && idx <= materialIndices.getSize())
        return (const U32 *) materialIndices.getPtr() + idx;
    else
        throw std::runtime_error("materialIndices out of bound");
}

const U32 &Scene::getMatIndex(int idx) {
    return *getMatIndexPtr(idx);
}

int Scene::getMaterialNum() const {
    return materials.getSize();
}

const Material **Scene::getMaterialPtr(int idx) {
    if (0 <= idx && idx <= materials.getSize())
        return (const Material **) materials.getPtr() + idx;
    else
        throw std::runtime_error("materials out of bound");
}

const Material *&Scene::getMaterial(int idx) {
    return *getMaterialPtr(idx);
}

int Scene::getGeometryNum() const {
    return geometries.getSize();
}

const Group **Scene::getGeometryPtr(int idx) {
    if (0 <= idx && idx <= geometries.getSize())
        return (const Group **) geometries.getPtr() + idx;
    else
        throw std::runtime_error("geometries out of bound");

}

const Group *&Scene::getGeometry(int idx) {
    return *getGeometryPtr(idx);
}

