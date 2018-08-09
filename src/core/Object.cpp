#include <core/Object.h>

Object::Object() = default;

Object::~Object() {
    delete this->pool;
    for (Group *group: this->groups) {
        delete group;
    }
}

void Object::addVertex(const Vec3f &vertex) {
    vertices.add(vertex);
}

void Object::addNormal(const Vec3f &normal) {
    normals.add(normal);
}

const Array<Vec3f> &Object::getVertices() {
    return vertices;
}

void Object::addGroup(Group *group) {
    this->groups.push_back(group);
}

void Object::setMaterialPool(MaterialPool *pool) {
    this->pool = pool;
}

MaterialPool *Object::getMaterialPool() {
    return this->pool;
}

std::vector<Group *> Object::getGroups() {
    return this->groups;
}

void Object::postProcess() {
    // Center scene at world's center
    Vec3f minp(FLT_MAX, FLT_MAX, FLT_MAX);
    Vec3f maxp(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    // calculate min and max bounds of object
    // loop over all triangles in all groups, grow minp and maxp

    for (Group *group: groups) {
        Array<Vec3i> &triangles = static_cast<Mesh *>(group->getGeometry())->getVertexIndices();
        for (int i = 0; i < triangles.getSize(); i++) {
            for (int vertexIndex : triangles[i]._v) {
                const Vec3f &vertex = vertices[vertexIndex];
                minp = min3f(minp, vertex);
                maxp = max3f(maxp, vertex);
            }
        }
    }
    // scene bounding box center before scaling and translating
    Vec3f origCenter = Vec3f(
            (maxp.x + minp.x) * 0.5f,
            (maxp.y + minp.y) * 0.5f,
            (maxp.z + minp.z) * 0.5f);
    minp -= origCenter;
    maxp -= origCenter;

    float maxi = 0;
    maxi = std::max(maxi, std::fabs(minp.x));
    maxi = std::max(maxi, std::fabs(minp.y));
    maxi = std::max(maxi, std::fabs(minp.z));
    maxi = std::max(maxi, std::fabs(maxp.x));
    maxi = std::max(maxi, std::fabs(maxp.y));
    maxi = std::max(maxi, std::fabs(maxp.z));

    float scaleFactor = 1.0f / maxi;
    std::cout << "Scaling factor: " << scaleFactor << std::endl;
    std::cout << "Center origin: " << origCenter << std::endl;

    std::cout << "\nCentering and scaling vertices..." << std::endl;
    for (unsigned i = 0; i < vertices.getSize(); i++) {
        vertices[i] -= origCenter;
        vertices[i] *= scaleFactor;
    }
}

void Object::apply(const Transform &transform) {
    for (int i = 0; i < vertices.getSize(); i++) {
        vertices[i] = transform.apply(vertices[i], VERTEX);
        normals[i] = transform.apply(normals[i], NORMAL);
        normals[i].normalize();
    }
}

const Array<Vec3f> &Object::getNormals() {
    return normals;
}
