#include <core/Object.h>

Object::Object() {
    this->vertices = new Vertices();
}

Object::~Object() {
    delete this->vertices;
    for (Group *group: this->groups) {
        delete group;
    }
}

void Object::addVertex(Vec3f vertex) {
    this->vertices->add(vertex);
}

void Object::addGroup(Group *group) {
    this->groups.push_back(group);
}

Vertices *Object::getVertices() {
    return this->vertices;
}

std::vector<Group *> Object::getGroups() {
    return this->groups;
}

void Object::postProcess() {
// Rescale input objects to have this size...
    const float MaxCoordAfterRescale = 1.2f;

    // Center scene at world's center
    Vec3f minp(FLT_MAX, FLT_MAX, FLT_MAX);
    Vec3f maxp(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    Array<Vec3f> &vertices = this->vertices->getVertex();
    // calculate min and max bounds of object
    // loop over all triangles in all groups, grow minp and maxp

    for (Group *group: groups) {
        Array<Vec3i> &triangles = group->getMesh()->getVertexIndices();
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
    maxi = std::max(maxi, (float) fabs(minp.x));
    maxi = std::max(maxi, (float) fabs(minp.y));
    maxi = std::max(maxi, (float) fabs(minp.z));
    maxi = std::max(maxi, (float) fabs(maxp.x));
    maxi = std::max(maxi, (float) fabs(maxp.y));
    maxi = std::max(maxi, (float) fabs(maxp.z));

    std::cout << "Scaling factor: " << (MaxCoordAfterRescale / maxi) << "\n";
    std::cout << "Center origin: " << origCenter.x << " " << origCenter.y << " " << origCenter.z << "\n";

    std::cout << "\nCentering and scaling vertices..." << std::endl;
    for (unsigned i = 0; i < vertices.getSize(); i++) {
        vertices[i] -= origCenter;
        //vertices[i].y += origCenter.y;
        //vertices[i] *= (MaxCoordAfterRescale / maxi);
        vertices[i] *= 0.1; // 0.25
    }
}
