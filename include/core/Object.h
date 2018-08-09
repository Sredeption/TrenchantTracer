#ifndef TRENCHANTTRACER_OBJECT_H
#define TRENCHANTTRACER_OBJECT_H


#include <vector>
#include <cfloat>
#include <iostream>
#include <cmath>

#include <core/Group.h>
#include <geometry/Transform.h>
#include <material/MaterialPool.h>

class Group;

class MaterialPool;

class Object {
private:
    Array<Vec3f> vertices;
    Array<Vec3f> normals;
    std::vector<Group *> groups;
    MaterialPool *pool;

public:
    Object();

    ~Object();

    void addVertex(const Vec3f &vertex);

    void addNormal(const Vec3f &normal);

    void addGroup(Group *group);

    void setMaterialPool(MaterialPool *pool);

    MaterialPool *getMaterialPool();

    const Array<Vec3f> &getVertices();

    const Array<Vec3f> &getNormals();

    std::vector<Group *> getGroups();

    void apply(const Transform &transform);

    void postProcess();
};


#endif //TRENCHANTTRACER_OBJECT_H
