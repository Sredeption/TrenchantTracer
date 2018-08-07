#ifndef TRENCHANTTRACER_OBJECT_H
#define TRENCHANTTRACER_OBJECT_H


#include <vector>
#include <cfloat>
#include <iostream>
#include <cmath>

#include <core/Group.h>
#include <geometry/Vertices.h>
#include <material/MaterialPool.h>

class Group;

class MaterialPool;

class Object {
private:
    Vertices *vertices;
    std::vector<Group *> groups;
    MaterialPool *pool;

public:
    Object();

    ~Object();

    void addVertex(Vec3f vertex);

    void addGroup(Group *group);

    void setMaterialPool(MaterialPool *pool);

    MaterialPool *getMaterialPool();

    Vertices *getVertices();

    std::vector<Group *> getGroups();

    void apply(const Transform &transform);

    void postProcess();
};


#endif //TRENCHANTTRACER_OBJECT_H
