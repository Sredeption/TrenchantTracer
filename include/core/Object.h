#ifndef TRENCHANTTRACER_OBJECT_H
#define TRENCHANTTRACER_OBJECT_H


#include <vector>
#include <cfloat>
#include <iostream>

#include <core/Group.h>
#include <geometry/Vertices.h>

class Object {
private:
    Vertices *vertices;
    std::vector<Group *> groups;

public:
    Object();

    ~Object();

    void addVertex(Vec3f vertex);

    void addGroup(Group * group);

    Vertices* getVertices();

    std::vector<Group *> getGroups();

    void postProcess();
};


#endif //TRENCHANTTRACER_OBJECT_H
