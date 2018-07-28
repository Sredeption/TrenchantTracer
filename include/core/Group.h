//
// Created by issac on 18-7-11.
//

#ifndef TRENCHANTTRACER_GROUP_H
#define TRENCHANTTRACER_GROUP_H

#include <string>
#include <utility>

#include <geometry/Mesh.h>
#include <material/Material.h>

class Group {
private:
    std::string name;
    Mesh *mesh;
    Material *material;
public:
    Group(std::string name, Mesh *mesh);

    Mesh *getMesh();

    void setMaterial(Material *material);

    Material* getMaterial();
};


#endif //TRENCHANTTRACER_GROUP_H
