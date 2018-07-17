//
// Created by issac on 18-7-11.
//

#include <core/Group.h>

Group::Group(std::string name, Mesh *mesh) :
        name(std::move(name)), mesh(mesh) {

}

Mesh *Group::getMesh() {
    return mesh;
}
