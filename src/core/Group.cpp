#include <core/Group.h>

Group::Group(std::string name, Geometry *geometry) :
        name(std::move(name)), geometry(geometry) {

}

Geometry *Group::getGeometry() const {
    return geometry;
}

void Group::setMaterial(Material *material) {
    this->material = material;
}

Material *Group::getMaterial() const {
    return this->material;
}
