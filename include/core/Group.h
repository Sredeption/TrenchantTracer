#ifndef TRENCHANTTRACER_GROUP_H
#define TRENCHANTTRACER_GROUP_H


#include <string>
#include <utility>

#include <geometry/Mesh.h>
#include <material/Material.h>

class Material;

class Group {
private:
    std::string name;
    Geometry *geometry;
    Material *material;
public:
    Group(std::string name, Geometry *geometry);

    Geometry *getGeometry() const;

    void setMaterial(Material *material);

    Material *getMaterial() const;
};


#endif //TRENCHANTTRACER_GROUP_H
