#ifndef TRENCHANTTRACER_MATERIALPOOL_H
#define TRENCHANTTRACER_MATERIALPOOL_H

#include <map>
#include <string>

#include <material/Material.h>
#include <util/Array.h>


class MaterialPool {
private:
    U32 size;
    std::map<std::string, Material *> materials;
public:
    MaterialPool();

    void add(const std::string &name, Material *material);

    Material *get(const std::string &name);

    Array<Material *> all();

};

#endif //TRENCHANTTRACER_MATERIALPOOL_H
