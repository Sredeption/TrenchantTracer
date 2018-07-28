#include <material/MaterialPool.h>

MaterialPool::MaterialPool() :
        size(0) {
}

void MaterialPool::add(const std::string &name, Material *material) {
    size++;
    materials[name] = material;
}

Material *MaterialPool::get(const std::string &name) {
    return materials[name];
}

Array<Material *> MaterialPool::all() {
    Array<Material *> allMaterials;
    for (auto &it : materials) {
        allMaterials.add(it.second);
    }

    return allMaterials;
}
