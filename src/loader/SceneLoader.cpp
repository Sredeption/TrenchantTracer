#include <loader/SceneLoader.h>
#include <geometry/Transform.h>

SceneLoader::SceneLoader(Config *config) {
    this->config = config;
}

Scene *SceneLoader::load() {
    auto scene = new Scene();

    OBJLoader objLoader;

    MaterialLoader materialLoader;
    MaterialPool *masterPool = materialLoader.load(config->materialFile);
    scene->add(masterPool);
    for (nlohmann::json &geometry : config->objects) {
        Transform transform(geometry);
        if (geometry[Geometry::TYPE] == Mesh::TYPE) {
            Object *object = objLoader.load(geometry["file"]);
            object->apply(transform);
            scene->add(object);
            delete object;
        } else {
            scene->add(loadGeometry(geometry, masterPool));
        }
    }

    return scene;
}

Group *SceneLoader::loadGeometry(nlohmann::json &text, MaterialPool *pool) {
    Geometry *geometry = nullptr;
    if (text[Geometry::TYPE] == Sphere::TYPE) {
        geometry = new Sphere(text);
    } else if (text[Geometry::TYPE] == Plane::TYPE) {
        geometry = new Plane(text);
    }
    auto group = new Group("", geometry);
    auto material = pool->get(text["material"]);
    group->setMaterial(material);
    return group;
}
