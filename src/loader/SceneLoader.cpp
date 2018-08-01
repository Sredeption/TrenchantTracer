#include <loader/SceneLoader.h>

SceneLoader::SceneLoader(Config *config) {
    this->config = config;
}

Scene *SceneLoader::load() {
    auto scene = new Scene();

    OBJLoader objLoader;
    Object *object = objLoader.load(config->objFileName);
    scene->add(object);
    delete object;

    MaterialLoader materialLoader;
    MaterialPool *masterPool = materialLoader.load(config->materialFile);
    scene->add(masterPool);
    for (nlohmann::json &geometry : config->objects) {
        if (geometry[Geometry::TYPE] == Mesh::TYPE) {

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
    }
    auto group = new Group("", geometry);
    auto material = pool->get(text["material"]);
    group->setMaterial(material);
    return group;
}
