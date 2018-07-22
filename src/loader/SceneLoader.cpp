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


    return scene;
}
