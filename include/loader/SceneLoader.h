//
// Created by issac on 18-7-15.
//

#ifndef TRENCHANTTRACER_SCENELOADER_H
#define TRENCHANTTRACER_SCENELOADER_H


#include <core/Scene.h>
#include <util/Config.h>
#include <loader/OBJLoader.h>
#include <loader/HDRLoader.h>

class SceneLoader {
    Config *config;

public:
    explicit SceneLoader(Config *config);

    Scene *load();
};


#endif //TRENCHANTTRACER_SCENELOADER_H
