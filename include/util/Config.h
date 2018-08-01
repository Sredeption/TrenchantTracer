#ifndef TRENCHANTTRACER_CONFIG_H
#define TRENCHANTTRACER_CONFIG_H

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include <json.hpp>

class Config {
public:
    int width;
    int height;
    std::string hdrFileName;
    std::string objFileName;
    std::string materialFile;
    int samples;
    bool bvhReload;
    bool materialReload;
    nlohmann::json objects;

    explicit Config(std::string fileName);
};

#endif //TRENCHANTTRACER_CONFIG_H
