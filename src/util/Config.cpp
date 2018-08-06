#include <util/Config.h>

Config::Config(std::string fileName) {

    std::ifstream ifs(fileName);
    std::stringstream buffer;

    buffer << ifs.rdbuf();
    auto config = nlohmann::json::parse(buffer.str());
    workDir = config.at("work-dir").get<std::string>();
    width = config.at("width").get<int>();
    height = config.at("height").get<int>();
    hdrFileName = config.at("hdr-file-name").get<std::string>();
    materialFile = config.at("material-file").get<std::string>();
    samples = config.at("samples").get<int>();
    bvhReload = config.at("bvh-reload").get<bool>();
    materialReload = config.at("material-reload").get<bool>();
    objects = config["objects"];

    ifs.close();
}
