#include <util/Config.h>

Config::Config(std::string fileName) {

    std::ifstream ifs(fileName);
    std::stringstream buffer;

    buffer << ifs.rdbuf();
    auto config = nlohmann::json::parse(buffer.str());
    width = config.at("width").get<int>();
    height = config.at("height").get<int>();
    hdrFileName = config.at("hdr-file-name").get<std::string>();
    objFileName = config.at("obj-file-name").get<std::string>();
    samples = config.at("samples").get<int>();

    ifs.close();
}
