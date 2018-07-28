#include <loader/MaterialLoader.h>

MaterialPool *MaterialLoader::load(std::string fileName) {

    auto pool = new MaterialPool();
    std::ifstream ifs(fileName);
    std::stringstream buffer;
    buffer << ifs.rdbuf();

    auto materials = nlohmann::json::parse(buffer.str());
    for (nlohmann::json::iterator it = materials.begin(); it != materials.end(); ++it) {
        nlohmann::json &text = it.value();
        Material *material;
        if (text[Material::TYPE] == Coat::TYPE) {
            material = new Coat(text);
        } else if (text[Material::TYPE] == Diff::TYPE) {
            material = new Diff(text);
        }
        pool->add(it.key(), material);
    }

    ifs.close();
    return pool;
}
