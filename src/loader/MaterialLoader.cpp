#include <loader/MaterialLoader.h>

#include <material/Coat.h>
#include <material/Diff.h>
#include <material/Metal.h>
#include <material/Spec.h>
#include <material/Refr.h>

MaterialPool *MaterialLoader::load(std::string fileName) {

    auto pool = new MaterialPool();
    std::ifstream ifs(fileName);
    std::stringstream buffer;
    buffer << ifs.rdbuf();

    auto materials = nlohmann::json::parse(buffer.str());
    for (nlohmann::json::iterator it = materials.begin(); it != materials.end(); ++it) {
        nlohmann::json &text = it.value();
        Material *material = nullptr;
        if (text[Material::TYPE] == Coat::TYPE) {
            material = new Coat(text);
        } else if (text[Material::TYPE] == Diff::TYPE) {
            material = new Diff(text);
        } else if (text[Material::TYPE] == Metal::TYPE) {
            material = new Metal(text);
        } else if (text[Material::TYPE] == Spec::TYPE) {
            material = new Spec(text);
        } else if (text[Material::TYPE] == Refr::TYPE) {
            material = new Refr(text);
        }
        pool->add(it.key(), material);
    }

    ifs.close();
    return pool;
}
