#include <geometry/Transform.h>

const std::string Transform::SCALE = "scale";
const std::string Transform::ORIENTATION = "orientation";
const std::string Transform::TRANSLATE = "scale";

Vec3f Transform::jsonToVec(const nlohmann::json &j) {
    return Vec3f(j[0], j[1], j[2]);
}

Transform::Transform(const nlohmann::json &j) {
    matrix.setIdentity();
    if (j.find(SCALE) != j.end()) {
        Vec3f scale = jsonToVec(j[SCALE]);
    }

    if (j.find(ORIENTATION) != j.end()) {
        const nlohmann::json &o = j[ORIENTATION];
    }

    if (j.find(TRANSLATE) != j.end()) {
        Vec3f translate = jsonToVec(j[TRANSLATE]);
    }
}
