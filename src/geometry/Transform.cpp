#include <geometry/Transform.h>
#include <geometry/Geometry.h>
#include <math/PoseMath.h>

const std::string Transform::SCALE = "scale";
const std::string Transform::ORIENTATION = "orientation";
const std::string Transform::TRANSLATE = "translate";

Vec3f Transform::jsonToVec(const nlohmann::json &j) {
    return Vec3f(j[0], j[1], j[2]);
}

Transform::Transform(const nlohmann::json &geometryJson) {
    matrix.setIdentity();

    if (geometryJson.find(Geometry::TRANSFORM) == geometryJson.end())
        return;
    const nlohmann::json &transformJson = geometryJson[Geometry::TRANSFORM];

    if (transformJson.find(SCALE) != transformJson.end()) {
        Vec3f scale = jsonToVec(transformJson[SCALE]);
        matrix = matrix * PoseMath::scale(scale);
    }

    if (transformJson.find(TRANSLATE) != transformJson.end()) {
        Vec3f translate = jsonToVec(transformJson[TRANSLATE]);
        matrix = matrix * PoseMath::translate(translate);
    }
    if (transformJson.find(ORIENTATION) != transformJson.end()) {
        Vec3f orientation = (jsonToVec(transformJson[ORIENTATION]) / 180.0f) * M_PI;
        matrix = matrix * PoseMath::orientation(orientation);
    }
}

Vec3f Transform::apply(Vec3f &vec, TransformType type) const {
    int a;
    switch (type) {
        case VERTEX:
            a = 1;
            break;
        case NORMAL:
            a = 0;
            break;
    }
    Vec4f v = matrix * Vec4f(vec, a);
    return Vec3f(v.x, v.y, v.z);
}
