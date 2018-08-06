#ifndef TRENCHANTTRACER_TRANSFORM_H
#define TRENCHANTTRACER_TRANSFORM_H


#include <json.hpp>

#include <math/LinearMath.h>

class Transform {
private:
    Mat4f matrix;
public:
    static const std::string SCALE;
    static const std::string ORIENTATION;
    static const std::string TRANSLATE;

    explicit Transform(const nlohmann::json &j);

    static Vec3f jsonToVec(const nlohmann::json &j);
};


#endif //TRENCHANTTRACER_TRANSFORM_H
