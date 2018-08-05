#include <geometry/Transform.h>

Transform::Transform(const nlohmann::json &j) {
    if (j.find("scale") != j.end()) {
    }

    if (j.find("orientation") != j.end()) {
        const nlohmann::json &o = j["orientation"];
    }
}
