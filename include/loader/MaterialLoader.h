#ifndef TRENCHANTTRACER_MATERIALLOADER_H
#define TRENCHANTTRACER_MATERIALLOADER_H


#include <string>
#include <fstream>
#include <sstream>

#include <json.hpp>

#include <material/MaterialPool.h>

class MaterialLoader {
public:

    MaterialPool *load(std::string fileName);
};


#endif //TRENCHANTTRACER_MATERIALLOADER_H
