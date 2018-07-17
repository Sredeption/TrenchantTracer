
#ifndef TRENCHANTTRACER_OBJLOADER_H
#define TRENCHANTTRACER_OBJLOADER_H

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include <core/Object.h>
#include <math/LinearMath.h>

class OBJLoader {
public:
    Object *load(std::string fileName);
};


#endif //TRENCHANTTRACER_OBJLOADER_H
