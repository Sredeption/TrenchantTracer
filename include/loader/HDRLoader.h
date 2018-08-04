//
// Created by issac on 18-7-7.
//

#ifndef TRENCHANTTRACER_HDRLOADER_H
#define TRENCHANTTRACER_HDRLOADER_H


#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <stdexcept>

#include <hdr/HDRImage.h>

class HDRLoader {
private:
    typedef unsigned char RGBE[4];
    const static int R = 0;
    const static int G = 1;
    const static int B = 2;
    const static int E = 3;
    const static int MIN_LEN = 8;
    const static int MAX_LEN = 0x7fff;


    float convertComponent(int expo, int val);

    void workOnRGBE(RGBE *scan, int len, float *cols);

    bool decrunch(RGBE *scanLine, int len, FILE *file);

    bool oldDecrunch(RGBE *scanLine, int len, FILE *file);

public:
    HDRLoader();

    HDRImage *load(std::string fileName);
};


#endif //TRENCHANTTRACER_HDRLOADER_H
