//
// Created by issac on 18-7-7.
//

#ifndef TRENCHANTTRACER_HDRIMAGE_H
#define TRENCHANTTRACER_HDRIMAGE_H

#include <cuda_runtime.h>

#include <math/LinearMath.h>

class HDRImage {
private:

    void copyToGPU(float *colors);

    void createTexture();

public:
    int width, height;
    // each pixel takes 3 32-bit floats, each component can be of any value...
    float4 *hdrEnv; //device Memory
    cudaTextureObject_t hdrTexture;

    HDRImage(long width, long height, float *colors);

    ~HDRImage();


};


#endif //TRENCHANTTRACER_HDRIMAGE_H
