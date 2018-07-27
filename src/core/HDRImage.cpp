#include <core/HDRImage.h>
#include <iostream>
#include <cstring>

HDRImage::HDRImage(long width, long height, float *colors) {
    this->width = static_cast<int>(width);
    this->height = static_cast<int>(height);
    copyToGPU(colors);
    createTexture();
}

HDRImage::~HDRImage() {
    cudaFree(hdrEnv);
}

void HDRImage::copyToGPU(float *colors) {
    auto cpuHDREnv = new Vec4f[width * height];

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int idx1 = width * j + i;
            int idx2 = 3 * idx1;
            cpuHDREnv[idx1] = Vec4f(colors[idx2], colors[idx2 + 1], colors[idx2 + 2], 0.0f);
        }
    }

    // copy HDR map to CUDA
    cudaMalloc(&hdrEnv, width * height * sizeof(float4));
    cudaMemcpy(hdrEnv, cpuHDREnv, width * height * sizeof(float4), cudaMemcpyHostToDevice);

    delete[] cpuHDREnv;
}

void HDRImage::createTexture() {
    cudaResourceDesc resDesc{};
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = hdrEnv;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // r-channel bits
    resDesc.res.linear.desc.y = 32; // g-channel bits
    resDesc.res.linear.desc.z = 32; // b-channel bits
    resDesc.res.linear.desc.w = 32; // a-channel bits
    resDesc.res.linear.sizeInBytes = width * height * sizeof(float4);

    cudaTextureDesc texDesc{};
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.filterMode = cudaFilterModeLinear;

    cudaCreateTextureObject(&hdrTexture, &resDesc, &texDesc, nullptr);
}

