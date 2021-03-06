#ifndef TRENCHANTTRACER_BVHHOLDER_H
#define TRENCHANTTRACER_BVHHOLDER_H


#include <cuda_runtime.h>

#include <bvh/BVH.h>
#include <math/LinearMath.h>
#include <util/Array.h>

class BVH;

class BVHNode;

class BVHCompact {
private:
    struct StackEntry {
        const BVHNode *node;
        S32 idx;

        StackEntry(const BVHNode *n = nullptr, int i = 0) : node(n), idx(i) {}
    };

    __host__ void createTexture();

    __host__ void getTriangle(const BVH &bvh, int triIdx, Vec4f *vertex, Vec4f *normal);

public:

    float4 *nodes; // device memory
    float4 *vertices; // device memory
    float4 *normals; // device memory
    int1 *triIndices; // device memory
    int1 *matIndices; // device memory

    cudaTextureObject_t nodesTexture;
    cudaTextureObject_t verticesTexture;
    cudaTextureObject_t normalsTexture;
    cudaTextureObject_t triIndicesTexture;
    cudaTextureObject_t matIndicesTexture;

    U32 nodesSize;
    U32 verticesSize;
    U32 normalsSize;
    U32 triIndicesSize;
    U32 matIndicesSize;
    U32 leafNodeCount;
    U32 triCount;

    __host__ explicit BVHCompact(const BVH &bvh);

    __host__ explicit BVHCompact(FILE *bvhFile);

    __host__ ~BVHCompact();

    __host__ void createCompact(const BVH &bvh, int nodeOffsetSizeDiv);

    __host__ void save(const std::string &fileName);

};


#endif //TRENCHANTTRACER_BVHHOLDER_H
