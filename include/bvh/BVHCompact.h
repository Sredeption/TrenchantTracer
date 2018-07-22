#ifndef TRENCHANTTRACER_BVHHOLDER_H
#define TRENCHANTTRACER_BVHHOLDER_H


#include <bvh/BVH.h>
#include <math/LinearMath.h>
#include <util/Array.h>


class BVH;

class BVHCompact {
private:
    struct StackEntry {
        const BVHNode *node;
        S32 idx;

        StackEntry(const BVHNode *n = nullptr, int i = 0) : node(n), idx(i) {}
    };


    Vec4i *nodes; // device memory
    Vec4i *woopTri; // device memory
    Vec4i *debugTri; // device memory
    S32 *triIndices; // device memory

    U32 nodesSize;
    U32 woopTriSize;
    U32 debugTriSize;
    U32 triIndicesSize;
    U32 leafNodeCount;
    U32 triCount;
public:

    __host__ explicit BVHCompact(const BVH &bvh);

    __host__ explicit BVHCompact(FILE *bvhFile);

    __host__ ~BVHCompact();

    __host__ void createCompact(const BVH &bvh, int nodeOffsetSizeDiv);

    __host__ void woopifyTri(const BVH &bvh, int idx, Vec4f *woopTri, Vec4f *debugTri);

    __host__ void save(const std::string &fileName);
};


#endif //TRENCHANTTRACER_BVHHOLDER_H
