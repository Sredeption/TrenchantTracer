//
// Created by issac on 18-7-18.
//

#ifndef TRENCHANTTRACER_BVHHOLDER_H
#define TRENCHANTTRACER_BVHHOLDER_H


#include <bvh/BVH.h>
#include <math/LinearMath.h>
#include <util/Array.h>


class BVH;

class BVHHolder {
private:
    struct StackEntry {
        const BVHNode *node;
        S32 idx;

        StackEntry(const BVHNode *n = nullptr, int i = 0) : node(n), idx(i) {}
    };

    const BVH &bvh;

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

    explicit BVHHolder(const BVH &bvh);

    ~BVHHolder();

    void createCompact(int nodeOffsetSizeDiv);

    void woopifyTri(int idx, Vec4f *woopTri, Vec4f *debugTri);
};


#endif //TRENCHANTTRACER_BVHHOLDER_H
