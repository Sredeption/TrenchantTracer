//
// Created by issac on 18-7-14.
//

#ifndef TRENCHANTTRACER_BVHNODE_H
#define TRENCHANTTRACER_BVHNODE_H


#include <math/LinearMath.h>
#include <bvh/AABB.h>
#include <bvh/SAHHelper.h>
#include <util/Array.h>


class BVHNode {
public:
    enum BVH_STAT {
        NODE_COUNT,
        INNER_COUNT,
        LEAF_COUNT,
        TRIANGLE_COUNT,
        CHILD_NODE_COUNT,
    };

    AABB bounding;

    // These are somewhat experimental, for some specific test and may be invalid...
    float probability;          // probability of coming here (widebvh uses this)
    float parentProbability;    // probability of coming to parent (widebvh uses this)

    int treelet;              // for queuing tests (qmachine uses this)
    int index;                // in linearized tree (qmachine uses this)
    BVHNode();

    virtual bool isLeaf() const = 0;

    virtual S32 getNumChildNodes() const = 0;

    virtual BVHNode *getChildNode(S32 i) const = 0;

    virtual S32 getNumTriangles() const;

    float getArea() const;

    // Subtree functions
    int getSubtreeSize(BVH_STAT stat = NODE_COUNT) const;

    void computeSubtreeProbabilities(const SAHHelper &helper, float parentProbability, float &sah);

    float computeSubtreeSAHCost(const SAHHelper &helper) const;     // NOTE: assumes valid probabilities

    void deleteSubtree();

    void assignIndicesDepthFirst(S32 index = 0, bool includeLeafNodes = true);

    void assignIndicesBreadthFirst(S32 index = 0, bool includeLeafNodes = true);
};


#endif //TRENCHANTTRACER_BVHNODE_H

