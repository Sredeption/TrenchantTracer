#ifndef TRENCHANTTRACER_LEAFNODE_H
#define TRENCHANTTRACER_LEAFNODE_H


#include <bvh/BVHNode.h>


class LeafNode : public BVHNode {
public:
    S32 lo;  // lower index in triangle list
    S32 hi;  // higher index in triangle list

    LeafNode(const AABB &bounds, int lo, int hi);

    LeafNode(const LeafNode &s);

    bool isLeaf() const override;

    S32 getNumChildNodes() const override;  // leafnode has 0 children
    BVHNode *getChildNode(S32) const override;

    S32 getNumTriangles() const override;

};


#endif //TRENCHANTTRACER_LEAFNODE_H
