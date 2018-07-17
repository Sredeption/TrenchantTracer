//
// Created by issac on 18-7-14.
//

#ifndef TRENCHANTTRACER_INNERNODE_H
#define TRENCHANTTRACER_INNERNODE_H


#include <bvh/BVHNode.h>

class InnerNode : public BVHNode {
public:
    InnerNode(const AABB &bounds, BVHNode *child0, BVHNode *child1);

    bool isLeaf() const override;

    S32 getNumChildNodes() const override;

    BVHNode *getChildNode(S32 i) const override;

    BVHNode *children[2];
};

#endif //TRENCHANTTRACER_INNERNODE_H
