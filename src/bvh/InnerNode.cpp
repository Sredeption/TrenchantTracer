#include <bvh/InnerNode.h>

InnerNode::InnerNode(const AABB &bounds, BVHNode *child0, BVHNode *child1) {
    bounding = bounds;
    children[0] = child0;
    children[1] = child1;
}

bool InnerNode::isLeaf() const {
    return false;
}

S32 InnerNode::getNumChildNodes() const {
    return 2;
}

BVHNode *InnerNode::getChildNode(S32 i) const {
    FW_ASSERT(i >= 0 && i < 2);
    return children[i];
}

