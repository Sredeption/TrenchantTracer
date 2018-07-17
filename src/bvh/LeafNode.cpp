#include <bvh/LeafNode.h>

LeafNode::LeafNode(const AABB &bounds, int lo, int hi) {
    this->bounding = bounds;
    this->lo = lo;
    this->hi = hi;
}

LeafNode::LeafNode(const LeafNode &s) : BVHNode(s) {
    *this = s;
}

bool LeafNode::isLeaf() const {
    return true;
}

S32 LeafNode::getNumChildNodes() const {
    return 0;
}

BVHNode *LeafNode::getChildNode(S32) const {
    return nullptr;
}

S32 LeafNode::getNumTriangles() const {
    return hi - lo;
}