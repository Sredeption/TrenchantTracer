#include <bvh/BVHNode.h>

BVHNode::BVHNode() : probability(1.f), parentProbability(1.f), treelet(-1), index(-1) {

}

S32 BVHNode::getNumTriangles() const {
    return 0;
}


float BVHNode::getArea() const {
    return bounding.area();
}

// recursively counts some type of nodes (either leafNodes, innerNodes, childNodes) or nunmber of triangles
int BVHNode::getSubtreeSize(BVH_STAT stat) const {
    int count;

    switch (stat) {
        default:
            FW_ASSERT(0);  // unknown mode
        case NODE_COUNT:
            count = 1;
            break; // counts all nodes including leafNodes
        case LEAF_COUNT:
            count = isLeaf() ? 1 : 0;
            break; // counts only leafNodes
        case INNER_COUNT:
            count = isLeaf() ? 0 : 1;
            break; // counts only innerNodes
        case TRIANGLE_COUNT:
            count = isLeaf() ? getNumTriangles() : 0;
            break; // counts all triangles
        case CHILD_NODE_COUNT:
            count = getNumChildNodes();
            break; //counts only childNodes
    }

    // if current node is not a leaf node, continue counting its childNodes recursively
    if (!isLeaf()) {
        for (int i = 0; i < getNumChildNodes(); i++)
            count += getChildNode(i)->getSubtreeSize(stat);
    }

    return count;
}

void BVHNode::deleteSubtree() {
    for (int i = 0; i < getNumChildNodes(); i++)
        getChildNode(i)->deleteSubtree();

    delete this;
}

void BVHNode::computeSubtreeProbabilities(const SAHHelper &helper, float probability, float &sah) {
    sah += probability * helper.getCost(this->getNumChildNodes(), this->getNumTriangles());
    this->probability = probability;

    // recursively compute probabilities and add to SAH
    for (int i = 0; i < getNumChildNodes(); i++) {
        BVHNode *child = getChildNode(i);
        child->parentProbability = probability;           // childNode area / parentNode area
        child->computeSubtreeProbabilities(helper, probability * child->bounding.area() / this->bounding.area(), sah);
    }
}

// TODO: requires valid probabilities...
float BVHNode::computeSubtreeSAHCost(const SAHHelper &helper) const {
    float sah = probability * helper.getCost(getNumChildNodes(), getNumTriangles());

    for (int i = 0; i < getNumChildNodes(); i++)
        sah += getChildNode(i)->computeSubtreeSAHCost(helper);

    return sah;
}

void assignIndicesDepthFirstRecursive(BVHNode *node, S32 &index, bool includeLeafNodes) {
    if (node->isLeaf() && !includeLeafNodes)
        return;

    node->index = index++;
    for (int i = 0; i < node->getNumChildNodes(); i++)
        assignIndicesDepthFirstRecursive(node->getChildNode(i), index, includeLeafNodes);
}

void BVHNode::assignIndicesDepthFirst(S32 index, bool includeLeafNodes) {
    assignIndicesDepthFirstRecursive(this, index, includeLeafNodes);
}

void BVHNode::assignIndicesBreadthFirst(S32 index, bool includeLeafNodes) {
    Array<BVHNode *> nodes;  // array acts like a stack
    nodes.add(this);
    S32 head = 0;

    while (head < nodes.getSize()) {
        // pop
        BVHNode *node = nodes[head++];
        // discard
        if (node->isLeaf() && !includeLeafNodes)
            continue;
        // assign
        node->index = index++;

        // push children
        for (int i = 0; i < node->getNumChildNodes(); i++)
            nodes.add(node->getChildNode(i));
    }
}

