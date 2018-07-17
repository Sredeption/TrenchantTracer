#include <bvh/SAHHelper.h>

const std::string &SAHHelper::getName() const {
    return name;
}

float SAHHelper::getSAHTriangleCost() const {
    return sahTriangleCost;
}

float SAHHelper::getSAHNodeCost() const {
    return sahNodeCost;
}

float SAHHelper::getCost(int numChildNodes, int numTris) const {
    return getNodeCost(numChildNodes) + getTriangleCost(numTris);
}

float SAHHelper::getTriangleCost(S32 n) const {
    return roundToTriangleBatchSize(n) * sahTriangleCost;
}

float SAHHelper::getNodeCost(S32 n) const {
    return roundToNodeBatchSize(n) * sahNodeCost;
}

S32 SAHHelper::getTriangleBatchSize() const {
    return triBatchSize;
}

S32 SAHHelper::getNodeBatchSize() const {
    return nodeBatchSize;
}

void SAHHelper::setTriangleBatchSize(S32 triBatchSize) {
    this->triBatchSize = triBatchSize;
}

void SAHHelper::setNodeBatchSize(S32 nodeBatchSize) {
    this->nodeBatchSize = nodeBatchSize;
}

S32 SAHHelper::roundToTriangleBatchSize(S32 n) const {
    return ((n + triBatchSize - 1) / triBatchSize) * triBatchSize;
}

S32 SAHHelper::roundToNodeBatchSize(S32 n) const {
    return ((n + nodeBatchSize - 1) / nodeBatchSize) * nodeBatchSize;
}

void SAHHelper::setLeafPreferences(S32 minSize, S32 maxSize) {
    minLeafSize = minSize;
    maxLeafSize = maxSize;
}

S32 SAHHelper::getMinLeafSize() const {
    return minLeafSize;
}

S32 SAHHelper::getMaxLeafSize() const {
    return maxLeafSize;
}
