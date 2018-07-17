#ifndef TRENCHANTTRACER_SAH_H
#define TRENCHANTTRACER_SAH_H


#include <string>
#include <math/LinearMath.h>

//Surface Area Heuristic
class SAHHelper {
public:
    SAHHelper() {
        this->name = std::string("Default");
        this->sahNodeCost = 1.f;
        this->sahTriangleCost = 1.f;
        this->nodeBatchSize = 1;
        this->triBatchSize = 1;
        this->minLeafSize = 1;
        this->maxLeafSize = 0x7FFFFFF;
    } // leafsize = aantal tris

    explicit SAHHelper(const std::string &name, float nodeCost = 1.f, float triCost = 1.f, S32 nodeBatchSize = 1,
                       S32 triBatchSize = 1) {
        this->name = name;
        this->sahNodeCost = nodeCost;
        this->sahTriangleCost = triCost;
        this->nodeBatchSize = nodeBatchSize;
        this->triBatchSize = triBatchSize;
        this->minLeafSize = 1;
        this->maxLeafSize = 0x7FFFFFF;
    }

    const std::string &getName() const;

    // SAH weights
    float getSAHTriangleCost() const;

    float getSAHNodeCost() const;

    // SAH costs, raw and batched
    float getCost(int numChildNodes, int numTris) const;

    float getTriangleCost(S32 n) const;

    float getNodeCost(S32 n) const;

    // batch processing (how many ops at the price of one)
    S32 getTriangleBatchSize() const;

    S32 getNodeBatchSize() const;

    void setTriangleBatchSize(S32 triBatchSize);

    void setNodeBatchSize(S32 nodeBatchSize);

    S32 roundToTriangleBatchSize(S32 n) const;

    S32 roundToNodeBatchSize(S32 n) const;

    // leaf preferences
    void setLeafPreferences(S32 minSize, S32 maxSize);

    S32 getMinLeafSize() const;

    S32 getMaxLeafSize() const;

private:
    std::string name;
    float sahNodeCost;
    float sahTriangleCost;
    S32 triBatchSize;
    S32 nodeBatchSize;
    S32 minLeafSize;
    S32 maxLeafSize;
};


#endif //TRENCHANTTRACER_SAH_H
