#ifndef TRENCHANTTRACER_BVH_H
#define TRENCHANTTRACER_BVH_H


#include <core/Scene.h>
#include <bvh/BVHNode.h>
#include <bvh/InnerNode.h>
#include <bvh/LeafNode.h>
#include <bvh/BVHHolder.h>
#include <math/LinearMath.h>
#include <util/Array.h>
#include <util/Sort.h>

class BVHHolder;

//Bounding Volume Hierarchies
class BVH {
public:
    struct Stats {
        Stats() { clear(); }

        void clear() { memset(this, 0, sizeof(Stats)); }

        F32 SAHCost;           // Surface Area Heuristic cost
        S32 branchingFactor;
        S32 numInnerNodes;
        S32 numLeafNodes;
        S32 numChildNodes;
        S32 numTris;
    };

    struct BuildParams {
        Stats *stats;
        bool enablePrints;
        F32 splitAlpha;     // spatial split area threshold, see Nvidia paper on SBVH by Martin Stich, usually 0.05

        BuildParams() {
            stats = nullptr;
            enablePrints = true;
            splitAlpha = 1.0e-5f;
        }

    };

private:

    enum {
        MaxDepth = 64,
        MaxSpatialDepth = 48,
        NumSpatialBins = 32,
    };

    // a AABB bounding box enclosing 1 triangle, a reference can be duplicated by a split to be contained in 2 AABB boxes
    struct Reference {
        S32 triIdx;
        AABB bounds;

        Reference() : triIdx(-1) {}
    };

    struct NodeSpec {
        S32 numRef;   // number of references contained by node
        AABB bounds;

        NodeSpec() : numRef(0) {}
    };

    struct ObjectSplit {
        F32 sah;   // cost
        S32 sortDim;  // axis along which triangles are sorted
        S32 numLeft;  // number of triangles (references) in left child
        AABB leftBounds;
        AABB rightBounds;

        ObjectSplit() : sah(FW_F32_MAX), sortDim(0), numLeft(0) {}
    };

    struct SpatialSplit {
        F32 sah;
        S32 dim;   // split axis
        F32 pos;   // position of split along axis (dim)

        SpatialSplit() : sah(FW_F32_MAX), dim(0), pos(0.0f) {}
    };

    struct SpatialBin {
        AABB bounds;
        S32 enter;
        S32 exit;
    };

    Scene *scene;
    SAHHelper sahHelper;
    BVHNode *root;
    Array<S32> triIndices;
    Stats *stats;

    Array<Reference> refStack;
    F32 minOverlap;
    Array<AABB> rightBounds;
    S32 sortDim;
    SpatialBin bins[3][NumSpatialBins];

    S32 numDuplicates;

public:
    BVH(Scene *scene, const SAHHelper &sahHelper, float splitAlpha = 1.0e-5f);

    ~BVH();

    Scene *getScene() const { return scene; }

    const SAHHelper &getPlatform() const { return sahHelper; }

    BVHNode *getRoot() const { return root; }

    Array<S32> &getTriIndices() { return triIndices; }

    const Array<S32> &getTriIndices() const { return triIndices; }

    BVHHolder *createHolder();

private:
    BVHNode *buildNode(const NodeSpec &spec, int level);


    BVHNode *createLeaf(const NodeSpec &spec);

    ObjectSplit findObjectSplit(const NodeSpec &spec, F32 nodeSAH);

    void performObjectSplit(NodeSpec &left, NodeSpec &right, const NodeSpec &spec, const ObjectSplit &split);

    SpatialSplit findSpatialSplit(const NodeSpec &spec, F32 nodeSAH);

    void performSpatialSplit(NodeSpec &left, NodeSpec &right, const NodeSpec &spec, const SpatialSplit &split);

    void splitReference(Reference &left, Reference &right, const Reference &ref, int dim, F32 pos);

    static int sortCompare(void *data, int idxA, int idxB);

    static void sortSwap(void *data, int idxA, int idxB);
};


#endif //TRENCHANTTRACER_BVH_H
