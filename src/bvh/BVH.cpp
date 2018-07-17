#include <bvh/BVH.h>

BVH::BVH(Scene *scene, const SAHHelper &sahHelper, const BVH::BuildParams &params) {
    FW_ASSERT(scene);
    this->scene = scene;
    this->sahHelper = sahHelper;
    this->params = params;

    if (params.enablePrints)
        printf("BVH builder: %d tris, %d vertices\n", scene->getNumTriangles(), scene->getNumVertices());

    //  builds the actual BVH

    // See SBVH paper by Martin Stich for details

    // Initialize reference stack and determine root bounds.
    const Vec3i *tris = scene->getTrianglePtr(); // list of all triangles in scene
    const Vec3f *verts = scene->getVertexPtr();  // list of all vertices in scene

    NodeSpec rootSpec;
    rootSpec.numRef = scene->getNumTriangles();  // number of triangles/references in entire scene (root)
    m_refStack.resize(rootSpec.numRef);

    // calculate the bounds of the rootnode by merging the AABBs of all the references
    for (int i = 0; i < rootSpec.numRef; i++) {
        // assign triangle to the array of references
        m_refStack[i].triIdx = i;
        // grow the bounds of each reference AABB in all 3 dimensions by including the vertex
        for (int j : tris[i]._v)
            m_refStack[i].bounds.grow(verts[j]);
        rootSpec.bounds.grow(m_refStack[i].bounds);

    }

    // Initialize rest of the members.
    m_minOverlap = rootSpec.bounds.area() * params.splitAlpha;
    // split alpha (maximum allowable overlap) relative to size of rootNode
    m_rightBounds.reset(max1i(rootSpec.numRef, (int) NumSpatialBins) - 1);
    m_numDuplicates = 0;

    // Build recursively.
    root = buildNode(rootSpec, 0, 0.0f, 1.0f);
    m_triIndices.compact();

    // Done.
    if (params.enablePrints)
        printf("BVH Builder: progress %.0f%%, duplicates %.0f%%\n",
               100.0f, (F32) m_numDuplicates / (F32) scene->getNumTriangles() * 100.0f);

    if (params.enablePrints)
        printf("BVH Scene bounds: (%.1f,%.1f,%.1f) - (%.1f,%.1f,%.1f)\n", root->bounding.min().x,
               root->bounding.min().y, root->bounding.min().z,
               root->bounding.max().x, root->bounding.max().y, root->bounding.max().z);

    float sah = 0.f;
    root->computeSubtreeProbabilities(sahHelper, 1.f, sah);
    if (params.enablePrints)
        printf("top-down sah: %.2f\n", sah);

    if (params.stats) {
        params.stats->SAHCost = sah;
        params.stats->branchingFactor = 2;
        params.stats->numLeafNodes = root->getSubtreeSize(BVHNode::LEAF_COUNT);
        params.stats->numInnerNodes = root->getSubtreeSize(BVHNode::INNER_COUNT);
        params.stats->numTris = root->getSubtreeSize(BVHNode::TRIANGLE_COUNT);
        params.stats->numChildNodes = root->getSubtreeSize(BVHNode::CHILD_NODE_COUNT);
    }
}

BVHNode *BVH::buildNode(const BVH::NodeSpec &spec, int level, F32 progressStart, F32 progressEnd) {
    if (spec.numRef <= sahHelper.getMinLeafSize() || level >= MaxDepth) {
        return createLeaf(spec);
    }

    // Find split candidates.
    F32 area = spec.bounds.area();
    F32 leafSAH = area * sahHelper.getTriangleCost(spec.numRef);
    F32 nodeSAH = area * sahHelper.getNodeCost(2);
    ObjectSplit object = findObjectSplit(spec, nodeSAH);

    SpatialSplit spatial;
    if (level < MaxSpatialDepth) {
        AABB overlap = object.leftBounds;
        overlap.intersect(object.rightBounds);
        if (overlap.area() >= m_minOverlap)
            spatial = findSpatialSplit(spec, nodeSAH);
    }

    F32 minSAH = min1f3(leafSAH, object.sah, spatial.sah);
    // Leaf SAH is the lowest => create leaf.
    if (minSAH == leafSAH && spec.numRef <= sahHelper.getMaxLeafSize()) {
        return createLeaf(spec);
    }

    // Leaf SAH is not the lowest => Perform spatial split.
    NodeSpec left, right;
    if (minSAH == spatial.sah) {
        performSpatialSplit(left, right, spec, spatial);
    }

    if (!left.numRef || !right.numRef) {
        // if either child contains no triangles/references
        performObjectSplit(left, right, spec, object);
    }

    // Create inner node.
    m_numDuplicates += left.numRef + right.numRef - spec.numRef;
    F32 progressMid = lerp(progressStart, progressEnd, (F32) right.numRef / (F32) (left.numRef + right.numRef));
    BVHNode *rightNode = buildNode(right, level + 1, progressStart, progressMid);
    BVHNode *leftNode = buildNode(left, level + 1, progressMid, progressEnd);
    return new InnerNode(spec.bounds, leftNode, rightNode);
}

BVHNode *BVH::createLeaf(const NodeSpec &spec) {
    Array<S32> &tris = m_triIndices;

    for (int i = 0; i < spec.numRef; i++)
        tris.add(m_refStack.removeLast().triIdx); // take a triangle from the stack and add it to tris array

    return new LeafNode(spec.bounds, tris.getSize() - spec.numRef, tris.getSize());
}

BVH::ObjectSplit BVH::findObjectSplit(const NodeSpec &spec, F32 nodeSAH) {

    ObjectSplit split;
    const Reference *refPtr = m_refStack.getPtr(m_refStack.getSize() - spec.numRef);

    // Sort along each dimension.
    for (m_sortDim = 0; m_sortDim < 3; m_sortDim++) {
        sort(m_refStack.getSize() - spec.numRef, m_refStack.getSize(), this, sortCompare, sortSwap);

        // Sweep right to left and determine bounds.
        AABB rightBounds;
        for (int i = spec.numRef - 1; i > 0; i--) {
            rightBounds.grow(refPtr[i].bounds);
            m_rightBounds[i - 1] = rightBounds;
        }

        // Sweep left to right and select lowest SAH.
        AABB leftBounds;
        for (int i = 1; i < spec.numRef; i++) {
            leftBounds.grow(refPtr[i - 1].bounds);
            F32 sah = nodeSAH + leftBounds.area() * sahHelper.getTriangleCost(i) +
                      m_rightBounds[i - 1].area() * sahHelper.getTriangleCost(spec.numRef - i);
            if (sah < split.sah) {
                split.sah = sah;
                split.sortDim = m_sortDim;
                split.numLeft = i;
                split.leftBounds = leftBounds;
                split.rightBounds = m_rightBounds[i - 1];
            }
        }
    }
    return split;
}

void BVH::performObjectSplit(NodeSpec &left, NodeSpec &right, const NodeSpec &spec, const ObjectSplit &split) {
    m_sortDim = split.sortDim;
    sort(m_refStack.getSize() - spec.numRef, m_refStack.getSize(), this, sortCompare, sortSwap);

    left.numRef = split.numLeft;
    left.bounds = split.leftBounds;
    right.numRef = spec.numRef - split.numLeft;
    right.bounds = split.rightBounds;
}

BVH::SpatialSplit BVH::findSpatialSplit(const BVH::NodeSpec &spec, F32 nodeSAH) {
    // Initialize bins.
    Vec3f origin = spec.bounds.min();
    Vec3f binSize = (spec.bounds.max() - origin) * (1.0f / (F32) NumSpatialBins);
    Vec3f invBinSize = Vec3f(1.0f / binSize.x, 1.0f / binSize.y, 1.0f / binSize.z);

    for (auto &m_bin : m_bins) {
        for (auto &bin : m_bin) {
            bin.bounds = AABB();
            bin.enter = 0;
            bin.exit = 0;
        }
    }

    // Chop references into bins.
    for (int refIdx = m_refStack.getSize() - spec.numRef; refIdx < m_refStack.getSize(); refIdx++) {
        const Reference &ref = m_refStack[refIdx];
        Vec3i firstBin = clamp3i(Vec3i((ref.bounds.min() - origin) * invBinSize), Vec3i(0, 0, 0),
                                 Vec3i(NumSpatialBins - 1, NumSpatialBins - 1, NumSpatialBins - 1));
        Vec3i lastBin = clamp3i(Vec3i((ref.bounds.max() - origin) * invBinSize), firstBin,
                                Vec3i(NumSpatialBins - 1, NumSpatialBins - 1, NumSpatialBins - 1));

        for (int dim = 0; dim < 3; dim++) {
            Reference currRef = ref;
            for (int i = firstBin._v[dim]; i < lastBin._v[dim]; i++) {
                Reference leftRef, rightRef;
                splitReference(leftRef, rightRef, currRef, dim, origin._v[dim] + binSize._v[dim] * (F32) (i + 1));
                m_bins[dim][i].bounds.grow(leftRef.bounds);
                currRef = rightRef;
            }
            m_bins[dim][lastBin._v[dim]].bounds.grow(currRef.bounds);
            m_bins[dim][firstBin._v[dim]].enter++;
            m_bins[dim][lastBin._v[dim]].exit++;
        }
    }

    // Select best split plane.
    SpatialSplit split;
    for (int dim = 0; dim < 3; dim++) {
        // Sweep right to left and determine bounds.
        AABB rightBounds;
        for (int i = NumSpatialBins - 1; i > 0; i--) {
            rightBounds.grow(m_bins[dim][i].bounds);
            m_rightBounds[i - 1] = rightBounds;
        }
        // Sweep left to right and select lowest SAH.
        AABB leftBounds;
        int leftNum = 0;
        int rightNum = spec.numRef;

        for (int i = 1; i < NumSpatialBins; i++) {
            leftBounds.grow(m_bins[dim][i - 1].bounds);
            leftNum += m_bins[dim][i - 1].enter;
            rightNum -= m_bins[dim][i - 1].exit;

            F32 sah = nodeSAH + leftBounds.area() * sahHelper.getTriangleCost(leftNum) +
                      m_rightBounds[i - 1].area() * sahHelper.getTriangleCost(rightNum);
            if (sah < split.sah) {
                split.sah = sah;
                split.dim = dim;
                split.pos = origin._v[dim] + binSize._v[dim] * (F32) i;
            }
        }
    }
    return split;
}

void BVH::performSpatialSplit(NodeSpec &left, NodeSpec &right, const NodeSpec &spec, const SpatialSplit &split) {
    // Categorize references and compute bounds.
    //
    // Left-hand side:      [leftStart, leftEnd[
    // Uncategorized/split: [leftEnd, rightStart[
    // Right-hand side:     [rightStart, refs.getSize()[
    Array<Reference> &refs = m_refStack;
    int leftStart = refs.getSize() - spec.numRef;
    int leftEnd = leftStart;
    int rightStart = refs.getSize();
    left.bounds = right.bounds = AABB();

    for (int i = leftEnd; i < rightStart; i++) {
        if (refs[i].bounds.max()._v[split.dim] <= split.pos) {
            // Entirely on the left-hand side?
            left.bounds.grow(refs[i].bounds);
            swap(refs[i], refs[leftEnd++]);
        } else if (refs[i].bounds.min()._v[split.dim] >= split.pos) {
            // Entirely on the right-hand side?
            right.bounds.grow(refs[i].bounds);
            swap(refs[i--], refs[--rightStart]);
        }
    }

    // Duplicate or unsplit references intersecting both sides.
    while (leftEnd < rightStart) {
        // Split reference.
        Reference lref, rref;
        splitReference(lref, rref, refs[leftEnd], split.dim, split.pos);

        // Compute SAH for duplicate/unsplit candidates.
        AABB lub = left.bounds;  // Unsplit to left:     new left-hand bounds.
        AABB rub = right.bounds; // Unsplit to right:    new right-hand bounds.
        AABB ldb = left.bounds;  // Duplicate:           new left-hand bounds.
        AABB rdb = right.bounds; // Duplicate:           new right-hand bounds.
        lub.grow(refs[leftEnd].bounds);
        rub.grow(refs[leftEnd].bounds);
        ldb.grow(lref.bounds);
        rdb.grow(rref.bounds);


        F32 lac = sahHelper.getTriangleCost(leftEnd - leftStart);
        F32 rac = sahHelper.getTriangleCost(refs.getSize() - rightStart);
        F32 lbc = sahHelper.getTriangleCost(leftEnd - leftStart + 1);
        F32 rbc = sahHelper.getTriangleCost(refs.getSize() - rightStart + 1);

        F32 unsplitLeftSAH = lub.area() * lbc + right.bounds.area() * rac;
        F32 unsplitRightSAH = left.bounds.area() * lac + rub.area() * rbc;
        F32 duplicateSAH = ldb.area() * lbc + rdb.area() * rbc;
        F32 minSAH = min1f3(unsplitLeftSAH, unsplitRightSAH, duplicateSAH);

        if (minSAH == unsplitLeftSAH) {
            // Unsplit to left?
            left.bounds = lub;
            leftEnd++;
        } else if (minSAH == unsplitRightSAH) {
            // Unsplit to right?
            right.bounds = rub;
            swap(refs[leftEnd], refs[--rightStart]);
        } else {
            // Duplicate?
            left.bounds = ldb;
            right.bounds = rdb;
            refs[leftEnd++] = lref;
            refs.add(rref);
        }
    }
    left.numRef = leftEnd - leftStart;
    right.numRef = refs.getSize() - rightStart;
}

void BVH::splitReference(Reference &left, Reference &right, const Reference &ref, int dim, F32 pos) {
    // Initialize references.
    left.triIdx = right.triIdx = ref.triIdx;
    left.bounds = right.bounds = AABB();

    // Loop over vertices/edges.
    const Vec3i &inds = scene->getTriangle(ref.triIdx);
    const Vec3f *verts = scene->getVertexPtr();
    const Vec3f *v1 = &verts[inds.z];

    for (int ind : inds._v) {
        const Vec3f *v0 = v1;
        v1 = &verts[ind];
        F32 v0p = (*v0)._v[dim];
        F32 v1p = (*v1)._v[dim];

        // Insert vertex to the boxes it belongs to.

        if (v0p <= pos)
            left.bounds.grow(*v0);
        if (v0p >= pos)
            right.bounds.grow(*v0);

        // Edge intersects the plane => insert intersection to both boxes.

        if ((v0p < pos && v1p > pos) || (v0p > pos && v1p < pos)) {
            Vec3f t = lerp(*v0, *v1, clamp1f((pos - v0p) / (v1p - v0p), 0.0f, 1.0f));
            left.bounds.grow(t);
            right.bounds.grow(t);
        }
    }

    // Intersect with original bounds.
    left.bounds.max()._v[dim] = pos;
    right.bounds.min()._v[dim] = pos;
    left.bounds.intersect(ref.bounds);
    right.bounds.intersect(ref.bounds);
}

int BVH::sortCompare(void *data, int idxA, int idxB) {
    const auto *ptr = (const BVH *) data;
    int dim = ptr->m_sortDim;
    const Reference &ra = ptr->m_refStack[idxA];  // ra is a reference (struct containing a triIdx and bounds)
    const Reference &rb = ptr->m_refStack[idxB];  //
    F32 ca = ra.bounds.min()._v[dim] + ra.bounds.max()._v[dim];
    F32 cb = rb.bounds.min()._v[dim] + rb.bounds.max()._v[dim];
    return (ca < cb) ? -1 : (ca > cb) ? 1 : (ra.triIdx < rb.triIdx) ? -1 : (ra.triIdx > rb.triIdx) ? 1 : 0;
}

void BVH::sortSwap(void *data, int idxA, int idxB) {
    auto *ptr = (BVH *) data;
    swap(ptr->m_refStack[idxA], ptr->m_refStack[idxB]);
}
