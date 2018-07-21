#include <bvh/BVHHolder.h>

//Nodes / BVHLayout_Compact  (12 floats + 4 ints = 64 bytes)
// inner node contains two child nodes c0 and c1, each having x,y,z coordinates for AABBhi and AABBlo, 2*2*3 = 12 floats
//
//		nodes[innerOfs + 0 ] = Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)  // 4 floats = 16 bytes
//      nodes[innerOfs + 16] = Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)  // increment nodes array index with 16
//      nodes[innerOfs + 32] = Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
//      nodes[innerOfs + 48] = Vec4i(c0.innerOfs or ~c0.triOfs, c1.innerOfs or ~c1.triOfs, 0, 0)  // either inner or leaf, two dummy zeros at the end
//		CudaBVH Compact: Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
//		CudaBVH Compact: Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
//		CudaBVH Compact: Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
//		CudaBVH Compact: Vec4f(c0.innerOfs or ~c0.triOfs, c1.innerOfs or ~c1.triOfs, 0, 0)
//		BVH node bounds: c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y, c0.lo.z, c0.hi.z


BVHHolder::BVHHolder(const BVH &bvh)
        : bvh(bvh) {
    createCompact(16);
}

BVHHolder::~BVHHolder() {
    cudaFree(this->nodes);
    cudaFree(this->woopTri);
    cudaFree(this->debugTri);
    cudaFree(this->triIndices);
}


void BVHHolder::createCompact(int nodeOffsetSizeDiv) {

    // construct and initialize data arrays which will be copied to CudaBVH buffers (last part of this function).
    Array<Vec4i> nodeData(nullptr, 4);
    Array<Vec4i> triWoopData;
    Array<Vec4i> triDebugData; // array for regular (non-woop) triangles
    Array<S32> triIndexData;

    // construct a stack (array of stack entries) to help in filling the data arrays
    Array<StackEntry> stack(StackEntry(bvh.getRoot(), 0)); // initialise stack to root node

    // while stack is not empty
    while (stack.getSize()) {

        StackEntry e = stack.removeLast(); // pop the stack
        FW_ASSERT(e.node->getNumChildNodes() == 2);
        const AABB *cbox[2];
        int cidx[2]; // stores indices to both children

        // Process children.

        // for each child in entry e
        for (int i = 0; i < 2; i++) {
            const BVHNode *child = e.node->getChildNode(i); // current child node
            cbox[i] = &child->bounding; // current child's AABB
            ////////////////////////////
            /// INNER NODE
            //////////////////////////////

            // Inner node => push to stack.

            // no leaf, thus an inner node
            if (!child->isLeaf()) {   // compute childIndex
                cidx[i] = nodeData.getNumBytes() /
                          nodeOffsetSizeDiv; // nodeOffsetSizeDiv is 1 for Fermi kernel, 16 for Kepler kernel

                // push the current child on the stack
                stack.add(StackEntry(child, nodeData.getSize()));
                nodeData.add(nullptr, 4);
                // adds 4 * Vec4i per inner node or 4 * 16 bytes/Vec4i = 64 bytes of empty data per inner node
                continue; // process remaining child node (if any)
            }

            //////////////////////
            /// LEAF NODE
            /////////////////////

            // Leaf => append triangles.

            const auto *leaf = reinterpret_cast<const LeafNode *>(child);

            // index of a leaf node is a negative number, hence the ~
            cidx[i] = ~triWoopData.getSize();
            // leafs must be stored as negative (bitwise complement) in order to be recognised by pathtracer as a leaf


            // for each triangle in leaf, range of triangle index j from m_lo to m_hi
            for (int j = leaf->lo; j < leaf->hi; j++) {
                // transform the triangle's vertices to Woop triangle (simple transform to right angled
                // triangle, see paper by Sven Woop)
                Vec4f woopTri[4], debugTri[4];
                // transform the triangle's vertices to Woop triangle (simple transform to right angled triangle, see paper by Sven Woop)
                woopifyTri(j, woopTri, debugTri);  // j is de triangle index in triIndex array
                this->triCount++;

                if (woopTri[0].x == 0.0f) woopTri[0].x = 0.0f;  // avoid degenerate coordinates

                // add the transformed woop triangle to triWoopData
                triWoopData.add((Vec4i *) woopTri, 3);
                triDebugData.add((Vec4i *) debugTri, 3);

                // add tri index for current triangle to triIndexData
                triIndexData.add(bvh.getTriIndices()[j]);
                triIndexData.add(0);
                // zero padding because CUDA kernel uses same index for vertex array (3 vertices per triangle)
                triIndexData.add(0);
                // and array of triangle indices
            }


            // Leaf node terminator to indicate end of leaf, stores hexadecimal value 0x80000000 (= 2147483648 in decimal)
            triWoopData.add(0x80000000);
            // leaf node terminator code indicates the last triangle of the leaf node
            triDebugData.add(0x80000000);

            // add extra zero to triangle indices array to indicate end of leaf
            triIndexData.add(0);  // terminates triIndex data for current leaf
            this->leafNodeCount++;
        }

        // Write entry for current node.
        // 4 Vec4i per node (according to compact bvh node layout)
        Vec4i *dst = nodeData.getPtr(e.idx);

        dst[0] = Vec4i(floatToBits(cbox[0]->min().x),
                       floatToBits(cbox[0]->max().x),
                       floatToBits(cbox[0]->min().y),
                       floatToBits(cbox[0]->max().y));
        dst[1] = Vec4i(floatToBits(cbox[1]->min().x),
                       floatToBits(cbox[1]->max().x),
                       floatToBits(cbox[1]->min().y),
                       floatToBits(cbox[1]->max().y));
        dst[2] = Vec4i(floatToBits(cbox[0]->min().z),
                       floatToBits(cbox[0]->max().z),
                       floatToBits(cbox[1]->min().z),
                       floatToBits(cbox[1]->max().z));
        dst[3] = Vec4i(cidx[0], cidx[1], 0, 0);
    } // end of while loop, will iteratively empty the stack


    // Write data arrays to arrays of CudaBVH
    auto cpuNodes = (Vec4i *) malloc((size_t) nodeData.getNumBytes());
    this->nodesSize = (U32) nodeData.getSize();
    for (int i = 0; i < nodeData.getSize(); i++) {
        cpuNodes[i].x = nodeData.get(i).x;
        cpuNodes[i].y = nodeData.get(i).y;
        cpuNodes[i].z = nodeData.get(i).z;
        cpuNodes[i].w = nodeData.get(i).w; // child indices
    }
    cudaMalloc(&this->nodes, this->nodesSize * sizeof(Vec4i));
    cudaMemcpy(this->nodes, cpuNodes, this->nodesSize * sizeof(Vec4i), cudaMemcpyHostToDevice);

    auto cpuWoopTri = (Vec4i *) malloc(triWoopData.getSize() * sizeof(Vec4i));
    this->woopTriSize = (U32) triWoopData.getSize();

    for (int i = 0; i < triWoopData.getSize(); i++) {
        cpuWoopTri[i].x = triWoopData.get(i).x;
        cpuWoopTri[i].y = triWoopData.get(i).y;
        cpuWoopTri[i].z = triWoopData.get(i).z;
        cpuWoopTri[i].w = triWoopData.get(i).w;
    }
    cudaMalloc(&this->woopTri, this->woopTriSize * sizeof(Vec4i));
    cudaMemcpy(this->woopTri, cpuWoopTri, this->woopTriSize * sizeof(Vec4i), cudaMemcpyHostToDevice);

    auto cpuDebugTri = (Vec4i *) malloc(triDebugData.getSize() * sizeof(Vec4i));
    this->debugTriSize = (U32) triDebugData.getSize();
    for (int i = 0; i < triDebugData.getSize(); i++) {
        cpuDebugTri[i].x = triDebugData.get(i).x;
        cpuDebugTri[i].y = triDebugData.get(i).y;
        cpuDebugTri[i].z = triDebugData.get(i).z;
        cpuDebugTri[i].w = triDebugData.get(i).w;
    }
    cudaMalloc(&this->debugTri, this->debugTriSize * sizeof(Vec4i));
    cudaMemcpy(this->debugTri, cpuWoopTri, this->debugTriSize * sizeof(Vec4i), cudaMemcpyHostToDevice);

    auto cpuTriIndices = (S32 *) malloc(triIndexData.getSize() * sizeof(S32));
    this->triIndicesSize = (U32) triIndexData.getSize();
    for (int i = 0; i < triIndexData.getSize(); i++) {
        cpuTriIndices[i] = triIndexData.get(i);
    }
    cudaMalloc(&this->triIndices, this->triIndicesSize * sizeof(U32));
    cudaMemcpy(this->triIndices, cpuTriIndices, this->triIndicesSize * sizeof(U32), cudaMemcpyHostToDevice);

    free(cpuNodes);
    free(cpuWoopTri);
    free(cpuDebugTri);
    free(cpuTriIndices);
}

void BVHHolder::woopifyTri(int idx, Vec4f *woopTri, Vec4f *debugTri) {
    // fetch the 3 vertex indices of this triangle
    const Vec3i &vtxInds = bvh.getScene()->getTriangle(bvh.getTriIndices()[idx]);
    const Vec3f &v0 = bvh.getScene()->getVertex(vtxInds.x);
    const Vec3f &v1 = bvh.getScene()->getVertex(vtxInds.y);
    const Vec3f &v2 = bvh.getScene()->getVertex(vtxInds.z);

    // regular triangles (for debugging only)
    debugTri[0] = Vec4f(v0.x, v0.y, v0.z, 0.0f);
    debugTri[1] = Vec4f(v1.x, v1.y, v1.z, 0.0f);
    debugTri[2] = Vec4f(v2.x, v2.y, v2.z, 0.0f);

    Mat4f mtx;
    // compute edges and transform them with a matrix
    mtx.setCol(0, Vec4f(v0 - v2, 0.0f)); // sets matrix column 0 equal to a Vec4f(Vec3f, 0.0f )
    mtx.setCol(1, Vec4f(v1 - v2, 0.0f));
    mtx.setCol(2, Vec4f(cross(v0 - v2, v1 - v2), 0.0f));
    mtx.setCol(3, Vec4f(v2, 1.0f));
    mtx = invert(mtx);

    // m_woop[3] stores 3 transformed triangle edges
    woopTri[0] = Vec4f(mtx(2, 0), mtx(2, 1), mtx(2, 2), -mtx(2, 3));
    // elements of 3rd row of inverted matrix
    woopTri[1] = mtx.getRow(0);
    woopTri[2] = mtx.getRow(1);
}
