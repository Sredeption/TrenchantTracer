#include <bvh/BVHCompact.h>
#include <driver_types.h>

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


__host__ BVHCompact::BVHCompact(const BVH &bvh):
        leafNodeCount(0), triCount(0) {
    createCompact(bvh, 16);
    createTexture();
}

BVHCompact::BVHCompact(FILE *bvhFile) {

    if (1 != fread(&nodesSize, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    if (1 != fread(&triCount, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    if (1 != fread(&leafNodeCount, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    if (1 != fread(&woopTriSize, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    if (1 != fread(&debugTriSize, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    if (1 != fread(&triIndicesSize, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");

    std::cout << "Number of nodes: " << nodesSize << "\n";
    std::cout << "Number of triangles: " << triCount << "\n";
    std::cout << "Number of BVH leaf nodes: " << leafNodeCount << "\n";

    auto cpuNodes = (Vec4i *) malloc(nodesSize * sizeof(Vec4i));
    auto cpuWoopTri = (Vec4i *) malloc(woopTriSize * sizeof(Vec4i));
    auto cpuDebugTri = (Vec4i *) malloc(debugTriSize * sizeof(Vec4i));
    auto cpuTriIndices = (S32 *) malloc(triIndicesSize * sizeof(S32));

    if (nodesSize != fread(cpuNodes, sizeof(Vec4i), nodesSize, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    if (woopTriSize != fread(cpuWoopTri, sizeof(Vec4i), woopTriSize, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    if (debugTriSize != fread(cpuDebugTri, sizeof(Vec4i), debugTriSize, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    if (triIndicesSize != fread(cpuTriIndices, sizeof(S32), triIndicesSize, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");

    cudaMalloc(&nodes, nodesSize * sizeof(Vec4i));
    cudaMemcpy(nodes, cpuNodes, nodesSize * sizeof(Vec4i), cudaMemcpyHostToDevice);
    cudaMalloc(&woopTri, woopTriSize * sizeof(Vec4i));
    cudaMemcpy(woopTri, cpuWoopTri, woopTriSize * sizeof(Vec4i), cudaMemcpyHostToDevice);
    cudaMalloc(&debugTri, debugTriSize * sizeof(Vec4i));
    cudaMemcpy(debugTri, cpuDebugTri, debugTriSize * sizeof(Vec4i), cudaMemcpyHostToDevice);
    cudaMalloc(&triIndices, triIndicesSize * sizeof(U32));
    cudaMemcpy(triIndices, cpuTriIndices, triIndicesSize * sizeof(U32), cudaMemcpyHostToDevice);

    free(cpuNodes);
    free(cpuWoopTri);
    free(cpuDebugTri);
    free(cpuTriIndices);

    createTexture();
}

__host__ BVHCompact::~BVHCompact() {
    cudaFree(this->nodes);
    cudaFree(this->woopTri);
    cudaFree(this->debugTri);
    cudaFree(this->triIndices);
}

__host__ void BVHCompact::createCompact(const BVH &bvh, int nodeOffsetSizeDiv) {

    // construct and initialize data arrays which will be copied to CudaBVH buffers (last part of this function).
    Array<Vec4i> nodeData(nullptr, 4);
    Array<Vec4i> triWoopData;
    Array<Vec4i> triDebugData; // array for regular (non-woop) triangles
    Array<S32> triIndexData;

    // construct a stack (array of stack entries) to help in filling the data arrays
    Array<StackEntry> stack(StackEntry(bvh.getRoot(), 0)); // initialise stack to root node

    // while stack is not empty
    while (stack.getSize()) {

        StackEntry entry = stack.removeLast(); // pop the stack
        FW_ASSERT(entry.node->getNumChildNodes() == 2);
        const AABB *childBox[2];
        int childIndex[2]; // stores indices to both children

        // Process children.

        // for each child in entry
        for (int i = 0; i < 2; i++) {
            const BVHNode *child = entry.node->getChildNode(i); // current child node
            childBox[i] = &child->bounding; // current child's AABB

            // Inner node => push to stack.

            // no leaf, thus an inner node
            if (!child->isLeaf()) {   // compute childIndex
                childIndex[i] = nodeData.getNumBytes() / nodeOffsetSizeDiv;
                // nodeOffsetSizeDiv is 1 for Fermi kernel, 16 for Kepler kernel

                // push the current child on the stack
                stack.add(StackEntry(child, nodeData.getSize()));
                nodeData.add(nullptr, 4);
                // adds 4 * Vec4i per inner node or 4 * 16 bytes/Vec4i = 64 bytes of empty data per inner node
                continue; // process remaining child node (if any)
            }

            // Leaf node => append triangles.

            const auto *leaf = reinterpret_cast<const LeafNode *>(child);

            // index of a leaf node is a negative number, hence the ~
            childIndex[i] = ~triWoopData.getSize();
            // leafs must be stored as negative (bitwise complement) in order to be recognised by pathtracer as a leaf

            // for each triangle in leaf, range of triangle index j from m_lo to m_hi
            for (int j = leaf->lo; j < leaf->hi; j++) {
                // transform the triangle's vertices to Woop triangle (simple transform to right angled
                // triangle, see paper by Sven Woop)
                Vec4f woopTri[4], debugTri[4];
                // transform the triangle's vertices to Woop triangle
                // (simple transform to right angled triangle, see paper by Sven Woop)
                woopifyTri(bvh, j, woopTri, debugTri);  // j is de triangle index in triIndex array
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

            // Leaf node terminator to indicate end of leaf, stores hexadecimal value
            // 0x80000000 (= 2147483648 in decimal)
            triWoopData.add(0x80000000);
            // leaf node terminator code indicates the last triangle of the leaf node
            triDebugData.add(0x80000000);

            // add extra zero to triangle indices array to indicate end of leaf
            triIndexData.add(0);  // terminates triIndex data for current leaf
            this->leafNodeCount++;
        }

        // Write entry for current node.
        // 4 Vec4i per node (according to compact bvh node layout)
        Vec4i *dst = nodeData.getPtr(entry.idx);

        dst[0] = Vec4i(floatToBits(childBox[0]->min().x),
                       floatToBits(childBox[0]->max().x),
                       floatToBits(childBox[0]->min().y),
                       floatToBits(childBox[0]->max().y));
        dst[1] = Vec4i(floatToBits(childBox[1]->min().x),
                       floatToBits(childBox[1]->max().x),
                       floatToBits(childBox[1]->min().y),
                       floatToBits(childBox[1]->max().y));
        dst[2] = Vec4i(floatToBits(childBox[0]->min().z),
                       floatToBits(childBox[0]->max().z),
                       floatToBits(childBox[1]->min().z),
                       floatToBits(childBox[1]->max().z));
        dst[3] = Vec4i(childIndex[0], childIndex[1], 0, 0);
    } // end of while loop, will iteratively empty the stack


    // Write data arrays to arrays of CudaBVH
    nodesSize = (U32) nodeData.getSize();
    auto cpuNodes = (Vec4i *) malloc(nodesSize * sizeof(Vec4i));
    for (int i = 0; i < nodeData.getSize(); i++) {
        cpuNodes[i].x = nodeData.get(i).x;
        cpuNodes[i].y = nodeData.get(i).y;
        cpuNodes[i].z = nodeData.get(i).z;
        cpuNodes[i].w = nodeData.get(i).w; // child indices
    }
    cudaMalloc(&nodes, nodesSize * sizeof(Vec4i));
    cudaMemcpy(nodes, cpuNodes, nodesSize * sizeof(Vec4i), cudaMemcpyHostToDevice);

    woopTriSize = (U32) triWoopData.getSize();
    auto cpuWoopTri = (Vec4i *) malloc(woopTriSize * sizeof(Vec4i));
    for (int i = 0; i < triWoopData.getSize(); i++) {
        cpuWoopTri[i].x = triWoopData.get(i).x;
        cpuWoopTri[i].y = triWoopData.get(i).y;
        cpuWoopTri[i].z = triWoopData.get(i).z;
        cpuWoopTri[i].w = triWoopData.get(i).w;
    }
    cudaMalloc(&woopTri, woopTriSize * sizeof(Vec4i));
    cudaMemcpy(woopTri, cpuWoopTri, woopTriSize * sizeof(Vec4i), cudaMemcpyHostToDevice);

    debugTriSize = (U32) triDebugData.getSize();
    auto cpuDebugTri = (Vec4i *) malloc(debugTriSize * sizeof(Vec4i));
    for (int i = 0; i < triDebugData.getSize(); i++) {
        cpuDebugTri[i].x = triDebugData.get(i).x;
        cpuDebugTri[i].y = triDebugData.get(i).y;
        cpuDebugTri[i].z = triDebugData.get(i).z;
        cpuDebugTri[i].w = triDebugData.get(i).w;
    }
    cudaMalloc(&debugTri, debugTriSize * sizeof(Vec4i));
    cudaMemcpy(debugTri, cpuDebugTri, debugTriSize * sizeof(Vec4i), cudaMemcpyHostToDevice);

    triIndicesSize = (U32) triIndexData.getSize();
    auto cpuTriIndices = (S32 *) malloc(triIndicesSize * sizeof(S32));
    for (int i = 0; i < triIndexData.getSize(); i++) {
        cpuTriIndices[i] = triIndexData.get(i);
    }
    cudaMalloc(&triIndices, triIndicesSize * sizeof(U32));
    cudaMemcpy(triIndices, cpuTriIndices, triIndicesSize * sizeof(U32), cudaMemcpyHostToDevice);

    free(cpuNodes);
    free(cpuWoopTri);
    free(cpuDebugTri);
    free(cpuTriIndices);
}

__host__ void BVHCompact::woopifyTri(const BVH &bvh, int idx, Vec4f *woopTri, Vec4f *debugTri) {
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

__host__ void BVHCompact::createTexture() {
    cudaResourceDesc resDesc{};
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = nodes;
    resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
    resDesc.res.linear.desc.x = 32; // r-channel bits
    resDesc.res.linear.desc.y = 32; // g-channel bits
    resDesc.res.linear.desc.z = 32; // b-channel bits
    resDesc.res.linear.desc.w = 32; // a-channel bits
    resDesc.res.linear.sizeInBytes = nodesSize * sizeof(float4);

    cudaTextureDesc texDesc{};
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.filterMode = cudaFilterModeLinear;

    cudaCreateTextureObject(&nodesTexture, &resDesc, &texDesc, nullptr);

    resDesc.res.linear.devPtr = woopTri;
    resDesc.res.linear.sizeInBytes = woopTriSize * sizeof(float4);

    cudaCreateTextureObject(&woopTriTexture, &resDesc, &texDesc, nullptr);

    resDesc.res.linear.devPtr = debugTri;
    resDesc.res.linear.sizeInBytes = debugTriSize * sizeof(float4);

    cudaCreateTextureObject(&debugTriTexture, &resDesc, &texDesc, nullptr);

    resDesc.res.linear.devPtr = triIndices;
    resDesc.res.linear.desc.x = 32; // r-channel bits
    resDesc.res.linear.desc.y = 0; // g-channel bits
    resDesc.res.linear.desc.z = 0; // b-channel bits
    resDesc.res.linear.desc.w = 0; // a-channel bits
    resDesc.res.linear.sizeInBytes = triIndicesSize * sizeof(int1);

    cudaCreateTextureObject(&triIndicesTexture, &resDesc, &texDesc, nullptr);
}

__host__ void BVHCompact::save(const std::string &fileName) {

    FILE *bvhFile = fopen(fileName.c_str(), "wb");
    if (!bvhFile)
        throw std::runtime_error("Error opening BVH cache file!");

    if (1 != fwrite(&nodesSize, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error writing BVH cache file!\n");
    if (1 != fwrite(&triCount, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error writing BVH cache file!\n");
    if (1 != fwrite(&leafNodeCount, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error writing BVH cache file!\n");
    if (1 != fwrite(&woopTriSize, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error writing BVH cache file!\n");
    if (1 != fwrite(&debugTriSize, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error writing BVH cache file!\n");
    if (1 != fwrite(&triIndicesSize, sizeof(unsigned), 1, bvhFile))
        std::runtime_error("Error writing BVH cache file!\n");

    std::cout << "Number of nodes: " << nodesSize << "\n";
    std::cout << "Number of triangles: " << triCount << "\n";
    std::cout << "Number of BVH leaf nodes: " << leafNodeCount << "\n";

    auto cpuNodes = (Vec4i *) malloc(nodesSize * sizeof(Vec4i));
    cudaMemcpy(cpuNodes, this->nodes, nodesSize * sizeof(Vec4i), cudaMemcpyDeviceToHost);
    if (nodesSize != fwrite(cpuNodes, sizeof(Vec4i), nodesSize, bvhFile))
        std::runtime_error("Error writing BVH cache file!\n");

    auto cpuWoopTri = (Vec4i *) malloc(this->woopTriSize * sizeof(Vec4i));
    cudaMemcpy(cpuWoopTri, this->woopTri, this->woopTriSize * sizeof(Vec4i), cudaMemcpyDeviceToHost);
    if (woopTriSize != fwrite(cpuWoopTri, sizeof(Vec4i), woopTriSize, bvhFile))
        std::runtime_error("Error writing BVH cache file!\n");

    auto cpuDebugTri = (Vec4i *) malloc(this->debugTriSize * sizeof(Vec4i));
    cudaMemcpy(cpuDebugTri, this->debugTri, this->debugTriSize * sizeof(Vec4i), cudaMemcpyDeviceToHost);
    if (debugTriSize != fwrite(cpuDebugTri, sizeof(Vec4i), debugTriSize, bvhFile))
        std::runtime_error("Error writing BVH cache file!\n");

    auto cpuTriIndices = (S32 *) malloc(this->triIndicesSize * sizeof(S32));
    cudaMemcpy(cpuTriIndices, this->triIndices, this->triIndicesSize * sizeof(U32), cudaMemcpyDeviceToHost);
    if (triIndicesSize != fwrite(cpuTriIndices, sizeof(S32), triIndicesSize, bvhFile))
        std::runtime_error("Error writing BVH cache file!\n");

    fclose(bvhFile);

    free(cpuNodes);
    free(cpuWoopTri);
    free(cpuDebugTri);
    free(cpuTriIndices);
}
