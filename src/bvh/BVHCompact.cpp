#include <bvh/BVHCompact.h>

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


__host__ BVHCompact::BVHCompact(const BVH &bvh) :
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
    if (1 != fread(&verticesSize, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    if (1 != fread(&normalsSize, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    if (1 != fread(&triIndicesSize, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    if (1 != fread(&matIndicesSize, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");

    std::cout << "Number of nodes: " << nodesSize << "\n";
    std::cout << "Number of triangles: " << triCount << "\n";
    std::cout << "Number of BVH leaf nodes: " << leafNodeCount << "\n";

    auto cpuNodes = (Vec4i *) malloc(nodesSize * sizeof(Vec4i));
    auto cpuVertices = (Vec4i *) malloc(verticesSize * sizeof(Vec4i));
    auto cpuNormals = (Vec4i *) malloc(normalsSize * sizeof(Vec4i));
    auto cpuTriIndices = (S32 *) malloc(triIndicesSize * sizeof(S32));
    auto cpuMatIndices = (U32 *) malloc(matIndicesSize * sizeof(S32));

    if (nodesSize != fread(cpuNodes, sizeof(Vec4i), nodesSize, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    if (verticesSize != fread(cpuVertices, sizeof(Vec4i), verticesSize, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    if (normalsSize != fread(cpuNormals, sizeof(Vec4i), normalsSize, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    if (triIndicesSize != fread(cpuTriIndices, sizeof(S32), triIndicesSize, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    if (matIndicesSize != fread(cpuMatIndices, sizeof(S32), matIndicesSize, bvhFile))
        throw std::runtime_error("Error reading BVH cache file!\n");

    cudaMalloc(&nodes, nodesSize * sizeof(Vec4i));
    cudaMemcpy(nodes, cpuNodes, nodesSize * sizeof(Vec4i), cudaMemcpyHostToDevice);
    cudaMalloc(&vertices, verticesSize * sizeof(Vec4i));
    cudaMemcpy(vertices, cpuVertices, verticesSize * sizeof(Vec4i), cudaMemcpyHostToDevice);
    cudaMalloc(&normals, normalsSize * sizeof(Vec4i));
    cudaMemcpy(normals, cpuNormals, normalsSize * sizeof(Vec4i), cudaMemcpyHostToDevice);
    cudaMalloc(&triIndices, triIndicesSize * sizeof(U32));
    cudaMemcpy(triIndices, cpuTriIndices, triIndicesSize * sizeof(U32), cudaMemcpyHostToDevice);
    cudaMalloc(&matIndices, matIndicesSize * sizeof(S32));
    cudaMemcpy(matIndices, cpuMatIndices, matIndicesSize * sizeof(S32), cudaMemcpyHostToDevice);

    free(cpuNodes);
    free(cpuVertices);
    free(cpuNormals);
    free(cpuTriIndices);

    createTexture();
    fclose(bvhFile);
}

__host__ BVHCompact::~BVHCompact() {
    cudaFree(this->nodes);
    cudaFree(this->vertices);
    cudaFree(this->normals);
    cudaFree(this->triIndices);
}

__host__ void BVHCompact::createCompact(const BVH &bvh, int nodeOffsetSizeDiv) {

    // construct and initialize data arrays which will be copied to CudaBVH buffers (last part of this function).
    Array<Vec4i> nodeData(nullptr, 4);
    Array<Vec4i> vertexData;
    Array<Vec4i> normalData;
    Array<S32> triIndexData;
    Array<U32> matIndexData;

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

            auto *leaf = (LeafNode *) child;

            // index of a leaf node is a negative number, hence the ~
            childIndex[i] = ~vertexData.getSize();
            // leafs must be stored as negative (bitwise complement) in order to be recognised by path tracer as a leaf

            // for each triangle in leaf, range of triangle index j from m_lo to m_hi
            for (int j = leaf->lo; j < leaf->hi; j++) {
                Vec4f vertex[4], normal[4];
                getTriangle(bvh, j, vertex, normal);  // j is de triangle index in triIndex array
                this->triCount++;

                if (vertex[0].x == 0.0f) vertex[0].x = 0.0f;  // avoid degenerate coordinates

                // add the vertex and normal
                vertexData.add((Vec4i *) vertex, 3);
                normalData.add((Vec4i *) normal, 3);

                // add tri index for current triangle to triIndexData
                triIndexData.add(bvh.getTriIndices()[j]);
                // zero padding because CUDA kernel uses same index for vertex array (3 vertices per triangle)
                triIndexData.add(0);
                triIndexData.add(0);
                // and array of triangle indices

                matIndexData.add(bvh.getScene()->getMatIndex(bvh.getTriIndices()[j]));
                matIndexData.add(0);
                matIndexData.add(0);
            }

            // Leaf node terminator to indicate end of leaf, stores hexadecimal value
            // 0x80000000 (= 2147483648 in decimal)
            vertexData.add(0x80000000);
            // leaf node terminator code indicates the last triangle of the leaf node
            normalData.add(0x80000000);

            // add extra zero to triangle indices array to indicate end of leaf
            triIndexData.add(0);  // terminates triIndex data for current leaf

            // add extra zero to material indices array to indicate end of leaf
            matIndexData.add(0);  // terminates matIndex data for current leaf
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

    verticesSize = (U32) vertexData.getSize();
    auto cpuVertices = (Vec4i *) malloc(verticesSize * sizeof(Vec4i));
    for (int i = 0; i < vertexData.getSize(); i++) {
        cpuVertices[i].x = vertexData.get(i).x;
        cpuVertices[i].y = vertexData.get(i).y;
        cpuVertices[i].z = vertexData.get(i).z;
        cpuVertices[i].w = vertexData.get(i).w;
    }
    cudaMalloc(&vertices, verticesSize * sizeof(Vec4i));
    cudaMemcpy(vertices, cpuVertices, verticesSize * sizeof(Vec4i), cudaMemcpyHostToDevice);

    normalsSize = (U32) normalData.getSize();
    auto cpuNormals = (Vec4i *) malloc(normalsSize * sizeof(Vec4i));
    for (int i = 0; i < normalData.getSize(); i++) {
        cpuNormals[i].x = normalData.get(i).x;
        cpuNormals[i].y = normalData.get(i).y;
        cpuNormals[i].z = normalData.get(i).z;
        cpuNormals[i].w = normalData.get(i).w;
    }
    cudaMalloc(&normals, normalsSize * sizeof(Vec4i));
    cudaMemcpy(normals, cpuNormals, normalsSize * sizeof(Vec4i), cudaMemcpyHostToDevice);

    triIndicesSize = (U32) triIndexData.getSize();
    auto cpuTriIndices = (S32 *) malloc(triIndicesSize * sizeof(S32));
    for (int i = 0; i < triIndexData.getSize(); i++) {
        cpuTriIndices[i] = triIndexData.get(i);
    }
    cudaMalloc(&triIndices, triIndicesSize * sizeof(U32));
    cudaMemcpy(triIndices, cpuTriIndices, triIndicesSize * sizeof(U32), cudaMemcpyHostToDevice);

    matIndicesSize = (U32) matIndexData.getSize();
    auto cpuMatIndices = (U32 *) malloc(matIndicesSize * sizeof(U32));
    for (int i = 0; i < matIndicesSize; i++) {
        cpuMatIndices[i] = matIndexData.get(i);
    }
    cudaMalloc(&matIndices, matIndicesSize * sizeof(U32));
    cudaMemcpy(matIndices, cpuMatIndices, matIndicesSize * sizeof(U32), cudaMemcpyHostToDevice);

    free(cpuNodes);
    free(cpuVertices);
    free(cpuNormals);
    free(cpuTriIndices);
    free(cpuMatIndices);
}

__host__ void BVHCompact::getTriangle(const BVH &bvh, int triIdx, Vec4f *vertex, Vec4f *normal) {
    // fetch the 3 vertex indices of this triangle
    int index = bvh.getTriIndices()[triIdx];
    const Vec3i &vertexIndex = bvh.getScene()->getTriangle(index);
    const Vec3f &v0 = bvh.getScene()->getVertex(vertexIndex.x);
    const Vec3f &v1 = bvh.getScene()->getVertex(vertexIndex.y);
    const Vec3f &v2 = bvh.getScene()->getVertex(vertexIndex.z);

    const Vec3i &normalIndex = bvh.getScene()->getNormalIndex(index);
    const Vec3f &n0 = bvh.getScene()->getNormal(normalIndex.x);
    const Vec3f &n1 = bvh.getScene()->getNormal(normalIndex.y);
    const Vec3f &n2 = bvh.getScene()->getNormal(normalIndex.z);

    // normals
    normal[0] = Vec4f(n0, 0.0f);
    normal[1] = Vec4f(n1, 0.0f);
    normal[2] = Vec4f(n2, 0.0f);

    // vertices
    vertex[0] = Vec4f(v0, 0.0f);
    vertex[1] = Vec4f(v1, 0.0f);
    vertex[2] = Vec4f(v2, 0.0f);
}

__host__ void BVHCompact::createTexture() {
    cudaResourceDesc resDesc{};
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = nodes;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
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

    resDesc.res.linear.devPtr = vertices;
    resDesc.res.linear.sizeInBytes = verticesSize * sizeof(float4);

    cudaCreateTextureObject(&verticesTexture, &resDesc, &texDesc, nullptr);

    resDesc.res.linear.devPtr = normals;
    resDesc.res.linear.sizeInBytes = normalsSize * sizeof(float4);

    cudaCreateTextureObject(&normalsTexture, &resDesc, &texDesc, nullptr);

    resDesc.res.linear.devPtr = triIndices;
    resDesc.res.linear.desc.x = 32; // r-channel bits
    resDesc.res.linear.desc.y = 0; // g-channel bits
    resDesc.res.linear.desc.z = 0; // b-channel bits
    resDesc.res.linear.desc.w = 0; // a-channel bits
    resDesc.res.linear.sizeInBytes = triIndicesSize * sizeof(int1);

    cudaCreateTextureObject(&triIndicesTexture, &resDesc, &texDesc, nullptr);

    resDesc.res.linear.devPtr = matIndices;
    resDesc.res.linear.sizeInBytes = matIndicesSize * sizeof(int1);
    cudaCreateTextureObject(&matIndicesTexture, &resDesc, &texDesc, nullptr);
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
    if (1 != fwrite(&verticesSize, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error writing BVH cache file!\n");
    if (1 != fwrite(&normalsSize, sizeof(unsigned), 1, bvhFile))
        throw std::runtime_error("Error writing BVH cache file!\n");
    if (1 != fwrite(&triIndicesSize, sizeof(unsigned), 1, bvhFile))
        std::runtime_error("Error writing BVH cache file!\n");
    if (1 != fwrite(&matIndicesSize, sizeof(unsigned), 1, bvhFile))
        std::runtime_error("Error writing BVH cache file!\n");

    std::cout << "Number of nodes: " << nodesSize << "\n";
    std::cout << "Number of triangles: " << triCount << "\n";
    std::cout << "Number of BVH leaf nodes: " << leafNodeCount << "\n";

    auto cpuNodes = (Vec4i *) malloc(nodesSize * sizeof(Vec4i));
    cudaMemcpy(cpuNodes, this->nodes, nodesSize * sizeof(Vec4i), cudaMemcpyDeviceToHost);
    if (nodesSize != fwrite(cpuNodes, sizeof(Vec4i), nodesSize, bvhFile))
        std::runtime_error("Error writing BVH cache file!\n");

    auto cpuVertices = (Vec4i *) malloc(this->verticesSize * sizeof(Vec4i));
    cudaMemcpy(cpuVertices, this->vertices, this->verticesSize * sizeof(Vec4i), cudaMemcpyDeviceToHost);
    if (verticesSize != fwrite(cpuVertices, sizeof(Vec4i), verticesSize, bvhFile))
        std::runtime_error("Error writing BVH cache file!\n");

    auto cpuNormals = (Vec4i *) malloc(this->normalsSize * sizeof(Vec4i));
    cudaMemcpy(cpuNormals, this->normals, this->normalsSize * sizeof(Vec4i), cudaMemcpyDeviceToHost);
    if (normalsSize != fwrite(cpuNormals, sizeof(Vec4i), normalsSize, bvhFile))
        std::runtime_error("Error writing BVH cache file!\n");

    auto cpuTriIndices = (S32 *) malloc(this->triIndicesSize * sizeof(S32));
    cudaMemcpy(cpuTriIndices, this->triIndices, this->triIndicesSize * sizeof(U32), cudaMemcpyDeviceToHost);
    if (triIndicesSize != fwrite(cpuTriIndices, sizeof(S32), triIndicesSize, bvhFile))
        std::runtime_error("Error writing BVH cache file!\n");

    auto cpuMatIndices = (U32 *) malloc(this->matIndicesSize * sizeof(U32));
    cudaMemcpy(cpuMatIndices, this->matIndices, this->matIndicesSize * sizeof(U32), cudaMemcpyDeviceToHost);
    if (matIndicesSize != fwrite(cpuMatIndices, sizeof(U32), matIndicesSize, bvhFile))
        std::runtime_error("Error writing BVH cache file!\n");
    fclose(bvhFile);


    free(cpuNodes);
    free(cpuVertices);
    free(cpuNormals);
    free(cpuTriIndices);
}
