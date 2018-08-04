#include <geometry/GeometryCompact.h>

__host__ GeometryCompact::GeometryCompact(Scene *scene) {
    geometriesSize = (U32) scene->getGeometryNum();
    cudaMalloc(&matIndices, geometriesSize * sizeof(int1));
    cudaMalloc(&geometries, geometriesSize * sizeof(GeometryUnion));

    for (int i = 0; i < geometriesSize; i++) {
        const Geometry *geometry = scene->getGeometry(i)->getGeometry();
        const Material *material = scene->getGeometry(i)->getMaterial();
        cudaMemcpy(geometries + i, geometry, geometry->size(), cudaMemcpyHostToDevice);
        cudaMemcpy(matIndices + i, &material->index, sizeof(int1), cudaMemcpyHostToDevice);
    }

    createTexture();
}

__host__ GeometryCompact::GeometryCompact(FILE *geoFile) {
    if (1 != fread(&geometriesSize, sizeof(unsigned), 1, geoFile))
        throw std::runtime_error("Error reading geometry cache file!\n");

    cudaMalloc(&geometries, geometriesSize * sizeof(GeometryUnion));
    auto cpuGeometries = (GeometryUnion *) malloc(geometriesSize * sizeof(GeometryUnion));
    if (geometriesSize != fread(cpuGeometries, sizeof(GeometryUnion), geometriesSize, geoFile))
        throw std::runtime_error("Error reading geometry cache file!\n");
    cudaMemcpy(geometries, cpuGeometries, geometriesSize * sizeof(GeometryUnion), cudaMemcpyHostToDevice);
    free(cpuGeometries);

    cudaMalloc(&matIndices, geometriesSize * sizeof(int1));
    auto cpuMatIndices = (int1 *) malloc(geometriesSize * sizeof(int1));
    if (geometriesSize != fread(cpuMatIndices, sizeof(int1), geometriesSize, geoFile))
        throw std::runtime_error("Error reading geometry cache file!\n");
    cudaMemcpy(matIndices, cpuMatIndices, geometriesSize * sizeof(int1), cudaMemcpyHostToDevice);
    free(cpuMatIndices);

    fclose(geoFile);

    createTexture();
}

__host__ GeometryCompact::~GeometryCompact() {
    cudaFree(matIndices);
    cudaFree(geometries);
}

__host__ void GeometryCompact::save(const std::string &fileName) {
    FILE *geoFile = fopen(fileName.c_str(), "wb");
    if (!geoFile)
        throw std::runtime_error("Error opening geometry cache file!");

    if (1 != fwrite(&geometriesSize, sizeof(unsigned), 1, geoFile))
        throw std::runtime_error("Error writing geometry cache file!\n");

    auto cpuGeometries = (GeometryUnion *) malloc(geometriesSize * sizeof(GeometryUnion));
    cudaMemcpy(cpuGeometries, geometries, geometriesSize * sizeof(GeometryUnion), cudaMemcpyDeviceToHost);
    if (geometriesSize != fwrite(cpuGeometries, sizeof(GeometryUnion), geometriesSize, geoFile))
        std::runtime_error("Error writing geometry cache file!\n");
    free(cpuGeometries);

    auto cpuMatIndices = (int1 *) malloc(geometriesSize * sizeof(int1));
    cudaMemcpy(cpuMatIndices, geometries, geometriesSize * sizeof(int1), cudaMemcpyDeviceToHost);
    if (geometriesSize != fwrite(cpuMatIndices, sizeof(int1), geometriesSize, geoFile))
        std::runtime_error("Error writing material cache file!\n");
    free(cpuMatIndices);

    fclose(geoFile);
}

void GeometryCompact::createTexture() {
    cudaResourceDesc resDesc{};
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = matIndices;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // r-channel bits
    resDesc.res.linear.desc.y = 0; // g-channel bits
    resDesc.res.linear.desc.z = 0; // b-channel bits
    resDesc.res.linear.desc.w = 0; // a-channel bits
    resDesc.res.linear.sizeInBytes = geometriesSize * sizeof(int1);

    cudaTextureDesc texDesc{};
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.filterMode = cudaFilterModeLinear;
    cudaCreateTextureObject(&matIndicesTexture, &resDesc, &texDesc, nullptr);
}
