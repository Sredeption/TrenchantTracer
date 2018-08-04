#include <geometry/GeometryCompact.h>

__host__ GeometryCompact::GeometryCompact(Scene *scene) {
    geometriesSize = (U32) scene->getGeometryNum();
    geometryLength = (U32 *) malloc(geometriesSize * sizeof(U32));
    cpuGeometries = (Geometry **) malloc(geometriesSize * sizeof(Geometry *));
    cudaMalloc(&matIndices, geometriesSize * sizeof(int1));

    for (int i = 0; i < geometriesSize; i++) {
        const Geometry *geometry = scene->getGeometry(i)->getGeometry();
        const Material *material = scene->getGeometry(i)->getMaterial();
        geometryLength[i] = geometry->size();
        cudaMalloc(cpuGeometries + i, geometryLength[i]);
        cudaMemcpy(cpuGeometries[i], geometry, geometryLength[i], cudaMemcpyHostToDevice);
        cudaMemcpy(matIndices + i, &material->index, sizeof(int1), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&geometries, geometriesSize * sizeof(Geometry *));
    cudaMemcpy(geometries, cpuGeometries, geometriesSize * sizeof(Geometry *), cudaMemcpyHostToDevice);
}

__host__ GeometryCompact::GeometryCompact(FILE *geoFile) {
    if (1 != fread(&geometriesSize, sizeof(unsigned), 1, geoFile))
        throw std::runtime_error("Error reading geometry cache file!\n");
    geometryLength = (U32 *) malloc(geometriesSize * sizeof(U32));
    cpuGeometries = (Geometry **) malloc(geometriesSize * sizeof(Geometry *));
    cudaMalloc(&matIndices, geometriesSize * sizeof(int1));

    for (int i = 0; i < geometriesSize; i++) {
        if (1 != fread(geometryLength + i, sizeof(unsigned), 1, geoFile))
            throw std::runtime_error("Error reading geometry cache file!\n");

        auto geometry = (Geometry *) malloc(geometryLength[i]);
        cudaMalloc(cpuGeometries + i, geometryLength[i]);

        if (1 != fread(geometry, geometryLength[i], 1, geoFile))
            throw std::runtime_error("Error reading geometry cache file!\n");
        cudaMemcpy(cpuGeometries[i], geometry, geometryLength[i], cudaMemcpyHostToDevice);

        int1 matIndex;
        if (1 != fread(&matIndex, sizeof(int1), 1, geoFile))
            std::runtime_error("Error reading geometry cache file!\n");
        cudaMemcpy(matIndices + i, &matIndex, sizeof(int1), cudaMemcpyHostToDevice);

        free(geometry);
    }
    cudaMalloc(&geometries, geometriesSize * sizeof(Geometry *));
    cudaMemcpy(geometries, cpuGeometries, geometriesSize * sizeof(Geometry *), cudaMemcpyHostToDevice);
    fclose(geoFile);
}

__host__ GeometryCompact::~GeometryCompact() {
    for (int i = 0; i < geometriesSize; i++)
        cudaFree(cpuGeometries[i]);
    free(cpuGeometries);
    free(geometryLength);
    cudaFree(matIndices);
    cudaFree(geometries);
}

__host__ void GeometryCompact::save(const std::string &fileName) {
    FILE *geoFile = fopen(fileName.c_str(), "wb");
    if (!geoFile)
        throw std::runtime_error("Error opening geometry cache file!");

    if (1 != fwrite(&geometriesSize, sizeof(unsigned), 1, geoFile))
        throw std::runtime_error("Error writing geometry cache file!\n");

    for (int i = 0; i < geometriesSize; i++) {
        if (1 != fwrite(geometryLength + i, sizeof(unsigned), 1, geoFile))
            throw std::runtime_error("Error writing geometry cache file!\n");
        auto geometry = (Geometry *) malloc(geometryLength[i]);

        cudaMemcpy(geometry, cpuGeometries[i], geometryLength[i], cudaMemcpyDeviceToHost);
        if (1 != fwrite(geometry, geometryLength[i], 1, geoFile))
            std::runtime_error("Error writing geometry cache file!\n");

        int1 matIndex;
        cudaMemcpy(&matIndex, matIndices + i, sizeof(int1), cudaMemcpyDeviceToHost);
        if (1 != fwrite(&matIndex, sizeof(int1), 1, geoFile))
            std::runtime_error("Error writing geometry cache file!\n");
        free(geometry);
    }
    fclose(geoFile);
}
