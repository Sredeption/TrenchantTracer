#include <geometry/GeometryCompact.h>

GeometryCompact::GeometryCompact(Scene *scene) {
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

GeometryCompact::~GeometryCompact() {
    for (int i = 0; i < geometriesSize; i++)
        cudaFree(cpuGeometries[i]);
    free(cpuGeometries);
    free(geometryLength);
    cudaFree(matIndices);
    cudaFree(geometries);
}
