#include <material/MaterialCompact.h>

MaterialCompact::MaterialCompact(Scene *scene) {
    materialsSize = (U32) scene->getMaterialNum();
    cudaMalloc(&materials, materialsSize * sizeof(MaterialUnion));

    for (int i = 0; i < materialsSize; i++) {
        const Material *material = scene->getMaterial(i);
        cudaMemcpy(materials + i, material, material->size(), cudaMemcpyHostToDevice);
    }
}

MaterialCompact::MaterialCompact(FILE *matFile) {
    if (1 != fread(&materialsSize, sizeof(unsigned), 1, matFile))
        throw std::runtime_error("Error reading material cache file!\n");

    auto cpuMaterials = (MaterialUnion *) malloc(materialsSize * sizeof(MaterialUnion));
    if (materialsSize != fread(cpuMaterials, sizeof(MaterialUnion), materialsSize, matFile))
        throw std::runtime_error("Error reading material cache file!\n");

    cudaMalloc(&materials, materialsSize * sizeof(MaterialUnion));
    cudaMemcpy(materials, cpuMaterials, materialsSize * sizeof(MaterialUnion), cudaMemcpyHostToDevice);

    free(cpuMaterials);
    fclose(matFile);
}

MaterialCompact::~MaterialCompact() {
    cudaFree(materials);
}

__host__ void MaterialCompact::save(const std::string &fileName) {
    FILE *matFile = fopen(fileName.c_str(), "wb");
    if (!matFile)
        throw std::runtime_error("Error opening material cache file!");

    if (1 != fwrite(&materialsSize, sizeof(unsigned), 1, matFile))
        throw std::runtime_error("Error writing material cache file!\n");

    auto cpuMaterials = (MaterialUnion *) malloc(materialsSize * sizeof(MaterialUnion));
    cudaMemcpy(cpuMaterials, materials, materialsSize * sizeof(MaterialUnion), cudaMemcpyDeviceToHost);
    if (materialsSize != fwrite(cpuMaterials, sizeof(MaterialUnion), materialsSize, matFile))
        std::runtime_error("Error writing material cache file!\n");
    free(cpuMaterials);
}
