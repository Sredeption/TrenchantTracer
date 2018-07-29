#include <material/MaterialCompact.h>

MaterialCompact::MaterialCompact(Scene *scene) {
    materialsSize = (U32) scene->getMaterialNum();
    materialLength = (U32 *) malloc(materialsSize * sizeof(U32));
    cpuMaterials = (Material **) malloc(materialsSize * sizeof(Material *));

    for (int i = 0; i < materialsSize; i++) {
        const Material *material = scene->getMaterial(i);
        materialLength[i] = material->size();
        cudaMalloc(cpuMaterials + i, materialLength[i]);
        cudaMemcpy(cpuMaterials[i], material, materialLength[i], cudaMemcpyHostToDevice);
    }

    cudaMalloc(&materials, materialsSize * sizeof(Material *));
    cudaMemcpy(materials, cpuMaterials, materialsSize * sizeof(Material *), cudaMemcpyHostToDevice);
}

MaterialCompact::MaterialCompact(FILE *matFile) {
    if (1 != fread(&materialsSize, sizeof(unsigned), 1, matFile))
        throw std::runtime_error("Error reading BVH cache file!\n");
    materialLength = (U32 *) malloc(materialsSize * sizeof(U32));
    cpuMaterials = (Material **) malloc(materialsSize * sizeof(Material *));

    for (int i = 0; i < materialsSize; i++) {
        if (1 != fread(materialLength + i, sizeof(unsigned), 1, matFile))
            throw std::runtime_error("Error reading BVH cache file!\n");

        auto material = (Material *) malloc(materialLength[i]);
        cudaMalloc(cpuMaterials + i, materialLength[i]);

        if (1 != fread(material, materialLength[i], 1, matFile))
            throw std::runtime_error("Error reading BVH cache file!\n");
        cudaMemcpy(cpuMaterials[i], material, materialLength[i], cudaMemcpyHostToDevice);

        free(material);
    }

    cudaMalloc(&materials, materialsSize * sizeof(Material *));
    cudaMemcpy(materials, cpuMaterials, materialsSize * sizeof(Material *), cudaMemcpyHostToDevice);
}

MaterialCompact::~MaterialCompact() {
    for (int i = 0; i < materialsSize; i++)
        cudaFree(cpuMaterials[i]);
    free(cpuMaterials);
    free(materialLength);
    cudaFree(materials);
}

__host__ void MaterialCompact::save(const std::string &fileName) {
    FILE *matFile = fopen(fileName.c_str(), "wb");
    if (!matFile)
        throw std::runtime_error("Error opening BVH cache file!");

    if (1 != fwrite(&materialsSize, sizeof(unsigned), 1, matFile))
        throw std::runtime_error("Error writing BVH cache file!\n");

    for (int i = 0; i < materialsSize; i++) {
        if (1 != fwrite(materialLength + i, sizeof(unsigned), 1, matFile))
            throw std::runtime_error("Error writing BVH cache file!\n");
        auto material = (Material *) malloc(materialLength[i]);

        cudaMemcpy(material, cpuMaterials[i], materialLength[i], cudaMemcpyDeviceToHost);
        if (1 != fwrite(material, materialLength[i], 1, matFile))
            std::runtime_error("Error writing BVH cache file!\n");

        free(material);
    }
}
