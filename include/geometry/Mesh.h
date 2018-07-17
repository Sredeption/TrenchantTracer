//
// Created by issac on 18-7-11.
//

#ifndef TRENCHANTTRACER_MESH_H
#define TRENCHANTTRACER_MESH_H


#include <util/Array.h>

class Mesh {
    Array<Vec3i> vertexIndices;
    Array<Vec3i> textureIndices;
    Array<Vec3i> normalIndices;

public:
    void addVertex(const Vec3i &vertex);

    void addTexture(const Vec3i &texture);

    void addNormal(const Vec3i &normal);

    Array<Vec3i> &getVertexIndices();

    Array<Vec3i> &getTextureIndices();

    Array<Vec3i> &getNormalIndices();

    bool empty();

    int size();
};


#endif //TRENCHANTTRACER_MESH_H
