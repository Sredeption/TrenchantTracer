#ifndef TRENCHANTTRACER_MESH_H
#define TRENCHANTTRACER_MESH_H


#include <geometry/Geometry.h>
#include <util/Array.h>

class Mesh : public Geometry {
    Array<Vec3i> vertexIndices;
    Array<Vec3i> textureIndices;
    Array<Vec3i> normalIndices;

public:
    static const std::string TYPE;

    Mesh();

    __host__ virtual U32 size() const;

    void addVertex(const Vec3i &vertex);

    void addTexture(const Vec3i &texture);

    void addNormal(const Vec3i &normal);

    Array<Vec3i> &getVertexIndices();

    Array<Vec3i> &getTextureIndices();

    Array<Vec3i> &getNormalIndices();

};


#endif //TRENCHANTTRACER_MESH_H
