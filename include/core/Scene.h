#ifndef TRENCHANTTRACER_SCENE_H
#define TRENCHANTTRACER_SCENE_H


#include <core/Object.h>
#include <core/HDRImage.h>
#include <util/Array.h>
#include <math/LinearMath.h>

class Scene {
private:
    Array<Vec3i> vertexIndices;
    Array<Vec3f> vertices;
    Array<U32> materialIndices;
    Array<Material *> materials;

public:
    Scene();

    ~Scene();

    void add(Object *object);

    int getTrianglesNum() const;

    const Vec3i *getTrianglePtr(int idx = 0);

    const Vec3i &getTriangle(int idx);

    int getVerticesNum() const;

    const Vec3f *getVertexPtr(int idx = 0);

    const Vec3f &getVertex(int idx);

    const U32 *getMatIndexPtr(int idx = 0);

    const U32 &getMatIndex(int idx);

    int getMaterialNum() const;

    const Material **getMaterialPtr(int idx = 0);

    const Material *&getMaterial(int idx);

};


#endif //TRENCHANTTRACER_SCENE_H
