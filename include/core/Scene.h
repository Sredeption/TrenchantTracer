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

public:
    Scene();

    ~Scene();

    void add(Object *object);

    int getNumTriangles() const;

    const Vec3i *getTrianglePtr(int idx = 0);

    const Vec3i &getTriangle(int idx);

    int getNumVertices() const;

    const Vec3f *getVertexPtr(int idx = 0);

    const Vec3f &getVertex(int idx);


};


#endif //TRENCHANTTRACER_SCENE_H
