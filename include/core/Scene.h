#ifndef TRENCHANTTRACER_SCENE_H
#define TRENCHANTTRACER_SCENE_H


#include <core/Object.h>
#include <util/Array.h>
#include <math/LinearMath.h>

class Scene {
private:
    Array<Vec3i> vertexIndices;
    Array<Vec3i> normalIndices;
    Array<Vec3f> vertices;
    Array<Vec3f> normals;
    Array<U32> materialIndices;
    Array<Material *> materials;
    Array<Group *> geometries;

public:
    Scene();

    ~Scene();

    void add(Object *object);

    void add(MaterialPool *pool);

    void add(Group *group);

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

    int getGeometryNum() const;

    const Group **getGeometryPtr(int idx = 0);

    const Group *&getGeometry(int idx);

    int getNormalIndexNum() const;

    const Vec3i *getNormalIndexPtr(int idx = 0);

    const Vec3i &getNormalIndex(int idx);

    int getNormalNum() const;

    const Vec3f *getNormalPtr(int idx = 0);

    const Vec3f &getNormal(int idx);

};


#endif //TRENCHANTTRACER_SCENE_H
