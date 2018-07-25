#ifndef TRENCHANTTRACER_RENDERER_H
#define TRENCHANTTRACER_RENDERER_H


#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <core/Scene.h>
#include <core/Camera.h>
#include <math/LinearMath.h>
#include <control/Controller.h>
#include <util/Config.h>
#include <util/WangHash.h>
#include <bvh/BVHCompact.h>

struct RenderMeta {
    int frameNumber;
    unsigned int hashedFrame;
    int hdrWidth;
    int hdrHeight;

    int SAMPLES;
    float PI;
    float TWO_PI;
};

class Renderer {
private:
    static Renderer *instance;

    GLuint vbo;
    Vec3f *accumulatedBuffer; // image buffer storing accumulated pixel samples
    Vec3f *outputBuffer; // stores averaged pixel samples
    CameraMeta *cameraMetaDevice; //device memory
    RenderMeta renderMeta; //host memory
    RenderMeta *renderMetaDevice; //device memory

    Config *config;
    BVHCompact *bvhCompact; //device memory
    HDRImage *hdrEnv;

    void render();

public:
    const static float PI;
    const static float TWO_PI;

    Renderer(Config *config, BVHCompact *bvhCompact, HDRImage *hdrImage);

    ~Renderer();

    static void init(Config *config, BVHCompact *bvhCompact, HDRImage *hdrImage);

    static Renderer *getInstance();

    static void clear();

    void display(Controller *controller);
};


#endif //TRENCHANTTRACER_RENDERER_H
