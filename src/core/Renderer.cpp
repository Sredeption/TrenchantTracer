#include <core/Renderer.h>

Renderer *Renderer::instance = nullptr;

const float Renderer::PI = 3.14156265f;
const float Renderer::TWO_PI = 6.2831853071795864769252867665590057683943f;

Renderer::Renderer(HDRImage *hdrEnv) {

    // store rendering resources
    this->hdrEnv = hdrEnv;

    // allocate GPU memory for accumulation buffer
    cudaMalloc(&accumulatedBuffer, Config::WIDTH * Config::HEIGHT * sizeof(Vec3f));

    // allocate GPU memory for interactive camera
    cudaMalloc(&cameraMetaDevice, sizeof(CameraMeta));

    // allocate GPU memory for rendering parameter
    cudaMalloc(&renderMetaDevice, sizeof(RenderMeta));

    // initialize HDR meta data
    renderMeta.hdrWidth = hdrEnv->width;
    renderMeta.hdrHeight = hdrEnv->height;

    // initialize constants
    renderMeta.SAMPLES = Config::SAMPLES;
    renderMeta.PI = Renderer::PI;
    renderMeta.TWO_PI = Renderer::TWO_PI;

    //Create vertex buffer object
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    //Initialize VBO
    unsigned int size = Config::WIDTH * Config::HEIGHT * sizeof(Vec3f);
    glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //Register VBO with CUDA
    cudaGLRegisterBufferObject(vbo);
}

Renderer::~Renderer() {
    cudaFree(accumulatedBuffer);
    cudaFree(outputBuffer);
    cudaFree(cameraMetaDevice);
    cudaFree(renderMetaDevice);
}

void Renderer::init(HDRImage *hdrEnv) {
    Renderer::instance = new Renderer(hdrEnv);
}

void Renderer::clear() {
    delete Renderer::instance;
}

Renderer *Renderer::getInstance() {
    return Renderer::instance;
}

void Renderer::display(Controller *controller) {
    if (controller->bufferReset) {
        cudaMemset(accumulatedBuffer, 1, Config::WIDTH * Config::HEIGHT * sizeof(Vec3f));
        renderMeta.frameNumber = 0;
    }

    controller->bufferReset = false;
    renderMeta.frameNumber++;

    // build a new camera for each frame on the CPU
    CameraMeta hostCamera = controller->getCamera()->getCameraMeta();

    // copy the CPU camera to a GPU camera
    cudaMemcpy(cameraMetaDevice, &hostCamera, sizeof(CameraMeta), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    cudaGLMapBufferObject((void **) &outputBuffer, vbo); // maps a buffer object for access by CUDA

    glClear(GL_COLOR_BUFFER_BIT); //clear all pixels

    // calculate a new seed for the random number generator, based on the frameNumber
    renderMeta.hashedFrame = WangHash::encode(static_cast<unsigned int>(renderMeta.frameNumber));

    // gateway from host to CUDA, passes all data needed to render frame (triangles, BVH tree, camera) to CUDA for execution
    this->render();

    cudaThreadSynchronize();
    cudaGLUnmapBufferObject(vbo);
    glFlush();
    glFinish();
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(2, GL_FLOAT, 12, nullptr);
    glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid *) 8);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glDrawArrays(GL_POINTS, 0, Config::WIDTH * Config::HEIGHT);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
}
