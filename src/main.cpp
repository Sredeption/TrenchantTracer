#include <GL/glew.h>
#include <GL/glut.h>

#include <iostream>
#include <cuda_gl_interop.h>

#include <core/Camera.h>
#include <core/Renderer.h>
#include <control/Controller.h>
#include <loader/HDRLoader.h>
#include <loader/OBJLoader.h>
#include <util/Config.h>
#include <bvh/BVH.h>
#include <loader/SceneLoader.h>
#include <material/MaterialCompact.h>

void Timer(int) {
    glutPostRedisplay();
    glutTimerFunc(10, Timer, 0);
}

void display() {
    auto controller = Controller::getInstance();
    auto renderer = Renderer::getInstance();
    renderer->display(controller);
}

int main(int argc, char **argv) {
    // set up config
//    std::string configFileName = "config/maserati.json";
    std::string configFileName = "config/dev.json";
    auto config = new Config(configFileName);

    // init camera
    auto *camera = new Camera(config->width, config->height);
    camera->setFovx(45);
    Controller::init(camera);

    // load assets
    HDRLoader hdrLoader;
    HDRImage *hdrEnv = hdrLoader.load(config->hdrFileName);

    std::string bvhFileName = config->objFileName + ".bvh";
    std::string matFileName = config->objFileName + ".mat";
    std::string geoFileName = config->objFileName + ".geo";
    FILE *bvhFile = nullptr;
    if (!config->bvhReload)
        bvhFile = fopen(bvhFileName.c_str(), "rb");
    FILE *matFile = nullptr;
    if (!config->materialReload)
        matFile = fopen(matFileName.c_str(), "rb");
    FILE *geoFile = nullptr;
    geoFile = fopen(geoFileName.c_str(), "rb");

    BVHCompact *bvhCompact;
    MaterialCompact *materialCompact;
    GeometryCompact *geometryCompact;

    if (!bvhFile || !matFile) {
        SceneLoader sceneLoader(config);
        Scene *scene = sceneLoader.load();

        geometryCompact = new GeometryCompact(scene);
        geometryCompact->save(geoFileName);

        materialCompact = new MaterialCompact(scene);
        materialCompact->save(matFileName);

        if (!bvhFile) {
            SAHHelper sahHelper;
            auto bvh = new BVH(scene, sahHelper);
            bvhCompact = bvh->createHolder();
            bvhCompact->save(bvhFileName);

            delete bvh;
        } else {
            bvhCompact = new BVHCompact(bvhFile);
        }

        delete scene;
    } else {
        bvhCompact = new BVHCompact(bvhFile);
        materialCompact = new MaterialCompact(matFile);
        geometryCompact = new GeometryCompact(geoFile);
    }

    // initialize GLUT
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB); // specify the display mode to be RGB and single buffering
    glutInitWindowPosition(100, 100); // specify the initial window position
    glutInitWindowSize(config->width, config->height); // specify the initial window size
    glutCreateWindow("Trenchant Tracer, CUDA path tracer using SplitBVH"); // create the window and set title

    cudaGLSetGLDevice(0);
    cudaSetDevice(0);

    // initialize OpenGL:
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, config->width, 0.0, config->height);
    fprintf(stderr, "OpenGL initialized \n");

    // register callback function to display graphics
    glutDisplayFunc(display);

    // functions for user interaction
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeys);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    // initialize GLEW
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 "))
        throw std::runtime_error("ERROR: Support for necessary OpenGL extensions missing.");
    fprintf(stderr, "GLEW initialized  \n");

    Renderer::init(config, bvhCompact, geometryCompact, materialCompact, hdrEnv);
    fprintf(stderr, "Renderer initialized \n");

    Timer(0);

    fprintf(stderr, "Entering glutMainLoop...  \n");
    printf("Rendering started...\n");
    glutMainLoop();

    delete config;
    delete camera;
    Controller::clear();
    Renderer::clear();
    delete bvhCompact;
    delete geometryCompact;
    delete materialCompact;
    delete hdrEnv;
}