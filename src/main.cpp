#include <GL/glew.h>
#include <GL/glut.h>

#include <iostream>

#include <cuda_gl_interop.h>

#include <Config.h>
#include <core/Camera.h>
#include <core/Renderer.h>
#include <control/Controller.h>
#include <loader/HDRLoader.h>

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
    auto *camera = new Camera(Config::WIDTH, Config::HEIGHT);
    Controller::init(camera);


    HDRLoader hdrLoader;
    HDRImage *hdrEnv = hdrLoader.load(Config::HDRFileName);

    // initialize GLUT
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB); // specify the display mode to be RGB and single buffering
    glutInitWindowPosition(100, 100); // specify the initial window position
    glutInitWindowSize(Config::WIDTH, Config::HEIGHT); // specify the initial window size
    glutCreateWindow("Trenchant Tracer, CUDA path tracer using SplitBVH"); // create the window and set title

    cudaGLSetGLDevice(0);
    cudaSetDevice(0);

    // initialize OpenGL:
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, Config::WIDTH, 0.0, Config::HEIGHT);
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
    fprintf(stderr, "glew initialized  \n");

    Renderer::init(hdrEnv);
    fprintf(stderr, "Renderer initialized \n");

    Timer(0);

    fprintf(stderr, "Entering glutMainLoop...  \n");
    printf("Rendering started...\n");
    glutMainLoop();

    delete camera;
    Controller::clear();
    Renderer::clear();
    delete hdrEnv;
}