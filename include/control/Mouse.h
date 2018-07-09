
#ifndef TRENCHANTTRACER_MOUSE_H
#define TRENCHANTTRACER_MOUSE_H

#include <GL/glut.h>

#include <core/Camera.h>
#include <control/Controller.h>


class Controller;

class Mouse {
private:
    int x;
    int y;
    int buttonState;
    int modifierState;
    Controller *controller;

public:

    explicit Mouse(Controller *controller);

    void motion(int x, int y);

    void mouse(int button, int state, int x, int y);
};

#endif //TRENCHANTTRACER_MOUSE_H
