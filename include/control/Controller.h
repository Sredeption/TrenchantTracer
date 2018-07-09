#ifndef TRENCHANTTRACER_CONTROLLER_H
#define TRENCHANTTRACER_CONTROLLER_H

#include<core/Camera.h>
#include <control/Mouse.h>
#include <control/Keyboard.h>

class Mouse;

class Keyboard;

class Controller {
private:
    friend Mouse;
    friend Keyboard;

    Mouse *mouse;
    Keyboard *keyboard;
    Camera *camera;

    static Controller *instance;

    explicit Controller(Camera *camera);

    ~Controller();

public:
    bool bufferReset;

    static void init(Camera *camera);

    static Controller *getInstance();

    static void clear();

    Mouse *getMouse();

    Keyboard *getKeyboard();

    Camera * getCamera();
};

void motion(int x, int y);

void mouse(int button, int state, int x, int y);

void keyboard(unsigned char key, int, int);

void specialKeys(int key, int, int);

#endif //TRENCHANTTRACER_CONTROLLER_H
