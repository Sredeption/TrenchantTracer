//
// Created by issac on 18-7-6.
//

#ifndef TRENCHANTTRACER_KEYBOARD_H
#define TRENCHANTTRACER_KEYBOARD_H

#include <cstdlib>
#include <GL/glut.h>

#include <control/Controller.h>

class Controller;

class Keyboard {

private:
    Controller *controller;

public:

    explicit Keyboard(Controller *controller);

    void keyboard(unsigned char key);

    void specialKeys(int key);
};


#endif //TRENCHANTTRACER_KEYBOARD_H
