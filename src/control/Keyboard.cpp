//
// Created by issac on 18-7-6.
//


#include <control/Keyboard.h>

Keyboard::Keyboard(Controller *controller) :
        controller(controller) {

}

void Keyboard::keyboard(unsigned char key) {
    Camera *camera = controller->camera;
    switch (key) {

        case (27) :
            exit(0);
        case ('a') :
            camera->strafe(-0.05f);
            controller->bufferReset = true;
            break;
        case ('d') :
            camera->strafe(0.05f);
            controller->bufferReset = true;
            break;
        case ('r') :
            camera->changeAltitude(0.05f);
            controller->bufferReset = true;
            break;
        case ('f') :
            camera->changeAltitude(-0.05f);
            controller->bufferReset = true;
            break;
        case ('w') :
            camera->goForward(0.05f);
            controller->bufferReset = true;
            break;
        case ('s') :
            camera->goForward(-0.05f);
            controller->bufferReset = true;
            break;
        case ('g') :
            camera->changeApertureDiameter(0.1f);
            controller->bufferReset = true;
            break;
        case ('h') :
            camera->changeApertureDiameter(-0.1f);
            controller->bufferReset = true;
            break;
        case ('t') :
            camera->changeFocalDistance(0.1f);
            controller->bufferReset = true;
            break;
        case ('y') :
            camera->changeFocalDistance(-0.1f);
            controller->bufferReset = true;
            break;
        default:
            break;
    }
}

void Keyboard::specialKeys(int key) {
    Camera *camera = controller->camera;
    switch (key) {

        case GLUT_KEY_LEFT:
            camera->changeYaw(0.02f);
            controller->bufferReset = true;
            break;
        case GLUT_KEY_RIGHT:
            camera->changeYaw(-0.02f);
            controller->bufferReset = true;
            break;
        case GLUT_KEY_UP:
            camera->changePitch(0.02f);
            controller->bufferReset = true;
            break;
        case GLUT_KEY_DOWN:
            camera->changePitch(-0.02f);
            controller->bufferReset = true;
            break;
        default:
            break;
    }
}
