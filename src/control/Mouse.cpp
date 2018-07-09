#include <control/Mouse.h>

Mouse::Mouse(Controller *controller) :
        x(0), y(0), buttonState(0), modifierState(0), controller(controller) {

}

// camera mouse controls in X and Y direction
void Mouse::motion(int x, int y) {
    Camera *camera = controller->camera;
    int deltaX = this->x - x;
    int deltaY = this->y - y;

    if (deltaX != 0 || deltaY != 0) {

        if (this->buttonState == GLUT_LEFT_BUTTON)  // Rotate
        {
            camera->changeYaw(deltaX * 0.01f);
            camera->changePitch(-deltaY * 0.01f);
        } else if (this->buttonState == GLUT_MIDDLE_BUTTON) // Zoom
        {
            camera->changeAltitude(-deltaY * 0.01f);
        }

        if (this->buttonState == GLUT_RIGHT_BUTTON) // camera move
        {
            camera->changeRadius(-deltaY * 0.01f);
        }

        this->x = x;
        this->y = y;
        controller->bufferReset = true;
        glutPostRedisplay();

    }
}

void Mouse::mouse(int button, int state, int x, int y) {
    this->buttonState = button;
    this->modifierState = glutGetModifiers();
    this->x = x;
    this->y = y;
    this->motion(x, y);
}
