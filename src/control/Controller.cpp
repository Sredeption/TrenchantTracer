#include <control/Controller.h>

Controller *Controller::instance = nullptr;

Controller::Controller(Camera *camera) :
        camera(camera) {
    this->mouse = new Mouse(this);
    this->keyboard = new Keyboard(this);
    this->bufferReset = false;
}

Controller::~Controller() {
    delete this->mouse;
    delete this->keyboard;
}

void Controller::init(Camera *camera) {
    Controller::instance = new Controller(camera);
}

Controller *Controller::getInstance() {
    return Controller::instance;
}

void Controller::clear() {
    delete Controller::instance;
}

Mouse *Controller::getMouse() {
    return this->mouse;
}

Keyboard *Controller::getKeyboard() {
    return this->keyboard;
}

Camera *Controller::getCamera() {
    return this->camera;
}

void motion(int x, int y) {
    Controller::getInstance()->getMouse()->motion(x, y);
}

void mouse(int button, int state, int x, int y) {
    Controller::getInstance()->getMouse()->mouse(button, state, x, y);
}

void keyboard(unsigned char key, int, int) {
    Controller::getInstance()->getKeyboard()->keyboard(key);
}

void specialKeys(int key, int, int) {
    Controller::getInstance()->getKeyboard()->specialKeys(key);
}
