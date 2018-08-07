#ifndef TRENCHANTTRACER_CAMERA_H
#define TRENCHANTTRACER_CAMERA_H


#include <math/LinearMath.h>
#include <util/Config.h>

struct CameraMeta {
    Vec2i resolution;
    Vec3f position;
    Vec3f view;
    Vec3f up;
    Vec2f fov;
    float apertureRadius;
    float focalDistance;
};

class Camera {
private:

    const static float PI_OVER_TWO;
    const static float PI;
    const static std::string POSITION;
    const static std::string YAW;
    const static std::string PITCH;

    Vec3f centerPosition;
    Vec3f viewDirection;
    float yaw;
    float pitch;
    float radius;
    float apertureRadius;
    float focalDistance;

    void fixYaw();

    void fixPitch();

    void fixRadius();

    void fixApertureRadius();

    void fixFocalDistance();

    static float radiansToDegrees(float radians);

    static float degreesToRadians(float degrees);


public:

    Camera(Config *config);

    virtual ~Camera();

    void changeYaw(float m);

    void changePitch(float m);

    void changeRadius(float m);

    void changeAltitude(float m);

    void changeFocalDistance(float m);

    void strafe(float m);

    void goForward(float m);

    void rotateRight(float m);

    void changeApertureDiameter(float m);

    void setResolution(int x, int y);

    void setFovx(float fovx);

    CameraMeta getCameraMeta();

    Vec2i resolution;
    Vec2f fov;
};


#endif //TRENCHANTTRACER_CAMERA_H
