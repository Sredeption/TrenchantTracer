#include <core/Camera.h>
#include <geometry/Geometry.h>

const float Camera::PI_OVER_TWO = 1.5707963267948966192313216916397514420985f;
const float Camera::PI = 3.14156265f;
const std::string Camera::POSITION = "position";
const std::string Camera::YAW = "yaw";
const std::string Camera::PITCH = "pitch";

Camera::Camera(Config *config) {
    const nlohmann::json &camera = config->camera;

    if (camera.find(POSITION) == camera.end())
        centerPosition = Vec3f(0, 0, 0);
    else
        centerPosition = Geometry::jsonToVec(camera[POSITION]);

    if (camera.find(YAW) == camera.end())
        yaw = 0;
    else
        yaw = camera[YAW];

    if (camera.find(PITCH) == camera.end())
        pitch = 0.3;
    else
        pitch = camera[PITCH];

    radius = 4;
    apertureRadius = 0.04;
    focalDistance = 4.0f;

    resolution = Vec2i(config->width, config->height);
    fov = Vec2f(40, 40);
}

Camera::~Camera() = default;

void Camera::changeYaw(float m) {
    yaw += m;
    fixYaw();
}

void Camera::changePitch(float m) {
    pitch += m;
    fixPitch();
}

void Camera::changeRadius(float m) {
    // Change proportional to current radius. Assuming radius isn't allowed to go to zero.
    radius += radius * m;
    fixRadius();
}

void Camera::changeAltitude(float m) {
    centerPosition.y += m;
    //fixCenterPosition();
}

void Camera::goForward(float m) {
    centerPosition += viewDirection * m;
}

void Camera::strafe(float m) {
    Vec3f strafeAxis = cross(viewDirection, Vec3f(0, 1, 0));
    strafeAxis.normalize();
    centerPosition += strafeAxis * m;
}

void Camera::rotateRight(float m) {
    float yaw2 = yaw;
    yaw2 += m;
    float pitch2 = pitch;
    float xDirection = std::sin(yaw2) * std::cos(pitch2);
    float yDirection = std::sin(pitch2);
    float zDirection = std::cos(yaw2) * std::cos(pitch2);
    Vec3f directionToCamera = Vec3f(xDirection, yDirection, zDirection);
    viewDirection = directionToCamera * (-1.0);
}

void Camera::changeApertureDiameter(float m) {
    apertureRadius += (apertureRadius + 0.01) * m; // Change proportional to current apertureRadius.
    fixApertureRadius();
}


void Camera::changeFocalDistance(float m) {
    focalDistance += m;
    fixFocalDistance();
}


void Camera::setResolution(int x, int y) {
    resolution = Vec2i(x, y);
    setFovx(fov.x);
}

float Camera::radiansToDegrees(float radians) {
    return radians * 180.0f / Camera::PI;
}

float Camera::degreesToRadians(float degrees) {
    return degrees / 180.0f * Camera::PI;
}

void Camera::setFovx(float fovx) {
    // resolution float division
    fov.x = fovx;
    fov.y = radiansToDegrees(std::atan(std::tan(degreesToRadians(fovx) * 0.5f) * resolution.y / resolution.x) * 2.0f);
}

CameraMeta Camera::getCameraMeta() {
    CameraMeta cameraMeta;
    float xDirection = std::sin(yaw) * std::cos(pitch);
    float yDirection = std::sin(pitch);
    float zDirection = std::cos(yaw) * std::cos(pitch);
    Vec3f directionToCamera = Vec3f(xDirection, yDirection, zDirection);
    viewDirection = directionToCamera * (-1.0);
    Vec3f eyePosition = centerPosition + directionToCamera * radius;
    // rotate camera from stationary viewpoint

    cameraMeta.position = eyePosition;
    cameraMeta.view = viewDirection;
    cameraMeta.up = Vec3f(0, 1, 0);
    cameraMeta.resolution = Vec2i(resolution.x, resolution.y);
    cameraMeta.fov = Vec2f(fov.x, fov.y);
    cameraMeta.apertureRadius = apertureRadius;
    cameraMeta.focalDistance = focalDistance;

    return cameraMeta;
}

float mod(float x, float y) { // Does this account for -y ???
    return x - y * floorf(x / y);
}

void Camera::fixYaw() {
    yaw = mod(yaw, 2 * Camera::PI); // Normalize the yaw.
}

float clamp2(float n, float low, float high) {
    n = fminf(n, high);
    n = fmaxf(n, low);
    return n;
}

void Camera::fixPitch() {
    float padding = 0.05;
    pitch = clamp2(pitch, -PI_OVER_TWO + padding, PI_OVER_TWO - padding); // Limit the pitch.
}

void Camera::fixRadius() {
    float minRadius = 0.2;
    float maxRadius = 100.0;
    radius = clamp2(radius, minRadius, maxRadius);
}

void Camera::fixApertureRadius() {
    float minApertureRadius = 0.0;
    float maxApertureRadius = 25.0;
    apertureRadius = clamp2(apertureRadius, minApertureRadius, maxApertureRadius);
}

void Camera::fixFocalDistance() {
    float minFocalDist = 0.2;
    float maxFocalDist = 100.0;
    focalDistance = clamp2(focalDistance, minFocalDist, maxFocalDist);
}

