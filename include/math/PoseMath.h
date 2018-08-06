#ifndef TRENCHANTTRACER_POSEMATH_H
#define TRENCHANTTRACER_POSEMATH_H

#include <math/LinearMath.h>

inline Mat4f translate(const Vec3f &v) {
    Mat4f matrix;
    matrix.setRow(0, Vec4f(1, 0, 0, v.x));
    matrix.setRow(1, Vec4f(0, 1, 0, v.y));
    matrix.setRow(2, Vec4f(0, 0, 1, v.z));
    matrix.setRow(3, Vec4f(0, 0, 0, 1));
    return matrix;
}

inline Mat4f scale(const Vec3f &v) {
    Mat4f matrix;
    matrix.setRow(0, Vec4f(v.x, 0, 0, 0));
    matrix.setRow(1, Vec4f(0, v.y, 0, 0));
    matrix.setRow(2, Vec4f(0, 0, v.z, 0));
    matrix.setRow(3, Vec4f(0, 0, 0, 1));
    return matrix;
}

inline Mat4f orientation(const Vec3f &v) {

}

//pitch
inline Mat4f rotateX(const float &pitch) {
    Mat4f matrix;
    matrix.setRow(0, Vec4f(1, 0, 0, 0));
    matrix.setRow(1, Vec4f(0, cosf(pitch), -sinf(pitch), 0));
    matrix.setRow(2, Vec4f(0, sinf(pitch), cosf(pitch), 0));
    matrix.setRow(3, Vec4f(0, 0, 0, 1));
    return matrix;
}

//yaw
inline Mat4f rotateY(const float &yaw) {
    Mat4f matrix;
    matrix.setRow(0, Vec4f(cosf(yaw), 0, sinf(yaw), 0));
    matrix.setRow(1, Vec4f(0, 1, 0, 0));
    matrix.setRow(2, Vec4f(-sinf(yaw), 0, cosf(yaw), 0));
    matrix.setRow(1, Vec4f(0, 0, 0, 1));
    return matrix;
}

//roll
inline Mat4f rotateZ(const float &roll) {
    Mat4f matrix;
    matrix.setRow(0, Vec4f(cosf(roll), -sinf(roll), 0, 0));
    matrix.setRow(1, Vec4f(sinf(roll), cosf(roll), 0, 0));
    matrix.setRow(2, Vec4f(0, 0, 1, 0));
    matrix.setRow(3, Vec4f(0, 0, 0, 1));
    return matrix;
}

#endif //TRENCHANTTRACER_POSEMATH_H
