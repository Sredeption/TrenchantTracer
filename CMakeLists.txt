cmake_minimum_required(VERSION 3.10)
project(TrenchantTracer)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${OPENGL_INCLUDE_DIR})

include_directories(include)

set(CUDA_SRC
        src/core/RenderKernel.cu)

CUDA_COMPILE(CUDA_O ${CUDA_SRC})

add_executable(TrenchantTracer
        src/main.cpp
        src/core/Camera.cpp
        include/core/Camera.h
        include/math/LinearMath.h
        include/math/CutilMath.h
        src/control/Mouse.cpp
        include/control/Mouse.h
        src/control/Keyboard.cpp
        include/control/Keyboard.h
        src/control/Controller.cpp
        include/control/Controller.h
        include/Config.h
        src/loader/HDRLoader.cpp
        include/loader/HDRLoader.h
        src/core/HDRImage.cpp
        include/core/HDRImage.h
        src/core/Renderer.cpp
        include/core/Renderer.h
        src/util/WangHash.cpp
        include/util/WangHash.h
        ${CUDA_O})

target_link_libraries(TrenchantTracer
        ${CUDA_LIBRARIES}
        ${GLEW_LIBRARIES}
        ${GLUT_LIBRARIES}
        ${OPENGL_LIBRARIES})