cmake_minimum_required(VERSION 3.10)
project(TrenchantTracer CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA QUIET REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${OPENGL_INCLUDE_DIR})

include_directories(include)
include_directories(lib/include)

set(CUDA_SRC
        src/core/RenderKernel.cu
        src/geometry/IntersectKernel.cu
        src/core/HDRKernel.cu
        src/material/Material.cu
        src/material/Coat.cu
        src/material/Diff.cu
        src/material/Metal.cu
        src/material/Spec.cu
        src/material/Refr.cu
        src/geometry/Sphere.cu)

CUDA_COMPILE(CUDA_O ${CUDA_SRC})

add_executable(TrenchantTracer
        ${CUDA_O}
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
        include/util/Config.h
        src/loader/HDRLoader.cpp
        include/loader/HDRLoader.h
        src/core/HDRImage.cpp
        include/core/HDRImage.h
        src/core/Renderer.cpp
        include/core/Renderer.h
        src/util/WangHash.cpp
        include/util/WangHash.h
        src/loader/OBJLoader.cpp
        include/loader/OBJLoader.h
        src/core/Object.cpp
        include/core/Object.h
        src/geometry/Mesh.cpp
        include/geometry/Mesh.h
        src/material/MaterialPool.cpp
        include/material/MaterialPool.h
        src/core/Group.cpp
        include/core/Group.h
        src/geometry/Vertices.cpp
        include/geometry/Vertices.h
        src/core/Scene.cpp
        include/core/Scene.h
        include/util/Array.h
        src/util/Config.cpp
        src/bvh/BVH.cpp
        include/bvh/BVH.h
        src/bvh/SAHHelper.cpp
        include/bvh/SAHHelper.h
        src/bvh/AABB.cpp
        include/bvh/AABB.h
        src/bvh/BVHNode.cpp
        include/bvh/BVHNode.h
        src/bvh/InnerNode.cpp
        include/bvh/InnerNode.h
        src/bvh/LeafNode.cpp
        include/bvh/LeafNode.h
        src/util/Sort.cpp
        include/util/Sort.h
        src/loader/SceneLoader.cpp
        include/loader/SceneLoader.h
        src/bvh/BVHCompact.cpp
        include/bvh/BVHCompact.h
        include/geometry/Ray.h
        include/geometry/Hit.h
        include/material/Coat.h
        include/material/Diff.h
        src/loader/MaterialLoader.cpp
        include/loader/MaterialLoader.h
        src/material/MaterialCompact.cpp
        include/material/MaterialCompact.h
        include/material/Material.h
        include/material/Metal.h
        include/material/Spec.h
        include/material/Refr.h
        src/geometry/Geometry.cpp
        include/geometry/Geometry.h
        include/geometry/Sphere.h
        src/geometry/GeometryCompact.cpp
        include/geometry/GeometryCompact.h)

target_link_libraries(TrenchantTracer
        ${CUDA_LIBRARIES}
        ${GLEW_LIBRARIES}
        ${GLUT_LIBRARIES}
        ${OPENGL_LIBRARIES})