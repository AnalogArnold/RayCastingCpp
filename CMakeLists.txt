cmake_minimum_required(VERSION 4.0)
project(RayCastingCpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS -O2)

add_executable(RayCastingCpp main.cpp
        math_utils.h
        eigen_types.h
        camera.h
        src/camera.cpp
        ray.h
        hit_record.h
        ray_intersection.h
        src/ray_intersection.cpp
        renderer.h
        src/renderer.cpp)
