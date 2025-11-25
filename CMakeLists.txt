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

find_package(OpenMP)
if (OPENMP_FOUND)
    target_link_libraries(RayCastingCpp PRIVATE "${OpenMP_CXX_FLAGS}")
    target_compile_options(RayCastingCpp PRIVATE "${OpenMP_CXX_FLAGS}")
    #set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    #set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")    #set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")    #target_link_libraries(RayCastingCpp PRIVATE OpenMP::OpenMP_CXX)    message(STATUS "OpenMP_CXX_VERSION ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP_CXX_INCLUDE_DIRS ${OpenMP_CXX_INCLUDE_DIRS}")
    message(STATUS "OpenMP_CXX_LIB_NAMES ${OpenMP_CXX_LIB_NAMES}")
    message(STATUS "OpenMP_CXX_LIBRARIES ${OpenMP_CXX_LIBRARIES}")
endif()