#pragma once

#include <vector>
#include <array>
#include "camera.h"
#include "eigen_types.h"
#include "ray.h"

inline EiVector3d get_face_color(Eigen::Index minRowIndex,
    const std::vector<std::array<double,3>> &face_colors);

EiVector3d return_ray_color(const Ray &ray,
                            const std::vector<std::vector<std::array<int,3>>> &scene_connectivity,
                            const std::vector<std::vector<std::array<double,3>>> &scene_coords,
                            const std::vector<std::vector<std::array<double,3>>> &scene_face_colors);

void render_ppm_image(const Camera& camera1,
                     const std::vector<std::vector<std::array<int,3>>> &scene_connectivity,
                     const std::vector<std::vector<std::array<double,3>>> &scene_coords,
                     const std::vector<std::vector<std::array<double,3>>> &scene_face_colors);
