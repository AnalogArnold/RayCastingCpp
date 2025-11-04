#pragma once

#include <vector>
#include <array>
#include "camera.h"
#include "eigen_types.h"
#include "ray.h"

EiVector3d return_ray_color(const Ray &ray,
                            const std::vector<std::array<int,3>> &connectivity,
                            const std::vector<std::array<double,3>> &node_coords);

void render_ppm_image(const Camera& camera1,
                     const std::vector<std::array<int,3>> &connectivity,
                     const std::vector<std::array<double,3>> &node_coords);
