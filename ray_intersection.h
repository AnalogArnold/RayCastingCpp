#pragma once

#include <vector>
#include <array>
#include "eigen_types.h"
#include "ray.h"

struct IntersectionOutput {
    Eigen::ArrayXXd barycentric_coordinates;
    EiVectorD3d plane_normals;
    Eigen::Array<double, Eigen::Dynamic, 1> t_values;
};

EiVectorD3d cross_rowwise(const EiVectorD3d &mat1, const EiVectorD3d &mat2);

IntersectionOutput intersect_plane(const Ray &ray,
                                   const std::vector<std::array<int,3>> &connectivity,
                                   const std::vector<std::array<double,3>> &node_coords);
