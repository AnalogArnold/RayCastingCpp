#pragma once
#include "eigen_types.h"

class Camera {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // Required for structures using Eigen members
    // Constructors
    Camera();
    Camera(const EiVector3d &camera_center_in,
           const EiVector3d &point_camera_target_in,
           const double angle_vertical_view_in);

    // Public members
    EiVector3d camera_center;
    EiVector3d point_camera_target;
    double angle_vertical_view;

    // Vectorised quantities
    EiMatrix4d matrix_camera_to_world;
    EiMatrix4d matrix_world_to_camera;
    Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor> matrix_pixel_spacing;
    EiVector3d viewport_upper_left;
    EiVector3d pixel_00_center;

private:
    void create_basis_matrices();
    void create_viewport(const EiVector3d &basis_vector_forward,
                        const EiVector3d &basis_vector_right,
                        const EiVector3d &basis_vector_up,
                        const double &focal_length);
};