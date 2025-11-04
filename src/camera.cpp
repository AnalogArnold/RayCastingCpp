#include "../camera.h"
#include "../math_utils.h"
#include <cmath>

// Global rendering parameters defined in main.cpp
extern unsigned short image_width;
extern unsigned short image_height;

// Default constructor - Set camera at the origin, looking down with FOV of 90 degrees
// We initialize all vectorised quantities with all coefficients set to 0
Camera::Camera()
    : camera_center(EiVector3d::Zero()),
      point_camera_target((EiVector3d() << 0.0, 0.0, -1.0).finished()),
      angle_vertical_view(degreesToRadians(90.0)),
      matrix_camera_to_world(EiMatrix4d::Zero()),
      matrix_world_to_camera(EiMatrix4d::Zero()),
      matrix_pixel_spacing(Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>::Zero()),
      viewport_upper_left(EiVector3d::Zero()),
      pixel_00_center(EiVector3d::Zero()) {
    create_basis_matrices();
}

// Full constructor for camera at an arbitrary position
Camera::Camera(const EiVector3d &camera_center_in,
               const EiVector3d &point_camera_target_in,
               const double angle_vertical_view_in)
    : camera_center(camera_center_in),
      point_camera_target(point_camera_target_in),
      angle_vertical_view(degreesToRadians(angle_vertical_view_in)),
      matrix_camera_to_world(EiMatrix4d::Zero()),
      matrix_world_to_camera(EiMatrix4d::Zero()),
      matrix_pixel_spacing(Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>::Zero()),
      viewport_upper_left(EiVector3d::Zero()),
      pixel_00_center(EiVector3d::Zero()) {
    create_basis_matrices();
}

void Camera::create_basis_matrices() {
    // Creates the camera-to-world and world-to-camera matrices.
    EiVector3d basis_vector_forward = camera_center - point_camera_target;
    double focal_length = sqrt(basis_vector_forward.dot(basis_vector_forward));
    basis_vector_forward = basis_vector_forward.normalized();
    EiVector3d vector_view_up = EiVector3d::UnitY(); // (0, 1, 0). # View up vector orthogonal to basis_vector_right. Defines sideways tilt. Value can be changed, this is the default for the camera to be straight.
    EiVector3d basis_vector_right = (vector_view_up.cross(basis_vector_forward)).normalized();
    EiVector3d basis_vector_up = (basis_vector_forward.cross(basis_vector_right)).normalized();

    // Fill the top-left 3x3 block with the 3 basis vectors as rows
    matrix_camera_to_world.block<1,3>(0,0) = basis_vector_right;
    matrix_camera_to_world.block<1,3>(1,0) = basis_vector_up;
    matrix_camera_to_world.block<1,3>(2,0) = basis_vector_forward;
    matrix_camera_to_world.block<1,3>(3,0) = camera_center;
    matrix_camera_to_world(3,3) = 1.0;
    // Take the inverse to get the world_to_camera matrix for free
    matrix_world_to_camera = matrix_camera_to_world.inverse();
    create_viewport(basis_vector_forward, basis_vector_right, basis_vector_up, focal_length);
}

void Camera::create_viewport(const EiVector3d &basis_vector_forward,
                            const EiVector3d &basis_vector_right,
                            const EiVector3d &basis_vector_up,
                            const double &focal_length) {
    // Creates the viewport from the camera basis vectors and the focal length.
    // Returns pixel spacing vectors and the 0,0-positions for the pixel and the upper left corner of the viewport.
    double h_temp = std::tan(angle_vertical_view / 2.0);
    double viewport_height = 2 * h_temp * focal_length; // world units (arbitrary)
    double viewport_width = viewport_height * (static_cast<double>(image_width) / image_height); // world units (arbitrary)
    // Viewport basis vectors
    EiVector3d vector_viewport_x_axis = viewport_width * basis_vector_right; //Vu
    EiVector3d vector_viewport_y_axis = (-viewport_height) * basis_vector_up; //Vw
    // Pixel spacing vectors (delta vectors)
    EiVector3d vector_pixel_spacing_x = vector_viewport_x_axis / image_width; // Delta u
    EiVector3d vector_pixel_spacing_y = vector_viewport_y_axis / image_height; // Delta v
    // Store delta vectors as a matrix
    matrix_pixel_spacing.block(0,0, 1, 3) = vector_pixel_spacing_x;
    matrix_pixel_spacing.block(1,0, 1, 3) = vector_pixel_spacing_y;
    // 0,0-positions
    viewport_upper_left = camera_center - basis_vector_forward - vector_viewport_x_axis / 2 - vector_viewport_y_axis / 2;
    pixel_00_center = viewport_upper_left + 0.5 * (vector_pixel_spacing_x + vector_pixel_spacing_y);
}