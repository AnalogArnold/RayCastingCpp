#include <cmath>
#include <fstream>
#include <iostream>
#include <iostream>
#include <limits>
#include "./Eigen/Dense" // Since pyvale uses Eigen, might as well use their implementation of a 3D vector at compile time? I saw it being used in pyvale (dicsmooth) anyway

double degreesToRadians(double angleDeg) {
    // Converts degrees to radians. Used to convert the angle of vertical view.
    return angleDeg * M_PI / 180;
}

// Input
int image_width = 400; // px
double aspect_ratio = 16.0/9.0;
int image_height = (image_width/aspect_ratio); // px
//double camera_center[] = {-0.5, 1.1, 1.1};
//double camera_target[] = {0, 0, -1};
//double angle_vertical_view = 90.0; // degrees


// Define aliases for the vectors and matrices from Eigen library.
// Can't use the convenience typedefs like Matrix4d or Vector3d because everything in Eigen is column-major, whereas
// C++, NumPy, and ScratchAPixel all use the row-major, so equations would be different and I don't want to tamper with that.
using EiMatrix4d = Eigen::Matrix<double, 4, 4, Eigen::StorageOptions::RowMajor>;
using EiVector3d = Eigen::Matrix<double, 3, 1, Eigen::StorageOptions::RowMajor>;

// Camera

class Camera {
public:
    // Default constructor - Set camera at the origin, looking down with FOV of 90 degrees
    Camera(): camera_center(Eigen::Vector3d(0.0, 0.0, 0.0)), point_camera_target(Eigen::Vector3d(0.0, 0.0, -1.0)), angle_vertical_view(degreesToRadians(90.0)) {create_basis_matrices();}
    // Full constructor for camera at an arbitrary position
    Camera(Eigen::Vector3d camera_center_in, Eigen::Vector3d point_camera_target_in, double angle_vertical_view_in): camera_center(camera_center_in), point_camera_target(point_camera_target_in), angle_vertical_view(degreesToRadians(angle_vertical_view_in)) {create_basis_matrices();}

    Eigen::Vector3d camera_center;
    Eigen::Vector3d point_camera_target;
    double angle_vertical_view;

    // Initialize all vectorised quantities with all coefficients set to 0

    Eigen::Matrix4d matrix_camera_to_world = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d matrix_world_to_camera = Eigen::Matrix4d::Zero();
    Eigen::Matrix<double, 3, 2> matrix_pixel_spacing = Eigen::Matrix<double, 3, 2>::Zero();
    Eigen::Vector3d viewport_upper_left = Eigen::Vector3d(0.0, 0.0, 0.0);
    Eigen::Vector3d pixel_00_center = Eigen::Vector3d(0.0, 0.0, 0.0);

    void create_basis_matrices() {
        // Creates the camera-to-world and world-to-camera matrices.
        Eigen::Vector3d basis_vector_forward = camera_center - point_camera_target;
        double focal_length = sqrt(basis_vector_forward.dot(basis_vector_forward));
        basis_vector_forward = basis_vector_forward.normalized();
        Eigen::Vector3d vector_view_up = Eigen::Vector3d::UnitY(); // (0, 1, 0). # View up vector orthogonal to basis_vector_right. Defines sideways tilt. Value can be changed, this is the default for the camera to be straight.
        Eigen::Vector3d basis_vector_right = (vector_view_up.cross(basis_vector_forward)).normalized();
        Eigen::Vector3d basis_vector_up = (basis_vector_forward.cross(basis_vector_right)).normalized();

        // Fill the top-left 3x3 block with the 3 basis vectors as columns
        matrix_camera_to_world.block<3,1>(0,0) = basis_vector_right;
        matrix_camera_to_world.block<3,1>(0,1) = basis_vector_up;
        matrix_camera_to_world.block<3,1>(0,2) = basis_vector_forward;
        matrix_camera_to_world.block<3,1>(0,3) = camera_center;
        // Take the inverse to get the world_to_camera matrix for free
        matrix_world_to_camera = matrix_camera_to_world.inverse();
        // Create the viewport
        create_viewport(basis_vector_forward, basis_vector_right, basis_vector_up, focal_length);
    }

    void create_viewport(Eigen::Vector3d basis_vector_forward, Eigen::Vector3d basis_vector_right, Eigen::Vector3d basis_vector_up, double focal_length) {
        // Creates the viewport from the camera basis vectors and the focal length.
        // Returns pixel spacing vectors and the 0,0-positions for the pixel and the upper left corner of the viewport.
        double h_temp = std::tan(angle_vertical_view / 2.0);
        double viewport_height = 2 * h_temp * focal_length; // world units (arbitrary)
        double viewport_width = viewport_height * (image_width / image_height); // world units (arbitrary)
        // Viewport basis vectors
        Eigen::Vector3d vector_viewport_x_axis = viewport_width * basis_vector_right; //Vu
        Eigen::Vector3d vector_viewport_y_axis = (-viewport_height) * basis_vector_up; //Vw
        // Pixel spacing vectors (delta vectors)
        Eigen::Vector3d vector_pixel_spacing_x = vector_viewport_x_axis / image_width; // Delta u
        Eigen::Vector3d vector_pixel_spacing_y = vector_viewport_y_axis / image_height; // Delta v
        // Store delta vectors as a matrix
        matrix_pixel_spacing.block(0,0, 3, 1) = vector_pixel_spacing_x;
        matrix_pixel_spacing.block(0,1, 3, 1) = vector_pixel_spacing_y;
        // 0,0-positions
        viewport_upper_left = camera_center - basis_vector_forward - vector_viewport_x_axis / 2 - vector_viewport_y_axis / 2;
        pixel_00_center = viewport_upper_left + 0.5 * (vector_pixel_spacing_x + vector_pixel_spacing_y);
    }
};

// Rays
class Ray {
    public:
        Ray(Eigen::Vector3d origin, Eigen::Vector3d direction, double t_min, double t_max) : origin(origin), direction(direction), t_min(t_min), t_max(t_max) {} // constructor
        double t_max;
        double t_min;
        Eigen::Vector3d origin;
        Eigen::Vector3d direction;

        Eigen::Vector3d normalised_direction() {
        // Normalizes the ray direction vector.
            return direction.normalized();
        }
        Eigen::Vector3d point_at_parameter(const double t) const {
        // Computes the ray parameters at given t
            return origin + t * direction;
        }
};





int main() {
    Camera camera1;
    std::cout << camera1.matrix_camera_to_world << std::endl << camera1.pixel_00_center;
   // Ray test_ray(Eigen::Vector3d(-0.5, 1.1, 1.1), Eigen::Vector3d(4.132331920978222, -2.603127666416139, 1.1937133836332001), 0, std::numeric_limits<double>::infinity());
   // std::cout << test_ray.direction;
    return 0;
}