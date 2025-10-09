#include <cmath>
#include <fstream>
#include <iostream>
#include <fstream>
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
using EiMatrix4d = Eigen::Matrix<double, 4, 4, Eigen::StorageOptions::RowMajor>; // 4x4 matrix
using EiVector3d = Eigen::Matrix<double, 1, 3, Eigen::StorageOptions::RowMajor>; // row vector (3D)
using EiVector3dC = Eigen::Matrix<double, 3, 1, Eigen::StorageOptions::RowMajor>; // column vector (3D)
using RmMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EiVectorXd = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>; // Matrix storing X x 3 elements; mostly for coordinates to avoid having to loop constantly in the intersection code to get cross products etc.


// Camera

class Camera {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // Required for structures using Eigen members
    // Default constructor - Set camera at the origin, looking down with FOV of 90 degrees
    Camera(): camera_center(EiVector3d::Zero()), point_camera_target((EiVector3d() << 0.0, 0.0, -1.0).finished()), angle_vertical_view(degreesToRadians(90.0)) {create_basis_matrices();}
    // Full constructor for camera at an arbitrary position
    Camera(const EiVector3d &camera_center_in, const EiVector3d &point_camera_target_in, const double angle_vertical_view_in): camera_center(camera_center_in), point_camera_target(point_camera_target_in), angle_vertical_view(degreesToRadians(angle_vertical_view_in)) {create_basis_matrices();}

    EiVector3d camera_center;
    EiVector3d point_camera_target;
    double angle_vertical_view;

    // Initialize all vectorised quantities with all coefficients set to 0

    EiMatrix4d matrix_camera_to_world = EiMatrix4d::Zero();
    EiMatrix4d matrix_world_to_camera = EiMatrix4d::Zero();
    Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor> matrix_pixel_spacing = Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>::Zero();
    EiVector3d viewport_upper_left = EiVector3d::Zero();
    EiVector3d pixel_00_center = EiVector3d::Zero();

private:
    void create_basis_matrices() {
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
        // Take the inverse to get the world_to_camera matrix for free
        matrix_world_to_camera = matrix_camera_to_world.inverse();
        // Create the viewport
        create_viewport(basis_vector_forward, basis_vector_right, basis_vector_up, focal_length);
    }

    void create_viewport(const EiVector3d &basis_vector_forward, const EiVector3d &basis_vector_right, const EiVector3d &basis_vector_up, const double &focal_length) {
        // Creates the viewport from the camera basis vectors and the focal length.
        // Returns pixel spacing vectors and the 0,0-positions for the pixel and the upper left corner of the viewport.
        double h_temp = std::tan(angle_vertical_view / 2.0);
        double viewport_height = 2 * h_temp * focal_length; // world units (arbitrary)
        double viewport_width = viewport_height * (image_width / image_height); // world units (arbitrary)
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
};

// Rays
class Ray {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // Required for structures using Eigen members
        Ray(EiVector3d origin, EiVector3d direction, double t_min, double t_max) : origin(origin), direction(direction), t_min(t_min), t_max(t_max) {} // constructor
        double t_max;
        double t_min;
        EiVector3d origin;
        EiVector3d direction;

        EiVector3d normalised_direction() const {
        // Normalizes the ray direction vector.
            return direction.normalized();
        }
        EiVector3d point_at_parameter(const double t) const {
        // Computes the ray parameters at given t
            return origin + t * direction;
        }
};


struct intersection_output {
    EiVector3d t_values; // size elements x 1
    RmMatrixXd plane_normals; // size elements x 3
    Eigen::ArrayXXd barycentric_coordinates; // size elements x 3
};

struct HitRecord {
    // Hit record, which is called every time we test ray for an intersection. Ultimately stores the values of the closest hits
    double t {std::numeric_limits<double>::infinity()};
    EiVector3d point_intersection {EiVector3d::Zero()};
    EiVector3d normal_surface {EiVector3d::Zero()};
    EiVector3d barycentric_coordinates {EiVector3d::Zero()};
    bool hit_anything {false};
};

void set_face_normal(const Ray &ray, EiVector3d &normal_surface) {
    // Normalises the surface normal at the intersection point and determines which way the ray hits the object. Flips the normal if it hits the back face
    normal_surface.normalize();
    if (ray.direction.dot(normal_surface) > 0.0) {
        normal_surface = -normal_surface; // Flip normal if it hits the back face
    }
}

EiVectorXd mat_cross_product(EiVectorXd &mat1, EiVectorXd &mat2) {
    // Row-wise cross product for 2 matrices (i.e., treating each row as a vector).
    // Also works for multiplying a matrix with a row vector, so the input order determines the multiplication order. Happy days.
    // Written because this otherwise can't be a one-liner like in NumPy - Eigen's cross product works only for vector types.
    if (mat1.cols() != 3 || mat2.cols() != 3) {
        std::cerr << "Error: matrices need to have exactly 3 columns to find the cross product" << std::endl;
        return {};
    }
    long long number_of_rows = mat1.rows(); // number of rows. Long long to match the type from Eigen::Index
    if (number_of_rows  != mat2.rows()) {
        if (number_of_rows == 1) {
            // Matrix 1 is a row vector, so we just won't iterate over it
            EiVectorXd cross_product_result(mat2.rows(), 3);
            EiVector3d v1_const = mat1.row(0); // It should only have one row anyway, but just to be sure
            for (int i = 0; i < mat2.rows(); i++) {
                Eigen::Vector3d v2 = mat2.row(0);
                cross_product_result.row(i) = v1_const.cross(v2);
            }
            return cross_product_result;
        }
        else if (mat2.rows() == 1) {
            // Matrix 2 is a row vector, so we just won't iterate over it
            EiVectorXd cross_product_result(number_of_rows, 3);
            EiVector3d v2_const = mat2.row(0);
            for (int i = 0; i < number_of_rows; i++) {
                Eigen::Vector3d v1 = mat1.row(0);
                cross_product_result.row(i) = v1.cross(v2_const);
            }
        }
        else {
            // Dimensional mismatch, and neither matrix is a row vector, so can't compute cross product
            std::cerr << "Error: cross product of vectors of different sizes" << std::endl;
            return {};
        }
    }
    EiVectorXd cross_product_result(number_of_rows, 3);
    for (int i = 0; i < number_of_rows; i++) {
        Eigen::Vector3d v1 = mat1.row(i);
        Eigen::Vector3d v2 = mat2.row(i);
        cross_product_result.row(i) = v1.cross(v2);
    }
    return cross_product_result;
}

intersection_output intersect_plane(Ray &ray, RmMatrixXd nodes) {
    EiVectorXd ray_direction = ray.direction;
    EiVector3d ray_origin = ray.origin;

    long long number_of_elements = nodes.rows(); // number of rows = number of triangles, will give us indices for some bits
    Eigen::ArrayXXd barycentric_coordinates(number_of_elements, 3);
    EiVectorXd plane_normals(number_of_elements, 3);

    // Define default negative output if there is no intersection
    intersection_output negative_output {
        EiVectorXd::Constant(number_of_elements, 1, std::numeric_limits<double>::infinity()), EiVectorXd::Zero(number_of_elements, 3), Eigen::ArrayXXd(number_of_elements, 3)};

    EiVectorXd edge0 = nodes.block(0, 3, nodes.rows(), 3) - nodes.block(0, 0, nodes.rows(), 3);
    EiVectorXd edge1 = nodes.block(0, 6, nodes.rows(), 3) - nodes.block(0, 3, nodes.rows(), 3);
    // Edge 2 = node(0) - node(2), but since we always use its negative value in calculations, calculate nEdge2 = node(2) - node(0)
    EiVectorXd nEdge2 = nodes.block(0, 6, nodes.rows(), 3) - nodes.block(0, 0, nodes.rows(), 3);

    plane_normals = mat_cross_product(edge0, nEdge2); // not normalised!

    // Step 1: Quantities for the Moller Trumbore method
    EiVectorXd p_vec = mat_cross_product(ray_direction, edge1);
    EiVectorXd determinants = edge0.cwiseProduct(p_vec);

    // Step 2: Culling.
    //Determinant negative -> triangle is back-facing. If det is close to 0, ray misses the triangle.
    // If determinant is close to 0, ray and triangle are parallel and ray misses the triangle.

    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask = (determinants.array() > 1e-6) && (determinants.array() > 0);
    if (!valid_mask.any()) {
        return negative_output; // No intersection - return infinity
    }

    EiVectorXd inverse_determinants = determinants.inverse();
    EiVectorXd t_vec = ray_origin - nodes.block(0, 0, nodes.rows(), 3);

    EiVector3d barycentric_u = t_vec.cwiseProduct(p_vec) * inverse_determinants;


    valid_mask = (barycentric_u.array() >= 0) && (barycentric_u.array() <= 1);
    if (!valid_mask.any()) {
        return negative_output; //  No intersection - return infinity
    }

    EiVectorXd q_vec = mat_cross_product(t_vec, edge0);
    EiVector3d barycentric_v = ray_direction.cwiseProduct(q_vec) * inverse_determinants;

    // need to add condition for barycentric_v < 0 or barycentric_u + barycentric_v > 1
    /*
    valid_mask &= (t_values >= ray.t_min) & (t_values <= ray.t_max)
    t_values[~valid_mask] = np.inf # Set invalid values to infinity
    */
    valid_mask = (barycentric_v.array() >= 0) && (barycentric_u.array() + barycentric_v.array() <= 1);
    if (!valid_mask.any()) {
        return negative_output; // No intersection - return infinity)
    }

    EiVectorXd t_values = nEdge2.cwiseProduct(q_vec) * inverse_determinants; // t value for the ray intersections
    valid_mask = (t_values.array() >= ray.t_min) && (t_values.array() <= ray.t_max);
    // Iterate through all t_values and set them to infinity if they don't satisfy the conditions imposed by the mask
    // add contidion for t_values > ray min and t_values < t_max
    for (int i = 0; i < t_values.rows(); ++i) {
        for (int j = 0; j < t_values.cols(); ++j) {
            if (!valid_mask(i, j)) {
                t_values(i, j) = std::numeric_limits<double>::infinity();
            }
        }
    }

    EiVector3d barycentric_w = barycentric_u - barycentric_v;
    barycentric_coordinates.col(0) = barycentric_u;
    barycentric_coordinates.col(1) = barycentric_v;
    barycentric_coordinates.col(2) = barycentric_w;

    return intersection_output{t_values, plane_normals, barycentric_coordinates};
}



EiVector3d return_ray_color(Ray &ray) {
    RmMatrixXd node_coords_test(2,9);
    node_coords_test.row(0) << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0;
    node_coords_test.row(1) <<  0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
// Returns the color for a given ray. If the ray intersects an object, return colour. Otherwise, return blue sky gradient.
    HitRecord intersection_record {ray.t_max}; // Create HitRecord struct, but only set t=tmax, the rest can be default.
    // Find all intersections at once
    // HOW DO I DO THISSSSSSSSSSS I'm overwhelmed, anyway, will have to return struct bc else I can't return multiple values in CPP. Sad times.
    intersection_output intersected_results = intersect_plane(ray, node_coords_test);
    // CURRENTLY WIP
    // Blue sky gradient
    double a = 0.5 * (ray.normalised_direction()(1) + 1.0);
    EiVector3d color = (1.0 - a) * (EiVector3d() << 1.0, 1.0, 1.0).finished() + a * (EiVector3d() << 0.5, 0.7, 1.0).finished();
    return color;
}

void render_ppm_image(const Camera& camera1) {
    std::ofstream image_file;
    image_file.open("test.ppm");
    image_file << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            EiVector3d pixel_center = camera1.pixel_00_center + i * camera1.matrix_pixel_spacing.row(0) + j * camera1.matrix_pixel_spacing.row(1);
            EiVector3d pixel_color = EiVector3d::Zero(); // update this
            EiVector3d ray_direction = pixel_center - camera1.camera_center;
            Ray current_ray = Ray(camera1.camera_center, ray_direction, 0, std::numeric_limits<double>::infinity());
            pixel_color = return_ray_color(current_ray); // write this
            // Get the RGB components of the pixel color (in [0,1] range) and convert them to a single-channel grayscale
            double gray = 0.2126 * pixel_color[0] + 0.7152 * pixel_color[1] + 0.0722 * pixel_color[2];
            int gray_byte = int(gray * 255.99);
            image_file << gray_byte << ' ' << gray_byte << ' ' << gray_byte << '\n';
        }
    }
    image_file.close();
    std::cout << "\r Done. \n";
}


int main() {
    Camera camera1;
    // Sample mesh data in the input format after flattening the NumPy array
    RmMatrixXd node_coords_test(2,9);
    node_coords_test.row(0) << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0;
    node_coords_test.row(1) <<  0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    render_ppm_image(camera1);
    //std::cout << camera1.matrix_camera_to_world << std::endl << camera1.pixel_00_center;
    //Ray test_ray((EiVector3d() << -0.5, 1.1, 1.1).finished(), (EiVector3d() << 4.132331920978222, -2.603127666416139, 1.1937133836332001).finished(), 0, std::numeric_limits<double>::infinity());
   // std::cout << test_ray.direction;
    return 0;
}