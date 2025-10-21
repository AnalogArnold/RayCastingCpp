#include <cmath>
#include <fstream>
#include <iostream>
#include <fstream>
#include <limits>
#include "./Eigen/Dense" // Since pyvale uses Eigen, might as well use their implementation of a 3D vector at compile time? I saw it being used in pyvale (dicsmooth) anyway
#include <array>
#include <iostream>
#include <vector>
#include <chrono>

inline double degreesToRadians(double angleDeg) {
    // Converts degrees to radians. Used to convert the angle of vertical view.
    return angleDeg * M_PI / 180;
}

// Define aliases for the vectors and matrices from Eigen library.
// Can't use the convenience typedefs like Matrix4d or Vector3d because everything in Eigen is column-major, whereas
// C++, NumPy, and ScratchAPixel all use the row-major, so equations would be different and I don't want to tamper with that.
using EiMatrix4d = Eigen::Matrix<double, 4, 4, Eigen::StorageOptions::RowMajor>; // Shape (4,4)
using EiVector3d = Eigen::Matrix<double, 1, 3, Eigen::StorageOptions::RowMajor>; // Vector; shape (3)
using EiMatrixDd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; // Dynamic-size matrix (Dd = dynamic double)
using EiVectorD3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>; // Matrix shaped (D,3); mostly for coordinates to avoid having to loop constantly in the intersection code to get cross products etc. Think coordinates stacked together

//////////////////////////////////// INPUT
double aspect_ratio = 16.0/9.0;
unsigned short image_width = 400; // px
unsigned short image_height = static_cast<unsigned short>(image_width / aspect_ratio); // px
unsigned short number_of_samples; // For anti-aliasing. Really don't expect we'll need more than a short

//////////////////////////////////// CAMERA
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
        matrix_camera_to_world(3,3) = 1.0;
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
};

//////////////////////////////////// RAYS
struct Ray {
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // Required for structures using Eigen members
    //Eigen::Matrix<double, 3, 1, Eigen::StorageOptions::RowMajor, Eigen:: Aligned8> origin;
    EiVector3d origin;
    EiVector3d direction;
    double t_min;
    double t_max {std::numeric_limits<double>::infinity()};
};

inline EiVector3d ray_at_t(const double t, const Ray &ray) {
    return ray.origin + t * ray.direction;
};
// return direction.normalized(); // for normalizing ray direction; can keep it inline, just have it here so I don't forget it's an option

//////////////////////////////////// HIT RECORD
struct HitRecord {
    // Hit record, which is called every time we test ray for an intersection. Ultimately stores the values of the closest hits
    double t {std::numeric_limits<double>::infinity()};
    EiVector3d point_intersection {EiVector3d::Zero()};
    EiVector3d normal_surface {EiVector3d::Zero()};
    EiVector3d barycentric_coordinates {EiVector3d::Zero()};
};

inline void set_face_normal(const Ray &ray, EiVector3d &normal_surface) {
    // Normalises the surface normal at the intersection point and determines which way the ray hits the object. Flips the normal if it hits the back face
    normal_surface = normal_surface.normalized();
    if (ray.direction.dot(normal_surface) > 0.0) {
        normal_surface = -normal_surface; // Flip normal if it hits the back face
    }
}

//////////////////////////////////// CROSS-PRODUCT
EiVectorD3d cross_rowwise(const EiVectorD3d &mat1, const EiVectorD3d &mat2) {
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
            EiVectorD3d cross_product_result(mat2.rows(), 3);
            EiVector3d v1_const = mat1.row(0); // It should only have one row anyway, but just to be sure
            for (int i = 0; i < mat2.rows(); i++) {
                Eigen::Vector3d v2 = mat2.row(i);
                cross_product_result.row(i) = v1_const.cross(v2);
            }
            return cross_product_result;
        }
        else if (mat2.rows() == 1) {
            // Matrix 2 is a row vector, so we just won't iterate over it
            EiVectorD3d cross_product_result(number_of_rows, 3);
            EiVector3d v2_const = mat2.row(0);
            for (int i = 0; i < number_of_rows; i++) {
                Eigen::Vector3d v1 = mat1.row(i);
                cross_product_result.row(i) = v1.cross(v2_const);
            }
            return cross_product_result;
        }
        else {
            // Dimensional mismatch, and neither matrix is a row vector, so can't compute cross product
            std::cerr << "Error: cross product of vectors of different sizes" << std::endl;
            return {};
        }
    }
    EiVectorD3d cross_product_result(number_of_rows, 3);
    for (int i = 0; i < number_of_rows; i++) {
        Eigen::Vector3d v1 = mat1.row(i);
        Eigen::Vector3d v2 = mat2.row(i);
        cross_product_result.row(i) = v1.cross(v2);
    }
    return cross_product_result;
}

//////////////////////////////////// INTERSECTION
struct IntersectionOutput {
    Eigen::ArrayXXd barycentric_coordinates; // size elements x 3. Array because I'll be interested in using it element-wise
    EiVectorD3d plane_normals; // size elements x 3
    Eigen::Vector<double, Eigen::Dynamic> t_values; // size elements x 1
};

int counter_con1 = 0;
int counter_con2 = 0;
int counter_con3 = 0;
int counter_pass = 0;
int total_counter = 0;

IntersectionOutput intersect_plane(const Ray &ray, EiMatrixDd nodes) {
//IntersectionOutput intersect_plane(const Ray &ray, double (&nodes)[44][9]) {
    // Declare everything on the top because else I get very confused
    //long long number_of_elements = nodes.rows(); // number of rows = number of triangles, will give us indices for some bits
    long long number_of_elements = 44;
    // Ray data
    EiVector3d ray_direction = ray.direction.normalized(); // TROUBLESHOOTING: forgot to normalize this...
    EiVector3d ray_origin = ray.origin; // shape (3,1)
    // Broadcasted to use in vectorised operations on matrices
    EiVectorD3d ray_directions = ray_direction.replicate(number_of_elements, 1);
    EiVectorD3d ray_origins = ray_origin.replicate(number_of_elements, 1);
    // std::cout << "ray_direction: " << ray_directions << std::endl; // Broadcasting works as expected
    // Edges
    EiMatrixDd edge0 (number_of_elements,3), edge1 (number_of_elements,3), nEdge2 (number_of_elements,3), nodes0; // shape (faces, 3) each
    //EiVectorD3d edge0, edge1, nEdge2; // shape (faces, 3) each
    // Intersections and barycentric coordinates
    EiVectorD3d p_vec, t_vec, q_vec; // shape (faces, 3) each
    EiVectorD3d plane_normals; // Shape (faces, 3)
    Eigen::Vector<double, Eigen::Dynamic> determinants, inverse_determinants; //shape (faces, 1) each
    Eigen::Vector<double, Eigen::Dynamic> barycentric_u, barycentric_v, barycentric_w; // shape (faces, 1) each
    Eigen::ArrayXXd barycentric_coordinates(number_of_elements, 3); // shape (faces, 3) Array so we can do things element-wise with those
    Eigen::Vector<double, Eigen::Dynamic> t_values; // Shape (faces, 1)
    /// Define default negative output if there is no intersection
    IntersectionOutput negative_output {
        Eigen::ArrayXXd(number_of_elements, 3),
        EiVectorD3d::Zero(number_of_elements, 3),
        Eigen::Vector<double, Eigen::Dynamic>::Constant(number_of_elements, 1, std::numeric_limits<double>::infinity())
    };

    //double edge0_arr[44][3];
    //double edge1_arr[44][3];
    //double nEdge2_arr[44][3];
    /*
    for (int i = 0; i < 44; i++) {
        for (int j = 0; j < 3; j++) {
            edge0(i,j) = nodes[i][j+3] - nodes[i][j];
            nodes0(i,j) = nodes[i][j];
            // edge1(i,j) = nodes[i][j+6] - nodes[i][j+3]; // We don't really use it in calculations, so keep it here... because, just don't waste memory on this
            nEdge2(i,j) = nodes[i][j+6] - nodes[i][j];
            //edge0_arr[i][j] = nodes[i][j+3] - nodes[i][j];
            //edge1_arr[i][j] = nodes[i][j+6] - nodes[i][j+3];
            //nEdge2_arr[i][j] = nodes[i][j+6] - nodes[i][j];
            //std::cout << edge0_arr[i][j]<< std::endl;
        }
    }
*/

    // Calculations - edges and normals
    total_counter ++;
    edge0 = nodes.block(0, 3, nodes.rows(), 3) - nodes.block(0, 0, nodes.rows(), 3);
    edge1 = nodes.block(0, 6, nodes.rows(), 3)  - nodes.block(0, 3, nodes.rows(), 3);
    nEdge2 = nodes.block(0, 6, nodes.rows(), 3) - nodes.block(0, 0, nodes.rows(), 3);
    plane_normals = cross_rowwise(edge0, nEdge2); // not normalised!
    //std::cout << edge0.row(0) << std::endl; // All calculated properly
    //std::cout << edge1.row(0) << std::endl;
    //std::cout <<nEdge2.row(0) << std::endl;
    // Step 1: Quantities for the Moller Trumbore method
    p_vec = cross_rowwise(ray_directions, nEdge2); // To my very own surprise, my function gives the correct output
    //std::cout << p_vec.array() << std::endl;
    determinants = (edge0.array() * p_vec.array()).rowwise().sum(); // Row-wise dot product // Correct
    if (total_counter == 1) {
        std::cout<<determinants<<std::endl; // this is already coming out wrong, even though it was fine previously. Curious.
    }
    //std::cout << "determinants: " << determinants.transpose() << std::endl;
    // Step 2: Culling.
    //Determinant negative -> triangle is back-facing. If det is close to 0, ray misses the triangle.
    // If determinant is close to 0, ray and triangle are parallel and ray misses the triangle.

    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask = (determinants.array() > 1e-6) && (determinants.array() > 0);
    //std::cout << "valid_mask: " << valid_mask.transpose() << std::endl;
    if (!valid_mask.any()) {
        //std::cout << "Condition 1 triggered" << std::endl;
        counter_con1++;
        return negative_output; // No intersection - return infinity
    }

    // Step 3: Test if ray is in front of the triangle
    inverse_determinants = determinants.array().inverse(); // Element-wise inverse. Correct
    //std::cout << "inverse_determinants: " << inverse_determinants.transpose() << std::endl;
    //t_vec = ray_origins.array() - nodes0.array();
    t_vec = ray_origins.array() - nodes.block(0, 0, nodes.rows(), 3).array(); // is ok, may return 0s depending on data passed, but tested with moving the camera center and works.
    //std::cout << "orig: " << ray_origin << std::endl;
    //std::cout << "block: " << nodes.block(0, 0, 1, 3) << std::endl;
    //std::cout << "t_vec: " << t_vec << std::endl; // t_vec is correct
    barycentric_u = ((t_vec.array() * p_vec.array()).rowwise().sum()).array() * inverse_determinants.array(); // comes out the same as benchmark
    //std::cout << barycentric_u << std::endl;
    // Check barycentric_u
    //valid_mask = valid_mask && (barycentric_u.array() >= 0) && (barycentric_u.array() <= 1);
    valid_mask = valid_mask && (barycentric_u.array() >= 0) && (barycentric_u.array() <= 1);
    if (!valid_mask.any()) {
        counter_con2++;
        //std::cout << "Condition 2 triggered" << std::endl;
        return negative_output; // No intersection - return infinity
    }

    q_vec = cross_rowwise(t_vec, edge0); // comes out like in Python. Happy days
    //std::cout << q_vec << std::endl;
    barycentric_v = (ray_directions.array() * q_vec.array()).rowwise().sum().matrix().array() * inverse_determinants.array(); // Comes out correctly
    // Check barycentric_v and sum
    //std::cout<<barycentric_v<<std::endl;
    valid_mask = valid_mask && (barycentric_v.array() >= 0) && ((barycentric_u.array() + barycentric_v.array()) <= 1);
    // t values
    t_values = (nEdge2.array() * q_vec.array()).rowwise().sum().matrix().array() * inverse_determinants.array(); // Same as Python
    //std::cout << "t_values: " << t_values << std::endl;
    valid_mask = valid_mask && (t_values.array() >= ray.t_min) && (t_values.array() <= ray.t_max);
    // Iterate through all t_values and set them to infinity if they don't satisfy the conditions imposed by the mask
    for (int i = 0; i < t_values.rows(); ++i) {
        for (int j = 0; j < t_values.cols(); ++j) {
            if (!valid_mask(i, j)) {
                t_values(i, j) = std::numeric_limits<double>::infinity();
                counter_con3++;
            }
        }
    }
    counter_pass++;
    barycentric_w = 1.0 - barycentric_u.array() - barycentric_v.array(); // Comes out the same as Python
    barycentric_coordinates.col(0) = barycentric_u;
    barycentric_coordinates.col(1) = barycentric_v;
    barycentric_coordinates.col(2) = barycentric_w;
    //std::cout << "barycentric_coordinates: " << barycentric_coordinates << std::endl; // Also same as Python. Troubleshooting of this section should be complete.
    return IntersectionOutput{barycentric_coordinates, plane_normals, t_values};

}

//////////////////////////////////// COLOR RAY
EiVector3d return_ray_color(const Ray &ray) {
// Returns the color for a given ray. If the ray intersects an object, return colour. Otherwise, return blue sky gradient.
    double node_coords_arr[2][9] = {1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0};

   // EiMatrixDd node_coords_test(2,9);
    EiVectorD3d color_test(3,3);
    color_test.row(0) << 1.0, 0.0, 0.0;
    color_test.row(1) << 0.0, 1.0, 1.0;
    color_test.row(2) << 1.0, 0.0, 1.0;
    //node_coords_test.row(0) << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0;
    //node_coords_test.row(1) <<  0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
     double node_coords_test1[44][9] = {
    0.0,0.0,0.0,0.0,0.49999999999999983,0.0,0.49999999999999994,0.0,0.0,
    0.0,0.0,0.0,0.0,0.49999999999999983,0.5,0.0,0.49999999999999983,0.0,
    0.0,0.0,0.0,0.49999999999999994,0.0,0.0,0.0,0.0,0.5,
    0.0,0.0,0.0,0.0,0.0,0.5,0.0,0.49999999999999983,0.5,
    0.49999999999999994,0.0,0.0,0.49999999999999994,0.0,0.5,0.0,0.0,0.5,
    0.49999999999999994,0.0,0.0,0.0,0.49999999999999983,0.0,0.5,0.49999999999999983,0.0,
    0.49999999999999994,0.0,0.0,0.5,0.49999999999999983,0.0,1.0,0.0,0.0,
    0.49999999999999994,0.0,0.0,1.0,0.0,0.0,0.49999999999999994,0.0,0.5,
    0.0,0.49999999999999983,0.0,0.0,1.0,0.0,0.5,0.49999999999999983,0.0,
    0.0,0.49999999999999983,0.0,0.0,1.0,0.5,0.0,1.0,0.0,
    0.0,0.49999999999999983,0.0,0.0,0.49999999999999983,0.5,0.0,1.0,0.5,
    0.0,0.49999999999999983,0.5,0.0,0.0,0.5,0.49999999999999994,0.0,0.5,
    0.0,0.49999999999999983,0.5,0.49999999999999994,0.0,0.5,0.5,0.49999999999999983,0.5,
    0.0,0.49999999999999983,0.5,0.5,0.49999999999999983,0.5,0.0,1.0,0.5,
    0.49999999999999994,0.0,0.5,1.0,0.0,0.0,1.0,0.0,0.5,
    0.49999999999999994,0.0,0.5,1.0,0.0,0.5,0.5,0.49999999999999983,0.5,
    0.5,0.49999999999999983,0.0,0.0,1.0,0.0,0.5,1.0,0.0,
    0.5,0.49999999999999983,0.0,1.0,0.49999999999999994,0.0,1.0,0.0,0.0,
    0.5,0.49999999999999983,0.0,0.5,1.0,0.0,1.0,0.49999999999999994,0.0,
    0.5,0.49999999999999983,0.5,0.5,1.0,0.5,0.0,1.0,0.5,
    0.5,0.49999999999999983,0.5,1.0,0.0,0.5,1.0,0.49999999999999994,0.5,
    0.5,0.49999999999999983,0.5,1.0,0.49999999999999994,0.5,0.5,1.0,0.5,
    0.0,1.0,0.0,0.0,1.5,0.0,0.5,1.0,0.0,
    0.0,1.0,0.0,0.0,1.5,0.5,0.0,1.5,0.0,
    0.0,1.0,0.0,0.0,1.0,0.5,0.0,1.5,0.5,
    0.0,1.0,0.5,0.5,1.0,0.5,0.0,1.5,0.5,
    0.5,1.0,0.0,0.0,1.5,0.0,0.5000000000000001,1.5,0.0,
    0.5,1.0,0.0,1.0,1.0,0.0,1.0,0.49999999999999994,0.0,
    0.5,1.0,0.0,0.5000000000000001,1.5,0.0,1.0,1.0,0.0,
    0.5,1.0,0.5,0.5000000000000001,1.5,0.5,0.0,1.5,0.5,
    0.5,1.0,0.5,1.0,0.49999999999999994,0.5,1.0,1.0,0.5,
    0.5,1.0,0.5,1.0,1.0,0.5,0.5000000000000001,1.5,0.5,
    0.0,1.5,0.0,0.5000000000000001,1.5,0.5,0.5000000000000001,1.5,0.0,
    0.0,1.5,0.0,0.0,1.5,0.5,0.5000000000000001,1.5,0.5,
    0.5000000000000001,1.5,0.0,1.0,1.5,0.0,1.0,1.0,0.0,
    0.5000000000000001,1.5,0.0,1.0,1.5,0.5,1.0,1.5,0.0,
    0.5000000000000001,1.5,0.0,0.5000000000000001,1.5,0.5,1.0,1.5,0.5,
    0.5000000000000001,1.5,0.5,1.0,1.0,0.5,1.0,1.5,0.5,
    1.0,0.0,0.0,1.0,0.49999999999999994,0.0,1.0,0.49999999999999994,0.5,
    1.0,0.0,0.0,1.0,0.49999999999999994,0.5,1.0,0.0,0.5,
    1.0,0.49999999999999994,0.0,1.0,1.0,0.0,1.0,1.0,0.5,
    1.0,0.49999999999999994,0.0,1.0,1.0,0.5,1.0,0.49999999999999994,0.5,
    1.0,1.0,0.0,1.0,1.5,0.0,1.0,1.5,0.5,
    1.0,1.0,0.0,1.0,1.5,0.5,1.0,1.0,0.5};
    EiMatrixDd node_coords_test(44,9);
    for (int i=0; i < 44; i++) {
        for (int j=0; j < 9; j++) {
            node_coords_test(i,j) = node_coords_arr[i][j];
        }}
    HitRecord intersection_record; // Create HitRecord struct
    IntersectionOutput intersection = intersect_plane(ray, node_coords_test);
    Eigen::Index minRowIndex, minColIndex;

    intersection.t_values.minCoeff(&minRowIndex, &minColIndex); // Find indices of the smallest t_value
    double closest_t = intersection.t_values(minRowIndex, minColIndex);
    if (closest_t < intersection_record.t) {
        intersection_record.t = closest_t;
        intersection_record.barycentric_coordinates = intersection.barycentric_coordinates.row(minRowIndex);
        intersection_record.point_intersection = ray_at_t(closest_t, ray);
        intersection_record.normal_surface = intersection.plane_normals.row(minRowIndex);
    }
    if (intersection_record.t != std::numeric_limits<double>::infinity()) { // Instead of keeping a bool hit_anything, check if t value has changed from the default
        set_face_normal(ray, intersection_record.normal_surface);
        std::cout<<"hit sth"<<std::endl;
        return intersection_record.barycentric_coordinates(0) * color_test.row(0) + intersection_record.barycentric_coordinates(1) * color_test.row(2) + intersection_record.barycentric_coordinates(2) * color_test.row(2);
        //return color
    }
    // Blue sky gradient
    double a = 0.5 * (ray.direction.normalized()(1) + 1.0);
    EiVector3d color = (1.0 - a) * (EiVector3d() << 1.0, 1.0, 1.0).finished() + a * (EiVector3d() << 0.5, 0.7, 1.0).finished();
    return color;
}




//////////////////////////////////// RENDERING
void render_ppm_image(const Camera& camera1) {
    std::ofstream image_file;
    image_file.open("test.ppm");
    image_file << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            //EiVector3d pixel_color = EiVector3d::Zero(); // for anti-aliasing later
            EiVector3d pixel_center = camera1.pixel_00_center + i * camera1.matrix_pixel_spacing.row(0) + j * camera1.matrix_pixel_spacing.row(1);
            EiVector3d pixel_color = EiVector3d::Zero(); // update this
            EiVector3d ray_direction = pixel_center - camera1.camera_center;
            Ray current_ray {camera1.camera_center, ray_direction};
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

void edgeArray(double (&node_coords_arr)[44][9]) {
    //double node_coords_arr[2][9] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    double edge0_arr[44][3];
    double edge1_arr[44][3];
    double nEdge2_arr[44][3];
    for (int i = 0; i < 44; i++) {
        for (int j = 0; j < 3; j++) {
            edge0_arr[i][j] = node_coords_arr[i][j+3] - node_coords_arr[i][j];
            edge1_arr[i][j] = node_coords_arr[i][j+6] - node_coords_arr[i][j+3];
            nEdge2_arr[i][j] = node_coords_arr[i][j+6] - node_coords_arr[i][j];
            //std::cout << edge0_arr[i][j]<< std::endl;
        }
    }
}

void edgeEigen(EiMatrixDd &node_coords_test) {
    // Eigen edge test
    //EiMatrixDd node_coords_test(2,9);
   //node_coords_test.row(0) << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0;
    //node_coords_test.row(1) <<  0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    EiMatrixDd edge0 = node_coords_test.block(0, 3, node_coords_test.rows(), 3) - node_coords_test.block(0, 0, node_coords_test.rows(), 3);
    EiMatrixDd edge1 = node_coords_test.block(0, 6, node_coords_test.rows(), 3)  - node_coords_test.block(0, 3, node_coords_test.rows(), 3);
    EiMatrixDd nEdge2 = node_coords_test.block(0, 6, node_coords_test.rows(), 3) - node_coords_test.block(0, 0, node_coords_test.rows(), 3);
}


int main() {
    //Camera test_camera{EiVector3d(0, 1, 1), EiVector3d(0, 0, -1), 90};
    Camera test_camera{EiVector3d(-0.5, 1.1, 1.1), EiVector3d(0, 0, -1), 90};
    // Mesh from simdata. Copied by force for now.
    unsigned long long number_of_elements = 44;
    unsigned long long number_of_coords = 9;

    double node_coords_arr[44][9] = {
    0.0,0.0,0.0,0.0,0.49999999999999983,0.0,0.49999999999999994,0.0,0.0,
    0.0,0.0,0.0,0.0,0.49999999999999983,0.5,0.0,0.49999999999999983,0.0,
    0.0,0.0,0.0,0.49999999999999994,0.0,0.0,0.0,0.0,0.5,
    0.0,0.0,0.0,0.0,0.0,0.5,0.0,0.49999999999999983,0.5,
    0.49999999999999994,0.0,0.0,0.49999999999999994,0.0,0.5,0.0,0.0,0.5,
    0.49999999999999994,0.0,0.0,0.0,0.49999999999999983,0.0,0.5,0.49999999999999983,0.0,
    0.49999999999999994,0.0,0.0,0.5,0.49999999999999983,0.0,1.0,0.0,0.0,
    0.49999999999999994,0.0,0.0,1.0,0.0,0.0,0.49999999999999994,0.0,0.5,
    0.0,0.49999999999999983,0.0,0.0,1.0,0.0,0.5,0.49999999999999983,0.0,
    0.0,0.49999999999999983,0.0,0.0,1.0,0.5,0.0,1.0,0.0,
    0.0,0.49999999999999983,0.0,0.0,0.49999999999999983,0.5,0.0,1.0,0.5,
    0.0,0.49999999999999983,0.5,0.0,0.0,0.5,0.49999999999999994,0.0,0.5,
    0.0,0.49999999999999983,0.5,0.49999999999999994,0.0,0.5,0.5,0.49999999999999983,0.5,
    0.0,0.49999999999999983,0.5,0.5,0.49999999999999983,0.5,0.0,1.0,0.5,
    0.49999999999999994,0.0,0.5,1.0,0.0,0.0,1.0,0.0,0.5,
    0.49999999999999994,0.0,0.5,1.0,0.0,0.5,0.5,0.49999999999999983,0.5,
    0.5,0.49999999999999983,0.0,0.0,1.0,0.0,0.5,1.0,0.0,
    0.5,0.49999999999999983,0.0,1.0,0.49999999999999994,0.0,1.0,0.0,0.0,
    0.5,0.49999999999999983,0.0,0.5,1.0,0.0,1.0,0.49999999999999994,0.0,
    0.5,0.49999999999999983,0.5,0.5,1.0,0.5,0.0,1.0,0.5,
    0.5,0.49999999999999983,0.5,1.0,0.0,0.5,1.0,0.49999999999999994,0.5,
    0.5,0.49999999999999983,0.5,1.0,0.49999999999999994,0.5,0.5,1.0,0.5,
    0.0,1.0,0.0,0.0,1.5,0.0,0.5,1.0,0.0,
    0.0,1.0,0.0,0.0,1.5,0.5,0.0,1.5,0.0,
    0.0,1.0,0.0,0.0,1.0,0.5,0.0,1.5,0.5,
    0.0,1.0,0.5,0.5,1.0,0.5,0.0,1.5,0.5,
    0.5,1.0,0.0,0.0,1.5,0.0,0.5000000000000001,1.5,0.0,
    0.5,1.0,0.0,1.0,1.0,0.0,1.0,0.49999999999999994,0.0,
    0.5,1.0,0.0,0.5000000000000001,1.5,0.0,1.0,1.0,0.0,
    0.5,1.0,0.5,0.5000000000000001,1.5,0.5,0.0,1.5,0.5,
    0.5,1.0,0.5,1.0,0.49999999999999994,0.5,1.0,1.0,0.5,
    0.5,1.0,0.5,1.0,1.0,0.5,0.5000000000000001,1.5,0.5,
    0.0,1.5,0.0,0.5000000000000001,1.5,0.5,0.5000000000000001,1.5,0.0,
    0.0,1.5,0.0,0.0,1.5,0.5,0.5000000000000001,1.5,0.5,
    0.5000000000000001,1.5,0.0,1.0,1.5,0.0,1.0,1.0,0.0,
    0.5000000000000001,1.5,0.0,1.0,1.5,0.5,1.0,1.5,0.0,
    0.5000000000000001,1.5,0.0,0.5000000000000001,1.5,0.5,1.0,1.5,0.5,
    0.5000000000000001,1.5,0.5,1.0,1.0,0.5,1.0,1.5,0.5,
    1.0,0.0,0.0,1.0,0.49999999999999994,0.0,1.0,0.49999999999999994,0.5,
    1.0,0.0,0.0,1.0,0.49999999999999994,0.5,1.0,0.0,0.5,
    1.0,0.49999999999999994,0.0,1.0,1.0,0.0,1.0,1.0,0.5,
    1.0,0.49999999999999994,0.0,1.0,1.0,0.5,1.0,0.49999999999999994,0.5,
    1.0,1.0,0.0,1.0,1.5,0.0,1.0,1.5,0.5,
    1.0,1.0,0.0,1.0,1.5,0.5,1.0,1.0,0.5};
    EiMatrixDd node_coords_test(44,9);
    for (int i=0; i < 44; i++) {
        for (int j=0; j < 9; j++) {
            node_coords_test(i,j) = node_coords_arr[i][j];
        }}

    //Camera camera1;
    // Sample data for testing
    //Ray test_ray{EiVector3d(-0.5, 1.1, 1.1), EiVector3d(4.132331920978222, -2.603127666416139, 1.1937133836332001), 0.0};
    //EiMatrixDd node_coords_test(2,9);
    //node_coords_test.row(0) << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0;
    //node_coords_test.row(1) <<  0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    //intersect_plane(test_ray, node_coords_test);

    render_ppm_image(test_camera);
    std::cout<<"Counter_con1: " << counter_con1 << "counter_con2: " << counter_con2 << "counter_con3: " << counter_con3 << "counter_pass: " << counter_pass << "total_counter:" << total_counter <<std::endl;

    // Code for timing
    /*
    std::chrono::high_resolution_clock::time_point begin1 = std::chrono::high_resolution_clock::now();
    edgeArray(node_coords_arr);
    std::chrono::high_resolution_clock::time_point end1 = std::chrono::high_resolution_clock::now();
    std::cout << "Array runtime: " << std::chrono::duration_cast<std::chrono::nanoseconds> (end1 - begin1) << std::endl;

    std::chrono::high_resolution_clock::time_point begin2 = std::chrono::high_resolution_clock::now();
    edgeEigen(node_coords_test);
    std::chrono::high_resolution_clock::time_point end2 = std::chrono::high_resolution_clock::now();
    std::cout << "Eigen runtime: " << std::chrono::duration_cast<std::chrono::nanoseconds> (end2 - begin2) << std::endl;
    */


    return 0;
}




