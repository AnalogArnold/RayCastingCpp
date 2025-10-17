#include <cmath>
#include <fstream>
#include <iostream>
#include <fstream>
#include <limits>
#include "./Eigen/Dense" // Since pyvale uses Eigen, might as well use their implementation of a 3D vector at compile time? I saw it being used in pyvale (dicsmooth) anyway

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
unsigned short image_width = 400; // px
double aspect_ratio = 16.0/9.0;
unsigned short image_height = static_cast<unsigned short>(image_width / aspect_ratio); // px
unsigned short number_of_samples; // For anti-aliasing. Really don't expect we'll need more than a short
//double camera_center[] = {-0.5, 1.1, 1.1};
//double camera_target[] = {0, 0, -1};
//double angle_vertical_view = 90.0; // degrees


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

//////////////////////////////////// RAYS
struct Ray {
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // Required for structures using Eigen members
    //Eigen::Matrix<double, 3, 1, Eigen::StorageOptions::RowMajor, Eigen:: Aligned8> origin; // might have to experiment with this later to make Rays smaller as atm the vectors themselves take up 48 bytes
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
EiVectorD3d mat_cross_product(const EiVectorD3d &mat1, const EiVectorD3d &mat2) {
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
    Eigen::Vector<double, Eigen::Dynamic> t_values; // size elements x 1
    EiVectorD3d plane_normals; // size elements x 3
    Eigen::ArrayXXd barycentric_coordinates; // size elements x 3
};

IntersectionOutput intersect_plane(const Ray &ray, EiMatrixDd nodes) {
    // Declare everything on the top because else I get very confused
    long long number_of_elements = nodes.rows(); // number of rows = number of triangles, will give us indices for some bits
    // Ray data
    EiVector3d ray_direction = ray.direction;
    EiVector3d ray_origin = ray.origin; // shape (3,1)
    // Broadcasted to use in vectorised operations on matrices
    EiVectorD3d ray_directions = ray_direction.replicate(number_of_elements, 1);
    EiVectorD3d ray_origins = ray_origin.replicate(number_of_elements, 1);
    // Edges
    EiVectorD3d edge0, edge1, nEdge2; // shape (faces, 3) each
    // Intersections and barycentric coordinates
    EiVectorD3d p_vec, t_vec, q_vec; // shape (faces, 3) each
    EiVectorD3d plane_normals; // Shape (faces, 3)
    Eigen::Vector<double, Eigen::Dynamic> determinants, inverse_determinants; //shape (faces, 1) each
    Eigen::Vector<double, Eigen::Dynamic> barycentric_u, barycentric_v, barycentric_w; // shape (faces, 1) each
    Eigen::ArrayXXd barycentric_coordinates(number_of_elements, 3); // shape (faces, 3) Array so we can do things element-wise with those
    Eigen::Vector<double, Eigen::Dynamic> t_values; // Shape (faces, 1)
    /// Define default negative output if there is no intersection
    IntersectionOutput negative_output {
        Eigen::Vector<double, Eigen::Dynamic>::Constant(number_of_elements, 1, std::numeric_limits<double>::infinity()),
        EiVectorD3d::Zero(number_of_elements, 3),
        Eigen::ArrayXXd(number_of_elements, 3)};

    // Calculations - edges and normals
    edge0 = nodes.block(0, 3, nodes.rows(), 3) - nodes.block(0, 0, nodes.rows(), 3);
    edge1 = nodes.block(0, 6, nodes.rows(), 3) - nodes.block(0, 3, nodes.rows(), 3);
    nEdge2 = nodes.block(0, 6, nodes.rows(), 3) - nodes.block(0, 0, nodes.rows(), 3);
    plane_normals = mat_cross_product(edge0, nEdge2); // not normalised!

    // Step 1: Quantities for the Moller Trumbore method
    p_vec = mat_cross_product(ray_directions, edge1);
    determinants = (edge0.array() * p_vec.array()).rowwise().sum(); // Row-wise dot product

    // Step 2: Culling.
    //Determinant negative -> triangle is back-facing. If det is close to 0, ray misses the triangle.
    // If determinant is close to 0, ray and triangle are parallel and ray misses the triangle.
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask = determinants.array().abs() > 1e-6;
    if (!valid_mask.any()) {
        return negative_output; // No intersection - return infinity
    }
    // Step 3: Test if ray is in front of the triangle
    inverse_determinants = determinants.array().inverse(); // Element-wise inverse
    t_vec = ray_origins - nodes.block(0, 0, nodes.rows(), 3);
    barycentric_u = (t_vec.array() * p_vec.array()).rowwise().sum().matrix().array() * inverse_determinants.array();    // need to add condition for barycentric_v < 0 or barycentric_u + barycentric_v > 1
    // Check barycentric_u
    valid_mask = valid_mask && (barycentric_u.array() >= 0) && (barycentric_u.array() <= 1);
    if (!valid_mask.any()) {
        return negative_output; // No intersection - return infinity
    }
    q_vec = mat_cross_product(t_vec, edge0);
    barycentric_v = (ray_directions.array() * q_vec.array()).rowwise().sum().matrix().array() * inverse_determinants.array();
    // Check barycentric_v and sum
    valid_mask = valid_mask && (barycentric_v.array() >= 0) && ((barycentric_u.array() + barycentric_v.array()) <= 1);
    if (!valid_mask.any()) {
        return negative_output; // No intersection - return infinity
    }
    // t values
    t_values = (nEdge2.array() * q_vec.array()).rowwise().sum().matrix().array() * inverse_determinants.array();
    valid_mask = (t_values.array() >= ray.t_min) && (t_values.array() <= ray.t_max);
    // Iterate through all t_values and set them to infinity if they don't satisfy the conditions imposed by the mask
    for (int i = 0; i < t_values.rows(); ++i) {
        for (int j = 0; j < t_values.cols(); ++j) {
            if (!valid_mask(i, j)) {
                t_values(i, j) = std::numeric_limits<double>::infinity();
            }
        }
    }
    barycentric_w = 1.0 - barycentric_u.array() - barycentric_v.array();
    barycentric_coordinates.col(0) = barycentric_u;
    barycentric_coordinates.col(1) = barycentric_v;
    barycentric_coordinates.col(2) = barycentric_w;
    return IntersectionOutput{t_values, plane_normals, barycentric_coordinates};

}

//////////////////////////////////// COLOR RAY
EiVector3d return_ray_color(const Ray &ray) {
// Returns the color for a given ray. If the ray intersects an object, return colour. Otherwise, return blue sky gradient.
    EiMatrixDd node_coords_test(2,9);
    EiVectorD3d color_test;
    color_test.row(0) << 1.0, 0.0, 0.0;
    color_test.row(1) << 0.0, 1.0, 1.0;
    color_test.row(2) << 1.0, 0.0, 1.0;
    node_coords_test.row(0) << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0;
    node_coords_test.row(1) <<  0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    std::cout << "nodes.rows() = " << node_coords_test.rows() << ", nodes.cols() = " << node_coords_test.cols() << std::endl;
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

int main() {
    Camera camera1;
    render_ppm_image(camera1);
    //std::cout << camera1.matrix_camera_to_world << std::endl << camera1.pixel_00_center;
    //Ray test_ray((EiVector3d() << -0.5, 1.1, 1.1).finished(), (EiVector3d() << 4.132331920978222, -2.603127666416139, 1.1937133836332001).finished(), 0, std::numeric_limits<double>::infinity());
   // std::cout << test_ray.direction;
    return 0;
}