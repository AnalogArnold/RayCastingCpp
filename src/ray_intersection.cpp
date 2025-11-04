#include <iostream>
#include <limits>
#include "../ray_intersection.h"

EiVectorD3d cross_rowwise(const EiVectorD3d &mat1, const EiVectorD3d &mat2) {
    // Row-wise cross product for 2 matrices (i.e., treating each row as a vector).
    // Also works for multiplying a matrix with a row vector, so the input order determines the multiplication order. Happy days.
    // Written because this otherwise can't be a one-liner like in NumPy - Eigen's cross product works only for vector types.
    if (mat1.cols() != 3 || mat2.cols() != 3) {
        std::cerr << "Error: matrices need to have exactly 3 columns to find the cross product" << std::endl;
        return {};
    }
    long long number_of_rows = mat1.rows(); // number of rows. Long long to match the type from Eigen::Index
    EiVectorD3d cross_product_result(number_of_rows, 3);
    cross_product_result.col(0) = mat1.col(1).cwiseProduct(mat2.col(2)) - mat1.col(2).cwiseProduct(mat2.col(1));
    cross_product_result.col(1) = mat1.col(2).cwiseProduct(mat2.col(0)) - mat1.col(0).cwiseProduct(mat2.col(2));
    cross_product_result.col(2) = mat1.col(0).cwiseProduct(mat2.col(1)) - mat1.col(1).cwiseProduct(mat2.col(0));
    return cross_product_result;
}

//IntersectionOutput intersect_plane(const Ray &ray, const double (&node_coords_arr)[44][9]) {
IntersectionOutput intersect_plane(const Ray &ray,
                                   const std::vector<std::array<int,3>> &connectivity,
                                   const std::vector<std::array<double,3>> &node_coords) {
    // Declare everything on the top because else I get very confused
    long long number_of_elements = connectivity.size(); // number of triangles, will give us indices for some bits
    // Ray data broadcasted to use in vectorised operations on matrices
    // This is faster than doing it in a loop
    EiVectorD3d ray_directions = ray.direction.replicate(number_of_elements, 1);
    EiArrayD3d ray_origins = ray.origin.replicate(number_of_elements, 1).array();
    // Edges
    EiMatrixDd edge0 (number_of_elements,3), nEdge2 (number_of_elements,3); // shape (faces, 3) each
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>  nodes0 (number_of_elements, 3);
    // Intersections and barycentric coordinates
    EiArrayD3d p_vec, q_vec, t_vec; // shape (faces, 3) each
    EiVectorD3d plane_normals; // Shape (faces, 3)
    Eigen::Array<double, Eigen::Dynamic, 1> determinants, inverse_determinants; //shape (faces, 1) each
    Eigen::Array<double, Eigen::Dynamic, 1> barycentric_u, barycentric_v, t_values; // shape (faces, 1) each
    Eigen::ArrayXXd barycentric_coordinates(number_of_elements, 3); // shape (faces, 3) Array so we can do things element-wise with those
    /// Define default negative output if there is no intersection
    IntersectionOutput negative_output {
        Eigen::ArrayXXd(number_of_elements, 3),
        EiVectorD3d::Zero(number_of_elements, 3),
        Eigen::Vector<double, Eigen::Dynamic>::Constant(number_of_elements, 1, std::numeric_limits<double>::infinity())
    };
    // Calculations - edges and normals
    for (int i=0; i < number_of_elements; i++) {
        int node_0 = connectivity[i][0];
        int node_1 = connectivity[i][1];
        int node_2 = connectivity[i][2];
        for (int j=0; j < 3; j++) {
            //std::cout<<node_coords_arr[i][j] << " ";
            edge0(i,j) = node_coords[node_1][j] - node_coords[node_0][j];
            nodes0(i,j) = node_coords[node_0][j];
            // Skip edge1 because it never gets used in the calculations anyway
            nEdge2(i,j) = node_coords[node_2][j] - node_coords[node_0][j];
        }
    }
    plane_normals = cross_rowwise(edge0, nEdge2); // not normalised!
    // Step 1: Quantities for the Moller Trumbore method
    p_vec = cross_rowwise(ray_directions, nEdge2); // Assigns a vector to an array variable, but Eigen automatically converts so long as the underlying sizes are correct at initialization
    determinants = (edge0.array() * p_vec).rowwise().sum(); // Row-wise dot product

    // Step 2: Culling.
    //Determinant negative -> triangle is back-facing. If det is close to 0, ray and triangle are parallel and ray misses the triangle.
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask = (determinants > 1e-6) && (determinants > 0);
    if (!valid_mask.any()) {
        //std::cout << "Condition 1 triggered" << std::endl;
        return negative_output; // No intersection - return infinity
    }
    // Step 3: Test if ray is in front of the triangle
    inverse_determinants = determinants.inverse(); // Element-wise inverse. Correct
    t_vec = ray_origins - nodes0;
    barycentric_u = ((t_vec * p_vec).rowwise().sum()).array() * inverse_determinants; // comes out the same as benchmark
    valid_mask = valid_mask && (barycentric_u >= 0) && (barycentric_u <= 1);
    if (!valid_mask.any()) {
        //std::cout << "Condition 2 triggered" << std::endl;
        return negative_output; // No intersection - return infinity
    }

    q_vec = cross_rowwise(t_vec.matrix(), edge0); //
    barycentric_v = (ray_directions.array() * q_vec).rowwise().sum().matrix().array() * inverse_determinants; // Comes out correctly
    // Check barycentric_v and sum
    valid_mask = valid_mask && (barycentric_v >= 0) && ((barycentric_u + barycentric_v) <= 1);
    // t values
    t_values = (nEdge2.array() * q_vec).rowwise().sum().array() * inverse_determinants;
    valid_mask = valid_mask && (t_values >= ray.t_min) && (t_values <= ray.t_max);
    // Iterate through all t_values and set them to infinity if they don't satisfy the conditions imposed by the mask
    for (int i = 0; i < t_values.rows(); ++i) {
        for (int j = 0; j < t_values.cols(); ++j) {
            if (!valid_mask(i, j)) {
                t_values(i, j) = std::numeric_limits<double>::infinity();
            }
        }
    }
    barycentric_coordinates.col(0) = barycentric_u;
    barycentric_coordinates.col(1) = barycentric_v;
    barycentric_coordinates.col(2) = 1.0 - barycentric_u - barycentric_v; // barycentric_w
    return IntersectionOutput{barycentric_coordinates, plane_normals, t_values};
}