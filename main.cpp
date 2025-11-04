#include <fstream>
#include <iostream>
#include <array>
#include <iostream>
#include <vector>
#include <chrono>
#include "eigen_types.h"
#include "camera.h"
#include "renderer.h"

//////////////////////////////////// INPUT
double aspect_ratio = 16.0/9.0;
unsigned short image_width = 400; // px
unsigned short image_height = static_cast<unsigned short>(image_width / aspect_ratio); // px
unsigned short number_of_samples = 1; // For anti-aliasing. Really don't expect we'll need more than a short


    //std::cout << "rows" << sizeof edge0_arr / sizeof edge0_arr[0] << std::endl;
    //std::cout << "cols" << sizeof edge0_arr[0] / sizeof(double) << std::endl;

int main() {
    //Camera test_camera{EiVector3d(0, 1, 1), EiVector3d(0, 0, -1), 90};
    Camera test_camera{EiVector3d(-0.5, 1.1, 1.1), EiVector3d(0, 0, -1), 90};
    // Mesh from simdata. Copied by force for now.

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
1.0,1.0,0.0,1.0,1.5,0.5,1.0,1.0,0.5
    };

// Nodal coords; 4, but 4th isn't of interest
double coords[24][3] = {
        0.0,0.0,0.0,
    0.0,0.49999999999999983,0.0,
    0.49999999999999994,0.0,0.0,
    0.0,0.49999999999999983,0.5,
    0.0,0.0,0.5,
    0.49999999999999994,0.0,0.5,
    0.5,0.49999999999999983,0.0,
    1.0,0.0,0.0,
    0.0,1.0,0.0,
    0.0,1.0,0.5,
    0.5,0.49999999999999983,0.5,
    1.0,0.0,0.5,
    0.5,1.0,0.0,
    1.0,0.49999999999999994,0.0,
    0.5,1.0,0.5,
    1.0,0.49999999999999994,0.5,
    0.0,1.5,0.0,
    0.0,1.5,0.5,
    0.5000000000000001,1.5,0.0,
    1.0,1.0,0.0,
    0.5000000000000001,1.5,0.5,
    1.0,1.0,0.5,
    1.0,1.5,0.0,
    1.0,1.5,0.5
    };

// Indices of nodes comprising each triangle
int connectivity [44][3] = {
    0,1,2,
    0,3,1,
    0,2,4,
    0,4,3,
    2,5,4,
    2,1,6,
    2,6,7,
    2,7,5,
    1,8,6,
    1,9,8,
    1,3,9,
    3,4,5,
    3,5,10,
    3,10,9,
    5,7,11,
    5,11,10,
    6,8,12,
    6,13,7,
    6,12,13,
    10,14,9,
    10,11,15,
    10,15,14,
    8,16,12,
    8,17,16,
    8,9,17,
    9,14,17,
    12,16,18,
    12,19,13,
    12,18,19,
    14,20,17,
    14,15,21,
    14,21,20,
    16,20,18,
    16,17,20,
    18,22,19,
    18,23,22,
    18,20,23,
    20,21,23,
    7,13,15,
    7,15,11,
    13,19,21,
    13,21,15,
    19,22,23,
    19,23,21
};
    std::vector<std::array<int,3>> connectivity_vec;
    std::vector<std::array<double,3>> node_coords_vec;
    for (int i = 0; i < 44; i++) {
        std::array<int,3> temp_arr {};
        for (int j = 0; j < 3; j++) {
            temp_arr[j] = connectivity[i][j];
        }
        connectivity_vec.push_back(temp_arr);
    }

    for (int i = 0; i < 24; i++) {
        std::array<double,3> temp_arr2 {};
        for (int j = 0; j < 3; j++) {
            temp_arr2[j] = coords[i][j];
        }
        node_coords_vec.push_back(temp_arr2);
    }
/*
    long long number_of_elements = connectivity_vec.size();
    EiMatrixDd edge0 (number_of_elements,3), nEdge2 (number_of_elements,3); // shape (faces, 3) each
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>  nodes0 (number_of_elements, 3);
    for (int i=0; i < number_of_elements; i++) {
        int node_0 = connectivity_vec[i][0];
        int node_1 = connectivity_vec[i][1];
        int node_2 = connectivity_vec[i][2];
        for (int j=0; j < 3; j++) {
            //std::cout<<node_coords_arr[i][j] << " ";
            edge0(i,j) = node_coords_vec[node_1][j] - node_coords_vec[node_0][j];
            nodes0(i,j) = node_coords_vec[node_0][j];
            // Skip edge1 because it never gets used in the calculations anyway
            nEdge2(i,j) = node_coords_vec[node_2][j] - node_coords_vec[node_0][j];
        }
    }
    //std::cout << "edge 0: " << edge0 << std::endl;
    std::cout << "nEdge2: " << nEdge2 << std::endl;
    //::cout << "nodes0:" << nodes0 << std::endl;
*/
    //std::cout << node_coords_test <<std::endl;
    //Camera camera1;
    // Sample data for testing
    //Ray test_ray{EiVector3d(-0.5, 1.1, 1.1), EiVector3d(4.132331920978222, -2.603127666416139, 1.1937133836332001), 0.0};
    //EiMatrixDd node_coords_test(2,9);
    //node_coords_test.row(0) << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0;
    //node_coords_test.row(1) <<  0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    //intersect_plane(test_ray, node_coords_test);

    std::chrono::high_resolution_clock::time_point begin1 = std::chrono::high_resolution_clock::now();
    render_ppm_image(test_camera, connectivity_vec, node_coords_vec);
    std::chrono::high_resolution_clock::time_point end1 = std::chrono::high_resolution_clock::now();
    std::cout << "runtime: " << std::chrono::duration_cast<std::chrono::milliseconds> (end1 - begin1) << std::endl;

    return 0;
}




