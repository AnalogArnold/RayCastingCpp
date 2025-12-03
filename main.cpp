#include <fstream>
#include <iostream>
#include <array>
#include <vector>
#include <chrono>
#include "eigen_types.h"
#include "camera.h"
#include "renderer.h"
#include <atomic>


//////////////////////////////////// INPUT
double aspect_ratio = 16.0/9.0;
unsigned short image_width = 400; // px
unsigned short image_height = static_cast<unsigned short>(image_width / aspect_ratio); // px
unsigned short number_of_samples = 50; // For anti-aliasing. Really don't expect we'll need more than a short
    //std::cout << "rows" << sizeof edge0_arr / sizeof edge0_arr[0] << std::endl;
    //std::cout << "cols" << sizeof edge0_arr[0] / sizeof(double) << std::endl;

// Variables for performance tests
std::atomic<uint64_t> num_intersection_tests = 0;
std::atomic<uint64_t> num_intersections_found = 0;
std::atomic<uint64_t> num_primary_rays = 0;

// Function that would receive the data from Pybind
void render_scene(const std::vector<std::vector<std::array<int,3>>> &scene_connectivity,
    const std::vector<std::vector<std::array<double,3>>> &scene_coords,
    const std::vector<std::vector<std::array<double,3>>> &scene_face_colors,
    const std::vector<Camera> &cameras) {

    for (const Camera &camera : cameras) {
        render_ppm_image(camera, scene_connectivity, scene_coords, scene_face_colors);
    }
}


inline void compute_triangle_centroid(const std::vector<std::vector<std::array<int,3>>> &scene_connectivity,
    const std::vector<std::vector<std::array<double,3>>> &scene_coords) {
    // Find the centroid of a triangle.
    // Might be worth rewriting this to take in the full array of triangles rather than just accepting one and
    // change the output type.

    size_t num_meshes = scene_coords.size(); // Get number of meshes
    std::vector<EiVector3d> centroids;
    // Iterate over meshes in the scene
    for (size_t mesh_idx = 0; mesh_idx < num_meshes; ++mesh_idx) {
        const std::vector<std::array<int,3>> connectivity = scene_connectivity[mesh_idx];
        const std::vector<std::array<double,3>> node_coords = scene_coords[mesh_idx];
        long long number_of_elements = connectivity.size();

        for (int i=0; i < number_of_elements; i++) {
            int node_0 = connectivity[i][0];
            int node_1 = connectivity[i][1];
            int node_2 = connectivity[i][2];
            double centroid_x = (node_coords[node_0][0] + node_coords[node_1][0] + node_coords[node_2][0])/3.0;
            double centroid_y = (node_coords[node_0][1] + node_coords[node_1][1] + node_coords[node_2][1])/3.0;
            double centroid_z = (node_coords[node_0][2] + node_coords[node_1][2] + node_coords[node_2][2])/3.0;
            centroids.push_back(EiVector3d(centroid_x, centroid_y, centroid_z));
            //::cout << centroids[i] << std::endl;
        }
    }
}

inline EiVector3d compute_triangle_centroid_sing(const std::vector<std::array<int,3>> &connectivity,
                                   const std::vector<std::array<double,3>> &node_coords,
                                   const int mesh_id){

    int node_0 = connectivity[mesh_id][0];
    int node_1 = connectivity[mesh_id][1];
    int node_2 = connectivity[mesh_id][2];
    double centroid_x = (node_coords[node_0][0] + node_coords[node_1][0] + node_coords[node_2][0])/3.0;
    double centroid_y = (node_coords[node_0][1] + node_coords[node_1][1] + node_coords[node_2][1])/3.0;
    double centroid_z = (node_coords[node_0][2] + node_coords[node_1][2] + node_coords[node_2][2])/3.0;
    return EiVector3d(centroid_x, centroid_y, centroid_z);

}

// Bounding volume structure - axis-aligned bounding boxes (AABB)
struct AABB {
    double corner_min[3]{};
    double corner_max[3]{};

    AABB() {
        corner_min[0] = corner_min[1] = corner_min[2] = std::numeric_limits<double>::infinity();
        corner_max[0] = corner_max[1] = corner_max[2] = -std::numeric_limits<double>::infinity();
    }
    void expand_to_include_point(const EiVector3d& point) {
        for (int i = 0; i < 3; ++i){
            if (point(i) < corner_min[i]) corner_min[i] = point(i);
            if (point(i) > corner_max[i]) corner_max[i] = point(i);
        }
    }
    void expand_to_include_triangle_node(const std::array<double,3>& triangle_coords) {
        for (int i = 0; i < 3; ++i){

            if (triangle_coords[i] < corner_min[i]) corner_min[i] = triangle_coords[i];
            if (triangle_coords[i] > corner_max[i]) corner_max[i] = triangle_coords[i];
        }
    }
    void expand_to_include_AABB(const AABB& other) {
        for (int i = 0; i < 3; ++i){
            if (other.corner_min[i] < corner_min[i]) corner_min[i] = other.corner_min[i];
            if (other.corner_max[i] > corner_max[i]) corner_max[i] = other.corner_max[i];
        }
    }
    inline double find_axis_extent(int axis) const {
        return corner_max[axis] - corner_min[axis];
    }
    double find_surface_area() const {
        double height = find_axis_extent(2);
        double width = find_axis_extent(1);
        double depth = find_axis_extent(0);
        return 2 * (height * width + width * depth + height * depth);
    }
};

int main() {

    // TEST DATA - as equivalent to what it looks like in Pybind as possible
    std::vector<Camera> cameras; // Stores camera center, pixel_00_center and matrix_pixel_spacings from relevant data
    std::vector<std::vector<std::array<int, 3>>> scene_connectivity;
    std::vector<std::vector<std::array<double, 3>>> scene_coords;
    std::vector<std::vector<std::array<double, 3>>> scene_face_colors;

    // TEST CAMERAS
    //Camera test_camera1{EiVector3d(0, 1, 1), EiVector3d(0, 0, -1), 90};
    // cameras.push_back(test_camera1);
    Camera test_camera2{EiVector3d(-0.5, 1.1, 1.1), EiVector3d(0, 0, -1), 90};
    cameras.push_back(test_camera2);

    // TEST MESH FROM SIMDATA
    std::vector<std::array<int,3>> mesh1_connectivity;
    std::vector<std::array<double,3>> mesh1_coords;
    std::vector<std::array<double,3>> mesh1_face_colors;
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
    // Normalised values of displacement in y to get the colour for each face
    double face_colors[44][3] = {
        0.3378407528533183,0.3378407528533183,0.3378407528533183,
    0.17465446093414858,0.17465446093414858,0.17465446093414858,
    0.49999999999071704,0.49999999999071704,0.49999999999071704,
    0.33681370807154737,0.33681370807154737,0.33681370807154737,
    0.49999999999071704,0.49999999999071704,0.49999999999071704,
    0.3231067213995538,0.3231067213995538,0.3231067213995538,
    0.48526596853695253,0.48526596853695253,0.48526596853695253,
    0.49999999999071704,0.49999999999071704,0.49999999999071704,
    0.1564400547359814,0.1564400547359814,0.1564400547359814,
    0.011479797359104616,0.011479797359104616,0.011479797359104616,
    0.01496017210350721,0.01496017210350721,0.01496017210350721,
    0.33681370807154726,0.33681370807154726,0.33681370807154726,
    0.3293557566576631,0.3293557566576631,0.3293557566576631,
    0.16966146782702177,0.16966146782702177,0.16966146782702177,
    0.49999999999071704,0.49999999999071704,0.49999999999071704,
    0.49254204857683287,0.49254204857683287,0.49254204857683287,
    0.3260572532852093,0.3260572532852093,0.3260572532852093,
    0.6449602573704004,0.6449602573704004,0.6449602573704004,
    0.6524182087822296,0.6524182087822296,0.6524182087822296,
    0.3475817912045913,0.3475817912045913,0.3475817912045913,
    0.6592087152465939,0.6592087152465939,0.6592087152465939,
    0.6739427467049937,0.6739427467049937,0.6739427467049937,
    0.34079128473897385,0.34079128473897385,0.34079128473897385,
    0.3333333333271447,0.3333333333271447,0.3333333333271447,
    0.17363904449650336,0.17363904449650336,0.17363904449650336,
    0.3550397426184755,0.3550397426184755,0.3550397426184755,
    0.5074579514025462,0.5074579514025462,0.5074579514025462,
    0.8303385321488467,0.8303385321488467,0.8303385321488467,
    0.6706442433153988,0.6706442433153988,0.6706442433153988,
    0.5147340314491169,0.5147340314491169,0.5147340314491169,
    0.8435599452623938,0.8435599452623938,0.8435599452623938,
    0.6768932785926328,0.6768932785926328,0.6768932785926328,
    0.49999999999071704,0.49999999999071704,0.49999999999071704,
    0.49999999999071704,0.49999999999071704,0.49999999999071704,
    0.6631862919035698,0.6631862919035698,0.6631862919035698,
    0.49999999999071704,0.49999999999071704,0.49999999999071704,
    0.49999999999071704,0.49999999999071704,0.49999999999071704,
    0.662159247134233,0.662159247134233,0.662159247134233,
    0.826360955493926,0.826360955493926,0.826360955493926,
    0.6666666666604781,0.6666666666604781,0.6666666666604781,
    0.9850398278805338,0.9850398278805338,0.9850398278805338,
    0.9885202026374419,0.9885202026374419,0.9885202026374419,
    0.6631862919035698,0.6631862919035698,0.6631862919035698,
    0.8253455390470857,0.8253455390470857,0.8253455390470857
    };

    // Stash the data inside vectors
    for (int i = 0; i < 44; i++) {
        std::array<int,3> temp_arr {};
        std::array<double,3> temp_arr2 {};
        for (int j = 0; j < 3; j++) {
            temp_arr[j] = connectivity[i][j];
            temp_arr2[j] = face_colors[i][j];
        }
        mesh1_connectivity.push_back(temp_arr);
        mesh1_face_colors.push_back(temp_arr2);
    }

    for (int i = 0; i < 24; i++) {
        std::array<double,3> temp_arr2 {};
        for (int j = 0; j < 3; j++) {
            temp_arr2[j] = coords[i][j];
        }
        mesh1_coords.push_back(temp_arr2);
    }
    // Add to the scene vectors so we can work with multiple meshes
    scene_connectivity.push_back(mesh1_connectivity);
    scene_coords.push_back(mesh1_coords);
    scene_face_colors.push_back(mesh1_face_colors);

    size_t num_meshes = scene_coords.size(); // Get number of meshes
    std::vector<AABB> aabbs;
    // Iterate over meshes in the scene
    for (size_t mesh_idx = 0; mesh_idx < num_meshes; ++mesh_idx) {
        const std::vector<std::array<int,3>> connectivity = scene_connectivity[mesh_idx];
        const std::vector<std::array<double,3>> node_coords = scene_coords[mesh_idx];
        long long number_of_elements = connectivity.size();

        for (int i=0; i < number_of_elements; i++) {
            AABB temp;
            for (int j =0; j < 3; j++){
                int node = connectivity[i][j];
                temp.expand_to_include_triangle_node(node_coords[node]);
            }
            aabbs.push_back(temp);
            std:: cout << "triangle number: " << i << std::endl;
            std::cout << "min: " << temp.corner_min[0] << " " << temp.corner_min[1] << " " << temp.corner_min[2] << std::endl;
            std::cout << "max: " << temp.corner_max[0] << " " << temp.corner_max[1] << " " << temp.corner_max[2] << std::endl;
            for (int j =0; j < 3; j++) {
                //std::cout << "Axis " << j << "extent: " << aabbs[i].find_axis_extent(j) << std::endl;
            }
            std::cout << "Surface area: " << aabbs[i].find_surface_area() << std::endl;
        }
    }

    int triangle_1 = 43;
    int triangle_2 = 0;

    aabbs[triangle_1].expand_to_include_AABB(aabbs[triangle_2]);
    //std:: cout << "triangle " << triangle_1 << " to include " << triangle_2 << std::endl;
    //std::cout << "min: " << aabbs[triangle_1].corner_min[0] << " " << aabbs[triangle_1].corner_min[1] << " " << aabbs[triangle_1].corner_min[2] << std::endl;
    //std::cout << "max: " << aabbs[triangle_1].corner_max[0] << " " << aabbs[triangle_1].corner_max[1] << " " << aabbs[triangle_1].corner_max[2] << std::endl;


    // Runtime tests
    //std::chrono::high_resolution_clock::time_point begin1 = std::chrono::high_resolution_clock::now();
    //render_scene(scene_connectivity, scene_coords, scene_face_colors, cameras);
    //std::chrono::high_resolution_clock::time_point end1 = std::chrono::high_resolution_clock::now();
    //std::cout << "Runtime: " << std::chrono::duration_cast<std::chrono::milliseconds> (end1 - begin1) << std::endl;
    //std::cout << "Number of primary rays: " << num_primary_rays << std::endl;
    //std::cout << "Number of intersection tests: " << num_intersection_tests << std::endl;
    //std::cout << "Number of intersections found: " << num_intersections_found << std::endl;

    return 0;
}

