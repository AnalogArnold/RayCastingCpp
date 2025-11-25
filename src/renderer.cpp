#include "../renderer.h"
#include "../hit_record.h"
#include "../ray_intersection.h"
#include "../math_utils.h"
#include <fstream>
#include <iostream>
#include <omp.h>
#include <atomic>

// Global rendering parameters
extern unsigned short image_width;
extern unsigned short image_height;
extern unsigned short number_of_samples;
extern double aspect_ratio;

// Global variables for performance tests
extern std::atomic<uint64_t> num_primary_rays;

inline EiVector3d get_face_color(Eigen::Index minRowIndex,
    const std::vector<std::array<double,3>> &face_colors) {
    double c1 = face_colors[minRowIndex][0];
    double c2 = face_colors[minRowIndex][1];
    double c3 = face_colors[minRowIndex][2];
    EiVector3d face_color;
    face_color << c1, c2, c3;
    return face_color;
}

EiVector3d return_ray_color(const Ray &ray,
                            const std::vector<std::vector<std::array<int,3>>> &scene_connectivity,
                            const std::vector<std::vector<std::array<double,3>>> &scene_coords,
                            const std::vector<std::vector<std::array<double,3>>> &scene_face_colors){
    EiVectorD3d color_test(3,3);
    color_test.row(0) << 1.0, 0.0, 0.0;
    color_test.row(1) << 0.0, 1.0, 1.0;
    color_test.row(2) << 1.0, 0.0, 1.0;
    HitRecord intersection_record; // Create HitRecord struct

    size_t num_meshes = scene_coords.size(); // Get number of meshes
    // Iterate over meshes in the scene
    for (size_t mesh_idx = 0; mesh_idx < num_meshes; ++mesh_idx) {
        IntersectionOutput intersection = intersect_plane(ray, scene_connectivity[mesh_idx], scene_coords[mesh_idx]);
        Eigen::Index minRowIndex, minColIndex;

        intersection.t_values.minCoeff(&minRowIndex, &minColIndex); // Find indices of the smallest t_value
        double closest_t = intersection.t_values(minRowIndex, minColIndex);
        if (closest_t < intersection_record.t) {
            intersection_record.t = closest_t;
            intersection_record.barycentric_coordinates = intersection.barycentric_coordinates.row(minRowIndex);
            intersection_record.point_intersection = ray_at_t(closest_t, ray);
            intersection_record.normal_surface = intersection.plane_normals.row(minRowIndex);
            intersection_record.face_color = get_face_color(minRowIndex, scene_face_colors[mesh_idx]);
        }
    }
    if (intersection_record.t != std::numeric_limits<double>::infinity()) { // Instead of keeping a bool hit_anything, check if t value has changed from the default
        set_face_normal(ray, intersection_record.normal_surface);
        //std::cout<<"hit sth"<<std::endl;
        //return intersection_record.barycentric_coordinates(0) * color_test.row(0) + intersection_record.barycentric_coordinates(1) * color_test.row(2) + intersection_record.barycentric_coordinates(2) * color_test.row(2);
        return intersection_record.barycentric_coordinates(0) * intersection_record.face_color + intersection_record.barycentric_coordinates(1) * intersection_record.face_color + intersection_record.barycentric_coordinates(2) * intersection_record.face_color;
        //return color
    }
    // Blue sky gradient
    double a = 0.5 * (ray.direction(1) + 1.0);
    static EiVector3d white, blue;
    white << 1.0, 1.0, 1.0;
    blue << 0.5, 0.7, 1.0;
    EiVector3d color = (1.0 - a) * white + a * blue;
    return color;
}

void render_ppm_image(const Camera& camera1,
                     const std::vector<std::vector<std::array<int,3>>> &scene_connectivity,
                     const std::vector<std::vector<std::array<double,3>>> &scene_coords,
                     const std::vector<std::vector<std::array<double,3>>> &scene_face_colors){

    std::vector<uint8_t> buffer;
    buffer.reserve(image_width * image_height * 12); // Preallocate memory for the image buffer (conservatively)

    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            EiVector3d pixel_color = EiVector3d::Zero();
            for (int k = 0; k < number_of_samples; k++) {
                double offset[2] = {random_double() - 0.5, random_double() - 0.5};
                EiVector3d pixel_sample = camera1.pixel_00_center +
                                         (i + offset[0]) * camera1.matrix_pixel_spacing.row(0) +
                                         (j + offset[1]) * camera1.matrix_pixel_spacing.row(1);
                EiVector3d ray_direction = pixel_sample - camera1.camera_center;
                num_primary_rays ++;
                Ray current_ray {camera1.camera_center, ray_direction.normalized()};
                pixel_color += return_ray_color(current_ray, scene_connectivity, scene_coords, scene_face_colors);
            }
            double gray = 0.2126 * pixel_color[0] + 0.7152 * pixel_color[1] + 0.0722 * pixel_color[2];
            int gray_byte = int(gray/number_of_samples * 255.99);
            buffer.push_back(static_cast<uint8_t>(gray_byte));
            buffer.push_back(static_cast<uint8_t>(gray_byte));
            buffer.push_back(static_cast<uint8_t>(gray_byte));
        }
    }

    // Write to file
    std::ofstream image_file;
    // WIP: Will have to make the filename change based on the camera number or some unique identifier, otherwise we will keep on overwriting the same file
    image_file.open("test.ppm");
    if (!image_file.is_open()) {
        std::cerr << "Failed to open the output file.\n";
        return;
    }
    image_file << "P6\n" << image_width << ' ' << image_height << "\n255\n";
    image_file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    image_file.close();
    std::cout << "\r Done. \n";
}