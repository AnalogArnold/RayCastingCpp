#include "../renderer.h"
#include "../hit_record.h"
#include "../ray_intersection.h"
#include "../math_utils.h"
#include <fstream>
#include <iostream>

// Global rendering parameters
extern unsigned short image_width;
extern unsigned short image_height;
extern unsigned short number_of_samples;
extern double aspect_ratio;

EiVector3d return_ray_color(const Ray &ray,
                            const std::vector<std::array<int,3>> &connectivity,
                            const std::vector<std::array<double,3>> &node_coords) {
    EiVectorD3d color_test(3,3);
    color_test.row(0) << 1.0, 0.0, 0.0;
    color_test.row(1) << 0.0, 1.0, 1.0;
    color_test.row(2) << 1.0, 0.0, 1.0;
    //node_coords_test.row(0) << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0;
    //node_coords_test.row(1) <<  0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    // Nodal coords; 4, but 4th isn't of interest
    HitRecord intersection_record; // Create HitRecord struct
    IntersectionOutput intersection = intersect_plane(ray, connectivity, node_coords);
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
        //std::cout<<"hit sth"<<std::endl;
        return intersection_record.barycentric_coordinates(0) * color_test.row(0) + intersection_record.barycentric_coordinates(1) * color_test.row(2) + intersection_record.barycentric_coordinates(2) * color_test.row(2);
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
                     const std::vector<std::array<int,3>> &connectivity,
                     const std::vector<std::array<double,3>> &node_coords) {
    std::ofstream image_file;
    image_file.open("test.ppm");
    image_file << "P3\n" << image_width << ' ' << image_height << "\n255\n";
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
                Ray current_ray {camera1.camera_center, ray_direction.normalized()};
                pixel_color += return_ray_color(current_ray, connectivity, node_coords);
            }
            double gray = 0.2126 * pixel_color[0] + 0.7152 * pixel_color[1] + 0.0722 * pixel_color[2];
            int gray_byte = int(gray/number_of_samples * 255.99);
            image_file << gray_byte << ' ' << gray_byte << ' ' << gray_byte << '\n';
        }
    }
    image_file.close();
    std::cout << "\r Done. \n";
}