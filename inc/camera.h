#ifndef CAMERA_H
#define CAMERA_H

#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>
#include <fstream>
#include <sstream>   

#include "hittable.h"
#include "material.h"

class camera {
  public:
    double aspect_ratio = 1.0;  // Ratio of image width over height
    int    image_width  = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 10;   // Count of random samples for each pixel
    int    max_depth         = 10;   // Maximum number of ray bounces into scene

    double vfov = 90;  // Vertical view angle (field of view)
    point3 lookfrom = point3(0,0,0);   // Point camera is looking from
    point3 lookat   = point3(0,0,-1);  // Point camera is looking at
    vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction

    double defocus_angle = 0;  // Variation angle of rays through each pixel
    double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

    void render(const hittable& world) {

        const std::string& filename = "image.ppm";
        initialize();

        std::ofstream out(filename, std::ios::out | std::ios::trunc);
        if (!out) {
            std::cerr << "[error] couldn't open output file: " << filename << "\n";
            return;
        }

        // PPM header
        out << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = 0; j < image_height; j++) {
            std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
            for (int i = 0; i < image_width; i++) {
                color pixel_color(0,0,0);
                for (int sample = 0; sample < samples_per_pixel; sample++) {
                    ray r = get_ray(i, j);
                    pixel_color += ray_color(r, max_depth, world);
                }
                write_color(out, pixel_samples_scale * pixel_color); // write to file
            }
        }
    
        out.close();
        std::clog << "\rDone.                 \n";
    }

    void render(const hittable& world, int num_threads) {
        initialize();

        if (num_threads <= 0) {
            const unsigned hc = std::thread::hardware_concurrency();
            num_threads = hc ? static_cast<int>(hc) : 1;
        }

        // Framebuffer: row-major [j * image_width + i]
        std::vector<color> framebuffer(static_cast<size_t>(image_width) * image_height);

        // Work function: compute rows [y0, y1)
        auto worker = [&](int y0, int y1) {
            for (int j = y0; j < y1; ++j) {
                for (int i = 0; i < image_width; ++i) {
                    color pixel_color(0, 0, 0);
                    for (int sample = 0; sample < samples_per_pixel; ++sample) {
                        ray r = get_ray(i, j);
                        pixel_color += ray_color(r, max_depth, world);
                    }
                    framebuffer[static_cast<size_t>(j) * image_width + i] =
                        pixel_samples_scale * pixel_color;
                }
            }
        };

        // Split rows into (almost) equal chunks
        const int T = std::min(num_threads, image_height);
        const int chunk = (image_height + T - 1) / T;

        std::vector<std::thread> threads;
        threads.reserve(T);

        std::atomic<int> rows_done{0};

        for (int t = 0; t < T; ++t) {
            const int y0 = t * chunk;
            const int y1 = std::min(image_height, y0 + chunk);
            if (y0 >= y1) break;

            threads.emplace_back([&, y0, y1] {
                worker(y0, y1);
                rows_done += (y1 - y0);
                std::clog << "\rScanlines remaining: " << (image_height - rows_done.load()) << ' ' << std::flush;
            });
        }

        for (auto& th : threads) th.join();

        // ---------- Write PPM to file ----------
        std::ostringstream name;
        name << "image_nt_" << num_threads << ".ppm";   // e.g., image_nt_5.ppm
        const std::string filename = name.str();

        std::ofstream out(filename, std::ios::out | std::ios::trunc);
        if (!out) {
            std::cerr << "\n[error] failed to open output file: " << filename << "\n";
            return;
        }

        // Header
        out << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        // Pixels (single-threaded write to keep order)
        for (int j = 0; j < image_height; ++j) {
            for (int i = 0; i < image_width; ++i) {
                write_color(out, framebuffer[static_cast<size_t>(j) * image_width + i]);
            }
        }

        out.close();
        std::clog << "\nWrote " << filename << "\n";
        std::clog << "Done.\n";
    }

  private:
    int    image_height;   // Rendered image height
    double pixel_samples_scale;  // Color scale factor for a sum of pixel samples
    point3 center;         // Camera center
    point3 pixel00_loc;    // Location of pixel 0, 0
    vec3   pixel_delta_u;  // Offset to pixel to the right
    vec3   pixel_delta_v;  // Offset to pixel below
    vec3   u, v, w;              // Camera frame basis vectors
    vec3   defocus_disk_u;       // Defocus disk horizontal radius
    vec3   defocus_disk_v;       // Defocus disk vertical radius

    void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;
        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        // Determine viewport dimensions.
        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta/2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    ray get_ray(int i, int j) const {
        // Construct a camera ray originating from the defocus disk and directed at a randomly
        // sampled point around the pixel location i, j.

        auto offset = sample_square();
        auto pixel_sample = pixel00_loc
                          + ((i + offset.x()) * pixel_delta_u)
                          + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    vec3 sample_square() const {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return vec3(random_double() - 0.5, random_double() - 0.5, 0);
    }

    point3 defocus_disk_sample() const {
        // Returns a random point in the camera defocus disk.
        auto p = random_in_unit_disk();
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    color ray_color(const ray& r, int depth, const hittable& world) const {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return color(0,0,0);

        hit_record rec;

        if (world.hit(r, interval(0.001, infinity), rec)){
            ray scattered;
            color attenuation;
            if (rec.mat->scatter(r, rec, attenuation, scattered))
                return attenuation * ray_color(scattered, depth-1, world);
            return color(0,0,0);
        }

        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5*(unit_direction.y() + 1.0);
        return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    }
};

#endif