#ifndef CAMERA_H
#define CAMERA_H

#include "cuda_compat.h"
#include "vec3.h"
#include "ray.h"
#include "rtweekend.h"
#include <cmath>

// --------------------------------------------------
// GPUCamera
// --------------------------------------------------
struct GPUCamera {
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;

    vec3 u;
    vec3 v;
    vec3 w;

    float lens_radius;

    int image_width;
    int image_height;

    int samples_per_pixel;
    int max_depth;
};

// --------------------------------------------------
// Device-side ray generator
// --------------------------------------------------
CUDA_D
inline ray generate_camera_ray_device(
    const GPUCamera& cam,
    int px,
    int py,
    uint32_t& rng_state
) {
    // Subpixel jitter for anti-aliasing
    float jitter_x = random_double_device(rng_state);
    float jitter_y = random_double_device(rng_state);

    float s = ((float(px) + jitter_x) / (float(cam.image_width)  - 1.0f));
    float t = ((float(py) + jitter_y) / (float(cam.image_height) - 1.0f));

    // Depth of field: sample a random point on the lens
    vec3 rd = cam.lens_radius * random_in_unit_disk_device(rng_state);
    vec3 offset = cam.u * rd.x() + cam.v * rd.y();

    vec3 pixel_pos =
        cam.lower_left_corner
        + s * cam.horizontal
        + t * cam.vertical;

    vec3 dir = pixel_pos - cam.origin - offset;

    return ray(cam.origin + offset, dir);
}

// --------------------------------------------------
// CPU-side camera
// --------------------------------------------------
class camera {
public:
    int   image_width        = 800;
    int   image_height       = 450;
    int   samples_per_pixel  = 10;
    int   max_depth          = 50;

    vec3  lookfrom;
    vec3  lookat;
    vec3  vup = vec3(0.0f, 1.0f, 0.0f);

    float vfov       = 40.0f;
    float aperture   = 0.0f;
    float focus_dist = 1.0f;

    vec3  origin;
    vec3  horizontal;
    vec3  vertical;
    vec3  lower_left_corner;
    vec3  u, v, w;

    float lens_radius = 0.0f;

    camera() = default;

    void initialize() {
        float aspect_ratio = float(image_width) / float(image_height);

        float theta = (float)degrees_to_radians((double)vfov);
        float h = tanf(theta / 2.0f);

        float viewport_height = 2.0f * h;
        float viewport_width  = aspect_ratio * viewport_height;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;

        horizontal = focus_dist * viewport_width  * u;
        vertical   = focus_dist * viewport_height * v;

        lower_left_corner =
            origin
          - horizontal * 0.5f
          - vertical   * 0.5f
          - focus_dist * w;

        lens_radius = aperture * 0.5f;
    }

    GPUCamera toGPUCamera() const {
        GPUCamera g;
        g.origin            = origin;
        g.lower_left_corner = lower_left_corner;
        g.horizontal        = horizontal;
        g.vertical          = vertical;
        g.u = u;
        g.v = v;
        g.w = w;
        g.lens_radius       = lens_radius;
        g.image_width       = image_width;
        g.image_height      = image_height;
        g.samples_per_pixel = samples_per_pixel;
        g.max_depth         = max_depth;
        return g;
    }
};

#endif // CAMERA_H
