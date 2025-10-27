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
// Plain data that we can memcpy to the GPU.
// NO std::vector, NO pointers to heap objects.

struct GPUCamera {
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;

    vec3 u;
    vec3 v;
    vec3 w;

    double lens_radius;

    int image_width;
    int image_height;

    int samples_per_pixel;
    int max_depth;

    // If you add motion blur, you'd add time0/time1 here.
};

// --------------------------------------------------
// Device-side ray generator
// --------------------------------------------------
//
// This runs per thread (per pixel/sample) on the GPU.
// It jitter-samples the pixel for AA and samples the lens for DOF.
// rng_state is a per-thread RNG seed that gets mutated.

CUDA_D
inline ray generate_camera_ray_device(
    const GPUCamera& cam,
    int px,
    int py,
    uint32_t& rng_state
) {
    // Subpixel jitter for anti-aliasing
    double jitter_x = random_double_device(rng_state);
    double jitter_y = random_double_device(rng_state);

    double s = ( (double(px) + jitter_x) / (cam.image_width  - 1) );
    double t = ( (double(py) + jitter_y) / (cam.image_height - 1) );

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
//
// You create this in main.cpp, set its fields
// (resolution, fov, lookfrom/lookat, aperture, etc.),
// then call initialize(), then call toGPUCamera().

class camera {
public:
    // Image settings
    int image_width        = 800;
    int image_height       = 450;
    int samples_per_pixel  = 10;
    int max_depth          = 50;

    // Camera transform / lens settings
    vec3 lookfrom;
    vec3 lookat;
    vec3 vup = vec3(0,1,0);

    double vfov       = 40.0;   // vertical field-of-view in degrees
    double aperture   = 0.0;    // aperture radius * 2 = lens diameter
    double focus_dist = 1.0;    // distance to focal plane

    // Precomputed internals
    vec3 origin;
    vec3 horizontal;
    vec3 vertical;
    vec3 lower_left_corner;
    vec3 u, v, w;

    double lens_radius = 0.0;

    camera() = default;

    // initialize() builds the ray generation basis from user params.
    void initialize() {
        // aspect ratio
        double aspect_ratio = double(image_width) / double(image_height);

        // viewport size in world units
        double theta = degrees_to_radians(vfov);
        double h = std::tan(theta / 2.0);

        double viewport_height = 2.0 * h;
        double viewport_width  = aspect_ratio * viewport_height;

        // camera basis (w points backwards from look dir)
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;

        // focus_dist controls the projected plane distance
        horizontal = focus_dist * viewport_width  * u;
        vertical   = focus_dist * viewport_height * v;

        lower_left_corner =
            origin
          - horizontal / 2.0
          - vertical   / 2.0
          - focus_dist * w;

        lens_radius = aperture * 0.5;
    }

    // Convert this CPU camera into the plain GPUCamera
    // that we can pass into CUDA.
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

    // (Optional) If you still want a CPU reference renderer,
    // you might still have:
    //
    // ray get_ray(double s, double t) const { ... }
    //
    // You can leave that here as a host-only helper, just DON'T mark it CUDA_D
    // and DON'T call it from the GPU path.
};

#endif // CAMERA_H
