#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include "cuda_compat.h"
#include "vec3.h"
#include <cmath>
#include <cstdlib>   // rand() for host RNG fallback
#include <cstdint>   // uint32_t
#include <limits>

// ======================================================
// Constants / math helpers
// ======================================================

// We wrap these in CUDA_HD so both host and device can call them.

CUDA_HD
inline double rt_infinity() {
    // Note: std::numeric_limits<>::infinity() is technically host-only constexpr,
    // but CUDA is usually fine with it. If your compiler warns, you can hardcode.
    return std::numeric_limits<double>::infinity();
    // If you still get noisy warnings from nvcc, swap to:
    // return 1.0e30;
}

CUDA_HD
inline double rt_pi() {
    return 3.1415926535897932385;
}

CUDA_HD
inline double degrees_to_radians(double degrees) {
    return degrees * rt_pi() / 180.0;
}

// Schlick-style Fresnel reflectance approximation.
// Used by dielectric/glass material. Must be callable on GPU.
CUDA_HD
inline double schlick_reflectance(double cosine, double ref_idx) {
    // Schlick's approximation for reflectance.
    double r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

// ======================================================
// HOST RNG (CPU-side)
// ======================================================
//
// These are ONLY for CPU code. We do NOT mark them CUDA_HD.
// We do NOT call them from device code.
//
// You can replace these with <random> + mt19937 if you prefer.
// I'm keeping it simple/portable here.

inline double random_double_host() {
    // Returns a random real in [0, 1).
    return rand() / (RAND_MAX + 1.0);
}

inline double random_double_host(double min, double max) {
    // Returns a random real in [min, max).
    return min + (max - min) * random_double_host();
}

inline vec3 random_vec3_host() {
    return vec3(
        random_double_host(),
        random_double_host(),
        random_double_host()
    );
}

inline vec3 random_vec3_host(double min, double max) {
    return vec3(
        random_double_host(min, max),
        random_double_host(min, max),
        random_double_host(min, max)
    );
}

inline vec3 random_in_unit_sphere_host() {
    while (true) {
        vec3 p = random_vec3_host(-1.0, 1.0);
        if (p.length_squared() < 1.0) return p;
    }
}

inline vec3 random_unit_vector_host() {
    return unit_vector(random_in_unit_sphere_host());
}

inline vec3 random_in_unit_disk_host() {
    while (true) {
        vec3 p(random_double_host(-1.0, 1.0),
               random_double_host(-1.0, 1.0),
               0.0);
        if (p.length_squared() < 1.0) return p;
    }
}

// cosine-weighted hemisphere direction (host version if you need it)
inline vec3 random_cosine_direction_host() {
    double r1 = random_double_host();
    double r2 = random_double_host();

    double z  = std::sqrt(1.0 - r2);
    double phi = 2.0 * rt_pi() * r1;
    double x = std::cos(phi) * std::sqrt(r2);
    double y = std::sin(phi) * std::sqrt(r2);

    return vec3(x, y, z); // local coords
}

// ======================================================
// DEVICE RNG (GPU-side)
// ======================================================
//
// CUDA_D (which becomes __device__ under NVCC, and nothing under MSVC)
// These *must not* be called from CPU-only code in .cpp files.
//
// Every thread gets its own uint32_t state. You update the state
// each time you pull a new random number. That gives you reproducible,
// independent RNG per pixel / per sample without <random>.

CUDA_D
inline float random_float_device(uint32_t& state) {
    // LCG (linear congruential generator)
    // simple, fast, not cryptographically strong, fine for path tracing
    state = state * 1664525u + 1013904223u;
    // extract 24 high-ish bits and scale to [0,1)
    return (state & 0x00FFFFFF) / float(0x01000000);
}

CUDA_D
inline double random_double_device(uint32_t& state) {
    return (double)random_float_device(state);
}

CUDA_D
inline double random_double_device(double min, double max, uint32_t& state) {
    return min + (max - min) * random_double_device(state);
}

CUDA_D
inline vec3 random_vec3_device(uint32_t& state) {
    return vec3(
        random_double_device(state),
        random_double_device(state),
        random_double_device(state)
    );
}

CUDA_D
inline vec3 random_vec3_device(double min, double max, uint32_t& state) {
    return vec3(
        random_double_device(min, max, state),
        random_double_device(min, max, state),
        random_double_device(min, max, state)
    );
}

CUDA_D
inline vec3 random_in_unit_sphere_device(uint32_t& state) {
    while (true) {
        vec3 p = random_vec3_device(-1.0, 1.0, state);
        if (p.length_squared() < 1.0)
            return p;
    }
}

CUDA_D
inline vec3 random_unit_vector_device(uint32_t& state) {
    return unit_vector(random_in_unit_sphere_device(state));
}

CUDA_D
inline vec3 random_in_hemisphere_device(const vec3& normal, uint32_t& state) {
    vec3 in_sphere = random_in_unit_sphere_device(state);
    if (dot(in_sphere, normal) > 0.0) {
        // same hemisphere as normal
        return in_sphere;
    } else {
        return -in_sphere;
    }
}

// cosine-weighted hemisphere sample in local coords
CUDA_D
inline vec3 random_cosine_direction_device(uint32_t& state) {
    float r1 = random_float_device(state);
    float r2 = random_float_device(state);

    double z = std::sqrt(1.0 - (double)r2);

    double phi = 2.0 * rt_pi() * (double)r1;
    double x = std::cos(phi) * std::sqrt((double)r2);
    double y = std::sin(phi) * std::sqrt((double)r2);

    return vec3(x, y, z); // local coords
}

// Random point in unit disk (for thin lens / depth of field)
CUDA_D
inline vec3 random_in_unit_disk_device(uint32_t& state) {
    while (true) {
        double x = random_double_device(state) * 2.0 - 1.0;
        double y = random_double_device(state) * 2.0 - 1.0;
        vec3 p(x, y, 0.0);
        if (p.length_squared() < 1.0) return p;
    }
}

#endif // RTWEEKEND_H
