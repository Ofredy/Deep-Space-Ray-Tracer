#ifndef COLOR_H
#define COLOR_H

#include "cuda_compat.h"
#include "vec3.h"
#include <cmath>

#ifndef __CUDACC__
#include <iostream>
#endif

using color = vec3;  // <<< NEW

CUDA_HD
inline double clamp_double(double x, double min_val, double max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

CUDA_HD
inline unsigned char linear_to_byte(double channel, int samples_per_pixel) {
    double scale = 1.0 / samples_per_pixel;
    double c = channel * scale;

    if (c < 0.0) c = 0.0;
    if (c > 1e9) c = 1e9;
    c = std::sqrt(c); // gamma 2.0

    c = clamp_double(c, 0.0, 0.999);
    return static_cast<unsigned char>(256.0 * c);
}

struct rgb8 {
    unsigned char r;
    unsigned char g;
    unsigned char b;

    CUDA_HD
    rgb8() : r(0), g(0), b(0) {}

    CUDA_HD
    rgb8(unsigned char R, unsigned char G, unsigned char B)
        : r(R), g(G), b(B) {}
};

CUDA_HD
inline rgb8 pack_color(const vec3& pixel_color, int samples_per_pixel) {
    unsigned char ir = linear_to_byte(pixel_color.x(), samples_per_pixel);
    unsigned char ig = linear_to_byte(pixel_color.y(), samples_per_pixel);
    unsigned char ib = linear_to_byte(pixel_color.z(), samples_per_pixel);
    return rgb8(ir, ig, ib);
}

#ifndef __CUDACC__
inline void write_color(std::ostream &out,
                        const vec3& pixel_color,
                        int samples_per_pixel)
{
    unsigned char ir = linear_to_byte(pixel_color.x(), samples_per_pixel);
    unsigned char ig = linear_to_byte(pixel_color.y(), samples_per_pixel);
    unsigned char ib = linear_to_byte(pixel_color.z(), samples_per_pixel);

    out << (int)ir << ' '
        << (int)ig << ' '
        << (int)ib << '\n';
}
#endif

#endif // COLOR_H
