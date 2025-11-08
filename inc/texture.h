#ifndef TEXTURE_H
#define TEXTURE_H

#include <memory>
#include <cmath>

#include "vec3.h"
#include "color.h"
#include "perlin.h"
#include "rtweekend.h"
#include "stb_image.h"

// Base texture class
class texture {
public:
    virtual ~texture() = default;

    virtual color value(
        double u, double v, const point3& p
    ) const = 0;
};

// Solid color texture
class solid_color : public texture {
public:
    color albedo;

    solid_color() : albedo(0,0,0) {}
    solid_color(color c) : albedo(c) {}
    solid_color(double r, double g, double b) : albedo(r,g,b) {}

    color value(double u, double v, const point3& p) const override {
        return albedo;
    }
};

// Checkerboard texture (3D)
class checker_texture : public texture {
public:
    std::shared_ptr<texture> even;
    std::shared_ptr<texture> odd;
    double scale;

    checker_texture() : scale(10.0) {}

    checker_texture(
        std::shared_ptr<texture> t0,
        std::shared_ptr<texture> t1,
        double sc = 10.0
    )
        : even(std::move(t0)), odd(std::move(t1)), scale(sc)
    {}

    checker_texture(color c1, color c2, double sc = 10.0)
        : scale(sc)
    {
        even = std::make_shared<solid_color>(c1);
        odd  = std::make_shared<solid_color>(c2);
    }

    color value(double u, double v, const point3& p) const override {
        double sines = std::sin(scale * p.x())
                     * std::sin(scale * p.y())
                     * std::sin(scale * p.z());
        if (sines < 0)
            return odd->value(u,v,p);
        else
            return even->value(u,v,p);
    }
};

// Perlin noise / marble-ish texture
class noise_texture : public texture {
public:
    perlin noise_src;
    double scale;

    noise_texture() : scale(1.0) {}
    noise_texture(double sc) : scale(sc) {}

    color value(double u, double v, const point3& p) const override {
        // Marble pattern
        return 0.5 * color(1,1,1) *
               (1.0 + std::sin(scale * p.z() + 10.0 * noise_src.turb(p)));
    }
};

inline double clampd(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

class image_texture : public texture {
public:
    explicit image_texture(const std::string& filename)
        : data(nullptr), width(0), height(0), bytes_per_pixel(0)
    {
        if (!filename.empty()) load(filename);
}

    ~image_texture() {
        if (data) stbi_image_free(data);
    }

    color value(double u, double v, const point3& /*p*/) const override {
        if (!data) return color(0,1,1); // cyan if missing

        // clamp + flip V to image convention
        u = clampd(u, 0.0, 1.0);
        v = 1.0 - clampd(v, 0.0, 1.0);

        int i = static_cast<int>(u * width);
        int j = static_cast<int>(v * height);
        if (i >= width)  i = width  - 1;
        if (j >= height) j = height - 1;

        const int idx = (j * width + i) * bytes_per_pixel;
        const double s = 1.0 / 255.0;

        // assume 3 channels (we force 3 at load)
        return color(s * data[idx + 0],
                     s * data[idx + 1],
                     s * data[idx + 2]);
    }

private:
    unsigned char* data;
    int width, height, bytes_per_pixel;

    void load(const std::string& filename) {
        int n = 0;
        stbi_set_flip_vertically_on_load(true);
        data = stbi_load(filename.c_str(), &width, &height, &n, 3); // force RGB
        bytes_per_pixel = 3;
        if (!data) {
            // silent fail; value() returns cyan to signal missing
            width = height = 0;
        }
    }
};
#endif // TEXTURE_H
