#ifndef TEXTURE_H
#define TEXTURE_H

#include <memory>
#include <cmath>

#include "vec3.h"
#include "color.h"
#include "perlin.h"
#include "rtweekend.h"

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

#endif // TEXTURE_H
