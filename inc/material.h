#ifndef MATERIAL_H
#define MATERIAL_H

#include <memory>
#include <cmath>

#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "hittable.h"
#include "pdf.h"
#include "rtweekend.h"
#include "texture.h"

// =======================================================
// scatter_record
// =======================================================
struct scatter_record {
    ray specular_ray;
    bool skip_pdf;
    color attenuation;
    std::shared_ptr<pdf> pdf_ptr;
};

// =======================================================
// Utility helpers
// =======================================================
inline double reflectance(double cosine, double ref_idx) {
    double r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * std::pow((1 - cosine), 5.0);
}

// =======================================================
// Base material
// =======================================================
class material {
public:
    virtual ~material() = default;

    virtual color emitted(
        const ray& r_in,
        const hit_record& rec,
        double u, double v,
        const point3& p
    ) const {
        return color(0,0,0);
    }

    virtual bool scatter(
        const ray& r_in,
        const hit_record& rec,
        scatter_record& srec
    ) const {
        return false;
    }

    virtual double scattering_pdf(
        const ray& r_in,
        const hit_record& rec,
        const ray& scattered
    ) const {
        return 0.0;
    }
};

// =======================================================
// Lambertian (diffuse)
// =======================================================
class lambertian : public material {
public:
    std::shared_ptr<texture> albedo;

    lambertian(const color& a) {
        albedo = std::make_shared<solid_color>(a);
    }

    lambertian(std::shared_ptr<texture> tex)
        : albedo(std::move(tex)) {}

    bool scatter(
        const ray& r_in,
        const hit_record& rec,
        scatter_record& srec
    ) const override {
        srec.skip_pdf   = false;
        srec.specular_ray = ray();
        srec.attenuation  = albedo->value(rec.u, rec.v, rec.p);
        srec.pdf_ptr = std::make_shared<cosine_pdf>(rec.normal);
        return true;
    }

    double scattering_pdf(
        const ray& r_in,
        const hit_record& rec,
        const ray& scattered
    ) const override {
        double cosine = dot(rec.normal, unit_vector(scattered.direction()));
        return (cosine <= 0.0) ? 0.0 : cosine / rt_pi();
    }

    // -------- Getter for GPU builder --------
    color albedo_value() const {
        auto solid = std::dynamic_pointer_cast<solid_color>(albedo);
        if (solid)
            return solid->value(0,0,point3());
        // fallback gray
        return color(0.8,0.8,0.8);
    }
};

// =======================================================
// Metal (reflective)
// =======================================================
class metal : public material {
public:
    color albedo;
    double fuzz;

    metal(const color& a, double f)
        : albedo(a), fuzz(f < 1 ? f : 1) {}

    bool scatter(
        const ray& r_in,
        const hit_record& rec,
        scatter_record& srec
    ) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        vec3 perturbed = reflected + fuzz * random_in_unit_sphere_host();

        srec.specular_ray = ray(rec.p, perturbed, r_in.time());
        srec.attenuation  = albedo;
        srec.skip_pdf     = true;
        srec.pdf_ptr      = nullptr;
        return (dot(srec.specular_ray.direction(), rec.normal) > 0.0);
    }

    // -------- Getters for GPU builder --------
    color albedo_value() const { return albedo; }
    double fuzz_value() const { return fuzz; }
};

// =======================================================
// Dielectric (glass)
// =======================================================
class dielectric : public material {
public:
    double ir; // index of refraction

    dielectric(double index_of_refraction)
        : ir(index_of_refraction) {}

    bool scatter(
        const ray& r_in,
        const hit_record& rec,
        scatter_record& srec
    ) const override {
        srec.skip_pdf = true;
        srec.pdf_ptr  = nullptr;
        srec.attenuation = color(1.0, 1.0, 1.0);

        double refraction_ratio = rec.front_face ? (1.0/ir) : ir;

        vec3 unit_dir = unit_vector(r_in.direction());
        double cos_theta = fmin(dot(-unit_dir, rec.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract ||
            reflectance(cos_theta, refraction_ratio) > random_double_host()) {
            direction = reflect(unit_dir, rec.normal);
        } else {
            direction = refract(unit_dir, rec.normal, refraction_ratio);
        }

        srec.specular_ray = ray(rec.p, direction, r_in.time());
        return true;
    }

    // -------- Getter for GPU builder --------
    double ior_value() const { return ir; }
};

// =======================================================
// Diffuse Light (emissive)
// =======================================================
class diffuse_light : public material {
public:
    std::shared_ptr<texture> emit;

    diffuse_light(const color& c) {
        emit = std::make_shared<solid_color>(c);
    }

    diffuse_light(std::shared_ptr<texture> tex)
        : emit(std::move(tex)) {}

    color emitted(
        const ray& r_in,
        const hit_record& rec,
        double u, double v,
        const point3& p
    ) const override {
        if (!rec.front_face)
            return color(0,0,0);
        return emit->value(u, v, p);
    }

    bool scatter(
        const ray& r_in,
        const hit_record& rec,
        scatter_record& srec
    ) const override {
        return false;
    }

    // -------- Getter for GPU builder --------
    color emit_value() const {
        auto solid = std::dynamic_pointer_cast<solid_color>(emit);
        if (solid)
            return solid->value(0,0,point3());
        return color(1.0,1.0,1.0);
    }
};

#endif // MATERIAL_H
