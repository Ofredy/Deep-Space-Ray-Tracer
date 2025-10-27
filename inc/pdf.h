#ifndef PDF_H
#define PDF_H

#include <memory>
#include <cmath>

#include "vec3.h"
#include "onb.h"
#include "rtweekend.h"
#include "hittable.h"

// Base PDF interface
class pdf {
public:
    virtual ~pdf() = default;

    virtual double value(const vec3& direction) const = 0;
    virtual vec3 generate() const = 0;
};

// cosine-weighted hemisphere around a normal
class cosine_pdf : public pdf {
public:
    onb uvw;

    cosine_pdf(const vec3& w) {
        uvw.build_from_w(w);
    }

    double value(const vec3& direction) const override {
        double cosine = dot(unit_vector(direction), uvw.w());
        return (cosine <= 0.0) ? 0.0 : cosine / rt_pi();
    }

    vec3 generate() const override {
        vec3 local_dir = random_cosine_direction_host();
        return uvw.local(local_dir);
    }
};

// samples directions toward a hittable (typically a light)
class hittable_pdf : public pdf {
public:
    std::shared_ptr<hittable> ptr;
    point3 origin;

    hittable_pdf(const std::shared_ptr<hittable>& p, const point3& o)
        : ptr(p), origin(o) {}

    double value(const vec3& direction) const override {
        return ptr->pdf_value(origin, direction);
    }

    vec3 generate() const override {
        return ptr->random(origin);
    }
};

// 50/50 mixture of two PDFs
class mixture_pdf : public pdf {
public:
    std::shared_ptr<pdf> p[2];

    mixture_pdf(std::shared_ptr<pdf> p0, std::shared_ptr<pdf> p1) {
        p[0] = p0;
        p[1] = p1;
    }

    double value(const vec3& direction) const override {
        return 0.5 * p[0]->value(direction)
             + 0.5 * p[1]->value(direction);
    }

    vec3 generate() const override {
        if (random_double_host() < 0.5)
            return p[0]->generate();
        return p[1]->generate();
    }
};

#endif // PDF_H
