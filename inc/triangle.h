#pragma once
#include "rtweekend.h"     // for unit_vector(...)
#include "hittable.h"
#include "material.h"
#include "aabb.h"

class triangle : public hittable {
public:
    vec3   v0, v1, v2;
    vec3   n0, n1, n2;
    vec3   uv0, uv1, uv2;    // vec3 used for UVs (u,v,0)
    std::shared_ptr<material> mat;

    triangle() = default;

    triangle(const vec3& a, const vec3& b, const vec3& c, std::shared_ptr<material> m)
        : v0(a), v1(b), v2(c), mat(std::move(m)) {
        compute_normals();
        uv0 = vec3(0.0, 0.0, 0.0);
        uv1 = vec3(0.0, 0.0, 0.0);
        uv2 = vec3(0.0, 0.0, 0.0);
    }

    triangle(const vec3& a, const vec3& b, const vec3& c,
             const vec3& uva, const vec3& uvb, const vec3& uvc,
             std::shared_ptr<material> m)
        : v0(a), v1(b), v2(c), uv0(uva), uv1(uvb), uv2(uvc), mat(std::move(m)) {
        compute_normals();
    }

    bool hit(const ray& r, const interval& ray_t, hit_record& rec) const override {
        const vec3 e1 = v1 - v0;
        const vec3 e2 = v2 - v0;
        const vec3 p  = cross(r.direction(), e2);
        const double det = dot(e1, p);
        if (fabs(det) < 1e-9) return false;
        const double invDet = 1.0 / det;

        const vec3 tvec = r.origin() - v0;
        const double u = dot(tvec, p) * invDet;
        if (u < 0.0 || u > 1.0) return false;

        const vec3 qvec = cross(tvec, e1);
        const double v = dot(r.direction(), qvec) * invDet;
        if (v < 0.0 || u + v > 1.0) return false;

        const double t = dot(e2, qvec) * invDet;
        if (!ray_t.surrounds(t)) return false;

        rec.t = t;
        rec.p = r.at(t);
        rec.set_face_normal(r, unit_vector(cross(e1, e2)));

        // NOTE: If your hit_record uses a different field name than mat_ptr, rename accordingly.
        rec.mat_ptr = mat;

        // Interpolate UV
        rec.u = (float)((1.0 - u - v) * uv0.x() + u * uv1.x() + v * uv2.x());
        rec.v = (float)((1.0 - u - v) * uv0.y() + u * uv1.y() + v * uv2.y());
        return true;
    }

    aabb bounding_box() const override {
        aabb box(v0, v1);
        box = surrounding_box(box, aabb(v2, v2));
        return box;
    }

private:
    void compute_normals() {
        const vec3 n = unit_vector(cross(v1 - v0, v2 - v0));
        n0 = n1 = n2 = n;
    }
};
