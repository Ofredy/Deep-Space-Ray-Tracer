#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "vec3.h"
#include "ray.h"
#include "interval.h"
#include "hittable.h"
#include "aabb.h"
#include <cmath>
#include <memory>

// A single triangle that can be hit by rays and that knows its material.
struct triangle : public hittable {
    vec3 v0, v1, v2;        // positions
    vec3 n0, n1, n2;        // vertex normals (can all be same)
    std::shared_ptr<material> mat;

    aabb bbox_cached;

    triangle() {}

    triangle(const vec3& _v0,
             const vec3& _v1,
             const vec3& _v2,
             const std::shared_ptr<material>& m)
        : v0(_v0), v1(_v1), v2(_v2), mat(m)
    {
        // flat normal for now
        vec3 face_n = unit_vector(cross(v1 - v0, v2 - v0));
        n0 = face_n;
        n1 = face_n;
        n2 = face_n;

        // precompute bbox
        auto min_x = std::fmin(std::fmin(v0.x(), v1.x()), v2.x());
        auto min_y = std::fmin(std::fmin(v0.y(), v1.y()), v2.y());
        auto min_z = std::fmin(std::fmin(v0.z(), v1.z()), v2.z());

        auto max_x = std::fmax(std::fmax(v0.x(), v1.x()), v2.x());
        auto max_y = std::fmax(std::fmax(v0.y(), v1.y()), v2.y());
        auto max_z = std::fmax(std::fmax(v0.z(), v1.z()), v2.z());

        bbox_cached = aabb(
            vec3(min_x, min_y, min_z),
            vec3(max_x, max_y, max_z)
        );
    }

    // Möller–Trumbore ray/triangle hit
    bool hit(const ray& r, const interval& ray_t, hit_record& rec) const override {
        const double EPS = 1e-8;

        vec3 e1 = v1 - v0;
        vec3 e2 = v2 - v0;

        vec3 pvec = cross(r.direction(), e2);
        double det = dot(e1, pvec);
        if (fabs(det) < EPS) return false;
        double invDet = 1.0 / det;

        vec3 tvec = r.origin() - v0;
        double u = dot(tvec, pvec) * invDet;
        if (u < 0.0 || u > 1.0) return false;

        vec3 qvec = cross(tvec, e1);
        double v = dot(r.direction(), qvec) * invDet;
        if (v < 0.0 || u + v > 1.0) return false;

        double t = dot(e2, qvec) * invDet;
        if (t < ray_t.min() || t > ray_t.max()) return false;

        // hit record fill
        rec.t = t;
        rec.p = r.at(t);

        vec3 shading_normal = unit_vector(
            (1.0 - u - v) * n0 +
            u * n1 +
            v * n2
        );
        rec.set_face_normal(r, shading_normal);

        rec.mat_ptr = mat;
        return true;
    }

    virtual aabb bounding_box() const override {
        return bbox_cached;
    }
};

#endif // TRIANGLE_H
