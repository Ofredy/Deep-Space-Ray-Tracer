#ifndef SPHERE_H
#define SPHERE_H

#include <memory>        // std::shared_ptr
#include <cmath>         // std::sqrt, std::acos, std::atan2, std::fmax
#include "hittable.h"
#include "rtweekend.h"

// We assume these types exist in your project:
//   - ray            (with origin(), direction(), time(), at(t))
//   - vec3 / point3  (point3 is typedef vec3 or similar)
//   - interval       (with surrounds(t))
//   - aabb
//   - material
//   - hit_record     (with t, p, u, v, mat_ptr, set_face_normal())
//   - pi             (in rtweekend.h)
// If any of those names are slightly different in *your* code, I'll call that out after.

static constexpr double pi = 3.1415926535897932385;

class sphere : public hittable {
public:
    // --------------------------
    // Constructors
    // --------------------------

    // --------------------------
    // Stationary Sphere
    // --------------------------
    sphere(const point3& static_center, double radius, std::shared_ptr<material> mat)
        : center(static_center, vec3(0,0,0)), radius(std::fmax(0.0, radius)), mat(mat)
    {
        vec3 rv(radius, radius, radius);
        bbox = aabb(static_center - rv, static_center + rv);
    }

    // Linearly-moving sphere
    sphere(const point3& c1,
           const point3& c2,
           double r,
           std::shared_ptr<material> m)
        : center(c1, c2 - c1),
          radius(std::fmax(0.0, r)),
          mat(m)
    {
        vec3 rv(radius, radius, radius);
    
        // bounding box at t = 0
        aabb box1(
            c1 - rv,
            c1 + rv
        );
    
        // bounding box at t = 1
        aabb box2(
            c2 - rv,
            c2 + rv
        );
    
        // merge them using your provided helper
        bbox = surrounding_box(box1, box2);
    }

    // --------------------------
    // hit()
    // --------------------------
    bool hit(const ray& r, const interval& ray_t, hit_record& rec) const override {
        // sphere center at this ray time
        point3 c = center.at(r.time());

        // geometric form
        vec3 oc = c - r.origin();
        double a = r.direction().length_squared();
        double h = dot(r.direction(), oc);
        double c_term = oc.length_squared() - radius * radius;

        double discriminant = h*h - a*c_term;
        if (discriminant < 0.0)
            return false;

        double sqrtd = std::sqrt(discriminant);

        // nearest root in range
        double root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);

        vec3 outward_normal = (rec.p - c) / radius;
        rec.set_face_normal(r, outward_normal);

        get_sphere_uv(outward_normal, rec.u, rec.v);

        rec.mat_ptr = mat;

        return true;
    }

    // --------------------------
    // bounding_box()
    // --------------------------
    aabb bounding_box() const override {
        return bbox;
    }

    // --------------------------
    // Lighting / sampling API hooks
    // We stub these because you're not using them
    // in the GPU path yet, but base class requires them.
    // --------------------------
    double pdf_value(const point3& /*origin*/, const vec3& /*direction*/) const override {
        return 0.0;
    }

    vec3 random(const point3& /*origin*/) const override {
        return vec3(1,0,0);
    }

    // --------------------------
    // Helpers for GPU scene builder
    // --------------------------
    point3 static_center() const {
        // position at time = 0
        return center.at(0);
    }

    double get_radius() const {
        return radius;
    }

    std::shared_ptr<material> get_mat() const {
        return mat;
    }

    std::shared_ptr<material> get_material() const { return mat; }

private:
    // Center is stored as a "ray": origin = c1, direction = (c2-c1),
    // so center.at(t) gives interpolated center. That's how your
    // moving-sphere constructor works.
    ray center;
    double radius;
    std::shared_ptr<material> mat;
    aabb bbox;

    // Spherical UV mapping for textures
    static void get_sphere_uv(const point3& p, double& u, double& v) {
        // p: point on the unit sphere (radius 1, centered at origin)
        // u: angle around Y from X = -1, range [0,1]
        // v: angle from Y=-1 to Y=+1, range [0,1]

        double theta = std::acos(-p.y());
        double phi   = std::atan2(-p.z(), p.x()) + pi;

        u = phi   / (2*pi);
        v = theta / pi;
    }
};

#endif // SPHERE_H
