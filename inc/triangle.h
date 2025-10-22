#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "hittable.h"
#include "rtweekend.h"

// Optional: only needed for random() if you want ONB transforms; not required here.
// #include "onb.h"

class triangle : public hittable {
  public:
    // ---- Stationary triangle ----
    triangle(const point3& a, const point3& b, const point3& c, shared_ptr<material> mat)
      : v0(a, vec3(0,0,0)), v1(b, vec3(0,0,0)), v2(c, vec3(0,0,0)), mat(mat)
    {
        build_bbox_for_time(0.0);
    }

    // ---- Moving triangle (linear motion over ray time in [0,1]) ----
    triangle(const point3& a0, const point3& b0, const point3& c0,
             const point3& a1, const point3& b1, const point3& c1,
             shared_ptr<material> mat)
      : v0(a0, a1 - a0), v1(b0, b1 - b0), v2(c0, c1 - c0), mat(mat)
    {
        // Union of boxes at t=0 and t=1
        aabb box0 = bbox_for_vertices(v0.at(0.0), v1.at(0.0), v2.at(0.0));
        aabb box1 = bbox_for_vertices(v0.at(1.0), v1.at(1.0), v2.at(1.0));
        bbox = aabb(box0, box1);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        // Evaluate vertices at ray time (matches your moving-sphere pattern)
        const point3 a = v0.at(r.time());
        const point3 b = v1.at(r.time());
        const point3 c = v2.at(r.time());

        // Möller–Trumbore
        const vec3 e1 = b - a;
        const vec3 e2 = c - a;

        const vec3 pvec = cross(r.direction(), e2);
        const double det = dot(e1, pvec);
        const double eps = 1e-12;

        if (fabs(det) < eps) return false; // Ray parallel to triangle
        const double invDet = 1.0 / det;

        const vec3 tvec = r.origin() - a;
        const double u = dot(tvec, pvec) * invDet;
        if (u < 0.0 || u > 1.0) return false;

        const vec3 qvec = cross(tvec, e1);
        const double v = dot(r.direction(), qvec) * invDet;
        if (v < 0.0 || u + v > 1.0) return false;

        const double t = dot(e2, qvec) * invDet;
        if (!ray_t.surrounds(t)) return false;

        rec.t = t;
        rec.p = r.at(rec.t);

        // Geometric normal (flat-shaded)
        vec3 outward_normal = unit_vector(cross(e1, e2));
        rec.set_face_normal(r, outward_normal);

        // Barycentric UVs: (u, v) from intersection; w = 1 - u - v
        rec.u = u;
        rec.v = v;

        rec.mat = mat;
        return true;
    }

    aabb bounding_box() const override { return bbox; }

    // pdf_value for area-sampled triangle:
    // pdf = distance^2 / (cos_theta * area) if the ray hits, else 0.
    double pdf_value(const point3& origin, const vec3& direction) const override {
        // Only reliable for stationary triangles (like your sphere note).
        // If moving, we evaluate at t=0 as an approximation.
        hit_record rec;
        if (!this->hit(ray(origin, unit_vector(direction)), interval(0.001, infinity), rec))
            return 0.0;

        // Triangle at t used in hit(), so its normal and point are consistent.
        // Compute area at the same time used for hit(): use t of ray (here unit ray with default time=0)
        // We’ll approximate with t=0 area to keep it simple and stable.
        const double A = area_at_time(0.0);
        if (A <= 0) return 0.0;

        const double dist2 = rec.t * rec.t * direction.length_squared(); // if dir normalized, = rec.t^2
        const double cos_theta = fabs(dot(unit_vector(direction), rec.normal));

        if (cos_theta <= 0) return 0.0;
        return dist2 / (cos_theta * A);
    }

    // Uniform-area direction toward a random point on the triangle
    vec3 random(const point3& origin) const override {
        // Only correct for stationary triangles. We use t=0 geometry.
        const point3 a = v0.at(0.0);
        const point3 b = v1.at(0.0);
        const point3 c = v2.at(0.0);

        // Uniform over triangle via sqrt trick
        const double r1 = random_double();
        const double r2 = random_double();
        const double su = std::sqrt(r1);
        const double u = 1.0 - su;
        const double v = r2 * su;
        const double w = 1.0 - u - v; (void)w; // for clarity; not needed explicitly

        point3 p = u*a + v*b + (1.0 - u - v)*c;
        return p - origin;
    }

  private:
    // Store per-vertex motion as rays (origin = start, direction = delta), like your sphere center
    ray v0, v1, v2;
    shared_ptr<material> mat;
    aabb bbox;

    static aabb bbox_for_vertices(const point3& a, const point3& b, const point3& c) {
        point3 minp(std::fmin(a.x(), std::fmin(b.x(), c.x())),
                    std::fmin(a.y(), std::fmin(b.y(), c.y())),
                    std::fmin(a.z(), std::fmin(b.z(), c.z())));
        point3 maxp(std::fmax(a.x(), std::fmax(b.x(), c.x())),
                    std::fmax(a.y(), std::fmax(b.y(), c.y())),
                    std::fmax(a.z(), std::fmax(b.z(), c.z())));

        // Expand a tad to avoid zero-thickness issues
        const double pad = 1e-6;
        minp = point3(minp.x()-pad, minp.y()-pad, minp.z()-pad);
        maxp = point3(maxp.x()+pad, maxp.y()+pad, maxp.z()+pad);

        return aabb(minp, maxp);
    }

    void build_bbox_for_time(double t) {
        bbox = bbox_for_vertices(v0.at(t), v1.at(t), v2.at(t));
    }

    double area_at_time(double t) const {
        const vec3 e1 = v1.at(t) - v0.at(t);
        const vec3 e2 = v2.at(t) - v0.at(t);
        return 0.5 * cross(e1, e2).length();
    }
};

#endif
