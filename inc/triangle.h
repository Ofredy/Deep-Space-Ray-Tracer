#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "hittable.h"
#include "rtweekend.h"

// Basic geometric triangle (no UVs). Good for solid color Kd.
class triangle : public hittable {
  public:
    triangle(const point3& a, const point3& b, const point3& c, shared_ptr<material> m)
      : p0(a), p1(b), p2(c), mat(m)
    {
        e1 = p1 - p0;
        e2 = p2 - p0;
        n  = unit_vector(cross(e1, e2));

        // AABB with tiny padding for thin triangles
        vec3 minv(fmin(fmin(p0.x(),p1.x()),p2.x()),
                  fmin(fmin(p0.y(),p1.y()),p2.y()),
                  fmin(fmin(p0.z(),p1.z()),p2.z()));
        vec3 maxv(fmax(fmax(p0.x(),p1.x()),p2.x()),
                  fmax(fmax(p0.y(),p1.y()),p2.y()),
                  fmax(fmax(p0.z(),p1.z()),p2.z()));
        const double pad = 1e-4;
        if (maxv.x()-minv.x()<pad) { minv[0]-=pad; maxv[0]+=pad; }
        if (maxv.y()-minv.y()<pad) { minv[1]-=pad; maxv[1]+=pad; }
        if (maxv.z()-minv.z()<pad) { minv[2]-=pad; maxv[2]+=pad; }
        bbox_ = aabb(point3(minv.x(),minv.y(),minv.z()),
                     point3(maxv.x(),maxv.y(),maxv.z()));
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        // Möller–Trumbore
        const vec3 pvec = cross(r.direction(), e2);
        const double det = dot(e1, pvec);
        if (fabs(det) < 1e-12) return false;
        const double invDet = 1.0 / det;

        const vec3 tvec = r.origin() - p0;
        const double u = dot(tvec, pvec) * invDet;
        if (u < 0.0 || u > 1.0) return false;

        const vec3 qvec = cross(tvec, e1);
        const double v = dot(r.direction(), qvec) * invDet;
        if (v < 0.0 || u + v > 1.0) return false;

        const double t = dot(e2, qvec) * invDet;
        if (!ray_t.contains(t)) return false;

        rec.t = t;
        rec.p = r.at(t);
        rec.set_face_normal(r, n);

        // These are barycentric (not texture UVs). Kept for compatibility.
        rec.u = u;
        rec.v = v;

        rec.mat = mat;
        return true;
    }

    aabb bounding_box() const override { return bbox_; }

  private:
    point3 p0, p1, p2;
    vec3   e1, e2, n;
    shared_ptr<material> mat;
    aabb bbox_;
};


// UV-capable triangle: interpolates OBJ vt into rec.u / rec.v so map_Kd works.
// UV-capable triangle: interpolates OBJ vt into rec.u / rec.v so map_Kd works.
class triangle_uv : public hittable {
  public:
    triangle_uv(const point3& a, const point3& b, const point3& c,
                const vec3& t0, const vec3& t1, const vec3& t2,
                shared_ptr<material> m)
      : p0(a), p1(b), p2(c), mat(m)
    {
        // set UVs (anonymous POD -> no ctor; assign fields)
        uv0.x = t0.x(); uv0.y = t0.y();
        uv1.x = t1.x(); uv1.y = t1.y();
        uv2.x = t2.x(); uv2.y = t2.y();

        // precompute edges and normal
        e1 = p1 - p0;
        e2 = p2 - p0;
        n  = unit_vector(cross(e1, e2));

        // AABB with tiny padding for thin triangles
        vec3 minv(fmin(fmin(p0.x(),p1.x()),p2.x()),
                  fmin(fmin(p0.y(),p1.y()),p2.y()),
                  fmin(fmin(p0.z(),p1.z()),p2.z()));
        vec3 maxv(fmax(fmax(p0.x(),p1.x()),p2.x()),
                  fmax(fmax(p0.y(),p1.y()),p2.y()),
                  fmax(fmax(p0.z(),p1.z()),p2.z()));
        const double pad = 1e-4;
        if (maxv.x()-minv.x()<pad) { minv[0]-=pad; maxv[0]+=pad; }
        if (maxv.y()-minv.y()<pad) { minv[1]-=pad; maxv[1]+=pad; }
        if (maxv.z()-minv.z()<pad) { minv[2]-=pad; maxv[2]+=pad; }
        bbox_ = aabb(point3(minv.x(),minv.y(),minv.z()),
                     point3(maxv.x(),maxv.y(),maxv.z()));
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        // Möller–Trumbore
        const vec3 pvec = cross(r.direction(), e2);
        const double det = dot(e1, pvec);
        if (fabs(det) < 1e-12) return false;
        const double invDet = 1.0 / det;

        const vec3 tvec = r.origin() - p0;
        const double u = dot(tvec, pvec) * invDet;
        if (u < 0.0 || u > 1.0) return false;

        const vec3 qvec = cross(tvec, e1);
        const double v = dot(r.direction(), qvec) * invDet;
        if (v < 0.0 || u + v > 1.0) return false;

        const double t = dot(e2, qvec) * invDet;
        if (!ray_t.contains(t)) return false;

        rec.t = t;
        rec.p = r.at(t);
        rec.set_face_normal(r, n);

        // Interpolate vt using barycentrics (w = 1-u-v)
        const double w = 1.0 - u - v;
        rec.u = w*uv0.x + u*uv1.x + v*uv2.x;
        rec.v = w*uv0.y + u*uv1.y + v*uv2.y;

        rec.mat = mat;
        return true;
    }

    aabb bounding_box() const override { return bbox_; }

  private:
    point3 p0, p1, p2;
    vec3   e1, e2, n;
    struct { double x, y; } uv0, uv1, uv2;  // anonymous PODs (no ctors)
    shared_ptr<material> mat;
    aabb bbox_;
};

#endif // TRIANGLE_H
