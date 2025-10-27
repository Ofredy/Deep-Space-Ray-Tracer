#ifndef HITTABLE_H
#define HITTABLE_H

#include <memory>
#include <cmath>

#include "cuda_compat.h"
#include "vec3.h"
#include "ray.h"
#include "aabb.h"
#include "interval.h"
#include "rtweekend.h"

class material; // forward declare, we just store pointer

// Info for a ray/object intersection
struct hit_record {
    point3 p;                      // hit point
    vec3   normal;                 // shading normal
    std::shared_ptr<material> mat_ptr;
    double t;                      // ray parameter
    double u;
    double v;
    bool   front_face;

    inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0.0;
        normal     = front_face ? outward_normal : -outward_normal;
    }
};

// Base interface for anything you can hit.
class hittable {
public:
    virtual ~hittable() = default;

    // geometric hit test
    virtual bool hit(
        const ray& r,
        const interval& ray_t,
        hit_record& rec
    ) const = 0;

    // axis-aligned bounding box
    virtual aabb bounding_box() const = 0;

    // optional: importance sampling helpers for lights / PDFs
    virtual double pdf_value(const point3& origin, const vec3& direction) const {
        // default: not a light
        return 0.0;
    }

    virtual vec3 random(const point3& origin) const {
        // default: arbitrary direction
        return vec3(1,0,0);
    }
};


// flip_face: wraps an object but flips the normal
class flip_face : public hittable {
public:
    std::shared_ptr<hittable> object;

    flip_face(std::shared_ptr<hittable> p)
        : object(std::move(p)) {}

    bool hit(const ray& r, const interval& ray_t, hit_record& rec) const override {
        if (!object->hit(r, ray_t, rec))
            return false;

        rec.front_face = !rec.front_face;
        rec.normal = -rec.normal;
        return true;
    }

    aabb bounding_box() const override {
        return object->bounding_box();
    }

    double pdf_value(const point3& origin, const vec3& direction) const override {
        return object->pdf_value(origin, direction);
    }

    vec3 random(const point3& origin) const override {
        return object->random(origin);
    }
};


// translate: wraps an object and offsets it in space
class translate : public hittable {
public:
    std::shared_ptr<hittable> object;
    vec3 offset;

    translate(std::shared_ptr<hittable> obj, const vec3& displacement)
        : object(std::move(obj)), offset(displacement) {}

    bool hit(const ray& r, const interval& ray_t, hit_record& rec) const override {
        // move the ray *back* into the object's local space
        ray moved_r(r.origin() - offset, r.direction(), r.time());

        if (!object->hit(moved_r, ray_t, rec))
            return false;

        // move intersection point back into world space
        rec.p += offset;
        rec.set_face_normal(moved_r, rec.normal);
        return true;
    }

    aabb bounding_box() const override {
        aabb child_box = object->bounding_box();
        return aabb(
            child_box.min() + offset,
            child_box.max() + offset
        );
    }

    double pdf_value(const point3& origin, const vec3& direction) const override {
        return object->pdf_value(origin - offset, direction);
    }

    vec3 random(const point3& origin) const override {
        return object->random(origin - offset);
    }
};


// rotate_y: rotate an object around world Y by some angle in degrees
class rotate_y : public hittable {
public:
    std::shared_ptr<hittable> object;
    double sin_theta;
    double cos_theta;
    aabb bbox;

    rotate_y(std::shared_ptr<hittable> obj, double angle_degrees)
        : object(std::move(obj))
    {
        double radians = degrees_to_radians(angle_degrees);
        sin_theta = std::sin(radians);
        cos_theta = std::cos(radians);

        // compute rotated bbox
        aabb b = object->bounding_box();

        point3 min_pt(  rt_infinity(),  rt_infinity(),  rt_infinity() );
        point3 max_pt( -rt_infinity(), -rt_infinity(), -rt_infinity() );

        for (int xi = 0; xi < 2; xi++) {
            double x = xi ? b.max().x() : b.min().x();
            for (int zi = 0; zi < 2; zi++) {
                double z = zi ? b.max().z() : b.min().z();
                for (int yi = 0; yi < 2; yi++) {
                    double y = yi ? b.max().y() : b.min().y();

                    double newx =  cos_theta * x + sin_theta * z;
                    double newz = -sin_theta * x + cos_theta * z;

                    vec3 tester(newx, y, newz);

                    min_pt = point3(
                        std::fmin(min_pt.x(), tester.x()),
                        std::fmin(min_pt.y(), tester.y()),
                        std::fmin(min_pt.z(), tester.z())
                    );
                    max_pt = point3(
                        std::fmax(max_pt.x(), tester.x()),
                        std::fmax(max_pt.y(), tester.y()),
                        std::fmax(max_pt.z(), tester.z())
                    );
                }
            }
        }

        bbox = aabb(min_pt, max_pt);
    }

    bool hit(const ray& r, const interval& ray_t, hit_record& rec) const override {
        // inverse-rotate the ray to test in object space
        vec3 orig = r.origin();
        vec3 dir  = r.direction();

        double rotx =  cos_theta * orig.x() - sin_theta * orig.z();
        double rotz =  sin_theta * orig.x() + cos_theta * orig.z();
        vec3 rotated_origin(rotx, orig.y(), rotz);

        double rdx =  cos_theta * dir.x() - sin_theta * dir.z();
        double rdz =  sin_theta * dir.x() + cos_theta * dir.z();
        vec3 rotated_dir(rdx, dir.y(), rdz);

        ray rotated_r(rotated_origin, rotated_dir, r.time());

        if (!object->hit(rotated_r, ray_t, rec))
            return false;

        // rotate hit point and normal back to world
        vec3 p_hit = rec.p;
        vec3 n_hit = rec.normal;

        double unrot_x =  cos_theta * p_hit.x() + sin_theta * p_hit.z();
        double unrot_z = -sin_theta * p_hit.x() + cos_theta * p_hit.z();
        rec.p = vec3(unrot_x, p_hit.y(), unrot_z);

        double unrot_nx =  cos_theta * n_hit.x() + sin_theta * n_hit.z();
        double unrot_nz = -sin_theta * n_hit.x() + cos_theta * n_hit.z();
        n_hit = vec3(unrot_nx, n_hit.y(), unrot_nz);

        rec.set_face_normal(rotated_r, n_hit);
        return true;
    }

    aabb bounding_box() const override {
        return bbox;
    }

    double pdf_value(const point3& origin, const vec3& direction) const override {
        // forward to child, but rotate origin opposite Y-rot
        double rotx =  cos_theta * origin.x() - sin_theta * origin.z();
        double rotz =  sin_theta * origin.x() + cos_theta * origin.z();
        point3 rotated_origin(rotx, origin.y(), rotz);
        return object->pdf_value(rotated_origin, direction);
    }

    vec3 random(const point3& origin) const override {
        double rotx =  cos_theta * origin.x() - sin_theta * origin.z();
        double rotz =  sin_theta * origin.x() + cos_theta * origin.z();
        point3 rotated_origin(rotx, origin.y(), rotz);
        return object->random(rotated_origin);
    }
};

#endif // HITTABLE_H
