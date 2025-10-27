#ifndef BVH_H
#define BVH_H

#include <algorithm>
#include <memory>
#include <vector>

#include "hittable.h"
#include "hittable_list.h"
#include "aabb.h"
#include "interval.h"

// BVH node for accelerating ray/object intersection
class bvh_node : public hittable {
public:
    std::shared_ptr<hittable> left;
    std::shared_ptr<hittable> right;
    aabb bbox;

    bvh_node() = default;

    // Build from a slice of objects [start, end)
    bvh_node(
        std::vector<std::shared_ptr<hittable>>& objects,
        size_t start,
        size_t end
    ) {
        // we’ll just sort on X for determinism
        int axis = 0;

        auto comparator =
            (axis == 0) ? box_x_compare
          : (axis == 1) ? box_y_compare
                        : box_z_compare;

        size_t object_span = end - start;

        if (object_span == 1) {
            left = right = objects[start];
        } else if (object_span == 2) {
            if (comparator(objects[start], objects[start+1])) {
                left  = objects[start];
                right = objects[start+1];
            } else {
                left  = objects[start+1];
                right = objects[start];
            }
        } else {
            std::sort(objects.begin() + start, objects.begin() + end, comparator);
            size_t mid = start + object_span/2;

            left  = std::make_shared<bvh_node>(objects, start, mid);
            right = std::make_shared<bvh_node>(objects, mid, end);
        }

        aabb box_left  = left->bounding_box();
        aabb box_right = right->bounding_box();
        bbox = surrounding_box(box_left, box_right);
    }

    // helper: build from a hittable_list
    bvh_node(const hittable_list& list)
        : bvh_node(
            const_cast<std::vector<std::shared_ptr<hittable>>&>(list.objects),
            0,
            list.objects.size()
        )
    {}

    bool hit(const ray& r, const interval& ray_t, hit_record& rec) const override {
        // fast reject using the bbox
        if (!bbox.hit(r, ray_t))
            return false;

        // try left
        bool hit_left = left->hit(r, ray_t, rec);

        // if we hit left, clamp tmax for right
        double tmax = hit_left ? rec.t : ray_t.max();
        interval right_range(ray_t.min(), tmax);

        hit_record right_rec;
        bool hit_right = right->hit(r, right_range, right_rec);

        if (hit_right) {
            rec = right_rec;
        }

        return hit_left || hit_right;
    }

    aabb bounding_box() const override {
        return bbox;
    }

    double pdf_value(const point3& origin, const vec3& direction) const override {
        // naive mixture of the two children
        return 0.5 * left->pdf_value(origin, direction)
             + 0.5 * right->pdf_value(origin, direction);
    }

    vec3 random(const point3& origin) const override {
        // choose left child for determinism
        return left->random(origin);
    }

private:
    static bool box_compare(
        const std::shared_ptr<hittable>& a,
        const std::shared_ptr<hittable>& b,
        int axis
    ) {
        aabb box_a = a->bounding_box();
        aabb box_b = b->bounding_box();
        return box_a.min()[axis] < box_b.min()[axis];
    }

    static bool box_x_compare(
        const std::shared_ptr<hittable>& a,
        const std::shared_ptr<hittable>& b
    ) {
        return box_compare(a,b,0);
    }
    static bool box_y_compare(
        const std::shared_ptr<hittable>& a,
        const std::shared_ptr<hittable>& b
    ) {
        return box_compare(a,b,1);
    }
    static bool box_z_compare(
        const std::shared_ptr<hittable>& a,
        const std::shared_ptr<hittable>& b
    ) {
        return box_compare(a,b,2);
    }
};

#endif // BVH_H
