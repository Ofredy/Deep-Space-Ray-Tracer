#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include <vector>
#include <memory>

#include "hittable.h"
#include "aabb.h"
#include "interval.h"

class hittable_list : public hittable {
public:
    std::vector<std::shared_ptr<hittable>> objects;

    hittable_list() = default;

    hittable_list(std::shared_ptr<hittable> object) {
        add(object);
    }

    void clear() {
        objects.clear();
    }

    void add(std::shared_ptr<hittable> object) {
        objects.push_back(object);
    }

    bool hit(
        const ray& r,
        const interval& ray_t,
        hit_record& rec
    ) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        double closest_so_far = ray_t.max();

        for (const auto& object : objects) {
            interval range(ray_t.min(), closest_so_far);
            if (object->hit(r, range, temp_rec)) {
                hit_anything   = true;
                closest_so_far = temp_rec.t;
                rec            = temp_rec;
            }
        }

        return hit_anything;
    }

    aabb bounding_box() const override {
        if (objects.empty()) {
            return aabb(point3(0,0,0), point3(0,0,0));
        }

        aabb box = objects[0]->bounding_box();
        for (size_t i = 1; i < objects.size(); i++) {
            box = surrounding_box(box, objects[i]->bounding_box());
        }
        return box;
    }

    double pdf_value(const point3& origin, const vec3& direction) const override {
        // avg pdf over all objects
        double weight = 1.0 / objects.size();
        double sum = 0.0;
        for (const auto& obj : objects) {
            sum += weight * obj->pdf_value(origin, direction);
        }
        return sum;
    }

    vec3 random(const point3& origin) const override {
        // pick first object to avoid RNG complexity
        return objects[0]->random(origin);
    }
};

#endif // HITTABLE_LIST_H
