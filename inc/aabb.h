#ifndef AABB_H
#define AABB_H

#include <algorithm> // for std::swap, std::fmin, std::fmax
#include <cmath>

#include "cuda_compat.h"
#include "vec3.h"
#include "ray.h"
#include "interval.h"

// Axis-Aligned Bounding Box
class aabb {
public:
    point3 minimum;
    point3 maximum;

    CUDA_HD
    aabb() {}

    CUDA_HD
    aabb(const point3& a, const point3& b)
        : minimum(a), maximum(b) {}

    CUDA_HD
    const point3& min() const { return minimum; }

    CUDA_HD
    const point3& max() const { return maximum; }

    // slab test against [ray_t.min(), ray_t.max()]
    CUDA_HD
    bool hit(const ray& r, const interval& ray_t) const {
        double tmin = ray_t.min();
        double tmax = ray_t.max();

        // x, y, z = indices 0,1,2
        for (int axis = 0; axis < 3; axis++) {
            double invD  = 1.0 / r.direction()[axis];
            double origA = r.origin()[axis];

            double t0 = (min()[axis] - origA) * invD;
            double t1 = (max()[axis] - origA) * invD;

            if (invD < 0.0)
                std::swap(t0, t1);

            if (t0 > tmin) tmin = t0;
            if (t1 < tmax) tmax = t1;

            if (tmax <= tmin)
                return false;
        }

        return true;
    }
};

// build the tightest box that contains both boxes
CUDA_HD
inline aabb surrounding_box(const aabb& box0, const aabb& box1) {
    point3 small(
        std::fmin(box0.min().x(), box1.min().x()),
        std::fmin(box0.min().y(), box1.min().y()),
        std::fmin(box0.min().z(), box1.min().z())
    );

    point3 big(
        std::fmax(box0.max().x(), box1.max().x()),
        std::fmax(box0.max().y(), box1.max().y()),
        std::fmax(box0.max().z(), box1.max().z())
    );

    return aabb(small, big);
}

#endif // AABB_H
