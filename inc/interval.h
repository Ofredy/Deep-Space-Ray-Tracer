#ifndef INTERVAL_H
#define INTERVAL_H

#include "cuda_compat.h"
#include <limits>

// A simple [min, max] numeric range used for t_min/t_max,
// clamping, etc. No iostream, no STL containers.

class interval {
public:
    double min_val;
    double max_val;

    // default = "empty" interval
    CUDA_HD
    interval()
        : min_val(+std::numeric_limits<double>::infinity()),
          max_val(-std::numeric_limits<double>::infinity()) {}

    CUDA_HD
    interval(double min_v, double max_v)
        : min_val(min_v), max_val(max_v) {}

    CUDA_HD
    double min() const { return min_val; }

    CUDA_HD
    double max() const { return max_val; }

    // closed interval check [min,max]
    CUDA_HD
    bool contains(double x) const {
        return (x >= min_val) && (x <= max_val);
    }

    // open interval check (min,max)
    CUDA_HD
    bool surrounds(double x) const {
        return (x > min_val) && (x < max_val);
    }

    // clamp x to [min_val, max_val]
    CUDA_HD
    double clamp(double x) const {
        if (x < min_val) return min_val;
        if (x > max_val) return max_val;
        return x;
    }
};

// A convenient "everything" interval
CUDA_HD
inline interval interval_universe() {
    return interval(
        -std::numeric_limits<double>::infinity(),
        +std::numeric_limits<double>::infinity()
    );
}

// Helpers to shrink one side of an interval, which is handy in BVH hit logic
CUDA_HD
inline interval interval_shrink_min(const interval& i, double new_min) {
    return interval(new_min, i.max_val);
}

CUDA_HD
inline interval interval_shrink_max(const interval& i, double new_max) {
    return interval(i.min_val, new_max);
}

#endif // INTERVAL_H
