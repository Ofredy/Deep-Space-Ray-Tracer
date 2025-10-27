#ifndef RAY_H
#define RAY_H

#include "cuda_compat.h"
#include "vec3.h"

class ray {
public:
    vec3 orig;
    vec3 dir;
    double tm; // time, if you support motion blur; can default to 0

    CUDA_HD
    ray() : orig(), dir(), tm(0.0) {}

    CUDA_HD
    ray(const vec3& origin, const vec3& direction, double time = 0.0)
        : orig(origin), dir(direction), tm(time) {}

    CUDA_HD
    const vec3& origin() const { return orig; }

    CUDA_HD
    const vec3& direction() const { return dir; }

    CUDA_HD
    double time() const { return tm; }

    CUDA_HD
    vec3 at(double t) const {
        return orig + t * dir;
    }
};

#endif // RAY_H
