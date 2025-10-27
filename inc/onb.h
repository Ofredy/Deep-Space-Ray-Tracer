#ifndef ONB_H
#define ONB_H

#include "cuda_compat.h"
#include "vec3.h"
#include <cmath>

class onb {
public:
    // axis[0] = u
    // axis[1] = v
    // axis[2] = w (usually the normal direction)
    vec3 axis[3];

    CUDA_HD
    onb() {
        axis[0] = vec3(1,0,0);
        axis[1] = vec3(0,1,0);
        axis[2] = vec3(0,0,1);
    }

    CUDA_HD vec3 u() const { return axis[0]; }
    CUDA_HD vec3 v() const { return axis[1]; }
    CUDA_HD vec3 w() const { return axis[2]; }

    // Convert local coordinates (a,b,c)
    // into world coordinates using this frame.
    CUDA_HD
    vec3 local(double a, double b, double c) const {
        return a * axis[0]
             + b * axis[1]
             + c * axis[2];
    }

    // Overload: takes a vec3 in local space (x,y,z)
    CUDA_HD
    vec3 local(const vec3& a) const {
        return a.x() * axis[0]
             + a.y() * axis[1]
             + a.z() * axis[2];
    }

    // Build this ONB from a single "w" direction (typically the surface normal).
    // We'll generate u,v to be orthonormal to w.
    CUDA_HD
    void build_from_w(const vec3& n) {
        axis[2] = unit_vector(n); // w = normalized n

        // choose a helper vector that's not almost parallel to w
        vec3 a = (fabs(axis[2].x()) > 0.9) ? vec3(0,1,0) : vec3(1,0,0);

        axis[1] = unit_vector(cross(axis[2], a)); // v = normalize(w x a)
        axis[0] = cross(axis[2], axis[1]);        // u = w x v
    }
};

#endif // ONB_H
