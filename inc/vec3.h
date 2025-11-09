#ifndef VEC3_H
#define VEC3_H

#include "cuda_compat.h"
#include <cmath>

#ifndef __CUDACC__
#include <iostream>
#endif

// ================================================
// float-based vec3 (GPU optimized)
// ================================================
class vec3 {
public:
    float e[3];

    CUDA_HD
    vec3() : e{0.0f, 0.0f, 0.0f} {}

    CUDA_HD
    vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    CUDA_HD float x() const { return e[0]; }
    CUDA_HD float y() const { return e[1]; }
    CUDA_HD float z() const { return e[2]; }

    CUDA_HD float r() const { return e[0]; }
    CUDA_HD float g() const { return e[1]; }
    CUDA_HD float b() const { return e[2]; }

    CUDA_HD vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

    CUDA_HD float operator[](int i) const { return e[i]; }
    CUDA_HD float& operator[](int i)      { return e[i]; }

    CUDA_HD vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    CUDA_HD vec3& operator*=(float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    CUDA_HD vec3& operator/=(float t) {
        float inv = 1.0f / t;
        e[0] *= inv;
        e[1] *= inv;
        e[2] *= inv;
        return *this;
    }

    CUDA_HD float length() const {
        return sqrtf(length_squared());
    }

    CUDA_HD float length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    CUDA_HD bool near_zero() const {
        const float s = 1e-6f;
        return (fabsf(e[0]) < s) && (fabsf(e[1]) < s) && (fabsf(e[2]) < s);
    }
};

// ================================================
// Free functions (float-based)
// ================================================
CUDA_HD
inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0]+v.e[0], u.e[1]+v.e[1], u.e[2]+v.e[2]);
}

CUDA_HD
inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0]-v.e[0], u.e[1]-v.e[1], u.e[2]-v.e[2]);
}

CUDA_HD
inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0]*v.e[0], u.e[1]*v.e[1], u.e[2]*v.e[2]);
}

CUDA_HD
inline vec3 operator*(float t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

CUDA_HD
inline vec3 operator*(const vec3& v, float t) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

CUDA_HD
inline vec3 operator/(const vec3& v, float t) {
    float inv = 1.0f / t;
    return vec3(v.e[0]*inv, v.e[1]*inv, v.e[2]*inv);
}

CUDA_HD
inline float dot(const vec3& u, const vec3& v) {
    return u.e[0]*v.e[0] + u.e[1]*v.e[1] + u.e[2]*v.e[2];
}

CUDA_HD
inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(
        u.e[1]*v.e[2] - u.e[2]*v.e[1],
        u.e[2]*v.e[0] - u.e[0]*v.e[2],
        u.e[0]*v.e[1] - u.e[1]*v.e[0]
    );
}

CUDA_HD
inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

// only compile this stream operator for non-CUDA translation units
#ifndef __CUDACC__
inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}
#endif

// ================================================
// Reflection / Refraction helpers
// ================================================
CUDA_HD
inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

CUDA_HD
inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0f);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

using point3 = vec3;

#endif // VEC3_H
