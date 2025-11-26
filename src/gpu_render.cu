#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gpu_scene.h"  // your header with GPUBVHNode, GPUScene, etc.

// ============================================================
// Basic float3 helpers
// ============================================================
__device__ inline float3 f3_make(float x, float y, float z) {
    return make_float3(x, y, z);
}

__device__ inline float3 f3_add(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 f3_sub(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 f3_mul(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline float3 f3_scale(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float  f3_dot(const float3& a, const float3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ inline float3 f3_cross(const float3& a, const float3& b) {
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

__device__ inline float  f3_len2(const float3& a) {
    return f3_dot(a, a);
}

__device__ inline float  f3_len(const float3& a) {
    return sqrtf(f3_len2(a));
}

__device__ inline float3 f3_norm(const float3& a) {
    float L = f3_len(a);
    if (L <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
    float invL = 1.0f / L;
    return make_float3(a.x * invL, a.y * invL, a.z * invL);
}

__device__ inline float3 f3_clamp01(const float3& a) {
    float3 r;
    r.x = fminf(1.0f, fmaxf(0.0f, a.x));
    r.y = fminf(1.0f, fmaxf(0.0f, a.y));
    r.z = fminf(1.0f, fmaxf(0.0f, a.z));
    return r;
}

__device__ inline float3 f3_lerp(const float3& a, const float3& b, float t) {
    return f3_add(f3_scale(a, 1.0f - t), f3_scale(b, t));
}

__device__ inline float3 to_float3(const vec3& v) {
    return make_float3((float)v.x(), (float)v.y(), (float)v.z());
}

// ============================================================
// RNG (LCG) per-thread
// ============================================================
__device__ inline float rand01(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return (state & 0x00FFFFFFu) / 16777216.0f;  // [0,1)
}

__device__ inline float3 random_in_unit_sphere(uint32_t& rng) {
    while (true) {
        float x = rand01(rng) * 2.0f - 1.0f;
        float y = rand01(rng) * 2.0f - 1.0f;
        float z = rand01(rng) * 2.0f - 1.0f;
        float3 p = make_float3(x, y, z);
        if (f3_len2(p) >= 1.0f) continue;
        return p;
    }
}

__device__ inline float3 random_unit_vector(uint32_t& rng) {
    return f3_norm(random_in_unit_sphere(rng));
}

__device__ inline float3 reflect(const float3& v, const float3& n) {
    return f3_sub(v, f3_scale(n, 2.0f * f3_dot(v, n)));
}

__device__ inline bool refract(const float3& v, const float3& n, float etai_over_etat, float3& refracted) {
    float3 uv = f3_norm(v);
    float cos_theta = fminf(f3_dot(f3_scale(uv, -1.0f), n), 1.0f);
    float3 r_out_perp = f3_scale(f3_add(uv, f3_scale(n, cos_theta)), etai_over_etat);
    float3 r_out_parallel = f3_scale(n, -sqrtf(fabsf(1.0f - f3_len2(r_out_perp))));
    refracted = f3_add(r_out_perp, r_out_parallel);
    return true;
}

__device__ inline float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

// ============================================================
// Ray
// ============================================================
struct RayD {
    float3 orig;
    float3 dir;

    __device__ RayD() {}
    __device__ RayD(const float3& o, const float3& d) : orig(o), dir(d) {}

    __device__ float3 at(float t) const {
        return make_float3(orig.x + t*dir.x, orig.y + t*dir.y, orig.z + t*dir.z);
    }
};

// ============================================================
// Texture sampling
// ============================================================
__device__ inline float3 tex2D(const GPUScene& scene, int tex_id, float u, float v) {
    if (tex_id < 0 || tex_id >= scene.num_textures ||
        !scene.textures || !scene.texture_pool) {
        return make_float3(1.0f, 1.0f, 1.0f);
    }

    const GPUTextureHeader& th = scene.textures[tex_id];
    int w = th.width;
    int h = th.height;

    // --- wrap to [0,1), works for negatives too
    u = u - floorf(u);
    v = v - floorf(v);

    // flip V (OBJ-style)
    int i = (int)(u * (w - 1));
    int j = (int)((1.0f - v) * (h - 1));

    int idx = th.offset + (j * w + i) * 3;
    if (idx < 0 || idx + 2 >= scene.texture_pool_floats) {
        return make_float3(1.0f, 1.0f, 1.0f);
    }

    float r = scene.texture_pool[idx + 0];
    float g = scene.texture_pool[idx + 1];
    float b = scene.texture_pool[idx + 2];
    return make_float3(r, g, b);
}

// ============================================================
// Hit record
// ============================================================
struct HitRecord {
    float  t;
    float3 p;
    float3 normal;
    int    mat_id;
    int    tri_tex_id;   // -1 if none
    int    tri_index;    // which triangle we hit
    float  u;            // barycentric u
    float  v;            // barycentric v
    bool   front_face;

    __device__ void set_face_normal(const RayD& r, const float3& outward_normal) {
        front_face = (f3_dot(r.dir, outward_normal) < 0.0f);
        normal = front_face ? outward_normal : f3_scale(outward_normal, -1.0f);
    }
};

// ============================================================
// BVH node AABB hit (uses your GPUBVHNode layout)
// GPUBVHNode has: vec3 bbox_min, bbox_max; int left,right,tri_offset,tri_count;
// ============================================================
__device__ inline bool bbox_hit(const GPUBVHNode& node,
                                const RayD& r,
                                float t_min,
                                float t_max)
{
    for (int a = 0; a < 3; ++a) {
        float orig = (a == 0 ? r.orig.x : (a == 1 ? r.orig.y : r.orig.z));
        float dir  = (a == 0 ? r.dir.x  : (a == 1 ? r.dir.y  : r.dir.z ));
        float invD = 1.0f / dir;

        // vec3 -> double; convert to float
        double dmin = (a == 0 ? node.bbox_min.x() :
                       (a == 1 ? node.bbox_min.y() : node.bbox_min.z()));
        double dmax = (a == 0 ? node.bbox_max.x() :
                       (a == 1 ? node.bbox_max.y() : node.bbox_max.z()));
        float minp = (float)dmin;
        float maxp = (float)dmax;

        float t0 = (minp - orig) * invD;
        float t1 = (maxp - orig) * invD;
        if (invD < 0.0f) {
            float tmp = t0; t0 = t1; t1 = tmp;
        }

        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;

        if (t_max <= t_min) return false;
    }
    return true;
}

// ============================================================
// Triangle hit by index (Möller–Trumbore)
// Your GPUTriangle: float3 v0,v1,v2; float3 n0,n1,n2; float3 uv0,uv1,uv2;
// int material_id; int albedo_tex;
// ============================================================
__device__ bool hit_triangle_index(
    const GPUScene& scene,
    int tri_index,
    const RayD& ray,
    float t_min,
    float t_max,
    HitRecord& rec)
{
    const GPUTriangle& tri = scene.triangles[tri_index];

    float3 v0 = tri.v0;
    float3 v1 = tri.v1;
    float3 v2 = tri.v2;

    float3 edge1 = f3_sub(v1, v0);
    float3 edge2 = f3_sub(v2, v0);

    float3 pvec = f3_cross(ray.dir, edge2);
    float det = f3_dot(edge1, pvec);
    if (fabsf(det) < 1e-8f) return false;
    float invDet = 1.0f / det;

    float3 tvec = f3_sub(ray.orig, v0);
    float u = f3_dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    float3 qvec = f3_cross(tvec, edge1);
    float v = f3_dot(ray.dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = f3_dot(edge2, qvec) * invDet;
    if (t < t_min || t > t_max) return false;

    // Hit
    rec.t = t;
    rec.p = ray.at(t);

    float w = 1.0f - u - v;

    // Interpolated normal
    float3 n0 = tri.n0;
    float3 n1 = tri.n1;
    float3 n2 = tri.n2;
    float3 normal = f3_add(
        f3_add(f3_scale(n0, w), f3_scale(n1, u)),
        f3_scale(n2, v)
    );
    normal = f3_norm(normal);
    rec.set_face_normal(ray, normal);

    rec.u = u;
    rec.v = v;

    rec.mat_id     = tri.material_id;
    rec.tri_tex_id = tri.albedo_tex;  // -1 if none
    rec.tri_index  = tri_index;

    return true;
}

// ============================================================
// BVH traversal: closest hit
// Uses GPUBVHNode{bbox_min/bbox_max,left,right,tri_offset,tri_count}
// and scene.tri_indices as the flatten index array
// ============================================================
__device__ bool bvh_hit_closest(
    const GPUScene& scene,
    const RayD& ray,
    float t_min,
    float t_max,
    HitRecord& out_rec)
{
    if (!scene.bvh_nodes || scene.num_bvh_nodes <= 0 ||
        !scene.tri_indices) {
        return false;
    }

    int stack[64];
    int stack_size = 0;
    int node_index = 0;

    bool  hit_anything = false;
    float closest      = t_max;
    HitRecord tmp_rec;

    while (true) {
        const GPUBVHNode& node = scene.bvh_nodes[node_index];

        if (bbox_hit(node, ray, t_min, closest)) {
            if (node.tri_count > 0) {
                // Leaf
                for (int i = 0; i < node.tri_count; ++i) {
                    int tri_idx = scene.tri_indices[node.tri_offset + i];
                    if (hit_triangle_index(scene, tri_idx, ray, t_min, closest, tmp_rec)) {
                        hit_anything = true;
                        closest      = tmp_rec.t;
                        out_rec      = tmp_rec;
                    }
                }

                // Pop
                if (stack_size == 0) break;
                node_index = stack[--stack_size];
            } else {
                // Internal node
                const GPUBVHNode& left  = scene.bvh_nodes[node.left];
                const GPUBVHNode& right = scene.bvh_nodes[node.right];

                bool hit_left  = bbox_hit(left,  ray, t_min, closest);
                bool hit_right = bbox_hit(right, ray, t_min, closest);

                if (hit_left && hit_right) {
                    // Choose nearer child based on bbox centers
                    float3 cL = make_float3(
                        0.5f * (float)(left.bbox_min.x() + left.bbox_max.x()),
                        0.5f * (float)(left.bbox_min.y() + left.bbox_max.y()),
                        0.5f * (float)(left.bbox_min.z() + left.bbox_max.z())
                    );
                    float3 cR = make_float3(
                        0.5f * (float)(right.bbox_min.x() + right.bbox_max.x()),
                        0.5f * (float)(right.bbox_min.y() + right.bbox_max.y()),
                        0.5f * (float)(right.bbox_min.z() + right.bbox_max.z())
                    );

                    float dL = f3_dot(f3_sub(cL, ray.orig), ray.dir);
                    float dR = f3_dot(f3_sub(cR, ray.orig), ray.dir);

                    int near_idx = (dL < dR) ? node.left  : node.right;
                    int far_idx  = (dL < dR) ? node.right : node.left;

                    stack[stack_size++] = far_idx;
                    node_index = near_idx;
                }
                else if (hit_left) {
                    node_index = node.left;
                }
                else if (hit_right) {
                    node_index = node.right;
                }
                else {
                    if (stack_size == 0) break;
                    node_index = stack[--stack_size];
                }
            }
        } else {
            if (stack_size == 0) break;
            node_index = stack[--stack_size];
        }
    }

    return hit_anything;
}

// ============================================================
// Sphere hit
// ============================================================
__device__ bool hit_sphere(
    const GPUSphere& sph,
    const RayD& ray,
    float t_min,
    float t_max,
    float& t_out,
    float3& normal_out)
{
    float3 oc = f3_sub(ray.orig, sph.center);
    float a = f3_dot(ray.dir, ray.dir);
    float half_b = f3_dot(oc, ray.dir);
    float c = f3_dot(oc, oc) - sph.radius * sph.radius;
    float discriminant = half_b*half_b - a*c;
    if (discriminant < 0.0f) return false;
    float sqrtd = sqrtf(discriminant);

    float root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) return false;
    }

    t_out = root;
    float3 p = ray.at(root);
    normal_out = f3_scale(f3_sub(p, sph.center), 1.0f / sph.radius);
    return true;
}

// ============================================================
// Scene hit: BVH triangles + brute-force spheres
// ============================================================
__device__ bool scene_hit(
    const GPUScene& scene,
    const RayD& ray,
    float t_min,
    float t_max,
    HitRecord& rec)
{
    HitRecord best_rec;
    bool  hit_any  = false;
    float closest  = t_max;

    // Triangles via BVH
    HitRecord tri_rec;
    if (bvh_hit_closest(scene, ray, t_min, closest, tri_rec)) {
        hit_any  = true;
        closest  = tri_rec.t;
        best_rec = tri_rec;
    }

    // Spheres
    for (int i = 0; i < scene.num_spheres; ++i) {
        const GPUSphere& sph = scene.spheres[i];
        float  t_hit;
        float3 n_hit;
        if (hit_sphere(sph, ray, t_min, closest, t_hit, n_hit)) {
            hit_any  = true;
            closest  = t_hit;
            best_rec.t         = t_hit;
            best_rec.p         = ray.at(t_hit);
            best_rec.set_face_normal(ray, n_hit);
            best_rec.mat_id    = sph.material_id;
            best_rec.tri_tex_id= -1;
            best_rec.tri_index = -1;
            best_rec.u         = 0.0f;
            best_rec.v         = 0.0f;
        }
    }

    if (hit_any) {
        rec = best_rec;
    }
    return hit_any;
}

__device__ bool scene_hit_bruteforce(
    const GPUScene& scene,
    const RayD& ray,
    float t_min,
    float t_max,
    HitRecord& rec)
{
    bool hit_anything = false;
    float closest = t_max;

    HitRecord temp_rec;

    // If you have an index buffer, use it. Otherwise just loop 0..num_tris-1
    for (int i = 0; i < scene.num_triangles; ++i) {
        int tri_index = scene.tri_indices ? scene.tri_indices[i] : i;

        if (hit_triangle_index(scene, tri_index, ray, t_min, closest, temp_rec)) {
            hit_anything = true;
            closest = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

// ============================================================
// Materials
// ============================================================
__device__ inline const GPUMaterial& get_mat(const GPUScene& scene, int mat_id) {
    return scene.materials[mat_id];
}

__device__ bool scatter_lambertian(
    const GPUScene& scene,
    const GPUMaterial& mat,
    const RayD& ray_in,
    const HitRecord& rec,
    uint32_t& rng,
    RayD& scattered,
    float3& attenuation,
    const float3& albedo)
{
    float3 scatter_dir = f3_add(rec.normal, random_unit_vector(rng));
    if (f3_len2(scatter_dir) < 1e-8f) scatter_dir = rec.normal;
    scattered  = RayD(rec.p, scatter_dir);
    attenuation= albedo;
    return true;
}

__device__ bool scatter_metal(
    const GPUScene& scene,
    const GPUMaterial& mat,
    const RayD& ray_in,
    const HitRecord& rec,
    uint32_t& rng,
    RayD& scattered,
    float3& attenuation,
    const float3& albedo)
{
    float3 reflected = reflect(f3_norm(ray_in.dir), rec.normal);
    float  fuzz      = fmaxf(0.0f, fminf(1.0f, mat.fuzz));
    float3 dir       = f3_add(reflected, f3_scale(random_in_unit_sphere(rng), fuzz));
    scattered        = RayD(rec.p, dir);
    attenuation      = albedo;
    return (f3_dot(scattered.dir, rec.normal) > 0.0f);
}

__device__ bool scatter_dielectric(
    const GPUScene& scene,
    const GPUMaterial& mat,
    const RayD& ray_in,
    const HitRecord& rec,
    uint32_t& rng,
    RayD& scattered,
    float3& attenuation)
{
    // Glass doesn't absorb by default
    attenuation = make_float3(1.0f, 1.0f, 1.0f);

    // ✅ Clamp IOR to a sane value so we don't divide by 0 or use garbage
    float eta = mat.ref_idx;
    if (eta <= 0.0f || !isfinite(eta)) {
        eta = 1.5f;   // fall back to typical glass IOR
    }

    float refraction_ratio = rec.front_face ? (1.0f / eta) : eta;

    float3 unit_dir = f3_norm(ray_in.dir);
    float  cos_theta = fminf(f3_dot(f3_scale(unit_dir, -1.0f), rec.normal), 1.0f);
    float  sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));

    bool   cannot_refract = refraction_ratio * sin_theta > 1.0f;
    float3 direction;

    // ✅ Use Schlick with IOR we actually used (eta), not some garbage
    float reflect_prob = schlick(cos_theta, refraction_ratio);

    if (cannot_refract || reflect_prob > rand01(rng)) {
        // Reflect
        direction = reflect(unit_dir, rec.normal);
    } else {
        // Refract – our refract() always returns true, just writes 'direction'
        refract(unit_dir, rec.normal, refraction_ratio, direction);
    }

    scattered = RayD(rec.p, direction);
    return true;
}

__device__ float3 debug_shade_hit(const GPUScene& scene, const HitRecord& rec) {
    const GPUMaterial& mat = get_mat(scene, rec.mat_id);

    // Lights: show as bright white
    if (mat.type == MAT_DIFFUSE_LIGHT) {
        return make_float3(1.0f, 1.0f, 1.0f);
    }

    // Start from material albedo
    float3 base = mat.albedo;

    // If this hit came from a textured triangle, modulate with texture
    if (rec.tri_tex_id >= 0) {
        float u = rec.u;
        float v = rec.v;

        float3 tex = tex2D(scene, rec.tri_tex_id, u, v);
        base = f3_mul(base, tex);
    }

    // Clamp for display
    base = f3_clamp01(base);
    return base;
}

__device__ float3 ray_color_debug(
    const GPUScene& scene,
    RayD ray,
    uint32_t& rng)
{
    HitRecord rec;
    if (!scene_hit(scene, ray, 0.001f, 1e30f, rec)) {
        // Simple sky debug
        if (scene.sky_type == SKY_SOLID) {
            return scene.sky_solid;
        } else {
            // gradient sky
            float3 unitdir = f3_norm(ray.dir);
            float t = 0.5f * (unitdir.y + 1.0f);
            return f3_add(
                f3_scale(scene.sky_bottom, (1.0f - t)),
                f3_scale(scene.sky_top, t)
            );
        }
    }

    return debug_shade_hit(scene, rec);
}

// ============================================================
// Ray color (path tracing with emissive + sky)
// ============================================================
__device__ float3 ray_color(
    const GPUScene& scene,
    RayD ray,
    uint32_t& rng)
{
    // Accumulated radiance along the path
    float3 L          = make_float3(0.0f, 0.0f, 0.0f);
    // Path throughput (how much light still “survives” after bounces)
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

    int max_depth = (scene.params.max_depth > 0) ? scene.params.max_depth : 8;

    for (int depth = 0; depth < max_depth; ++depth) {
        HitRecord rec;
        if (!scene_hit(scene, ray, 0.001f, 1.0e9f, rec)) {
            // No environment/sky: stop the path
            break;
        }

        const GPUMaterial& mat = get_mat(scene, rec.mat_id);

        // ----------------------------------------------------
        // 1) Emission (for emissive materials, NOT the Sun)
        // ----------------------------------------------------
        if (mat.type == MAT_DIFFUSE_LIGHT) {
            // Pure light source: add emission and stop
            L = f3_add(L, f3_mul(throughput, mat.emissive));
            break;
        } else {
            // Non-light materials can still have emissive tint
            if (mat.emissive.x > 0.0f ||
                mat.emissive.y > 0.0f ||
                mat.emissive.z > 0.0f)
            {
                L = f3_add(L, f3_mul(throughput, mat.emissive));
            }
        }

        // ----------------------------------------------------
        // 2) Base albedo (+ optional texture)
        // ----------------------------------------------------
        float3 albedo = mat.albedo;

        int tex_id = mat.albedo_tex;
        if (rec.tri_tex_id >= 0)
            tex_id = rec.tri_tex_id;

        if (tex_id >= 0 && rec.tri_index >= 0) {
            const GPUTriangle& tri = scene.triangles[rec.tri_index];
            float w = 1.0f - rec.u - rec.v;

            float u_tex = w * tri.uv0.x + rec.u * tri.uv1.x + rec.v * tri.uv2.x;
            float v_tex = w * tri.uv0.y + rec.u * tri.uv1.y + rec.v * tri.uv2.y;

            float3 tex = tex2D(scene, tex_id, u_tex, v_tex);
            albedo = f3_mul(albedo, tex);
        }

        #if 0

        // ----------------------------------------------------
        // 2.5) Direct Sun lighting with SHADOW RAYS
        // ----------------------------------------------------
        {
            // scene.sun_dir points FROM origin (ISS) TO the Sun.
            // Light travels in the opposite direction:
            float3 Ldir = make_float3(
                -scene.sun_dir.x,
                -scene.sun_dir.y,
                -scene.sun_dir.z
            );

            float ndotl = fmaxf(f3_dot(rec.normal, Ldir), 0.0f);

            if (ndotl > 0.0f) {
                // Cast a shadow ray to see if something blocks the Sun
                bool in_shadow = false;

                // Small offset along normal to avoid self-intersection
                float3 shadow_origin = f3_add(
                    rec.p,
                    f3_scale(rec.normal, 1.0e-3f)
                );

                RayD shadow_ray(shadow_origin, Ldir);
                HitRecord shadow_hit;

                // If we hit anything along the light direction, we’re in shadow
                if (scene_hit(scene, shadow_ray, 0.001f, 1.0e9f, shadow_hit)) {
                    in_shadow = true;
                }

                if (!in_shadow) {
                    // Diffuse term: throughput * albedo * sun_radiance * cos(theta)
                    float3 sun_term = f3_mul(
                        f3_mul(throughput, albedo),
                        f3_scale(scene.sun_radiance, ndotl)
                    );
                    L = f3_add(L, sun_term);
                }
            }
        }

        #endif

        // ----------------------------------------------------
        // 3) Scatter to generate the next ray
        // ----------------------------------------------------
        RayD   scattered;
        float3 atten;
        bool   ok = false;

        if (mat.type == MAT_DIELECTRIC) {
            ok = scatter_dielectric(
                scene, mat, ray, rec, rng,
                scattered, atten
            );
        } else {
            ok = scatter_lambertian(
                scene, mat, ray, rec, rng,
                scattered, atten, albedo
            );
        }

        if (!ok) {
            break;
        }

        throughput = f3_mul(throughput, atten);
        ray        = scattered;
    }

    return f3_clamp01(L);
}

// ============================================================
// Camera ray
// ============================================================
__device__ RayD make_camera_ray_jittered(
    const GPUScene& scene,
    int px,
    int py,
    int W,
    int H,
    float jx,
    float jy)
{
    const GPUCamera& cam = scene.camera;

    float u = ((float)px + jx) / (float)(W - 1);
    float v = ((float)py + jy) / (float)(H - 1);

    // Do math in vec3 (double-based)
    vec3 origin_d = cam.origin;
    vec3 dir_d =
        cam.lower_left_corner
      + u * cam.horizontal
      + v * cam.vertical
      - cam.origin;

    // Convert to float3 for the rest of the GPU pipeline
    float3 origin = to_float3(origin_d);
    float3 dir    = to_float3(dir_d);

    return RayD(origin, dir);
}

// ============================================================
// Kernel
// ============================================================
__global__ void render_kernel(
    const GPUScene* __restrict__ dscene,
    unsigned char* out_rgb,
    int W,
    int H,
    float inv_gamma,
    float exposure)
{
    const GPUScene& scene = *dscene;  // alias to keep the rest of your code unchanged

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int spp = scene.params.samples_per_pixel;
    if (spp < 1) spp = 1;

    uint32_t rng = (uint32_t)(x + y * W) ^ (uint32_t)(scene.seed & 0xFFFFFFFFu);

    float3 accum = make_float3(0,0,0);
    for (int s = 0; s < spp; ++s) {
        float jx = rand01(rng);
        float jy = rand01(rng);
        RayD ray = make_camera_ray_jittered(scene, x, y, W, H, jx, jy);
        accum = f3_add(accum, ray_color(scene, ray, rng));
    }

    // 1) average over samples (no exposure)
    float inv_spp = 1.0f / (float)spp;
    float3 color  = f3_scale(accum, inv_spp);

    // 2) clamp negatives
    color.x = fmaxf(color.x, 0.0f);
    color.y = fmaxf(color.y, 0.0f);
    color.z = fmaxf(color.z, 0.0f);

    // 3) gamma correct – pass inv_gamma = 1.0f / 2.0f from the host
    color.x = powf(color.x, inv_gamma);
    color.y = powf(color.y, inv_gamma);
    color.z = powf(color.z, inv_gamma);

    // 4) clamp to [0,1]
    color = f3_clamp01(color);

    // 5) store to 8-bit framebuffer
    int idx = ((H - 1 - y) * W + x) * 3;
    out_rgb[idx + 0] = (unsigned char)(255.99f * color.x);
    out_rgb[idx + 1] = (unsigned char)(255.99f * color.y);
    out_rgb[idx + 2] = (unsigned char)(255.99f * color.z);
}

// int flipped_y = (H - 1 - y);
// ============================================================
// Host entry point
// ============================================================
extern "C"
void gpu_render_scene(const GPUScene& scene, int width, int height)
{
    int W = width;
    int H = height;

    const float gamma     = (scene.params.gamma    > 0.0f) ? scene.params.gamma    : 1.0f;
    const float exposure  = (scene.params.exposure > 0.0f) ? scene.params.exposure : 2.0f;
    const float inv_gamma = 1.0f / gamma;

    const size_t pixels = static_cast<size_t>(W) * static_cast<size_t>(H);
    const size_t bytes  = pixels * 3;

    // Device framebuffer
    unsigned char* d_fb = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_fb, bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc framebuffer failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Device-side copy of the scene struct
    GPUScene* d_scene = nullptr;
    err = cudaMalloc((void**)&d_scene, sizeof(GPUScene));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(d_scene) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_fb);
        return;
    }
    err = cudaMemcpy(d_scene, &scene, sizeof(GPUScene), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy(d_scene) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_scene);
        cudaFree(d_fb);
        return;
    }

    dim3 block(8, 8);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    render_kernel<<<grid, block>>>(d_scene, d_fb, W, H, inv_gamma, exposure);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "render_kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_scene);
        cudaFree(d_fb);
        return;
    }

    std::vector<unsigned char> h_fb(bytes);
    err = cudaMemcpy(h_fb.data(), d_fb, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy framebuffer failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_scene);
        cudaFree(d_fb);
        return;
    }

    cudaFree(d_scene);
    cudaFree(d_fb);

    // Write PPM
    FILE* f = std::fopen("image_gpu.ppm", "wb");
    if (!f) {
        std::fprintf(stderr, "Failed to open image_gpu.ppm for writing\n");
        return;
    }
    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    std::fwrite(h_fb.data(), 1, bytes, f);
    std::fclose(f);
}
