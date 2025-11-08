// src/gpu_render.cu
#include "gpu_scene.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

// ------------------ helpers ------------------
static inline void checkCuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA ERROR at %s: %s\n", where, cudaGetErrorString(e));
#ifndef NDEBUG
        asm("trap;");
#endif
    }
}

__device__ inline double3 make_d3(double x, double y, double z) { double3 d{ x,y,z }; return d; }
__device__ inline double3 to_d3(const vec3& v) { return make_d3(v.x(), v.y(), v.z()); }
__device__ inline double3 d3_add(const double3& a, const double3& b){ return make_d3(a.x+b.x,a.y+b.y,a.z+b.z); }
__device__ inline double3 d3_sub(const double3& a, const double3& b){ return make_d3(a.x-b.x,a.y-b.y,a.z-b.z); }
__device__ inline double3 d3_mul(const double3& a, double s){ return make_d3(a.x*s,a.y*s,a.z*s); }
__device__ inline double  d3_dot(const double3& a, const double3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ inline double3 d3_cross(const double3& a, const double3& b){
    return make_d3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__device__ inline double  d3_len(const double3& a){ return sqrt(d3_dot(a,a)); }
__device__ inline double3 d3_norm(const double3& a){ double L=d3_len(a); return (L>0.0)? d3_mul(a,1.0/L):make_d3(0,0,0); }

__device__ inline float clamp01(float x){ return fminf(1.f,fmaxf(0.f,x)); }

__device__ inline float3 tex2D(const GPUScene& sc, int tex_id, float u, float v) {
    if (tex_id < 0 || tex_id >= sc.num_textures || sc.textures == nullptr || sc.texture_pool == nullptr)
        return make_float3(1,1,1);

    const GPUTextureHeader& T = sc.textures[tex_id];
    if (T.width <= 0 || T.height <= 0)
        return make_float3(1,1,1);

    // wrap UVs
    u = fmodf(u, 1.0f); if (u < 0) u += 1.0f;
    v = fmodf(v, 1.0f); if (v < 0) v += 1.0f;
    int x = (int)(u * (T.width  - 1));
    int y = (int)(v * (T.height - 1));

    int idx = (T.offset + (y * T.width + x) * 3);
    if (idx + 2 >= sc.texture_pool_floats)
        return make_float3(1,0,1);  // magenta = bad index

    const float* pool = sc.texture_pool;
    return make_float3(pool[idx+0], pool[idx+1], pool[idx+2]);
}

// ------------------ rays ------------------
struct RayD { double3 o; double3 d; };
__device__ inline double3 ray_at(const RayD& r, double t){ return d3_add(r.o, d3_mul(r.d,t)); }

// ------------------ intersections ------------------
// Triangle (Möller–Trumbore)
__device__ inline bool hit_triangle(const RayD& ray, const GPUTriangle& tri, double t_min, double t_max,
                                    double& t_out, double& u_out, double& v_out)
{
    const double3 v0 = to_d3(tri.v0);
    const double3 v1 = to_d3(tri.v1);
    const double3 v2 = to_d3(tri.v2);
    const double3 e1 = d3_sub(v1, v0);
    const double3 e2 = d3_sub(v2, v0);

    const double3 p  = d3_cross(ray.d, e2);
    const double det = d3_dot(e1, p);
    const double EPS = 1e-9;
    if (fabs(det) < EPS) return false;
    const double invDet = 1.0 / det;

    const double3 tv = d3_sub(ray.o, v0);
    const double u = d3_dot(tv, p) * invDet;
    if (u < 0.0 || u > 1.0) return false;

    const double3 q = d3_cross(tv, e1);
    const double v = d3_dot(ray.d, q) * invDet;
    if (v < 0.0 || (u + v) > 1.0) return false;

    const double t = d3_dot(e2, q) * invDet;
    if (t < t_min || t > t_max) return false;

    t_out = t; u_out = u; v_out = v;
    return true;
}

__device__ inline bool hit_aabb(const AABB& box,
                                const RayD& r,
                                double t_min,
                                double t_max)
{
    const double3 o = r.o;
    const double3 d = r.d;

    // X
    {
        double invD = 1.0 / d.x;
        double t0 = (box.minp.x() - o.x) * invD;
        double t1 = (box.maxp.x() - o.x) * invD;
        if (invD < 0.0) { double tmp = t0; t0 = t1; t1 = tmp; }
        if (t0 > t_min) t_min = t0;
        if (t1 < t_max) t_max = t1;
        if (t_max <= t_min) return false;
    }
    // Y
    {
        double invD = 1.0 / d.y;
        double t0 = (box.minp.y() - o.y) * invD;
        double t1 = (box.maxp.y() - o.y) * invD;
        if (invD < 0.0) { double tmp = t0; t0 = t1; t1 = tmp; }
        if (t0 > t_min) t_min = t0;
        if (t1 < t_max) t_max = t1;
        if (t_max <= t_min) return false;
    }
    // Z
    {
        double invD = 1.0 / d.z;
        double t0 = (box.minp.z() - o.z) * invD;
        double t1 = (box.maxp.z() - o.z) * invD;
        if (invD < 0.0) { double tmp = t0; t0 = t1; t1 = tmp; }
        if (t0 > t_min) t_min = t0;
        if (t1 < t_max) t_max = t1;
        if (t_max <= t_min) return false;
    }
    return true;
}

__device__ bool bvh_hit_triangles(
    const GPUScene& scene,
    const RayD& ray,
    double t_min,
    double t_max,
    int&    out_tri_index,
    double& out_t,
    double& out_u,
    double& out_v)
{
    if (!scene.bvh_nodes || scene.num_bvh_nodes == 0 ||
        !scene.tri_indices || scene.num_triangles == 0)
        return false;

    const GPUBVHNode* nodes = scene.bvh_nodes;
    const int*        idx   = scene.tri_indices;

    int stack[64];
    int sp = 0;
    stack[sp++] = 0; // assume node 0 is root

    bool hit_any = false;
    double t_hit = t_max;

    while (sp > 0) {
        int ni = stack[--sp];
        const GPUBVHNode& node = nodes[ni];

        if (!hit_aabb(node.box, ray, t_min, t_hit))
            continue;

        if (node.prim_count > 0) {
            // Leaf: intersect triangles in this range
            for (int i = 0; i < node.prim_count; ++i) {
                int tri_index = idx[node.first_prim + i];
                const GPUTriangle& tri = scene.triangles[tri_index];

                double t, u, v;
                if (hit_triangle(ray, tri, t_min, t_hit, t, u, v)) {
                    hit_any = true;
                    t_hit   = t;
                    out_t   = t;
                    out_u   = u;
                    out_v   = v;
                    out_tri_index = tri_index;
                }
            }
        } else {
            // Internal node: push children
            if (node.left  >= 0) stack[sp++] = node.left;
            if (node.right >= 0) stack[sp++] = node.right;
        }
    }
    return hit_any;
}

__device__ inline double3 tri_normal(const GPUTriangle& tri, double u, double v) {
    // Vertex normal blend or fallback to flat
    const double3 n0 = to_d3(tri.n0);
    const double3 n1 = to_d3(tri.n1);
    const double3 n2 = to_d3(tri.n2);
    double3 n = d3_add(d3_add(d3_mul(n0, 1.0 - u - v), d3_mul(n1, u)), d3_mul(n2, v));
    if (d3_len(n) < 1e-12) {
        const double3 e1 = d3_sub(to_d3(tri.v1), to_d3(tri.v0));
        const double3 e2 = d3_sub(to_d3(tri.v2), to_d3(tri.v0));
        n = d3_cross(e1, e2);
    }
    return d3_norm(n);
}

// Sphere
__device__ inline bool hit_sphere(const RayD& r, const GPUSphere& s, double t_min, double t_max,
                                  double& t_out, double3& n_out)
{
    const double3 C = to_d3(s.center);
    const double3 oc = d3_sub(r.o, C);
    const double a = d3_dot(r.d, r.d);
    const double b = 2.0 * d3_dot(oc, r.d);
    const double c = d3_dot(oc, oc) - s.radius * s.radius;
    const double disc = b*b - 4*a*c;
    if (disc < 0.0) return false;
    const double sqrtd = sqrt(disc);

    double t = (-b - sqrtd) / (2*a);
    if (t < t_min || t > t_max) {
        t = (-b + sqrtd) / (2*a);
        if (t < t_min || t > t_max) return false;
    }
    t_out = t;
    const double3 p = ray_at(r, t);
    n_out = d3_norm(d3_mul(d3_sub(p, C), 1.0 / s.radius));
    return true;
}

// ------------------ camera ------------------
__device__ inline RayD make_camera_ray(const GPUCamera& C, int x, int y, int W, int H) {
    const double u = (double(x) + 0.5) / double(W);
    const double v = (double(y) + 0.5) / double(H);
    const double3 origin = to_d3(C.origin);
    const double3 llc    = to_d3(C.lower_left_corner);
    const double3 horiz  = to_d3(C.horizontal);
    const double3 vert   = to_d3(C.vertical);
    double3 pixel = d3_add(d3_add(llc, d3_mul(horiz, u)), d3_mul(vert, v));
    RayD r; r.o = origin; r.d = d3_norm(d3_sub(pixel, origin));
    return r;
}

// ------------------ shading ------------------
__device__ inline bool is_emissive(const GPUMaterial& m) {
    return (m.type == MAT_DIFFUSE_LIGHT);
}

// Always give emissives a strong default in case host builder didn't set it.
__device__ inline float3 material_emissive(const GPUMaterial& m) {
    if (m.type == MAT_DIFFUSE_LIGHT) {
        // If your GPUMaterial has emissive stored, you can prefer it:
        // const float3 e = make_float3((float)m.emissive.x(), (float)m.emissive.y(), (float)m.emissive.z());
        // if (e.x > 0 || e.y > 0 || e.z > 0) return e;
        // Fallback default:
        return make_float3(20.f, 20.f, 20.f);
    }
    return make_float3(0.f, 0.f, 0.f);
}

__device__ inline float3 material_albedo(const GPUMaterial& m) {
    return make_float3((float)m.albedo.x(), (float)m.albedo.y(), (float)m.albedo.z());
}

// Shadow ray test against everything
__device__ inline bool occluded(const GPUScene& scene, const RayD& r, double t_max) {
    // triangles
    for (int i = 0; i < scene.num_triangles; ++i) {
        const GPUTriangle& tri = scene.triangles[i];
        double t, u, v;
        if (hit_triangle(r, tri, 1e-4, t_max, t, u, v)) {
            // --- NEW: simple interpolated normal + mat_id (optional, for debugging) ---
            double nx = (1.0 - u - v) * tri.n0.x() + u * tri.n1.x() + v * tri.n2.x();
            double ny = (1.0 - u - v) * tri.n0.y() + u * tri.n1.y() + v * tri.n2.y();
            double nz = (1.0 - u - v) * tri.n0.z() + u * tri.n1.z() + v * tri.n2.z();
            double nlen = sqrt(nx*nx + ny*ny + nz*nz);
            if (nlen > 1e-18) { nx /= nlen; ny /= nlen; nz /= nlen; }

            // (optional) you could access the material if needed
            // const GPUMaterial& m = scene.materials[tri.material_id];

            return true; // occluded
        }
    }

    // spheres
    for (int i = 0; i < scene.num_spheres; ++i) {
        const GPUSphere& sp = scene.spheres[i];
        double t; double3 n;
        if (hit_sphere(r, sp, 1e-4, t_max, t, n)) {
            // sp.material_id available if you ever need it
            return true;
        }
    }
    return false;
}

__device__ inline double3 nearest_point_on_sphere_toward_P(const GPUSphere& s, const double3& P) {
    const double3 C = to_d3(s.center);
    double3 PC = d3_sub(C, P);           // from P toward center
    double L = d3_len(PC);
    if (L <= 1e-12) return C;            // degenerate; shouldn't happen
    double3 dir = d3_mul(PC, 1.0 / L);   // unit toward center
    // nearest point on sphere surface “facing” P
    return d3_sub(C, d3_mul(dir, s.radius));
}

__device__ inline float3 sky_color(const GPUScene& scene, const double3& dir) {
    (void)dir;
    if (scene.sky_type == SKY_SOLID) {
        return make_float3((float)scene.sky_solid.x(), (float)scene.sky_solid.y(), (float)scene.sky_solid.z());
    } else if (scene.sky_type == SKY_GRADIENT) {
        // simple vertical gradient based on ray.y
        float t = (float)clamp01(0.5f * (float)dir.y + 0.5f);
        float3 bot = make_float3((float)scene.sky_bottom.x(), (float)scene.sky_bottom.y(), (float)scene.sky_bottom.z());
        float3 top = make_float3((float)scene.sky_top.x(), (float)scene.sky_top.y(), (float)scene.sky_top.z());
        return make_float3(bot.x + (top.x-bot.x)*t,
                           bot.y + (top.y-bot.y)*t,
                           bot.z + (top.z-bot.z)*t);
    }
    // env map not implemented here
    return make_float3(0.0f, 0.0f, 0.0f);
}

__device__ inline float3 shade_lambert_pointlights(
    const GPUScene& scene, const double3& P, const double3& N)
{
    float3 sum = make_float3(0,0,0);

    for (int i = 0; i < scene.num_spheres; ++i) {
        const GPUSphere& light_s = scene.spheres[i];
        const GPUMaterial& lm = scene.materials[light_s.material_id];
        if (!is_emissive(lm)) continue;

        // aim at the sphere SURFACE point closest to P (acts like a tiny area-light)
        const double3 Lsurf = nearest_point_on_sphere_toward_P(light_s, P);
        double3 Lvec = d3_sub(Lsurf, P);
        double dist = d3_len(Lvec);
        if (dist <= 1e-6) continue;
        const double3 Ldir = d3_mul(Lvec, 1.0 / dist);

        // hard shadow to that surface point
        RayD shadow; shadow.o = d3_add(P, d3_mul(N, 1e-4)); shadow.d = Ldir;
        if (occluded(scene, shadow, dist - 1e-4)) continue;

        const double ndotl = fmax(0.0, d3_dot(N, Ldir));

        // geometric factor: small “patch” approximation
        // scale by apparent size: R^2 / dist^2 (clamped to avoid going to 0 too fast)
        double d2 = dist * dist;
        double solid = (light_s.radius * light_s.radius) / fmax(1.0, d2);
        double gain  = 6.0; // slightly stronger so it pops

        const float3 I = material_emissive(lm);
        sum.x += (float)(ndotl * solid * gain) * I.x;
        sum.y += (float)(ndotl * solid * gain) * I.y;
        sum.z += (float)(ndotl * solid * gain) * I.z;
    }
    return sum;
}

// ---- spec helpers ----
__device__ inline double3 reflect_d3(const double3& I, const double3& N) {
    // I - 2 * dot(I,N) * N
    double dn = I.x*N.x + I.y*N.y + I.z*N.z;
    return make_double3(I.x - 2.0*dn*N.x, I.y - 2.0*dn*N.y, I.z - 2.0*dn*N.z);
}

__device__ inline bool refract_d3(const double3& Iu, const double3& N, double eta, double3& T_out) {
    // Snell: T = eta * I + (eta * c - sqrt(k)) * N, with k = 1 - eta^2*(1-c^2)
    double c = -(Iu.x*N.x + Iu.y*N.y + Iu.z*N.z);
    double k = 1.0 - eta*eta*(1.0 - c*c);
    if (k < 0.0) return false;
    double a = eta;
    double b = eta*c - sqrt(k);
    T_out = make_double3(a*Iu.x + b*N.x, a*Iu.y + b*N.y, a*Iu.z + b*N.z);
    return true;
}

__device__ inline double schlick(double cosTheta, double ior) {
    double r0 = (1.0 - ior) / (1.0 + ior);
    r0 *= r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);
}

__device__ inline float3 shade_material(
    const GPUScene& scene,
    const RayD&     view_ray,
    const double3&  P,
    const double3&  N,
    int             mat_id,
    int             tri_albedo_tex,   // triangle's albedo texture id (or -1)
    float           uvx, float uvy,   // interpolated UVs
    float3          fallback_albedo)  // Kd from material (CPU side)
{
    // Guard: bad material index -> magenta debug color
    if (mat_id < 0 || mat_id >= scene.num_materials) {
        return make_float3(1, 0, 1);
    }

    const GPUMaterial& m = scene.materials[mat_id];

    // ------------------------------------------------------------
    // 1) Base color = texture(if any) * material albedo
    // ------------------------------------------------------------
    float3 base = fallback_albedo;  // from CPU material

    // If the triangle has an albedo texture, sample it
    if (tri_albedo_tex >= 0) {
        float3 tex = tex2D(scene, tri_albedo_tex, uvx, uvy);
        base = tex2D(scene, tri_albedo_tex, uvx, uvy);
    } else {
        // Otherwise use GPUMaterial.albedo
        base = make_float3(
            (float)m.albedo.x(),
            (float)m.albedo.y(),
            (float)m.albedo.z()
        );
    }

    // ------------------------------------------------------------
    // 2) If this material is itself a light, just emit
    // ------------------------------------------------------------
    if (m.type == MAT_DIFFUSE_LIGHT) {
        // Use emissive stored in GPUMaterial (built in gpu_scene_builder)
        float3 e = make_float3(
            (float)m.emissive.x(),
            (float)m.emissive.y(),
            (float)m.emissive.z()
        );
        return e;
    }

    // ------------------------------------------------------------
    // 3) Direct diffuse lighting from emissive spheres only
    // ------------------------------------------------------------
    // shade_lambert_pointlights() already loops over *all spheres*
    // and uses only those whose material is MAT_DIFFUSE_LIGHT.
    // Background stays black because sky_color() isn't added here.
    float3 Lo = shade_lambert_pointlights(scene, P, N);

    // Simple Lambert shading: outgoing radiance = Lo * base color
    return make_float3(Lo.x * base.x,
                       Lo.y * base.y,
                       Lo.z * base.z);
}

// ------------------ kernel ------------------
// ------------------ kernel (only tiny changes marked NEW) ------------------
// Simple per-thread RNG -----------------------------------
__device__ inline uint32_t wang_hash(uint32_t s) {
    s = (s ^ 61u) ^ (s >> 16);
    s *= 9u;
    s = s ^ (s >> 4);
    s *= 0x27d4eb2du;
    s = s ^ (s >> 15);
    return s;
}

__device__ inline float rand01(uint32_t &state) {
    state = state * 1664525u + 1013904223u;   // LCG
    // use lower 24 bits
    return (state & 0x00FFFFFFu) / 16777216.0f;  // [0,1)
}

// Jittered camera ray -------------------------------------
__device__ inline RayD make_camera_ray_jittered(
    const GPUCamera& cam,
    int px, int py,
    int W, int H,
    float rx, float ry)
{
    // normalized screen coords in [0,1]
    double u = ((double)px + (double)rx) / (double)(W - 1);
    double v = ((double)py + (double)ry) / (double)(H - 1);

    // GPUCamera uses vec3
    vec3 dir_v = cam.lower_left_corner
               + u * cam.horizontal
               + v * cam.vertical
               - cam.origin;

    double3 origin = make_double3(cam.origin.x(), cam.origin.y(), cam.origin.z());
    double3 dir    = make_double3(dir_v.x(),     dir_v.y(),     dir_v.z());

    RayD r;
    r.o = origin;
    r.d = dir;
    return r;
}

// ============================================================
// BVH triangle intersection helper
// ============================================================
__device__ bool intersect_triangles_bvh(
    const GPUScene& scene,
    const RayD& ray,
    double t_min,
    double t_max,
    // OUT:
    double& out_t_hit,
    double3& out_N_hit,
    int&     out_mat_id,
    int&     out_tri_tex_id,
    float&   out_uvx,
    float&   out_uvy)
{
    if (!scene.bvh_nodes || scene.num_bvh_nodes <= 0 ||
        !scene.triangles || scene.num_triangles <= 0)
        return false;

    // Explicit small traversal stack
    int stack[64];
    int stack_size = 0;
    stack[stack_size++] = 0; // root = node 0

    double  closest_t = t_max;
    bool    hit_any   = false;
    double3 best_N    = make_double3(0,0,0);
    int     best_mat  = -1;
    int     best_tex  = -1;
    float   best_u    = 0.f, best_v = 0.f;

    while (stack_size > 0) {
        int node_idx = stack[--stack_size];
        const GPUBVHNode& node = scene.bvh_nodes[node_idx];

        if (!hit_aabb(node.box, ray, t_min, closest_t))
            continue;

        if (node.prim_count > 0) {
            // Leaf node: check triangles
            for (int i = 0; i < node.prim_count; ++i) {
                int tri_index = node.first_prim + i;
                if (scene.tri_indices) tri_index = scene.tri_indices[tri_index];

                const GPUTriangle& tri = scene.triangles[tri_index];
                double t, u, v;
                if (hit_triangle(ray, tri, t_min, closest_t, t, u, v)) {
                    closest_t = t;
                    hit_any   = true;

                    best_N   = tri_normal(tri, u, v);
                    best_mat = tri.material_id;
                    best_tex = tri.albedo_tex;
                    best_u   = (float)((1.0 - u - v) * tri.uv0.x()
                                     + u * tri.uv1.x()
                                     + v * tri.uv2.x());
                    best_v   = (float)((1.0 - u - v) * tri.uv0.y()
                                     + u * tri.uv1.y()
                                     + v * tri.uv2.y());
                }
            }
        } else {
            if (node.left  >= 0) stack[stack_size++] = node.left;
            if (node.right >= 0) stack[stack_size++] = node.right;
        }
    }

    if (!hit_any) return false;

    out_t_hit      = closest_t;
    out_N_hit      = best_N;
    out_mat_id     = best_mat;
    out_tri_tex_id = best_tex;
    out_uvx        = best_u;
    out_uvy        = best_v;
    return true;
}

__global__ void render_kernel(unsigned char* out_rgb, int W, int H,
                              GPUScene scene, float inv_gamma, float exposure)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    // How many samples per pixel (from CPU camera)
    int spp = scene.params.samples_per_pixel;
    if (spp < 1) spp = 1;

    // Per-pixel RNG seed (deterministic but different per pixel)
    uint32_t rng = (uint32_t)(x + y * W) ^ 0x9e3779b9u;

    float3 accum = make_float3(0.f, 0.f, 0.f);

    for (int s = 0; s < spp; ++s) {
        // Jitter inside the pixel in [0,1)
        float jx = rand01(rng);
        float jy = rand01(rng);

        // Build a jittered primary ray
        RayD ray = make_camera_ray_jittered(scene.camera, x, y, W, H, jx, jy);

        double  t_hit  = 1e30;
        double3 N_hit  = make_double3(0,0,0);
        int     mat_id_hit = -1;
        bool    hit = false;

        int   tri_tex_id = -1;
        float uvx = 0.f, uvy = 0.f;

        // ---- triangles (your existing code) ----
        for (int i = 0; i < scene.num_triangles; ++i) {
            const GPUTriangle& tri = scene.triangles[i];
            double t, u, v;
            if (hit_triangle(ray, tri, 1e-4, t_hit, t, u, v)) {
                t_hit = t; hit = true;
                N_hit = tri_normal(tri, u, v);
                mat_id_hit = tri.material_id;

                uvx = (float)((1.0 - u - v) * tri.uv0.x()
                            + u * tri.uv1.x()
                            + v * tri.uv2.x());
                uvy = (float)((1.0 - u - v) * tri.uv0.y()
                            + u * tri.uv1.y()
                            + v * tri.uv2.y());
                tri_tex_id = tri.albedo_tex;
            }
        }

        // ---- spheres (your existing code) ----
        for (int i = 0; i < scene.num_spheres; ++i) {
            const GPUSphere& sphr = scene.spheres[i];
            double t; double3 n;
            if (hit_sphere(ray, sphr, 1e-4, t_hit, t, n)) {
                t_hit = t; hit = true;
                N_hit = n;
                mat_id_hit = sphr.material_id;
                tri_tex_id = -1;
            }
        }

        float3 sample_color;
        if (!hit) {
            sample_color = sky_color(scene, ray.d);
        } else {
            const GPUMaterial& m = scene.materials[mat_id_hit];
            float3 alb = make_float3(
                (float)m.albedo.x(),
                (float)m.albedo.y(),
                (float)m.albedo.z()
            );
            const double3 P = ray_at(ray, t_hit);
            sample_color = shade_material(scene, ray, P, N_hit,
                                          mat_id_hit,
                                          tri_tex_id,
                                          uvx, uvy,
                                          alb);
        }

        accum.x += sample_color.x;
        accum.y += sample_color.y;
        accum.z += sample_color.z;
    }

    // Average over all samples
    float inv_spp = 1.0f / (float)spp;
    float3 color = make_float3(accum.x * inv_spp,
                               accum.y * inv_spp,
                               accum.z * inv_spp);

    // Tonemap
    color.x = powf(clamp01(color.x * exposure), inv_gamma);
    color.y = powf(clamp01(color.y * exposure), inv_gamma);
    color.z = powf(clamp01(color.z * exposure), inv_gamma);

    int idx = (y * W + x) * 3;
    out_rgb[idx+0] = (unsigned char)(255.99f * color.x);
    out_rgb[idx+1] = (unsigned char)(255.99f * color.y);
    out_rgb[idx+2] = (unsigned char)(255.99f * color.z);
}

// ------------------ entry point ------------------
extern "C"
void gpu_render_scene(const GPUScene& scene, int width, int height)
{
    const double gamma_d    = (scene.params.gamma    > 0.0) ? scene.params.gamma    : 2.2;
    const double exposure_d = (scene.params.exposure > 0.0) ? scene.params.exposure : 1.0;
    const float  inv_gamma  = (float)(1.0 / gamma_d);
    const float  exposure   = (float)exposure_d;

    const size_t bytes = (size_t)width * (size_t)height * 3;
    unsigned char* d_rgb = nullptr;
    checkCuda(cudaMalloc(&d_rgb, bytes), "cudaMalloc(d_rgb)");

    dim3 block(16,16);
    dim3 grid((width + block.x - 1)/block.x,
              (height+ block.y - 1)/block.y);

    render_kernel<<<grid, block>>>(d_rgb, width, height, scene, inv_gamma, exposure);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "sync");

    std::vector<unsigned char> h(bytes);
    checkCuda(cudaMemcpy(h.data(), d_rgb, bytes, cudaMemcpyDeviceToHost), "memcpy d2h");
    cudaFree(d_rgb);

    // write PPM
    FILE* f = std::fopen("output.ppm", "wb");
    if (!f) { std::fprintf(stderr, "Cannot open output.ppm\n"); return; }
    std::fprintf(f, "P6\n%d %d\n255\n", width, height);
    std::fwrite(h.data(), 1, bytes, f);
    std::fclose(f);
}
