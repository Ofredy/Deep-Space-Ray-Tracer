#ifndef GPU_SCENE_H
#define GPU_SCENE_H

#include "cuda_compat.h"
#include "vec3.h"
#include "camera.h"   // for GPUCamera
#include <cstdint>

// =============================
// GPU-side triangle
// =============================
struct GPUTriangle {
    vec3 v0;
    vec3 v1;
    vec3 v2;

    vec3 n0;
    vec3 n1;
    vec3 n2;

    int  mat_id; // index into GPUMaterial array
};

// =============================
// GPU-side material description
// =============================
enum MaterialType : int {
    MAT_LAMBERTIAN = 0,
    MAT_METAL      = 1,
    MAT_DIELECTRIC = 2
};

struct GPUMaterial {
    int    type;      // MaterialType
    vec3   albedo;    // base color / tint
    double fuzz;      // metal fuzz
    double ref_idx;   // dielectric IOR
};

// =============================
// (placeholder) GPU BVH node
// =============================
struct GPUNode {
    vec3 aabb_min;
    vec3 aabb_max;

    int  left_first;  // child index OR first triangle index
    int  tri_count;   // 0 = internal node, >0 = leaf tri count
};

// GPU representation of a sphere
struct GPUSphere {
    vec3  center;       // sphere center in world space
    float radius;       // radius
    int   material_id;  // index into d_mats
    int   _pad;         // padding so the struct stays aligned to 16 bytes (optional)
};

// =============================
// Full scene info to pass to kernels
// =============================
struct GPUScene {
    // triangles
    GPUTriangle* d_tris;
    int          num_tris;

    // spheres  <-- NEW
    GPUSphere*   d_spheres;
    int          num_spheres;

    // materials
    GPUMaterial* d_mats;
    int          num_mats;

    // BVH nodes, if you’re already uploading them
    GPUNode*     d_nodes;
    int          num_nodes;

    // camera
    GPUCamera    cam;

    // render params
    int image_width;
    int image_height;
    int samples_per_pixel;
    int max_depth;
};

#endif // GPU_SCENE_H
