#ifndef GPU_SCENE_H
#define GPU_SCENE_H

#include "cuda_compat.h"
#include "vec3.h"
#include "camera.h"   // defines GPUCamera or camera::toGPUCamera()
#include <cstdint>

// -----------------------------
// GPU-side triangle -- MUST match gpu_render.cu
// -----------------------------
struct GPUTriangle {
    vec3 v0;
    vec3 v1;
    vec3 v2;
    int  material_id;   // name and order MUST match what the kernel reads
};

// -----------------------------
// GPU-side material
// -----------------------------
enum MaterialType : int {
    MAT_LAMBERTIAN = 0,
    MAT_METAL      = 1,
    MAT_DIELECTRIC = 2,
    MAT_DIFFUSE_LIGHT = 3 // optional, for lights later
};

struct GPUMaterial {
    int    type;      // MaterialType
    vec3   albedo;    // base color / tint
    double fuzz;      // metal fuzz (unused for lambertian)
    double ref_idx;   // IOR for dielectric
};

// -----------------------------
// GPU-side BVH node (placeholder)
// -----------------------------
struct GPUNode {
    vec3 aabb_min;
    vec3 aabb_max;
    int  left_first;
    int  tri_count;
};

// -----------------------------
// GPU-side sphere -- MUST match gpu_render.cu
// -----------------------------
struct GPUSphere {
    vec3  center;        // sphere center in world space
    float radius;        // radius
    int   material_id;   // index into d_mats
    int   _pad;          // ok to keep for alignment
};

// -----------------------------
// GPUScene passed by value to the kernel
// -----------------------------
struct GPUScene {
    // triangles
    GPUTriangle* d_tris;
    int          num_tris;

    // spheres
    GPUSphere*   d_spheres;
    int          num_spheres;

    // materials
    GPUMaterial* d_mats;
    int          num_mats;

    // BVH nodes (not used yet on GPU)
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
