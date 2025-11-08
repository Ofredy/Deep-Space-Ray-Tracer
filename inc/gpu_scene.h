#ifndef GPU_SCENE_H
#define GPU_SCENE_H

#include "cuda_compat.h"
#include "vec3.h"
#include "camera.h"
#include <cstdint>

// -----------------------------
// Small PODs
// -----------------------------
struct AABB {
    vec3 minp;
    vec3 maxp;
};

// -----------------------------
// Materials & Textures
// -----------------------------
enum MaterialType : int {
    MAT_LAMBERTIAN     = 0,
    MAT_METAL          = 1,
    MAT_DIELECTRIC     = 2,
    MAT_DIFFUSE_LIGHT  = 3
};

struct GPUTextureHeader {
    int width;     // pixels
    int height;    // pixels
    int offset;    // float index into texture_pool (RGB packed)
};

struct GPUMaterial {
    int   type;         // MaterialType
    int   albedo_tex;   // index into textures[] or -1 if none
    int   _pad0;
    int   _pad1;

    vec3  albedo;       // base color / tint (used if no albedo_tex)
    vec3  emissive;     // emissive color

    double fuzz;        // metal fuzz [0..1]
    double ref_idx;     // IOR for dielectric
};

// -----------------------------
// Geometry
// -----------------------------
struct GPUSphere {
    vec3   center;
    double radius;
    int    material_id;
    int    _pad;
};

struct GPUTriangle {
    vec3   v0;
    vec3   v1;
    vec3   v2;

    vec3   n0;
    vec3   n1;
    vec3   n2;

    vec3   uv0;         // using vec3 for UV (u,v,0)
    vec3   uv1;
    vec3   uv2;

    int    material_id; // index into GPUMaterial
    int    albedo_tex;  // texture id for this tri (-1 if none)
};

// Optional BVH node
struct GPUBVHNode {
    AABB  box;
    int   left;
    int   right;
    int   first_prim;
    int   prim_count;
};

// -----------------------------
// Render / Scene Parameters
// -----------------------------
enum SkyType : int {
    SKY_SOLID    = 0,
    SKY_GRADIENT = 1,
    SKY_ENV_MAP  = 2
};

struct GPURenderParams {
    int    img_width;
    int    img_height;
    int    samples_per_pixel;
    int    max_depth;

    int    use_bvh;
    int    rng_mode;
    int    tile_size;
    int    _pad0;

    double gamma;
    double exposure;
    double env_rotation;
    double _pad1;
};

// -----------------------------
// Top-level GPU scene descriptor
// -----------------------------
struct GPUScene {
    // Geometry
    const GPUSphere*    spheres;       // [num_spheres]
    int                 num_spheres;
    int                 _pad_sph0;
    int                 _pad_sph1;

    const GPUTriangle*  triangles;     // [num_triangles]
    const int*          tri_indices;   // [num_triangles] (BVH order; optional)
    int                 num_triangles;
    int                 _pad_geo;

    // BVH
    const GPUBVHNode*   bvh_nodes;     // [num_bvh_nodes]
    int                 num_bvh_nodes;
    int                 _pad_bvh0;
    int                 _pad_bvh1;

    // Materials
    const GPUMaterial*  materials;     // [num_materials]
    int                 num_materials;
    int                 _pad_mat0;
    int                 _pad_mat1;

    // Textures (optional)
    const GPUTextureHeader* textures;      // [num_textures]
    int                     num_textures;
    int                     _pad_tex0;
    int                     _pad_tex1;

    const float*        texture_pool;      // float3-packed RGBs
    int                 texture_pool_floats;
    int                 _pad_pool0;
    int                 _pad_pool1;

    // Camera
    GPUCamera           camera;

    // Sky
    int     sky_type;     
    int     env_tex_id;   
    int     _pad_sky0;
    int     _pad_sky1;
    vec3    sky_solid;
    vec3    sky_top;
    vec3    sky_bottom;

    // Render params
    GPURenderParams     params;

    uint64_t            seed;
};

#endif // GPU_SCENE_H
