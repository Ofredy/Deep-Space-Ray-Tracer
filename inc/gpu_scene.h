#ifndef GPU_SCENE_H
#define GPU_SCENE_H

#include "cuda_compat.h"
#include "vec3.h"
#include <cuda_runtime.h>  
#include "camera.h"
#include <cstdint>

// =============================
// Basic PODs (float-based)
// =============================
struct AABB {
    float3 minp;
    float3 maxp;
};

// =============================
// Materials & Textures
// =============================
enum MaterialType : int {
    MAT_LAMBERTIAN    = 0,
    MAT_METAL         = 1,
    MAT_DIELECTRIC    = 2,
    MAT_DIFFUSE_LIGHT = 3
};

struct GPUTextureHeader {
    int width;     // pixels
    int height;    // pixels
    int offset;    // float index into texture_pool (RGB packed)
};

struct GPUMaterial {
    int   type;        // MaterialType
    int   albedo_tex;  // index into textures[] or -1 if none
    int   _pad0;
    int   _pad1;

    float3 albedo;     // base color / tint
    float3 emissive;   // emissive color

    float fuzz;        // metal fuzz [0..1]
    float ref_idx;     // IOR for dielectric
};

// =============================
// Geometry
// =============================
struct GPUSphere {
    float3 center;
    float  radius;
    int    material_id;
    int    _pad;
};

struct GPUTriangle {
    float3 v0;
    float3 v1;
    float3 v2;

    float3 n0;
    float3 n1;
    float3 n2;

    float3 uv0;    // using float3 for UV (u,v,0)
    float3 uv1;
    float3 uv2;

    int material_id; // index into GPUMaterial
    int albedo_tex;  // texture id for this tri (-1 if none)
};

// =============================
// BVH Node
// =============================
struct GPUBVHNode {
    vec3 bbox_min;
    vec3 bbox_max;

    int  left;       // index of left child, or -1 if leaf
    int  right;      // index of right child, or -1 if leaf
    int  tri_offset; // index into bvh_tri_indices
    int  tri_count;  // >0 => leaf, 0 => internal
};

// =============================
// Render / Scene Parameters
// =============================
enum SkyType : int {
    SKY_SOLID    = 0,
    SKY_GRADIENT = 1,
    SKY_ENV_MAP  = 2
};

struct GPURenderParams {
    int   img_width;
    int   img_height;
    int   samples_per_pixel;
    int   max_depth;

    int   use_bvh;
    int   rng_mode;
    int   tile_size;
    int   _pad0;

    float gamma;
    float exposure;
    float env_rotation;
    float _pad1;
};

// =============================
// Top-level GPU Scene
// =============================
struct GPUScene {
    // Geometry
    const GPUSphere*   spheres;
    int                num_spheres;
    int                _pad_sph0;
    int                _pad_sph1;

    const GPUTriangle* triangles;
    const int*         tri_indices;
    int                num_triangles;
    int                _pad_geo;

    GPUBVHNode*  bvh_nodes;        // device array of nodes
    int          num_bvh_nodes;

    int*         bvh_tri_indices;  // device array mapping BVH leaf ranges -> triangle indices

    // Materials
    const GPUMaterial* materials;
    int                num_materials;
    int                _pad_mat0;
    int                _pad_mat1;

    // Textures
    const GPUTextureHeader* textures;
    int                     num_textures;
    int                     _pad_tex0;
    int                     _pad_tex1;

    const float*       texture_pool;
    int                texture_pool_floats;
    int                _pad_pool0;
    int                _pad_pool1;

    // Camera
    GPUCamera          camera;

    // Sky
    int     sky_type;
    int     env_tex_id;
    int     _pad_sky0;
    int     _pad_sky1;
    float3  sky_solid;
    float3  sky_top;
    float3  sky_bottom;

    // Render params
    GPURenderParams    params;

    uint64_t           seed;

    // ================================
    // Directional Sun Light (NEW)
    // ================================
    bool sun_enabled;
    float3 sun_dir;        // normalized direction, ISS → Sun
    float3 sun_radiance;   // Sun brightness (e.g. {20,20,20})
};

#endif // GPU_SCENE_H
