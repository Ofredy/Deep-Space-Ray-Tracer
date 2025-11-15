// src/gpu_scene_builder.cpp
#include "gpu_scene_builder.h"

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdio>
#include <type_traits>
#include <string>
#include <algorithm>   // for std::nth_element

// Project headers
#include "gpu_scene.h"
#include "hittable_list.h"
#include "triangle.h"
#include "triangle_mesh.h"
#include "sphere.h"
#include "material.h"
#include "camera.h"
#include "rtweekend.h"   // for vec3 utilities (length, etc.)
#include "stb_image.h"

// ------------------------------------------------------------
// CUDA error helper
// ------------------------------------------------------------
static inline void checkCuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA ERROR at %s: %s\n", where, cudaGetErrorString(e));
#ifndef NDEBUG
        asm("trap;");
#endif
    }
}

// ------------------------------------------------------------
// CPU → GPU packing helpers
// ------------------------------------------------------------
static inline GPUTriangle make_gpu_triangle_from_cpu(const triangle& t, int mat_id, int albedo_tex_id = -1) {
    GPUTriangle gt{};
    gt.v0 = make_float3((float)t.v0.x(), (float)t.v0.y(), (float)t.v0.z());
    gt.v1 = make_float3((float)t.v1.x(), (float)t.v1.y(), (float)t.v1.z());
    gt.v2 = make_float3((float)t.v2.x(), (float)t.v2.y(), (float)t.v2.z());

    gt.n0 = make_float3((float)t.n0.x(), (float)t.n0.y(), (float)t.n0.z());
    gt.n1 = make_float3((float)t.n1.x(), (float)t.n1.y(), (float)t.n1.z());
    gt.n2 = make_float3((float)t.n2.x(), (float)t.n2.y(), (float)t.n2.z());

    gt.uv0 = make_float3((float)t.uv0.x(), (float)t.uv0.y(), 0.f);
    gt.uv1 = make_float3((float)t.uv1.x(), (float)t.uv1.y(), 0.f);
    gt.uv2 = make_float3((float)t.uv2.x(), (float)t.uv2.y(), 0.f);

    gt.material_id = mat_id;
    gt.albedo_tex  = albedo_tex_id;
    return gt;
}

static inline GPUSphere make_gpu_sphere_from_cpu(const sphere& s, int mat_id) {
    GPUSphere gs{};
    auto c = s.static_center();
    gs.center      = make_float3((float)c.x(), (float)c.y(), (float)c.z());
    gs.radius      = (float)s.get_radius();
    gs.material_id = mat_id;
    gs._pad        = 0;
    return gs;
}

// ------------------------------------------------------------
// Material table – map shared_ptr<material> to a unique GPU index
// ------------------------------------------------------------
static int upsert_material(
    const std::shared_ptr<material>& mptr,
    std::vector<GPUMaterial>& out,
    std::unordered_map<const material*, int>& idx_cache)
{
    if (!mptr) {
        GPUMaterial gm{};
        gm.type       = MAT_LAMBERTIAN;
        gm.albedo     = make_float3(0.8f, 0.8f, 0.8f);
        gm.emissive   = make_float3(0.f, 0.f, 0.f);
        gm.fuzz       = 0.f;
        gm.ref_idx    = 1.5f;
        gm.albedo_tex = -1;
        out.push_back(gm);
        return (int)out.size() - 1;
    }

    if (auto it = idx_cache.find(mptr.get()); it != idx_cache.end())
        return it->second;

    GPUMaterial gm{};
    gm.albedo_tex = -1;

    const material* raw = mptr.get();

    if (auto lam = dynamic_cast<const lambertian*>(raw)) {
        gm.type   = MAT_LAMBERTIAN;
        auto c    = lam->albedo_value();
        gm.albedo = make_float3((float)c.x(), (float)c.y(), (float)c.z());
        gm.emissive = make_float3(0.f, 0.f, 0.f);
        gm.fuzz   = 0.f;
        gm.ref_idx= 1.5f;
    }
    else if (auto met = dynamic_cast<const metal*>(raw)) {
        gm.type   = MAT_METAL;
        auto c    = met->albedo_value();
        gm.albedo = make_float3((float)c.x(), (float)c.y(), (float)c.z());
        gm.emissive = make_float3(0.f, 0.f, 0.f);
        gm.fuzz   = (float)met->fuzz_value();
        gm.ref_idx= 1.5f;
    }
    else if (auto diel = dynamic_cast<const dielectric*>(raw)) {
        gm.type   = MAT_DIELECTRIC;
        gm.albedo = make_float3(1.f, 1.f, 1.f);
        gm.emissive = make_float3(0.f, 0.f, 0.f);
        gm.fuzz   = 0.f;
        gm.ref_idx= (float)diel->ior_value();
    }
    else if (auto em = dynamic_cast<const diffuse_light*>(raw)) {
        gm.type   = MAT_DIFFUSE_LIGHT;
        gm.albedo = make_float3(1.f, 1.f, 1.f);
        auto e    = em->emit_value();
        gm.emissive = make_float3((float)e.x(), (float)e.y(), (float)e.z());
        gm.fuzz   = 0.f;
        gm.ref_idx= 1.f;
    }
    else {
        gm.type   = MAT_LAMBERTIAN;
        gm.albedo = make_float3(0.73f, 0.73f, 0.73f);
        gm.emissive = make_float3(0.f, 0.f, 0.f);
        gm.fuzz   = 0.f;
        gm.ref_idx= 1.5f;
    }

    out.push_back(gm);
    int idx = (int)out.size() - 1;
    idx_cache.emplace(mptr.get(), idx);
    return idx;
}

// ------------------------------------------------------------
// BVH helpers (CPU side)
// ------------------------------------------------------------
static AABB tri_bounds(const GPUTriangle& t) {
    AABB box;
    float min_x = fminf(fminf(t.v0.x, t.v1.x), t.v2.x);
    float min_y = fminf(fminf(t.v0.y, t.v1.y), t.v2.y);
    float min_z = fminf(fminf(t.v0.z, t.v1.z), t.v2.z);
    float max_x = fmaxf(fmaxf(t.v0.x, t.v1.x), t.v2.x);
    float max_y = fmaxf(fmaxf(t.v0.y, t.v1.y), t.v2.y);
    float max_z = fmaxf(fmaxf(t.v0.z, t.v1.z), t.v2.z);
    box.minp = make_float3(min_x, min_y, min_z);
    box.maxp = make_float3(max_x, max_y, max_z);
    return box;
}

static float3 tri_centroid(const GPUTriangle& t) {
    return make_float3(
        (t.v0.x + t.v1.x + t.v2.x) / 3.f,
        (t.v0.y + t.v1.y + t.v2.y) / 3.f,
        (t.v0.z + t.v1.z + t.v2.z) / 3.f
    );
}

static AABB union_box(const AABB& a, const AABB& b) {
    AABB out;
    out.minp = make_float3(
        fminf(a.minp.x, b.minp.x),
        fminf(a.minp.y, b.minp.y),
        fminf(a.minp.z, b.minp.z)
    );
    out.maxp = make_float3(
        fmaxf(a.maxp.x, b.maxp.x),
        fmaxf(a.maxp.y, b.maxp.y),
        fmaxf(a.maxp.z, b.maxp.z)
    );
    return out;
}

// ------------------------------------------------------------
// World flattener – pull triangles / spheres out of world
// ------------------------------------------------------------
struct HostBuild {
    std::vector<GPUTriangle>  h_tris;
    std::vector<GPUSphere>    h_spheres;
    std::vector<GPUMaterial>  h_mats;

    // Keep material indices so identical pointers map to one GPUMaterial
    std::unordered_map<const material*, int> mat_index;
};

struct HostTexture {
    int width  = 0;
    int height = 0;
    std::vector<float> data;   // RGB floats, size = width * height * 3
    std::string path;
};

struct HostTextureRegistry {
    std::vector<HostTexture> textures;
    std::unordered_map<std::string, int> path_to_index;

    int get_or_load(const std::string& path) {
        if (path.empty()) return -1;

        auto it = path_to_index.find(path);
        if (it != path_to_index.end()) {
            return it->second;
        }

        HostTexture ht;
        ht.path = path;

        int w = 0, h = 0, n = 0;
        unsigned char* img = stbi_load(path.c_str(), &w, &h, &n, 3); // force RGB
        if (!img) {
            std::fprintf(stderr, "WARN: failed to load texture '%s'\n", path.c_str());
            // Fallback to solid white 1×1
            ht.width  = 1;
            ht.height = 1;
            ht.data   = { 1.0f, 1.0f, 1.0f };
        } else {
            ht.width  = w;
            ht.height = h;
            ht.data.resize((size_t)w * h * 3);
            for (int p = 0; p < w * h; ++p) {
                unsigned char r = img[p * 3 + 0];
                unsigned char g = img[p * 3 + 1];
                unsigned char b = img[p * 3 + 2];
                auto to_linear = [](float c) {
                    return powf(c / 255.0f, 2.2f);   // convert sRGB → linear
                };
                
                ht.data[p * 3 + 0] = to_linear(r);
                ht.data[p * 3 + 1] = to_linear(g);
                ht.data[p * 3 + 2] = to_linear(b);
            }
            stbi_image_free(img);
        }

        int idx = (int)textures.size();
        textures.push_back(std::move(ht));
        path_to_index[path] = idx;
        return idx;
    }
};

static std::shared_ptr<material> get_material_from_sphere(const sphere& s) {
    return s.get_material();   // just return the material pointer directly
}

static void collect_from_hittable(const std::shared_ptr<hittable>& obj,
                                  HostBuild& B,
                                  HostTextureRegistry& texreg)
{
    if (!obj) return;

    // Triangle mesh
    if (auto mesh = std::dynamic_pointer_cast<triangle_mesh>(obj)) {
        const auto& tris     = mesh->triangles;
        const auto& tri_maps = mesh->tri_map_Kd;   // per-triangle map_Kd paths
        
        for (size_t i = 0; i < tris.size(); ++i) {
            const triangle& t = tris[i];
        
            int mid = upsert_material(t.mat, B.h_mats, B.mat_index);
        
            int tex_id = -1;
            if (i < tri_maps.size() && !tri_maps[i].empty()) {
                tex_id = texreg.get_or_load(tri_maps[i]);
                if (tex_id < 0) tex_id = -1;
            }
        
            // 💡 If this triangle has a texture, treat the material albedo as white
            if (tex_id >= 0) {
                GPUMaterial& gm = B.h_mats[mid];
                gm.albedo = make_float3(1.0f, 1.0f, 1.0f);
            }
        
            B.h_tris.emplace_back(make_gpu_triangle_from_cpu(t, mid, tex_id));
        }
        return;
    }

    // Single triangle (if ever added to world directly)
    if (auto tri = std::dynamic_pointer_cast<triangle>(obj)) {
        int mid = upsert_material(tri->mat, B.h_mats, B.mat_index);
        B.h_tris.emplace_back(make_gpu_triangle_from_cpu(*tri, mid, -1));
        return;
    }

    // Sphere
    if (auto sph = std::dynamic_pointer_cast<sphere>(obj)) {
        int mid = upsert_material(get_material_from_sphere(*sph), B.h_mats, B.mat_index);
        B.h_spheres.emplace_back(make_gpu_sphere_from_cpu(*sph, mid));
        return;
    }

    // If obj is a container (like hittable_list), descend into it
    if (auto list = std::dynamic_pointer_cast<hittable_list>(obj)) {
        for (const auto& h : list->objects) {
            collect_from_hittable(h, B, texreg);
        }
        return;
    }

    // Otherwise: not a known type; ignore
}

static void collect_world(const hittable_list& world,
                          HostBuild& B,
                          HostTextureRegistry& texreg)
{
    for (const auto& obj : world.objects) {
        collect_from_hittable(obj, B, texreg);
    }
}

// ------------------------------------------------------------
// Device allocation & upload helpers
// ------------------------------------------------------------
template <typename T>
static T* upload_vector(const std::vector<T>& v, const char* tag) {
    if (v.empty()) return nullptr;
    T* d_ptr = nullptr;
    size_t bytes = v.size() * sizeof(T);
    checkCuda(cudaMalloc((void**)&d_ptr, bytes), (std::string("cudaMalloc ") + tag).c_str());
    checkCuda(cudaMemcpy(d_ptr, v.data(), bytes, cudaMemcpyHostToDevice),
              (std::string("cudaMemcpy ") + tag).c_str());
    return d_ptr;
}

// ------------------------------------------------------------
// Camera & params
// ------------------------------------------------------------
static GPUCamera to_gpu_camera(const camera& c) {
    return c.toGPUCamera();
}

// ----------------------
// BVH build (CPU side)
// ----------------------
static int build_bvh_recursive(
    const std::vector<GPUTriangle>& tris,
    std::vector<int>& indices,
    std::vector<GPUBVHNode>& nodes,
    int start,
    int end)
{
    const int node_index = (int)nodes.size();
    nodes.emplace_back();
    GPUBVHNode& node = nodes.back();

    // Compute bounds of all primitives in [start, end)
    AABB box = tri_bounds(tris[indices[start]]);
    for (int i = start + 1; i < end; ++i) {
        box = union_box(box, tri_bounds(tris[indices[i]]));
    }

    // Fill node's bbox (AABB uses float3; GPUBVHNode uses vec3)
    node.bbox_min = vec3(box.minp.x, box.minp.y, box.minp.z);
    node.bbox_max = vec3(box.maxp.x, box.maxp.y, box.maxp.z);

    node.left  = -1;
    node.right = -1;

    const int prim_count    = end - start;
    const int max_leaf_size = 4;

    // By default, treat as leaf: record triangle range here
    node.tri_offset = start;
    node.tri_count  = prim_count;

    // If small enough, keep as leaf
    if (prim_count <= max_leaf_size) {
        return node_index;
    }

    // Compute centroid bounds
    AABB centroid_box;
    {
        float3 c0 = tri_centroid(tris[indices[start]]);
        centroid_box.minp = c0;
        centroid_box.maxp = c0;

        for (int i = start + 1; i < end; ++i) {
            float3 c = tri_centroid(tris[indices[i]]);
            centroid_box.minp.x = fminf(centroid_box.minp.x, c.x);
            centroid_box.minp.y = fminf(centroid_box.minp.y, c.y);
            centroid_box.minp.z = fminf(centroid_box.minp.z, c.z);

            centroid_box.maxp.x = fmaxf(centroid_box.maxp.x, c.x);
            centroid_box.maxp.y = fmaxf(centroid_box.maxp.y, c.y);
            centroid_box.maxp.z = fmaxf(centroid_box.maxp.z, c.z);
        }
    }

    // Choose split axis by largest extent
    float3 diag = make_float3(
        centroid_box.maxp.x - centroid_box.minp.x,
        centroid_box.maxp.y - centroid_box.minp.y,
        centroid_box.maxp.z - centroid_box.minp.z
    );
    int axis = 0;
    if (diag.y > diag.x && diag.y >= diag.z) axis = 1;
    else if (diag.z > diag.x && diag.z >= diag.y) axis = 2;

    // If degenerate (all centroids the same), keep as leaf
    if ((axis == 0 && diag.x == 0.0f) ||
        (axis == 1 && diag.y == 0.0f) ||
        (axis == 2 && diag.z == 0.0f)) {
        // node.tri_offset / tri_count already set above
        return node_index;
    }

    int mid = (start + end) / 2;

    auto cent_less = [&](int tri_a, int tri_b) {
        float3 ca = tri_centroid(tris[tri_a]);
        float3 cb = tri_centroid(tris[tri_b]);
        if (axis == 0) return ca.x < cb.x;
        if (axis == 1) return ca.y < cb.y;
        return ca.z < cb.z;
    };

    // Partition indices based on centroid along chosen axis
    std::nth_element(
        indices.begin() + start,
        indices.begin() + mid,
        indices.begin() + end,
        [&](int a, int b) { return cent_less(a, b); }
    );

    // Mark this as an internal node: triangles are in children
    node.tri_offset = 0;
    node.tri_count  = 0;

    node.left  = build_bvh_recursive(tris, indices, nodes, start, mid);
    node.right = build_bvh_recursive(tris, indices, nodes, mid,   end);

    return node_index;
}

static void build_bvh_for_triangles(
    const std::vector<GPUTriangle>& tris,
    std::vector<int>& out_indices,
    std::vector<GPUBVHNode>& out_nodes)
{
    const int N = (int)tris.size();
    if (N == 0) return;

    out_indices.resize(N);
    for (int i = 0; i < N; ++i) out_indices[i] = i;

    out_nodes.clear();
    out_nodes.reserve(N * 2);

    build_bvh_recursive(tris, out_indices, out_nodes, 0, N);
}

// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------
GPUScene build_gpu_scene(const hittable_list& world, const camera& cam)
{
    HostBuild B;
    HostTextureRegistry texreg;
    collect_world(world, B, texreg);

    GPUScene scene{};

    // ------------------------
    // Geometry & materials
    // ------------------------
    GPUTriangle* d_tris   = upload_vector(B.h_tris,    "tris");
    GPUSphere*   d_sph    = upload_vector(B.h_spheres, "spheres");
    GPUMaterial* d_mats   = upload_vector(B.h_mats,    "materials");

    scene.spheres       = d_sph;
    scene.num_spheres   = (int)B.h_spheres.size();
    scene._pad_sph0 = scene._pad_sph1 = 0;

    scene.triangles     = d_tris;
    scene.num_triangles = (int)B.h_tris.size();
    scene._pad_geo      = 0;

    // ------------------------
    // Build BVH on CPU
    // ------------------------
    std::vector<int>        h_indices;
    std::vector<GPUBVHNode> h_nodes;

    build_bvh_for_triangles(B.h_tris, h_indices, h_nodes);

    // Upload BVH data to GPU
    int*        d_indices = upload_vector(h_indices, "tri_indices");
    GPUBVHNode* d_bvh     = upload_vector(h_nodes,   "bvh_nodes");

    // Hook into scene (this is what bvh_hit_closest() expects)
    scene.tri_indices     = d_indices;                     // <-- NEW: used in gpu_render.cu
    scene.bvh_nodes       = d_bvh;                         // was d_bvh_nodes (wrong)
    scene.num_bvh_nodes   = (int)h_nodes.size();           // was num_bvh_nodes (undefined)
    scene.bvh_tri_indices = d_indices;                     // optional; kernel uses tri_indices

    // Materials
    scene.materials     = d_mats;
    scene.num_materials = (int)B.h_mats.size();
    scene._pad_mat0 = scene._pad_mat1 = 0;

    // ------------------------
    // Textures
    // ------------------------
    if (!texreg.textures.empty()) {
        std::vector<GPUTextureHeader> headers;
        headers.reserve(texreg.textures.size());

        std::vector<float> pool;
        // optional: pre-reserve some space
        for (const auto& ht : texreg.textures) {
            pool.reserve(pool.size() + ht.data.size());
        }

        int float_offset = 0;  // index into pool[] (floats)

        for (const auto& ht : texreg.textures) {
            GPUTextureHeader h{};
            h.width  = ht.width;
            h.height = ht.height;
            h.offset = float_offset;              // float index into texture_pool

            headers.push_back(h);

            pool.insert(pool.end(), ht.data.begin(), ht.data.end());
            float_offset += (int)ht.data.size();
        }

        GPUTextureHeader* d_headers = upload_vector(headers, "texture_headers");
        float*            d_pool    = upload_vector(pool,    "texture_pool");

        scene.textures             = d_headers;
        scene.num_textures         = (int)headers.size();
        scene._pad_tex0 = scene._pad_tex1 = 0;

        scene.texture_pool         = d_pool;
        scene.texture_pool_floats  = (int)pool.size();
        scene._pad_pool0 = scene._pad_pool1 = 0;
    } else {
        scene.textures             = nullptr;
        scene.num_textures         = 0;
        scene._pad_tex0 = scene._pad_tex1 = 0;

        scene.texture_pool         = nullptr;
        scene.texture_pool_floats  = 0;
        scene._pad_pool0 = scene._pad_pool1 = 0;
    }

    // ------------------------
    // Camera
    // ------------------------
    scene.camera = to_gpu_camera(cam);

    // Sky defaults (solid black for your "only sphere is light" setup)
    scene.sky_type   = SKY_SOLID;
    scene.env_tex_id = -1;
    scene._pad_sky0 = scene._pad_sky1 = 0;
    scene.sky_solid  = make_float3(0.0f, 0.0f, 0.0f);
    scene.sky_top    = make_float3(0.0f, 0.0f, 0.0f);
    scene.sky_bottom = make_float3(0.0f, 0.0f, 0.0f);

    // Render params from camera
    GPURenderParams P{};
    P.img_width         = cam.image_width;
    P.img_height        = cam.image_height;
    P.samples_per_pixel = cam.samples_per_pixel;
    P.max_depth         = cam.max_depth;
    P.use_bvh           = 1;
    P.rng_mode          = 0;
    P.tile_size         = 0;
    P.gamma             = 2.0f;
    P.exposure          = 1.0f;
    P.env_rotation      = 0.0f;
    scene.params = P;

    // Seed
    scene.seed = 1337ULL;

    return scene;
}

void free_gpu_scene(GPUScene& scene)
{
    // Only free what we allocated
    if (scene.triangles)      checkCuda(cudaFree((void*)scene.triangles), "cudaFree triangles");
    if (scene.spheres)        checkCuda(cudaFree((void*)scene.spheres),   "cudaFree spheres");
    if (scene.materials)      checkCuda(cudaFree((void*)scene.materials), "cudaFree materials");

    if (scene.bvh_nodes)      checkCuda(cudaFree((void*)scene.bvh_nodes),   "cudaFree bvh_nodes");
    if (scene.tri_indices)    checkCuda(cudaFree((void*)scene.tri_indices), "cudaFree tri_indices");

    if (scene.textures)       checkCuda(cudaFree((void*)scene.textures),      "cudaFree textures");
    if (scene.texture_pool)   checkCuda(cudaFree((void*)scene.texture_pool),  "cudaFree texture_pool");

    // Null out pointers/counts to avoid double-free and stale prints
    scene.triangles = nullptr;   scene.num_triangles = 0;
    scene.spheres   = nullptr;   scene.num_spheres   = 0;
    scene.materials = nullptr;   scene.num_materials = 0;

    scene.bvh_nodes = nullptr;   scene.num_bvh_nodes = 0;
    scene.tri_indices = nullptr;

    scene.textures = nullptr;    scene.num_textures = 0;
    scene.texture_pool = nullptr; scene.texture_pool_floats = 0;
}
