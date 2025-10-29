#include "gpu_scene_builder.h"

#include "gpu_scene.h"        // GPUScene, GPUCamera, GPUSphere, GPUTriangle, GPUMaterial, GPUNode
#include "hittable_list.h"    // hittable_list
#include "sphere.h"           // class sphere
#include "triangle_mesh.h"    // class triangle_mesh
#include "triangle.h"         // class triangle
#include "material.h"         // lambertian, metal, dielectric, diffuse_light, etc.
#include "camera.h"           // class camera (with toGPUCamera())

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdlib>
#include <cstdio>

// =========================================================
// Small CUDA error helper (host-side only)
// =========================================================
static void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        printf("CUDA ERROR %s: %s\n", msg, cudaGetErrorString(result));
        std::abort();
    }
}

// =========================================================
// Device upload helper for std::vector<T>
// Returns device pointer or nullptr if empty
// =========================================================
template <typename T>
static T* uploadVectorToDevice(const std::vector<T>& hostVec) {
    if (hostVec.empty()) return nullptr;

    T* d_ptr = nullptr;
    size_t bytes = hostVec.size() * sizeof(T);

    checkCuda(cudaMalloc(&d_ptr, bytes), "cudaMalloc uploadVectorToDevice");
    checkCuda(cudaMemcpy(d_ptr, hostVec.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy uploadVectorToDevice");

    return d_ptr;
}

// =========================================================
// Material packing
// We dedupe materials by pointer identity.
// We'll assign each material an integer ID and produce a GPUMaterial.
// =========================================================

struct MatRecord {
    int id;
    GPUMaterial gpuMat;
};

static int get_material_id(
    const std::shared_ptr<material>& m,
    std::unordered_map<material*, MatRecord>& mat_table,
    std::vector<GPUMaterial>& mats_out
) {
    if (!m) {
        return -1;
    }

    // Did we already assign this material an ID?
    auto it = mat_table.find(m.get());
    if (it != mat_table.end()) {
        return it->second.id;
    }

    GPUMaterial g{};
    // defaults so it's visible even if we don't know what it is yet
    g.type    = MAT_LAMBERTIAN;
    g.albedo  = vec3(0.8, 0.1, 0.1); // fallback: reddish
    g.fuzz    = 0.0f;
    g.ref_idx = 1.0f;

    // Try to detect subclasses.
    // NOTE: tweak these dynamic_pointer_cast targets to match YOUR actual class names.
    if (auto lam = std::dynamic_pointer_cast<lambertian>(m)) {
        // lambertian in your code uses a texture instead of a raw color,
        // so we can't grab lam->albedo as a vec3.
        // Just bake a neutral gray so we can see geometry.
        g.type   = MAT_LAMBERTIAN;
        g.albedo = vec3(0.7, 0.7, 0.7); // fake diffuse color
    }
    else if (auto met = std::dynamic_pointer_cast<metal>(m)) {
        g.type   = MAT_METAL;
        // assuming 'metal' has 'albedo' as vec3 and 'fuzz' as double/float
        g.albedo = met->albedo;
        g.fuzz   = (float)met->fuzz;
    }
    else if (auto die = std::dynamic_pointer_cast<dielectric>(m)) {
        g.type    = MAT_DIELECTRIC;
        g.albedo  = vec3(1.0, 1.0, 1.0); // glass = white highlight
        g.ref_idx = (float)die->ir;      // index of refraction
    }
    else if (auto light = std::dynamic_pointer_cast<diffuse_light>(m)) {
        g.type   = MAT_LAMBERTIAN;
        // many diffuse_light classes store emit(color) or similar;
        // if yours uses a texture too, same problem, so just fake bright white
        g.albedo = vec3(20.0, 20.0, 20.0);
    }

    MatRecord rec;
    rec.id     = (int)mats_out.size();
    rec.gpuMat = g;

    mat_table[m.get()] = rec;
    mats_out.push_back(g);

    return rec.id;
}

// =========================================================
// push_sphere(): convert CPU sphere -> GPUSphere
//
// EXPECTED GPUSphere FIELDS (must match gpu_render.cu):
//   vec3 center;
//   float radius;
//   int   material_id;
// =========================================================
static void push_sphere(
    const std::shared_ptr<sphere>& s,
    std::vector<GPUSphere>& gpuSpheres,
    std::unordered_map<material*, MatRecord>& mat_table,
    std::vector<GPUMaterial>& gpuMats
) {
    GPUSphere gs{};

    // Your sphere class might expose different getters.
    // I'm guessing from your code:
    //   s->static_center()   returns vec3/point3
    //   s->get_radius()      returns double
    //   s->get_mat()         returns shared_ptr<material>
    gs.center = s->static_center();
    gs.radius = (float)s->get_radius();

    int mat_id = get_material_id(s->get_mat(), mat_table, gpuMats);
    gs.material_id = mat_id;

    gpuSpheres.push_back(gs);
}

// =========================================================
// push_triangle(): convert CPU triangle -> GPUTriangle
//
// EXPECTED GPUTriangle FIELDS (must match gpu_render.cu):
//   vec3 v0;
//   vec3 v1;
//   vec3 v2;
//   int  material_id;
// =========================================================
static void push_triangle(
    const triangle& tri,
    std::vector<GPUTriangle>& gpuTris,
    std::unordered_map<material*, MatRecord>& mat_table,
    std::vector<GPUMaterial>& gpuMats
) {
    GPUTriangle gt{};

    // Your triangle class appears to have public v0,v1,v2, and 'mat'
    gt.v0 = tri.v0;
    gt.v1 = tri.v1;
    gt.v2 = tri.v2;

    int mid = get_material_id(tri.mat, mat_table, gpuMats);
    gt.material_id = mid;

    gpuTris.push_back(gt);
}

// =========================================================
// collect_primitives(): walk hittable_list and gather
// spheres + triangle meshes + loose triangles
//
// NOTE: we do NOT recurse into bvh_node yet.
// So in main(), DON'T wrap world in bvh_node for now.
// =========================================================
static void collect_primitives(
    const hittable_list& world,
    std::vector<GPUSphere>& outSpheres,
    std::vector<GPUTriangle>& outTris,
    std::unordered_map<material*, MatRecord>& mat_table,
    std::vector<GPUMaterial>& outMats
) {
    for (const auto& obj : world.objects) {
        if (!obj) continue;

        // sphere
        if (auto sp = std::dynamic_pointer_cast<sphere>(obj)) {
            push_sphere(sp, outSpheres, mat_table, outMats);
            continue;
        }

        // triangle_mesh (many tris)
        if (auto mesh_ptr = std::dynamic_pointer_cast<triangle_mesh>(obj)) {
            for (const triangle& tri : mesh_ptr->triangles) {
                push_triangle(tri, outTris, mat_table, outMats);
            }
            continue;
        }

        // single triangle
        if (auto tri_ptr = std::dynamic_pointer_cast<triangle>(obj)) {
            push_triangle(*tri_ptr, outTris, mat_table, outMats);
            continue;
        }

        // TODO (later): if obj is a bvh_node, recurse into its children
        // For now we ignore BVH nodes in GPU path.
    }
}

// =========================================================
// PUBLIC: build_gpu_scene()
// - flattens objects
// - uploads them
// - packs camera
// - returns a GPUScene ready for gpu_render_scene()
// =========================================================
GPUScene build_gpu_scene(const hittable_list& world, const camera& cam)
{
    // 1. Gather CPU-side primitives
    std::vector<GPUSphere>   hostSpheres;
    std::vector<GPUTriangle> hostTris;
    std::vector<GPUMaterial> hostMats;
    std::unordered_map<material*, MatRecord> mat_table;

    collect_primitives(world, hostSpheres, hostTris, mat_table, hostMats);

    // (Optional future) build BVH nodes for GPU. Right now: empty.
    std::vector<GPUNode> hostNodes;

    // 2. Fill scene struct
    GPUScene scene{};

    // upload spheres
    scene.d_spheres   = uploadVectorToDevice(hostSpheres);
    scene.num_spheres = (int)hostSpheres.size();

    // upload tris
    scene.d_tris   = uploadVectorToDevice(hostTris);
    scene.num_tris = (int)hostTris.size();

    // upload mats
    scene.d_mats   = uploadVectorToDevice(hostMats);
    scene.num_mats = (int)hostMats.size();

    // upload BVH nodes (currently empty)
    scene.d_nodes   = uploadVectorToDevice(hostNodes);
    scene.num_nodes = (int)hostNodes.size();

    // 3. Camera packing
    //
    // You either:
    //   (A) have camera::toGPUCamera() that returns GPUCamera with float3 members
    // or
    //   (B) need to pack manually here.
    //
    // I'll assume you already wrote cam.toGPUCamera() that builds a GPUCamera
    // compatible with the kernel (origin, lower_left_corner, horizontal, vertical as float3).
    scene.cam = cam.toGPUCamera();

    // 4. Global render settings
    scene.image_width        = cam.image_width;
    scene.image_height       = cam.image_height;
    scene.samples_per_pixel  = cam.samples_per_pixel;
    scene.max_depth          = cam.max_depth;

    return scene;
}

// =========================================================
// PUBLIC: free_gpu_scene()
// - frees device allocations
// =========================================================
void free_gpu_scene(GPUScene& scene)
{
    if (scene.d_spheres) {
        cudaFree(scene.d_spheres);
        scene.d_spheres = nullptr;
    }

    if (scene.d_tris) {
        cudaFree(scene.d_tris);
        scene.d_tris = nullptr;
    }

    if (scene.d_mats) {
        cudaFree(scene.d_mats);
        scene.d_mats = nullptr;
    }

    if (scene.d_nodes) {
        cudaFree(scene.d_nodes);
        scene.d_nodes = nullptr;
    }

    scene.num_spheres = 0;
    scene.num_tris    = 0;
    scene.num_mats    = 0;
    scene.num_nodes   = 0;
}
