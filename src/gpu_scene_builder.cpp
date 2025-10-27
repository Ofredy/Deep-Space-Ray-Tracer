#include "gpu_scene_builder.h"

#include "triangle_mesh.h"
#include "triangle.h"
#include "sphere.h"
#include "material.h"
#include "bvh.h"            // (BVH todo for later)
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <cstring> // for memcpy if needed

// ----------------------
// helper: upload std::vector<T> to device
// ----------------------
template <typename T>
static T* uploadVectorToDevice(const std::vector<T>& hostVec) {
    if (hostVec.empty()) return nullptr;

    T* d_ptr = nullptr;
    size_t bytes = hostVec.size() * sizeof(T);

    cudaMalloc(&d_ptr, bytes);
    cudaMemcpy(d_ptr, hostVec.data(), bytes, cudaMemcpyHostToDevice);

    return d_ptr;
}

// ----------------------
// Convert CPU material -> GPUMaterial
// We also dedupe by pointer identity so we don't duplicate the same mat.
// ----------------------
static int push_material_if_new(
    const std::shared_ptr<material>& m,
    std::vector<GPUMaterial>& gpuMats,
    std::vector<std::shared_ptr<material>>& matRefs
) {
    // Reuse if we've already seen this material pointer
    for (size_t i = 0; i < matRefs.size(); ++i) {
        if (matRefs[i].get() == m.get()) {
            return static_cast<int>(i);
        }
    }

    GPUMaterial gm{};
    gm.type    = MAT_LAMBERTIAN;
    gm.albedo  = vec3(0.8, 0.1, 0.1); // BRIGHT RED so we SEE IT
    gm.fuzz    = 0.0;
    gm.ref_idx = 1.0;

    // If it's actually metal or glass, override:
    if (auto met = std::dynamic_pointer_cast<metal>(m)) {
        gm.type   = MAT_METAL;
        gm.albedo = met->albedo;  // metal usually stores vec3 directly
        gm.fuzz   = met->fuzz;
    } else if (auto die = std::dynamic_pointer_cast<dielectric>(m)) {
        gm.type    = MAT_DIELECTRIC;
        gm.albedo  = vec3(1.0, 1.0, 1.0); // white
        gm.ref_idx = die->ir;
    }

    matRefs.push_back(m);
    gpuMats.push_back(gm);

    return static_cast<int>(gpuMats.size() - 1);
}

static void push_sphere(
    const std::shared_ptr<sphere>& s,
    std::vector<GPUSphere>& gpuSpheres,
    std::vector<GPUMaterial>& gpuMats,
    std::vector<std::shared_ptr<material>>& matRefs
) {
    GPUSphere gs{};

    gs.center = s->static_center();
    gs.radius = (float)s->get_radius();

    int mat_id = push_material_if_new(s->get_mat(), gpuMats, matRefs);
    gs.material_id = mat_id;

    gpuSpheres.push_back(gs);
}

// ----------------------
// Walk the hittable_list and pull out triangles
// (triangle or triangle_mesh). We'll extend this later for spheres, etc.
// ----------------------
static void collect_triangles(
    const hittable_list& world,
    std::vector<GPUTriangle>& outTris,
    std::vector<GPUMaterial>& outMats,
    std::vector<std::shared_ptr<material>>& matRefs
) {
    for (const auto& obj : world.objects) {
        if (!obj) continue;

        // whole mesh?
        if (auto mesh_ptr = std::dynamic_pointer_cast<triangle_mesh>(obj)) {
            for (const triangle& tri : mesh_ptr->triangles) {
                GPUTriangle gt;

                gt.v0 = tri.v0;
                gt.v1 = tri.v1;
                gt.v2 = tri.v2;

                gt.n0 = tri.n0;
                gt.n1 = tri.n1;
                gt.n2 = tri.n2;

                int mid = push_material_if_new(tri.mat, outMats, matRefs);
                gt.mat_id = mid;

                outTris.push_back(gt);
            }
        }
        // single loose triangle?
        else if (auto tri_ptr = std::dynamic_pointer_cast<triangle>(obj)) {
            const triangle& tri = *tri_ptr;

            GPUTriangle gt;
            gt.v0 = tri.v0;
            gt.v1 = tri.v1;
            gt.v2 = tri.v2;

            gt.n0 = tri.n0;
            gt.n1 = tri.n1;
            gt.n2 = tri.n2;

            int mid = push_material_if_new(tri.mat, outMats, matRefs);
            gt.mat_id = mid;

            outTris.push_back(gt);
        }
        else {
            // TODO: support spheres, quads, lights, etc.
            // For now we just skip them in the GPU scene builder.
        }
    }
}

// ----------------------
// PUBLIC: build_gpu_scene
// ----------------------
GPUScene build_gpu_scene(const hittable_list& world, const camera& cam)
{
    // 1. Flatten CPU scene into host arrays
    std::vector<GPUTriangle> hostTris;
    std::vector<GPUMaterial> hostMats;
    std::vector<std::shared_ptr<material>> matRefs; // parallel to hostMats

    collect_triangles(world, hostTris, hostMats, matRefs);

    // TODO later: build BVH on CPU and flatten into GPUNode list.
    std::vector<GPUNode> hostNodes; // empty for now

    // 2. Upload to device
    GPUScene scene{};

    scene.d_tris   = uploadVectorToDevice(hostTris);
    scene.num_tris = static_cast<int>(hostTris.size());

    scene.d_mats   = uploadVectorToDevice(hostMats);
    scene.num_mats = static_cast<int>(hostMats.size());

    scene.d_nodes    = uploadVectorToDevice(hostNodes);
    scene.num_nodes  = static_cast<int>(hostNodes.size());

    // 3. Copy camera
    scene.cam = cam.toGPUCamera();

    // 4. Render settings
    scene.image_width        = cam.image_width;
    scene.image_height       = cam.image_height;
    scene.samples_per_pixel  = cam.samples_per_pixel;
    scene.max_depth          = cam.max_depth;

    return scene;
}

// ----------------------
// PUBLIC: free_gpu_scene
// ----------------------
void free_gpu_scene(GPUScene& scene)
{
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

    scene.num_tris  = 0;
    scene.num_mats  = 0;
    scene.num_nodes = 0;
}
