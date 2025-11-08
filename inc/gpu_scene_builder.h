// inc/gpu_scene_builder.h
#pragma once
#include <vector>
#include <memory>
#include "gpu_scene.h"

#include "hittable_list.h"
#include "camera.h"

#include <unordered_map>
#include <type_traits>
#include "hittable_list.h"
#include "triangle_mesh.h"
#include "triangle.h"
#include "sphere.h"
#include "material.h"


struct HostSceneInputs {
    // geometry
    // ADD THIS:
    std::vector<GPUSphere>   spheres;
    std::vector<GPUTriangle> triangles;
    std::vector<int>         tri_indices;  // optional if no BVH
    std::vector<GPUBVHNode>  bvh_nodes;

    // materials / textures
    std::vector<GPUMaterial>      materials;
    std::vector<GPUTextureHeader> textures;
    std::vector<float>            texture_pool; // concatenated texels

    // camera
    GPUCamera camera{};
    bool      camera_invalid = false;

    // sky
    int   sky_type = SKY_SOLID;
    int   env_tex_id = -1;
    vec3  sky_solid = vec3(0.7, 0.8, 1.0);
    vec3  sky_top   = vec3(0.7, 0.8, 1.0);
    vec3  sky_bottom= vec3(1.0, 1.0, 1.0);

    // render params
    GPURenderParams params{};

    // seed
    uint64_t seed = 1337ULL;
};

class GPUSceneBuilder {
public:
    GPUSceneBuilder();
    ~GPUSceneBuilder();

    // Build uploads all arrays and returns a device pointer to a fully-initialized GPUScene.
    void build(const HostSceneInputs& in, GPUScene*& scene_device_out);

    // Frees all device allocations created by build()
    void destroy();

    // Optional: host mirror of the header
    const GPUScene& host_scene_header() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

// ------------------------------------------------------------------
// Convenience API used by main.cpp
// ------------------------------------------------------------------
GPUScene build_gpu_scene(const hittable_list& world, const camera& cam);
void     free_gpu_scene(GPUScene& scene);