#include <iostream>
#include <cstdio>
#include <memory>

#include "rtweekend.h"
#include "camera.h"
#include "hittable_list.h"
#include "triangle_mesh.h"
#include "bvh.h"
#include "material.h"
#include "sphere.h"

#include "gpu_scene_builder.h"   // GPUScene build_gpu_scene(), free_gpu_scene()
#include "gpu_scene.h"           // struct GPUScene

// CUDA entry point (gpu_render.cu)
extern "C"
void gpu_render_scene(const GPUScene& scene, int width, int height);

// --- optional PPM->PNG helper (same as before):
static inline int ppm_to_png(const std::string& ppm, const std::string& png) {
    std::string cmd = "magick \"" + ppm + "\" \"" + png + "\"";
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::string cmd2 = "magick convert \"" + ppm + "\" \"" + png + "\"";
        rc = std::system(cmd2.c_str());
    }
    return rc;
}

int main() {
    // ------------------------------------------------------------
    // 1. LOAD OBJ MESH INTO CPU WORLD
    // ------------------------------------------------------------

    const char* OBJ_PATH = "../../iss_model/ISS_stationary.obj";
    const char* OBJ_DIR  = "../../iss_model";  // folder with .obj, .mtl, textures

    hittable_list world;
    hittable_list lights;

    // fallback material for triangles
    auto fallbackM = std::make_shared<lambertian>(vec3(0.73, 0.73, 0.73));

    // load mesh directly using new constructor
    auto mesh_ptr = std::make_shared<triangle_mesh>(
        std::string(OBJ_PATH),
        fallbackM,
        1.0 // scale
    );

    world.add(mesh_ptr);

    // ------------------------------------------------------------
    // 2. ADD A LIGHT (CPU SIDE)
    // ------------------------------------------------------------
    // Note: Your GPU code right now DOES NOT support emissive materials
    // or sampling of lights. We'll still add it to the world so it ends
    // up in the BVH / triangle list, but on GPU it'll probably just show
    // up as a white ball, not actual lighting.

    auto bright_light_material = std::make_shared<diffuse_light>(color(200, 200, 200));

    auto ceiling_light = std::make_shared<sphere>(
        point3(0, 500, 2),
        100.0,
        bright_light_material
    );

    world.add(ceiling_light);
    lights.add(ceiling_light);

    // ------------------------------------------------------------
    // 3. ACCELERATION STRUCTURE (BVH)
    // ------------------------------------------------------------
    // This wraps all hittables into one hittable (BVH root).
    // Your build_gpu_scene() needs to handle BVH nodes or flatten them.
    // If build_gpu_scene() already walks the world (including bvh_node),
    // leave this. If not, you can skip BVH for now.
    world = hittable_list(std::make_shared<bvh_node>(world));

    // ------------------------------------------------------------
    // 4. CAMERA SETUP
    // ------------------------------------------------------------
    camera cam;
    cam.image_width        = 800;
    cam.image_height       = 450;
    cam.samples_per_pixel  = 1;   // keep 1 for now for speed / debugging
    cam.max_depth          = 1;   // single bounce / primary-only for now

    cam.vfov     = 40;
    cam.lookfrom = point3(0, 0, 100);    // same as your CPU scene
    cam.lookat   = point3(0, 1.0, 0);
    cam.vup      = vec3(0, 1, 0);

    cam.aperture   = 0.0;
    cam.focus_dist = (cam.lookfrom - cam.lookat).length();

    cam.initialize(); // sets internal GPUCamera fields

    // ------------------------------------------------------------
    // 5. BUILD GPU SCENE FROM CPU WORLD
    // ------------------------------------------------------------
    GPUScene gpu_scene = build_gpu_scene(world, cam);

    // quick sanity prints
    std::cout << "GPUScene.num_tris  = " << gpu_scene.num_tris  << "\n";
    std::cout << "GPUScene.num_mats  = " << gpu_scene.num_mats  << "\n";
    std::cout << "GPUScene.image_w/h = " << gpu_scene.image_width
              << " x " << gpu_scene.image_height << "\n";

    // ------------------------------------------------------------
    // 6. RENDER ON GPU
    // ------------------------------------------------------------
    gpu_render_scene(
        gpu_scene,
        cam.image_width,
        cam.image_height
    );

    // We expect gpu_render_scene to write "output.ppm".
    // We'll try to convert it to PNG after:
    ppm_to_png("output.ppm", "output.png");

    // ------------------------------------------------------------
    // 7. CLEANUP GPU BUFFERS
    // ------------------------------------------------------------
    free_gpu_scene(gpu_scene);

    std::cout << "Done.\n";
    return 0;
}
