#include <iostream>
#include <cstdio>
#include <memory>
#include <chrono>  // <-- for timing

#include "rtweekend.h"
#include "camera.h"
#include "hittable_list.h"
#include "triangle_mesh.h"
#include "bvh.h"
#include "material.h"
#include "sphere.h"

#include "gpu_scene_builder.h"
#include "gpu_scene.h"

// CUDA entry point (implemented in your gpu_render.cu or bridge)
extern "C"
void gpu_render_scene(const GPUScene& scene, int width, int height);

// --- optional PPM->PNG helper:
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
    using namespace std::chrono;

    auto total_start = high_resolution_clock::now();

    // ------------------------------------------------------------
    // 1) LOAD OBJ MESH INTO CPU WORLD
    // ------------------------------------------------------------
    auto t1 = high_resolution_clock::now();

    const char* OBJ_PATH = "../../iss_model/ISS_stationary.obj";
    const char* OBJ_DIR  = "../../iss_model";

    hittable_list world;
    hittable_list lights;

    auto fallbackM = std::make_shared<lambertian>(vec3(0.73, 0.73, 0.73));

    auto mesh_ptr = std::make_shared<triangle_mesh>(
        std::string(OBJ_PATH),
        fallbackM,
        1.0 // scale
    );
    world.add(mesh_ptr);

    // Add light
    auto bright_light_material = std::make_shared<diffuse_light>(color(200.0, 200.0, 200.0));
    auto ceiling_light = std::make_shared<sphere>(
        point3(0, -1000, 100),
        100.0,
        bright_light_material
    );
    world.add(ceiling_light);
    lights.add(ceiling_light);

    auto t2 = high_resolution_clock::now();
    std::cout << "OBJ + Scene load time: "
              << duration_cast<seconds>(t2 - t1).count() << " s\n";

    // ------------------------------------------------------------
    // 2) CAMERA SETUP
    // ------------------------------------------------------------
    camera cam;
    cam.image_width        = 800;
    cam.image_height       = 450;
    cam.samples_per_pixel  = 1000;
    cam.max_depth          = 50;

    cam.vfov     = 40;
    cam.lookfrom = point3(0, 0, 100);
    cam.lookat   = point3(0, 1.0, 0);
    cam.vup      = vec3(0, 1, 0);

    cam.aperture   = 0.0;
    cam.focus_dist = (cam.lookfrom - cam.lookat).length();
    cam.initialize();

    // ------------------------------------------------------------
    // 3) BUILD GPU SCENE
    // ------------------------------------------------------------
    auto t3 = high_resolution_clock::now();

    GPUScene gpu_scene = build_gpu_scene(world, cam);

    auto t4 = high_resolution_clock::now();
    std::cout << "GPU scene build time: "
              << duration_cast<milliseconds>(t4 - t3).count() << " ms\n";

    std::cout << "GPUScene.num_triangles = " << gpu_scene.num_triangles << "\n";
    std::cout << "GPUScene.num_spheres   = " << gpu_scene.num_spheres << "\n";
    std::cout << "GPUScene.num_materials = " << gpu_scene.num_materials << "\n";
    std::cout << "GPUScene.num_textures  = " << gpu_scene.num_textures << "\n";
    std::cout << "GPUScene.texture_pool_floats = " << gpu_scene.texture_pool_floats << "\n";

    // ------------------------------------------------------------
    // 4) RENDER ON GPU
    // ------------------------------------------------------------
    auto render_start = high_resolution_clock::now();

    gpu_render_scene(gpu_scene, cam.image_width, cam.image_height);

    auto render_end = high_resolution_clock::now();
    std::cout << "GPU render time: "
              << duration_cast<seconds>(render_end - render_start).count() << " s\n";

    // ------------------------------------------------------------
    // 5) CONVERT TO PNG (optional)
    // ------------------------------------------------------------
    auto convert_start = high_resolution_clock::now();
    ppm_to_png("output.ppm", "output.png");
    auto convert_end = high_resolution_clock::now();
    std::cout << "Image conversion time: "
              << duration_cast<milliseconds>(convert_end - convert_start).count() << " ms\n";

    // ------------------------------------------------------------
    // 6) CLEANUP
    // ------------------------------------------------------------
    auto free_start = high_resolution_clock::now();
    free_gpu_scene(gpu_scene);
    auto free_end = high_resolution_clock::now();
    std::cout << "GPU free time: "
              << duration_cast<milliseconds>(free_end - free_start).count() << " ms\n";

    auto total_end = high_resolution_clock::now();
    std::cout << "Total runtime: "
              << duration_cast<seconds>(total_end - total_start).count() << " s\n";

    std::cout << "Done.\n";
    return 0;
}
