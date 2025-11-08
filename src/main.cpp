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
    // ------------------------------------------------------------
    // 1) LOAD OBJ MESH INTO CPU WORLD
    // ------------------------------------------------------------
    const char* OBJ_PATH = "../../iss_model/ISS_stationary.obj";
    const char* OBJ_DIR  = "../../iss_model"; // (unused here, but keep for reference)

    hittable_list world;
    hittable_list lights;

    auto fallbackM = std::make_shared<lambertian>(vec3(0.73, 0.73, 0.73));

    auto mesh_ptr = std::make_shared<triangle_mesh>(
        std::string(OBJ_PATH),
        fallbackM,
        1.0 // scale
    );
    world.add(mesh_ptr);

    // ------------------------------------------------------------
    // 2) ADD A LIGHT ABOVE (emissive sphere)
    // ------------------------------------------------------------
    // Use a high-intensity diffuse light so it actually illuminates with path tracing.
    // (Your GPU pipeline supports MAT_DIFFUSE_LIGHT.)
    auto bright_light_material = std::make_shared<diffuse_light>(color(200.0, 200.0, 200.0)); // strong white light

    auto ceiling_light = std::make_shared<sphere>(
        point3(0, 1000, 100),    // position above the model
        100.0,                // radius
        bright_light_material
    );
    world.add(ceiling_light);
    lights.add(ceiling_light);

    // ------------------------------------------------------------
    // 3) CAMERA SETUP
    // ------------------------------------------------------------
    camera cam;
    cam.image_width        = 800;
    cam.image_height       = 450;
    cam.samples_per_pixel  = 20;  // >1 to see soft lighting
    cam.max_depth          = 8;   // allow multiple bounces

    cam.vfov     = 40;
    cam.lookfrom = point3(0, 0, 100);
    cam.lookat   = point3(0, 1.0, 0);
    cam.vup      = vec3(0, 1, 0);

    cam.aperture   = 0.0;
    cam.focus_dist = (cam.lookfrom - cam.lookat).length();

    cam.initialize(); // sets internal GPUCamera fields based on image size & intrinsics

    // ------------------------------------------------------------
    // 4) BUILD GPU SCENE FROM CPU WORLD
    // ------------------------------------------------------------
    // NOTE: This assumes your build_gpu_scene(world, cam) fills a GPUScene header
    // (device-resident arrays already uploaded internally).
    GPUScene gpu_scene = build_gpu_scene(world, cam);

    // Quick sanity prints
    std::cout << "GPUScene.num_triangles = " << gpu_scene.num_triangles << "\n";
    std::cout << "GPUScene.num_spheres   = " << gpu_scene.num_spheres << "\n";
    std::cout << "GPUScene.num_materials = " << gpu_scene.num_materials << "\n";
    std::cout << "GPUScene.num_textures  = " << gpu_scene.num_textures << "\n";
    std::cout << "GPUScene.texture_pool_floats = " << gpu_scene.texture_pool_floats << "\n";
    std::cout << "Render " << cam.image_width << "x" << cam.image_height
              << " spp=" << cam.samples_per_pixel
              << " depth=" << cam.max_depth << "\n";

    auto c = gpu_scene.camera;
    std::cout << "cam.origin            = (" << c.origin.x()            << ", " << c.origin.y()            << ", " << c.origin.z()            << ")\n";
    std::cout << "cam.lower_left_corner = (" << c.lower_left_corner.x() << ", " << c.lower_left_corner.y() << ", " << c.lower_left_corner.z() << ")\n";
    std::cout << "cam.horizontal        = (" << c.horizontal.x()        << ", " << c.horizontal.y()        << ", " << c.horizontal.z()        << ")\n";
    std::cout << "cam.vertical          = (" << c.vertical.x()          << ", " << c.vertical.y()          << ", " << c.vertical.z()          << ")\n";

    // ------------------------------------------------------------
    // 5) RENDER ON GPU
    // ------------------------------------------------------------
    gpu_render_scene(
        gpu_scene,
        cam.image_width,
        cam.image_height
    );

    // Expect "output.ppm"; convert to PNG if ImageMagick is present
    ppm_to_png("output.ppm", "output.png");

    // ------------------------------------------------------------
    // 6) CLEANUP GPU BUFFERS
    // ------------------------------------------------------------
    free_gpu_scene(gpu_scene);

    std::cout << "Done.\n";
    return 0;
}
