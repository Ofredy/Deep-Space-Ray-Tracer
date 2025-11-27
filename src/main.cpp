#include <iostream>
#include <cstdio>
#include <memory>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>

#include "rtweekend.h"
#include "camera.h"
#include "hittable_list.h"
#include "triangle_mesh.h"
#include "bvh.h"
#include "material.h"
#include "sphere.h"

#include "gpu_scene_builder.h"
#include "gpu_scene.h"

using std::make_shared;

// CUDA entry point (implemented in gpu_render.cu)
extern "C"
void gpu_render_scene(const GPUScene& scene, int width, int height);

// PPM -> PNG via ImageMagick
static inline int ppm_to_png(const std::string& ppm, const std::string& png) {
    std::string cmd = "magick \"" + ppm + "\" \"" + png + "\"";
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::string cmd2 = "magick convert \"" + ppm + "\" \"" + png + "\"";
        rc = std::system(cmd2.c_str());
    }
    return rc;
}

namespace fs = std::filesystem;

// Ensure directory exists and is empty
static void prepare_output_dir(const std::string& dir) {
    if (fs::exists(dir)) {
        // Remove everything inside BUT keep the directory itself
        for (auto& entry : fs::directory_iterator(dir)) {
            fs::remove_all(entry.path());
        }
    } else {
        fs::create_directories(dir);
    }
}

// ------------------------------------------------------------
// CLI argument parsing for:
//   --output_dir <folder>
// (we ignore --input_txt now, but keep parsing it so your scripts don't break)
// ------------------------------------------------------------
static void parse_args(int argc, char** argv,
                       std::string& txt_path,
                       std::string& out_dir)
{
    txt_path = "";
    out_dir  = "output";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--input_txt" && i + 1 < argc) {
            txt_path = argv[++i];   // ignored for this simple-sphere debug
        }
        else if (a == "--output_dir" && i + 1 < argc) {
            out_dir = argv[++i];
        }
    }
}

int main(int argc, char** argv) {
    using namespace std::chrono;

    std::string pose_file;
    std::string output_dir;
    parse_args(argc, argv, pose_file, output_dir);

    // Clean existing output_dir or create if it doesn't exist
    prepare_output_dir(output_dir);

    std::cout << "Using (ignored) input_txt : " << (pose_file.empty() ? "(none)" : pose_file) << "\n";
    std::cout << "Using output_dir: " << output_dir << "\n";

    auto total_start = high_resolution_clock::now();

    // ------------------------------------------------------------
    // Build SIMPLE SPHERE SCENE (CPU side) — matches the CPU test main
    // ------------------------------------------------------------
    hittable_list world;
    hittable_list lights;

    // Materials
    auto ground_mat = make_shared<lambertian>(color(0.8, 0.8, 0.8));
    auto center_mat = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    auto left_mat   = make_shared<dielectric>(1.5);
    auto right_mat  = make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);

    auto light_mat  = make_shared<diffuse_light>(color(20.0, 20.0, 20.0));

    // Ground sphere
    world.add(make_shared<sphere>(
        point3(0.0, -1000.0, 0.0),
        1000.0,
        ground_mat
    ));

    // Three test spheres
    world.add(make_shared<sphere>(
        point3(-2.0, 1.0, 0.0),
        1.0,
        left_mat          // dielectric
    ));

    world.add(make_shared<sphere>(
        point3(0.0, 1.0, 0.0),
        1.0,
        center_mat        // lambertian
    ));

    world.add(make_shared<sphere>(
        point3(2.0, 1.0, 0.0),
        1.0,
        right_mat         // metal
    ));

    // Emissive light sphere above
    auto ceiling_light = make_shared<sphere>(
        point3(0.0, 8.0, 0.0),   // overhead
        2.0,                     // radius
        light_mat
    );
    world.add(ceiling_light);
    lights.add(ceiling_light);

    // (Optional) Wrap in BVH if your GPU pipeline assumes a BVH node at root
    //world = hittable_list(make_shared<bvh_node>(world));

    // ------------------------------------------------------------
    // Camera (match CPU simple-sphere settings)
    // ------------------------------------------------------------
    camera cam;
    cam.image_width        = 800;
    cam.image_height       = 450;         // 16:9
    cam.samples_per_pixel  = 200;         // match CPU test
    cam.max_depth          = 50;
    cam.vfov               = 40.0;
    cam.aperture           = 0.0;

    cam.lookfrom = point3(0.0, 2.0, 10.0);
    cam.lookat   = point3(0.0, 1.0, 0.0);
    cam.vup      = vec3(0, 1, 0);

    cam.focus_dist = (cam.lookfrom - cam.lookat).length();
    cam.initialize();

    // ------------------------------------------------------------
    // Sun direction — not really needed here since the emissive
    // sphere is the main light, but builder may still expect it.
    // Point it downwards (like an overhead directional fill light).
    // ------------------------------------------------------------
    vec3 sun_dir_model = unit_vector(vec3(0.0f, -1.0f, 0.0f));

    // ------------------------------------------------------------
    // Build GPU scene & render ONE frame
    // ------------------------------------------------------------
    auto build_start = high_resolution_clock::now();
    GPUScene gpu_scene = build_gpu_scene(world, cam, sun_dir_model);
    gpu_scene.params.samples_per_pixel = cam.samples_per_pixel;
    gpu_scene.params.max_depth         = cam.max_depth;
    gpu_scene.params.gamma             = 2.2f;  // or whatever your CPU path uses
    gpu_scene.params.exposure          = 1.0f;  // start with 1.0 to match CPU
    
    auto build_end = high_resolution_clock::now();

    std::cout << "GPU scene build time: "
              << duration_cast<milliseconds>(build_end - build_start).count()
              << " ms\n";

    gpu_render_scene(gpu_scene, cam.image_width, cam.image_height);

    // ------------------------------------------------------------
    // Save frame image as frame_0000.{ppm,png}
    // ------------------------------------------------------------
    char name_buf[64];
    std::snprintf(name_buf, sizeof(name_buf), "frame_%04d.ppm", 0);
    std::string ppm = output_dir + "/" + std::string(name_buf);

    std::snprintf(name_buf, sizeof(name_buf), "frame_%04d.png", 0);
    std::string png = output_dir + "/" + std::string(name_buf);

    std::rename("image_gpu.ppm", ppm.c_str());
    ppm_to_png(ppm, png);

    free_gpu_scene(gpu_scene);

    std::cout << "Saved " << png << "\n";

    auto total_end = high_resolution_clock::now();
    std::cout << "\nTotal runtime: "
              << duration_cast<std::chrono::seconds>(total_end - total_start).count()
              << " s\n";

    std::cout << "Done.\n";
    return 0;
}
