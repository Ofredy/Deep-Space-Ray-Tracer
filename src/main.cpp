#include <iostream>
#include <cstdio>
#include <memory>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>

#include "rtweekend.h"
#include "camera.h"
#include "hittable_list.h"
#include "triangle_mesh.h"
#include "bvh.h"
#include "material.h"
#include "sphere.h"

#include "gpu_scene_builder.h"
#include "gpu_scene.h"

// CUDA entry point
extern "C"
void gpu_render_scene(const GPUScene& scene, int width, int height);

static inline int ppm_to_png(const std::string& ppm, const std::string& png) {
    std::string cmd = "magick \"" + ppm + "\" \"" + png + "\"";
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::string cmd2 = "magick convert \"" + ppm + "\" \"" + png + "\"";
        rc = std::system(cmd2.c_str());
    }
    return rc;
}

struct PoseEntry {
    vec3 cam_pos;
    vec3 iss_pos;
    vec3 iss_euler;
};

static bool read_pose_file(const std::string& filename,
                           std::vector<PoseEntry>& poses)
{
    std::ifstream in(filename);
    if (!in) return false;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        double cx, cy, cz;
        double ix, iy, iz;
        double yaw, pitch, roll;

        if (!(iss >> cx >> cy >> cz >> ix >> iy >> iz >> yaw >> pitch >> roll))
            continue;

        PoseEntry p;
        p.cam_pos = vec3(cx, cy, cz);
        p.iss_pos = vec3(ix, iy, iz);
        p.iss_euler = vec3(yaw, pitch, roll);

        poses.push_back(p);
    }
    return !poses.empty();
}

static void point_camera_at(camera& cam,
                            const vec3& cam_pos,
                            const vec3& target_pos)
{
    cam.lookfrom = cam_pos;
    cam.lookat   = target_pos;
    cam.vup      = vec3(0, 1, 0);
    cam.focus_dist = (cam.lookfrom - cam.lookat).length();
    cam.initialize();
}

// ------------------------------------------------------------
// parse CLI args simple
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
            txt_path = argv[++i];
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

    std::filesystem::create_directory(output_dir);

    std::cout << "Using input_txt:   " << pose_file << "\n";
    std::cout << "Using output_dir:  " << output_dir << "\n";

    auto total_start = high_resolution_clock::now();

    // ------------------------------------------------------------
    // Load scene
    // ------------------------------------------------------------
    auto t1 = high_resolution_clock::now();

    const char* OBJ_PATH = "../../iss_model/ISS_stationary.obj";

    hittable_list world;
    hittable_list lights;

    auto fallbackM = std::make_shared<lambertian>(vec3(0.73, 0.73, 0.73));
    auto mesh_ptr = std::make_shared<triangle_mesh>(
        std::string(OBJ_PATH), fallbackM, 1.0);
    world.add(mesh_ptr);

    auto bright_light = std::make_shared<diffuse_light>(color(200,200,200));
    auto ceiling_light = std::make_shared<sphere>(
        point3(0, 500, 2), 100.0, bright_light);
    world.add(ceiling_light);

    auto t2 = high_resolution_clock::now();
    std::cout << "Scene load: " <<
        duration_cast<milliseconds>(t2 - t1).count() << " ms\n";

    // ------------------------------------------------------------
    // Camera
    // ------------------------------------------------------------
    camera cam;
    cam.image_width        = 800;
    cam.image_height       = 450;
    cam.samples_per_pixel  = 1000;
    cam.max_depth          = 50;
    cam.vfov               = 40;

    cam.lookfrom = point3(0,0,100);
    cam.lookat   = point3(0,1,0);
    cam.vup      = vec3(0,1,0);
    cam.focus_dist = (cam.lookfrom - cam.lookat).length();
    cam.initialize();

    // ------------------------------------------------------------
    // Poses
    // ------------------------------------------------------------
    std::vector<PoseEntry> poses;
    bool have_poses = (!pose_file.empty() && read_pose_file(pose_file, poses));

    // ------------------------------------------------------------
    // Rendering loop
    // ------------------------------------------------------------
    if (!have_poses) {
        std::cout << "No pose file → rendering single frame.\n";

        GPUScene gpu_scene = build_gpu_scene(world, cam);
        gpu_render_scene(gpu_scene, cam.image_width, cam.image_height);

        std::string ppm = output_dir + "/frame_0000.ppm";
        std::string png = output_dir + "/frame_0000.png";
        std::rename("image_gpu.ppm", ppm.c_str());
        ppm_to_png(ppm, png);

        free_gpu_scene(gpu_scene);
    }
    else {
        std::cout << "Rendering " << poses.size() << " frames...\n";

        for (size_t i = 0; i < poses.size(); i++) {
            const PoseEntry& p = poses[i];

            point_camera_at(cam, p.cam_pos, p.iss_pos);

            GPUScene gpu_scene = build_gpu_scene(world, cam);
            gpu_render_scene(gpu_scene, cam.image_width, cam.image_height);

            char name[64];
            sprintf(name, "frame_%04zu.ppm", i);
            std::string ppm = output_dir + "/" + name;

            sprintf(name, "frame_%04zu.png", i);
            std::string png = output_dir + "/" + name;

            std::rename("image_gpu.ppm", ppm.c_str());
            ppm_to_png(ppm, png);

            free_gpu_scene(gpu_scene);

            std::cout << "Rendered frame " << i << "\n";
        }
    }

    auto total_end = high_resolution_clock::now();
    std::cout << "Total time: " <<
        duration_cast<seconds>(total_end - total_start).count() << " s\n";

    return 0;
}
