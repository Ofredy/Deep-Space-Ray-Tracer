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

// ============================================================
// Pose: camera and model (ISS) pose in WORLD frame
//   - world frame origin is the LIGHT position
//   - cam_pos_world   : camera position in world frame
//   - model_pos_world : model origin in world frame
//   - model_euler_deg : yaw, pitch, roll in DEGREES (currently yaw used)
// ============================================================
struct PoseEntry {
    vec3 cam_pos_world;
    vec3 model_pos_world;
    vec3 model_euler_deg; // (yaw, pitch, roll)
};

// ------------------------------------------------------------
// Simple yaw rotation around +Y in degrees
// Right-handed: positive yaw rotates +Z toward +X
// ------------------------------------------------------------
static vec3 rotate_yaw_deg(const vec3& v, double yaw_deg) {
    double rad = degrees_to_radians(yaw_deg);
    double c = std::cos(rad);
    double s = std::sin(rad);

    return vec3(
        c * v.x() + s * v.z(),
        v.y(),
        -s * v.x() + c * v.z()
    );
}

// ------------------------------------------------------------
// Read poses from a text file.
// Format per non-comment line:
//
// cam_x cam_y cam_z   model_x model_y model_z   yaw pitch roll
//
// All in the WORLD frame whose origin is the LIGHT.
// ------------------------------------------------------------
static bool read_pose_file(const std::string& filename,
                           std::vector<PoseEntry>& poses)
{
    std::ifstream in(filename);
    if (!in) {
        return false;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::istringstream iss(line);
        double cx, cy, cz;
        double mx, my, mz;
        double yaw, pitch, roll;

        if (!(iss >> cx >> cy >> cz
                  >> mx >> my >> mz
                  >> yaw >> pitch >> roll)) {
            // malformed line, skip
            continue;
        }

        PoseEntry p;
        p.cam_pos_world   = vec3(cx, cy, cz);
        p.model_pos_world = vec3(mx, my, mz);
        p.model_euler_deg = vec3(yaw, pitch, roll);

        poses.push_back(p);
    }

    return !poses.empty();
}

// ------------------------------------------------------------
// Aim camera from cam_pos to target_pos, simple pinhole
// ------------------------------------------------------------
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
// CLI argument parsing for:
//   --input_txt <file>
//   --output_dir <folder>
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

    // Clean existing output_dir or create if it doesn't exist
    prepare_output_dir(output_dir);

    std::cout << "Using input_txt : " << (pose_file.empty() ? "(none)" : pose_file) << "\n";
    std::cout << "Using output_dir: " << output_dir << "\n";

    auto total_start = high_resolution_clock::now();

    // ------------------------------------------------------------
    // Load base ISS mesh ONCE in its OWN (model) frame at origin
    // ------------------------------------------------------------
    auto t1 = high_resolution_clock::now();

    const char* OBJ_PATH = "../../iss_model/ISS_stationary.obj";

    auto fallbackM = std::make_shared<lambertian>(vec3(0.73, 0.73, 0.73));
    auto iss_base_mesh = std::make_shared<triangle_mesh>(
        std::string(OBJ_PATH),
        fallbackM,
        1.0
    );

    // Light material (we'll place the light per-frame)
    auto bright_light_material = std::make_shared<diffuse_light>(color(200.0, 200.0, 200.0));
    double light_radius = 100.0;

    auto t2 = high_resolution_clock::now();
    std::cout << "Base ISS + material load time: "
              << duration_cast<milliseconds>(t2 - t1).count() << " ms\n";

    // ------------------------------------------------------------
    // Camera base configuration (overridden per pose)
    // ------------------------------------------------------------
    camera cam;
    cam.image_width        = 800;
    cam.image_height       = 450;
    cam.samples_per_pixel  = 1000;
    cam.max_depth          = 50;
    cam.vfov               = 40;
    cam.aperture           = 0.0;

    // Some default (unused once we have poses, but needed for initialize)
    cam.lookfrom = point3(0, 0, 100);
    cam.lookat   = point3(0, 0, 0);
    cam.vup      = vec3(0, 1, 0);
    cam.focus_dist = (cam.lookfrom - cam.lookat).length();
    cam.initialize();

    // ------------------------------------------------------------
    // Load poses or create a single default pose
    // ------------------------------------------------------------
    std::vector<PoseEntry> poses;
    bool have_poses = (!pose_file.empty() && read_pose_file(pose_file, poses));

    if (!have_poses) {
        std::cout << "No valid pose file found; using single default pose.\n";

        PoseEntry p;
        // world frame origin is light, so put light at (0,0,0),
        // model somewhere "below", camera somewhere above.
        p.cam_pos_world   = vec3(0, 50, 200);
        p.model_pos_world = vec3(0, -100, 0);
        p.model_euler_deg = vec3(0, 0, 0);
        poses.push_back(p);
    } else {
        std::cout << "Loaded " << poses.size() << " poses.\n";
    }

    // ------------------------------------------------------------
    // World frame:
    //   - origin is the LIGHT position
    //   - we treat the light as fixed at (0,0,0) in the WORLD frame
    // Model frame (ISS frame):
    //   - origin at ISS center (mesh at origin)
    //   - we build the GPU scene in this frame
    //
    // Given:
    //   model pose in WORLD: R_world_model (from model->world) and p_world_model
    //   point x in WORLD
    //
    //   x_model = R_world_model^T * (x_world - p_world_model)
    //
    // Here R_world_model is approximated using yaw about +Y.
    // We implement R_world_model^T by rotate_yaw_deg(., -yaw_deg).
    // ------------------------------------------------------------

    // The light in the WORLD frame:
    vec3 light_pos_world(0.0, 0.0, 0.0); // by definition of the world frame

    for (size_t i = 0; i < poses.size(); ++i) {
        const PoseEntry& p = poses[i];

        double yaw_deg   = p.model_euler_deg.x();
        // pitch, roll are parsed but not yet used:
        // double pitch_deg = p.model_euler_deg.y();
        // double roll_deg  = p.model_euler_deg.z();

        std::cout << "\n=== Frame " << i << " ===\n";
        std::cout << "Camera world: (" << p.cam_pos_world.x()   << ", "
                                       << p.cam_pos_world.y()   << ", "
                                       << p.cam_pos_world.z()   << ")\n";
        std::cout << "Model world : (" << p.model_pos_world.x() << ", "
                                       << p.model_pos_world.y() << ", "
                                       << p.model_pos_world.z() << ")\n";
        std::cout << "Model yaw/pitch/roll (deg): ("
                  << p.model_euler_deg.x() << ", "
                  << p.model_euler_deg.y() << ", "
                  << p.model_euler_deg.z() << ")\n";

        // --------------------------------------------------------
        // Transform camera & light from WORLD frame to MODEL frame
        // --------------------------------------------------------

        // camera relative to model in WORLD
        vec3 cam_rel_world   = p.cam_pos_world   - p.model_pos_world;
        // light relative to model in WORLD
        vec3 light_rel_world = light_pos_world   - p.model_pos_world;

        // Apply R_world_model^T ≈ yaw about +Y with negative angle
        vec3 cam_in_model   = rotate_yaw_deg(cam_rel_world,   -yaw_deg);
        vec3 light_in_model = rotate_yaw_deg(light_rel_world, -yaw_deg);

        // --------------------------------------------------------
        // Build per-frame world in MODEL frame
        //   - ISS mesh at origin
        //   - Light sphere at light_in_model
        // --------------------------------------------------------
        hittable_list frame_world;
        frame_world.add(iss_base_mesh);

        auto frame_light = std::make_shared<sphere>(
            point3(light_in_model.x(), light_in_model.y(), light_in_model.z()),
            light_radius,
            bright_light_material
        );
        frame_world.add(frame_light);

        // --------------------------------------------------------
        // Camera in MODEL frame: look at ISS origin
        // --------------------------------------------------------
        point_camera_at(cam, cam_in_model, vec3(0, 0, 0));

        // --------------------------------------------------------
        // Build GPU scene & render
        // --------------------------------------------------------
        auto build_start = high_resolution_clock::now();
        GPUScene gpu_scene = build_gpu_scene(frame_world, cam);
        auto build_end = high_resolution_clock::now();

        std::cout << "GPU scene build time: "
                  << duration_cast<milliseconds>(build_end - build_start).count()
                  << " ms\n";

        gpu_render_scene(gpu_scene, cam.image_width, cam.image_height);

        // --------------------------------------------------------
        // Save frame image
        // --------------------------------------------------------
        char name_buf[64];
        std::snprintf(name_buf, sizeof(name_buf), "frame_%04zu.ppm", i);
        std::string ppm = output_dir + "/" + std::string(name_buf);

        std::snprintf(name_buf, sizeof(name_buf), "frame_%04zu.png", i);
        std::string png = output_dir + "/" + std::string(name_buf);

        std::rename("image_gpu.ppm", ppm.c_str());
        ppm_to_png(ppm, png);

        free_gpu_scene(gpu_scene);

        std::cout << "Saved " << png << "\n";
    }

    auto total_end = high_resolution_clock::now();
    std::cout << "\nTotal runtime: "
              << duration_cast<seconds>(total_end - total_start).count()
              << " s\n";

    std::cout << "Done.\n";
    return 0;
}
