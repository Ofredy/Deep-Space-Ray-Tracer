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
// Double-precision 3D vector for large world coordinates
// ============================================================
struct dvec3 {
    double x, y, z;

    dvec3() : x(0.0), y(0.0), z(0.0) {}
    dvec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
};

inline dvec3 operator-(const dvec3& a, const dvec3& b) {
    return dvec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline dvec3 operator+(const dvec3& a, const dvec3& b) {
    return dvec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline dvec3 operator*(double s, const dvec3& v) {
    return dvec3(s * v.x, s * v.y, s * v.z);
}

inline double dlength(const dvec3& v) {
    return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

inline dvec3 dnormalize(const dvec3& v) {
    double L = dlength(v);
    if (L == 0.0) return dvec3(0.0, 0.0, 0.0);
    return (1.0 / L) * v;
}

inline vec3 to_float_vec3(const dvec3& v) {
    return vec3((float)v.x, (float)v.y, (float)v.z);
}

// ============================================================
// Pose: camera and model (ISS) pose in WORLD frame
//   - world frame origin is the LIGHT position
//   - cam_pos_world   : camera position in world frame
//   - model_pos_world : model origin in world frame
//   - model_euler_deg : yaw, pitch, roll in DEGREES (currently yaw used)
// ============================================================
struct PoseEntry {
    dvec3 cam_pos_world_d;    // camera position in WORLD frame [m] (double)
    dvec3 model_pos_world_d;  // model position in WORLD frame [m] (double)
    vec3  model_euler_deg;    // (yaw, pitch, roll) in degrees
};

// ------------------------------------------------------------
// Simple yaw rotation around +Y in degrees (float version)
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

// Double-precision version of yaw rotation for large coordinates
static dvec3 rotate_yaw_deg_d(const dvec3& v, double yaw_deg) {
    double rad = degrees_to_radians(yaw_deg);
    double c = std::cos(rad);
    double s = std::sin(rad);

    return dvec3(
        c * v.x + s * v.z,
        v.y,
        -s * v.x + c * v.z
    );
}

// ------------------------------------------------------------
// Read poses from a text file.
// Format per non-comment line:
//
// cam_x cam_y cam_z   model_x model_y model_z   yaw pitch roll
//
// All in the WORLD frame whose origin is the LIGHT.
// Positions are assumed to be in METERS (double precision).
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
        p.cam_pos_world_d   = dvec3(cx, cy, cz);
        p.model_pos_world_d = dvec3(mx, my, mz);
        p.model_euler_deg   = vec3((float)yaw, (float)pitch, (float)roll);

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
                       std::string& out_dir,
                       bool& do_upscale)
{
    txt_path = "";
    out_dir  = "output";
    do_upscale = false;   // default OFF

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--input_txt" && i + 1 < argc) {
            txt_path = argv[++i];
        }
        else if (a == "--output_dir" && i + 1 < argc) {
            out_dir = argv[++i];
        }
        else if (a == "--upscale") {
            do_upscale = true;
        }
    }
}

int main(int argc, char** argv) {
    using namespace std::chrono;

    std::string pose_file;
    std::string output_dir;
    bool do_upscale;
    parse_args(argc, argv, pose_file, output_dir, do_upscale);

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

    auto t2 = high_resolution_clock::now();
    std::cout << "Base ISS + material load time: "
              << duration_cast<milliseconds>(t2 - t1).count() << " ms\n";

    // ------------------------------------------------------------
    // Camera base configuration (overridden per pose)
    // ------------------------------------------------------------
    camera cam;
    cam.image_width        = 800;
    cam.image_height       = 450;
    cam.samples_per_pixel  = 1000;   // you can bump this back up
    cam.max_depth          = 50;   // same here
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
        p.cam_pos_world_d   = dvec3(0.0, 50.0, 200.0);
        p.model_pos_world_d = dvec3(0.0, -100.0, 0.0);
        p.model_euler_deg   = vec3(0, 0, 0);
        poses.push_back(p);
    } else {
        std::cout << "Loaded " << poses.size() << " poses.\n";
    }

    // ------------------------------------------------------------
    // World frame:
    //   - origin is the LIGHT position (Sun at origin)
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
    // We implement R_world_model^T by rotate_yaw_deg_d(., -yaw_deg).
    // ------------------------------------------------------------

    // The light in the WORLD frame (Sun at origin):
    dvec3 light_pos_world_d(0.0, 0.0, 0.0); // by definition of the world frame

    for (size_t i = 0; i < poses.size(); ++i) {
        const PoseEntry& p = poses[i];

        double yaw_deg   = p.model_euler_deg.x();
        // pitch, roll are parsed but not yet used:
        // double pitch_deg = p.model_euler_deg.y();
        // double roll_deg  = p.model_euler_deg.z();

        // Frame header printed below
        std::cout << "\n=== Frame " << i << " ===\n";
        std::cout << "Camera world: (" << p.cam_pos_world_d.x << ", "
                                       << p.cam_pos_world_d.y << ", "
                                       << p.cam_pos_world_d.z << ")\n";
        std::cout << "Model world : (" << p.model_pos_world_d.x << ", "
                                       << p.model_pos_world_d.y << ", "
                                       << p.model_pos_world_d.z << ")\n";
        std::cout << "Model yaw/pitch/roll (deg): ("
                  << p.model_euler_deg.x() << ", "
                  << p.model_euler_deg.y() << ", "
                  << p.model_euler_deg.z() << ")\n";

        // --------------------------------------------------------
        // camera and light relative to model in WORLD (double precision)
        // --------------------------------------------------------
        dvec3 cam_w   = p.cam_pos_world_d;
        dvec3 model_w = p.model_pos_world_d;
        dvec3 cam_rel_world_d   = cam_w   - model_w;
        dvec3 light_rel_world_d = light_pos_world_d - model_w;

        // Separation in meters (still in world frame)
        double sep_m = dlength(cam_rel_world_d);
        std::cout << "  sep(cam, model) = " << sep_m << " m\n";
        if (sep_m < 1.0) {
            std::cout << "  [!] Camera is inside/too close to ISS mesh. Skipping frame.\n";
            continue;
        }

        // Apply R_world_model^T ≈ yaw about +Y with negative angle (double)
        dvec3 cam_in_model_d   = rotate_yaw_deg_d(cam_rel_world_d,   -yaw_deg);
        dvec3 light_in_model_d = rotate_yaw_deg_d(light_rel_world_d, -yaw_deg);

        // Convert to float for GPU / camera
        vec3 cam_in_model   = to_float_vec3(cam_in_model_d);
        vec3 light_in_model = to_float_vec3(light_in_model_d);

        // Directional Sun light: direction from ISS (model origin) to Sun
        dvec3 sun_dir_model_d = dnormalize(light_in_model_d);
        vec3  sun_dir_model   = to_float_vec3(sun_dir_model_d);

        // Optional debug
        std::cout << "  sun_dir_model   = ("
                  << sun_dir_model.x() << ", "
                  << sun_dir_model.y() << ", "
                  << sun_dir_model.z() << ")\n";

        if (i < 3) {
            std::cout << "  cam_rel_world_d = ("
                      << cam_rel_world_d.x << ", "
                      << cam_rel_world_d.y << ", "
                      << cam_rel_world_d.z << ")\n";

            std::cout << "  cam_in_model_d  = ("
                      << cam_in_model_d.x << ", "
                      << cam_in_model_d.y << ", "
                      << cam_in_model_d.z << ")\n";

            std::cout << "  light_rel_world_d = ("
                      << light_rel_world_d.x << ", "
                      << light_rel_world_d.y << ", "
                      << light_rel_world_d.z << ")\n";

            std::cout << "  light_in_model_d  = ("
                      << light_in_model_d.x << ", "
                      << light_in_model_d.y << ", "
                      << light_in_model_d.z << ")\n";
        }

        // --------------------------------------------------------
        // Build per-frame world in MODEL frame
        //   - ISS mesh at origin
        // --------------------------------------------------------
        hittable_list frame_world;
        frame_world.add(iss_base_mesh);

        // NOTE: Sun is now treated as a directional light, not a geometry sphere.

        // --------------------------------------------------------
        // Camera in MODEL frame: look at ISS origin
        // --------------------------------------------------------
        point_camera_at(cam, cam_in_model, vec3(0, 0, 0));

        // --------------------------------------------------------
        // Build GPU scene & render
        // --------------------------------------------------------
        auto build_start = high_resolution_clock::now();
        GPUScene gpu_scene = build_gpu_scene(frame_world, cam, sun_dir_model);
        auto build_end = high_resolution_clock::now();

        std::cout << "GPU scene build time: "
                  << duration_cast<milliseconds>(build_end - build_start).count()
                  << " ms\n";

        // NOTE: enable this when you want the GPU render:
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
              << duration_cast<std::chrono::seconds>(total_end - total_start).count()
              << " s\n";

    if (do_upscale) {
        std::string upsample_cmd =
            "powershell -Command \"& 'C:/Users/Fredy Orellana/.conda/envs/rt_env/python.exe' "
            "'C:/Users/Fredy Orellana/Desktop/gpu programming/Ray-Tracer/scripts/upsample.py' "
            "--in '" + output_dir + "' "
            "--out '" + output_dir + "_upscaled' "
            "--scale 4\"";
        
        std::cout << "Running upsample command:\n" << upsample_cmd << "\n";
        std::system(upsample_cmd.c_str());
    }
    else {
        std::cout << "Upscaling disabled (use --upscale to enable).\n";
    }

    std::cout << "Done.\n";
    return 0;
}
