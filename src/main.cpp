#include "../inc/rtweekend.h"
#include "../inc/camera.h"
#include "../inc/hittable_list.h"
#include "../inc/material.h"
#include "../inc/sphere.h"
#include "../inc/bvh.h"
#include "../inc/triangle_mesh.h"

#include <iostream>
#include <chrono>
#include <fstream>
#include <cstdio>
#include <cstdlib>

// Optional helper to convert .ppm to .png using ImageMagick
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
    // ---- choose your OBJ path here ----
    const char* OBJ_PATH = "../../iss_model/ISS_stationary.obj";
    const char* OBJ_DIR  = "../../iss_model";

    // ---- world + lights containers ----
    hittable_list world;
    hittable_list lights;

    // ---- materials ----
    auto fallbackM = make_shared<lambertian>(color(.73, .73, .73));
    auto lightM    = make_shared<diffuse_light>(color(15, 15, 15));

    // ---- load OBJ as a triangle_mesh with per-material support ----
    FILE* f = std::fopen(OBJ_PATH, "rb");
    if (!f) {
        std::cerr << "[error] could not open OBJ: " << OBJ_PATH << "\n";
        return 1;
    }

    auto mesh = make_shared<triangle_mesh>(
        f,
        OBJ_DIR,       // folder containing OBJ, MTL, and textures
        fallbackM,     // fallback material
        1.0            // scale
    );
    std::fclose(f);

    world.add(mesh);

    // ---- bright spherical light above the object ----
    auto bright_light_material = make_shared<diffuse_light>(color(200, 200, 200));
    auto ceiling_light = make_shared<sphere>(
        point3(0, 500, 2),   // position of the light
        100.0,               // radius of the light
        bright_light_material
    );

    world.add(ceiling_light);
    lights.add(ceiling_light);

    // ---- accelerate the scene ----
    world = hittable_list(make_shared<bvh_node>(world));

    // ---- camera setup (same as before) ----
    camera cam;
    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 800;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0,0,0);

    cam.vfov     = 40;
    cam.lookfrom = point3(0, 0, 100);   // your original camera position
    cam.lookat   = point3(0, 1.0, 0);   // look toward model center
    cam.vup      = vec3(0, 1, 0);

    cam.defocus_angle = 0.0;  // no depth of field blur

    // ---- render ----
    cam.render(world, lights);

    // ---- optional PNG conversion ----
    if (ppm_to_png("image.ppm", "image.png") != 0) {
        std::cerr << "[warn] ImageMagick conversion failed. Is `magick` in PATH?\n";
    }

    return 0;
}
