#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <filesystem>

#include "hittable.h"
#include "triangle.h"
#include "aabb.h"
#include "material.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"

class triangle_mesh : public hittable {
public:
    std::vector<triangle> triangles;
    aabb bbox_all;

    triangle_mesh() = default;

    triangle_mesh(const std::vector<triangle>& tris_in)
        : triangles(tris_in)
    {
        build_bbox();
    }

    // NEW: build directly from .obj on disk, with .mtl support
    triangle_mesh(const std::string& obj_path,
                  std::shared_ptr<material> fallback_m,
                  double scale)
    {
        load_obj_from_file(obj_path, fallback_m, scale);
        build_bbox();
    }

    bool hit(const ray& r,
             const interval& ray_t,
             hit_record& rec) const override
    {
        hit_record temp_rec;
        bool hit_anything = false;
        double closest = ray_t.max();

        for (const auto& tri : triangles) {
            if (tri.hit(r, interval(ray_t.min(), closest), temp_rec)) {
                hit_anything = true;
                closest = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    aabb bounding_box() const override {
        return bbox_all;
    }

    double pdf_value(const point3& /*origin*/,
                     const vec3&   /*direction*/) const override
    {
        return 0.0;
    }

    vec3 random(const point3& /*origin*/) const override {
        return vec3(1,0,0);
    }

    const std::vector<triangle>& get_triangles() const {
        return triangles;
    }

private:
    // we’ll fill this from the .mtl file(s)
    std::unordered_map<std::string, std::shared_ptr<material>> mtl_lookup;

    void build_bbox() {
        if (!triangles.empty()) {
            bbox_all = triangles[0].bounding_box();
            for (size_t i = 1; i < triangles.size(); i++) {
                bbox_all = surrounding_box(bbox_all, triangles[i].bounding_box());
            }
        }
    }

    // -----------------------------------------
    // parse MTL: very basic
    //
    // supports:
    //   newmtl <name>
    //   Kd r g b   (diffuse color)
    //   Ks r g b   (specular color)
    //   Ns s       (shininess ~ fuzz/roughness-ish)
    //   d alpha    (opacity; if <1 maybe treat as dielectric/glass)
    //
    // We'll map:
    // - if d < 1.0 -> dielectric with ref_idx=1.5 (lazy glass guess)
    // - else if Ks is strong / Ns high -> metal
    // - else -> lambertian(Kd)
    //
    // This is super crude but enough to get different panels looking different.
    // -----------------------------------------
    void load_mtl_file(const std::string& mtl_path) {
        std::ifstream in(mtl_path);
        if (!in.is_open()) {
            std::cerr << "[triangle_mesh] WARNING: couldn't open MTL: "
                      << mtl_path << "\n";
            return;
        }

        std::string line;
        std::string current_name;
        // temp accumulators for one material
        vec3 Kd(0.8,0.8,0.8);
        vec3 Ks(0.0,0.0,0.0);
        double Ns = 0.0;
        double d  = 1.0;

        auto commit_current = [&]() {
            if (current_name.empty()) return;

            std::shared_ptr<material> mat_ptr;

            bool transparent = (d < 0.999);
            bool metallicish = (Ks.length() > 0.1 && Ns > 10.0);

            if (transparent) {
                // pretend it's glass
                double fake_ior = 1.5;
                mat_ptr = std::make_shared<dielectric>(fake_ior);
            } else if (metallicish) {
                // metal takes Ks as tint
                double fuzz = 0.1; // you can map Ns to fuzz if you want
                mat_ptr = std::make_shared<metal>(Ks, fuzz);
            } else {
                // default: lambertian with Kd
                mat_ptr = std::make_shared<lambertian>(Kd);
            }

            mtl_lookup[current_name] = mat_ptr;
        };

        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream iss(line);
            std::string tag;
            iss >> tag;

            if (tag == "newmtl") {
                // push last one first
                commit_current();

                current_name.clear();
                iss >> current_name;

                // reset defaults for the new block
                Kd = vec3(0.8,0.8,0.8);
                Ks = vec3(0.0,0.0,0.0);
                Ns = 0.0;
                d  = 1.0;
            }
            else if (tag == "Kd") {
                double r,g,b;
                if (iss >> r >> g >> b) {
                    Kd = vec3(r,g,b);
                }
            }
            else if (tag == "Ks") {
                double r,g,b;
                if (iss >> r >> g >> b) {
                    Ks = vec3(r,g,b);
                }
            }
            else if (tag == "Ns") {
                double ns_tmp;
                if (iss >> ns_tmp) {
                    Ns = ns_tmp;
                }
            }
            else if (tag == "d" || tag == "Tr") {
                // transparency. "d" is dissolve, "Tr" sometimes is 1-d.
                double alpha;
                if (iss >> alpha) {
                    d = alpha;
                }
            }
        }

        // commit the final material
        commit_current();
        in.close();
    }

    // helper: safely join directory + relative path
    static std::string join_path(const std::string& dir,
                                 const std::string& rel) {
        namespace fs = std::filesystem;
        fs::path base(dir);
        fs::path child(rel);
        return (base / child).string();
    }

    // -----------------------------------------
    // OBJ loader with material support
    //
    // We now parse:
    // - mtllib file.mtl      -> load_mtl_file(...)
    // - usemtl name          -> set current_material_name
    // - v x y z              -> push verts
    // - f ...                -> make triangles w/ that material
    //
    // We’ll still apply `scale`.
    // -----------------------------------------
    void load_obj_from_file(const std::string& obj_path,
                            const std::shared_ptr<material>& fallback_m,
                            double scale)
    {
        std::ifstream in(obj_path);
        if (!in.is_open()) {
            std::cerr << "[triangle_mesh] ERROR: couldn't open OBJ: "
                      << obj_path << "\n";
            return;
        }

        // figure out parent dir of obj to resolve mtllib
        std::string obj_dir = std::filesystem::path(obj_path).parent_path().string();

        std::vector<vec3> verts;
        std::string current_mtl_name; // name from "usemtl"

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#')
                continue;

            std::istringstream iss(line);
            std::string tag;
            iss >> tag;

            if (tag == "mtllib") {
                // e.g. "mtllib ISS_stationary.mtl"
                std::string mtl_file;
                iss >> mtl_file;
                if (!mtl_file.empty()) {
                    std::string full_mtl_path = join_path(obj_dir, mtl_file);
                    load_mtl_file(full_mtl_path);
                }
            }
            else if (tag == "usemtl") {
                // activate a named material
                std::string name;
                iss >> name;
                current_mtl_name = name;
            }
            else if (tag == "v") {
                double x,y,z;
                if (!(iss >> x >> y >> z)) continue;
                verts.push_back(vec3(
                    x * scale,
                    y * scale,
                    z * scale
                ));
            }
            else if (tag == "f") {
                // We'll support both "f i j k" and "f i/j/k i/j/k i/j/k"
                // parse tokens after "f"
                std::vector<std::string> tokens;
                {
                    std::string tmp;
                    while (iss >> tmp) tokens.push_back(tmp);
                }
                if (tokens.size() < 3) continue;

                // helper to grab only the vertex index from stuff like "12/9/3"
                auto parse_vi = [](const std::string& tok) {
                    size_t slash = tok.find('/');
                    if (slash == std::string::npos) {
                        return std::stoi(tok);
                    }
                    return std::stoi(tok.substr(0, slash));
                };

                // fan triangulation in case of n-gon
                for (size_t t = 1; t + 1 < tokens.size(); t++) {
                    int ia = parse_vi(tokens[0]);
                    int ib = parse_vi(tokens[t]);
                    int ic = parse_vi(tokens[t+1]);

                    add_face_with_mtl(
                        ia, ib, ic,
                        verts,
                        current_mtl_name,
                        fallback_m
                    );
                }
            }
            // you could also parse "vn" (normals) and store them per triangle for shading later
        }

        in.close();
    }

    // Create a triangle with material resolved from mtl_lookup.
    void add_face_with_mtl(int ia, int ib, int ic,
                           const std::vector<vec3>& verts,
                           const std::string& mtl_name,
                           const std::shared_ptr<material>& fallback_m)
    {
        int a = ia - 1;
        int b = ib - 1;
        int c = ic - 1;

        if (a < 0 || b < 0 || c < 0) return;
        if (a >= (int)verts.size())  return;
        if (b >= (int)verts.size())  return;
        if (c >= (int)verts.size())  return;

        std::shared_ptr<material> use_mat = fallback_m;

        auto it = mtl_lookup.find(mtl_name);
        if (it != mtl_lookup.end()) {
            use_mat = it->second;
        }

        triangle tri(
            verts[a],
            verts[b],
            verts[c],
            use_mat
        );

        triangles.push_back(tri);
    }
};
