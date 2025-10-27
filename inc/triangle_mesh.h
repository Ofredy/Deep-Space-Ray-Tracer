#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>

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
    // ------------------------------------------------------------
    // default ctor
    // ------------------------------------------------------------
    triangle_mesh() = default;

    // ------------------------------------------------------------
    // build from an already prepared list of triangles
    // (this is what you already had)
    // ------------------------------------------------------------
    triangle_mesh(const std::vector<triangle>& tris_in)
        : triangles(tris_in)
    {
        // build bbox_all = surrounding box of all triangles' bbox
        if (!triangles.empty()) {
            bbox_all = triangles[0].bounding_box();
            for (size_t i = 1; i < triangles.size(); i++) {
                bbox_all = surrounding_box(bbox_all, triangles[i].bounding_box());
            }
        }
    }

    // ------------------------------------------------------------
    // NEW: build directly from an .obj on disk
    //
    //  obj_path   .obj file path
    //  fallback_m material to use for all faces if no per-face material
    //  scale      uniform scale to apply to all verts
    //
    // lets you do:
    //     auto mesh = std::make_shared<triangle_mesh>("model.obj", someMat, 1.0);
    // ------------------------------------------------------------
    triangle_mesh(const std::string& obj_path,
                  std::shared_ptr<material> fallback_m,
                  double scale)
    {
        load_obj_from_file(obj_path, fallback_m, scale);

        // compute bbox_all after loading
        if (!triangles.empty()) {
            bbox_all = triangles[0].bounding_box();
            for (size_t i = 1; i < triangles.size(); i++) {
                bbox_all = surrounding_box(bbox_all, triangles[i].bounding_box());
            }
        }
    }

    // ------------------------------------------------------------
    // hit(): required by hittable
    //
    // IMPORTANT:
    // must match EXACT signature of hittable::hit:
    //   virtual bool hit(const ray &, const interval &, hit_record &) const = 0;
    // ------------------------------------------------------------
    bool hit(const ray& r,
             const interval& ray_t,
             hit_record& rec) const override
    {
        hit_record temp_rec;
        bool hit_anything = false;
        double closest = ray_t.max();

        // brute force triangle hit
        for (const auto& tri : triangles) {
            if (tri.hit(r, interval(ray_t.min(), closest), temp_rec)) {
                hit_anything = true;
                closest = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    // ------------------------------------------------------------
    // bounding_box(): required by hittable
    // ------------------------------------------------------------
    aabb bounding_box() const override {
        return bbox_all;
    }

    // ------------------------------------------------------------
    // pdf_value / random(): required by hittable in your project
    // we don't need fancy light sampling yet, so stub them.
    // ------------------------------------------------------------
    double pdf_value(const point3& /*origin*/,
                     const vec3&   /*direction*/) const override
    {
        return 0.0;
    }

    vec3 random(const point3& /*origin*/) const override {
        return vec3(1,0,0);
    }

    // ------------------------------------------------------------
    // public data access (gpu_scene_builder uses this)
    // ------------------------------------------------------------
    const std::vector<triangle>& get_triangles() const {
        return triangles;
    }

private:
    // ------------------------------------------------------------
    // Tiny OBJ loader: loads positions and faces,
    // builds per-face triangle objects using fallback_m.
    //
    // Supports:
    //   v x y z
    //   f i j k
    // or
    //   f i/j/k i/j/k i/j/k
    //
    // No normals/UVs/material libraries yet.
    // ------------------------------------------------------------
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

        std::vector<vec3> verts;

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') continue;

            std::istringstream iss(line);
            std::string tag;
            iss >> tag;

            // vertex
            if (tag == "v") {
                double x,y,z;
                if (!(iss >> x >> y >> z)) continue;
                verts.push_back(vec3(
                    x * scale,
                    y * scale,
                    z * scale
                ));
            }
            // face
            else if (tag == "f") {
                // try simple "f a b c"
                int i0,i1,i2;
                if (iss >> i0 >> i1 >> i2) {
                    add_face(i0, i1, i2, verts, fallback_m);
                } else {
                    // reset stream and try "f a/b/c d/e/f g/h/i"
                    std::istringstream iss2(line.substr(2)); // after 'f '
                    std::string t0,t1,t2;
                    if (!(iss2 >> t0 >> t1 >> t2)) continue;

                    auto parse_index = [](const std::string& token) {
                        // token "12/9/3", "12//7", or "12"
                        size_t slash = token.find('/');
                        if (slash == std::string::npos)
                            return std::stoi(token);
                        return std::stoi(token.substr(0, slash));
                    };

                    int j0 = parse_index(t0);
                    int j1 = parse_index(t1);
                    int j2 = parse_index(t2);

                    add_face(j0, j1, j2, verts, fallback_m);
                }
            }
        }

        in.close();
    }

    // helper: turn 1-based OBJ face indices into a triangle and push_back
    void add_face(int ia, int ib, int ic,
                  const std::vector<vec3>& verts,
                  const std::shared_ptr<material>& fallback_m)
    {
        // OBJ indices are 1-based
        int a = ia - 1;
        int b = ib - 1;
        int c = ic - 1;

        if (a < 0 || b < 0 || c < 0) return;
        if (a >= (int)verts.size())  return;
        if (b >= (int)verts.size())  return;
        if (c >= (int)verts.size())  return;

        triangle tri(
            verts[a],
            verts[b],
            verts[c],
            fallback_m
        );

        triangles.push_back(tri);
    }
};
