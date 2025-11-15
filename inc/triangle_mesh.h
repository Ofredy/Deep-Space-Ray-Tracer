#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cstdio>

#include "triangle.h"
#include "material.h"
#include "texture.h"

class triangle_mesh : public hittable {
public:
    std::vector<vec3>   verts;
    std::vector<vec3>   uvs;          // store (u,v,0)
    std::vector<triangle> triangles;
    std::vector<std::string> tri_map_Kd;
    std::shared_ptr<material> fallback;

    explicit triangle_mesh(const std::string& obj_path,
                           std::shared_ptr<material> fallback_mat,
                           double scale = 1.0)
        : fallback(std::move(fallback_mat))
    {
        load_obj_from_file(obj_path, scale);
    }

    bool hit(const ray& r, const interval& ray_t, hit_record& rec) const override {
        hit_record temp;
        bool hit_any = false;

        // NOTE: interval has methods min()/max(), not fields
        double closest = ray_t.max();

        for (const auto& tri : triangles) {
            // Use the current [ray_t.min(), closest] window for early-out refinement
            if (tri.hit(r, interval(ray_t.min(), closest), temp)) {
                hit_any  = true;
                closest  = temp.t;
                rec      = temp;
            }
        }
        return hit_any;
    }

    aabb bounding_box() const override {
        if (verts.empty()) return {};
        aabb box(verts[0], verts[0]);
        for (const auto& v : verts) box = surrounding_box(box, aabb(v, v));
        return box;
    }

private:
    struct MtlProps {
        std::string name;

        // colors
        vec3 Kd = vec3(0.8, 0.8, 0.8);  // diffuse
        vec3 Ks = vec3(0.0, 0.0, 0.0);  // specular
        vec3 Ke = vec3(0.0, 0.0, 0.0);  // emission

        // scalars
        double Ns = 0.0;   // shininess
        double d  = 1.0;   // opacity
        double Ni = 1.5;   // IOR

        // textures
        std::string map_Kd;
        std::string map_Ke;
    };

    static std::shared_ptr<material>
    to_cpu_material_from_mtl(const MtlProps& m, const std::string& base_dir) {
        // 1. Emissive / light materials
        const bool has_emissive = (m.Ke.x() != 0.0 || m.Ke.y() != 0.0 || m.Ke.z() != 0.0);
        if (has_emissive || !m.map_Ke.empty()) {
            if (!m.map_Ke.empty()) {
                auto tex = std::make_shared<image_texture>(base_dir + "/" + m.map_Ke);
                return std::make_shared<diffuse_light>(tex);
            }
            return std::make_shared<diffuse_light>(color(m.Ke.x(), m.Ke.y(), m.Ke.z()));
        }

        // 2. If there is a diffuse texture, treat as textured Lambertian.
        //    (Do this BEFORE dielectric/metal so textures aren't ignored.)
        if (!m.map_Kd.empty()) {
            std::string full_path = base_dir + "/" + m.map_Kd;
            auto tex = std::make_shared<image_texture>(full_path);
            return std::make_shared<lambertian>(tex);
        }

        // 3. Transparent materials → dielectric (no texture case)
        if (m.d < 0.999) {
            double ior = (m.Ni > 0.1 && m.Ni < 10.0) ? m.Ni : 1.5;
            return std::make_shared<dielectric>(ior);
        }

        // 4. Metals (based on Ks / Ns)
        const double ks_mag = m.Ks.length();
        if (ks_mag > 0.05) {
            double fuzz = 100.0 / (m.Ns + 100.0);
            fuzz = std::clamp(fuzz, 0.0, 1.0);
            vec3 c = (ks_mag > 0.05) ? m.Ks : m.Kd;
            return std::make_shared<metal>(color(c.x(), c.y(), c.z()), fuzz);
        }

        // 5. Plain diffuse (Lambertian with solid color)
        return std::make_shared<lambertian>(color(m.Kd.x(), m.Kd.y(), m.Kd.z()));
    }

    static std::unordered_map<std::string, MtlProps>
    load_mtl(const std::string& mtl_path) {
        std::unordered_map<std::string, MtlProps> out;
        std::ifstream in(mtl_path);
        if (!in) return out;

        std::string line, tag, cur;
        MtlProps props;

        auto flush = [&]() {
            if (!cur.empty()) {
                props.name = cur;
                out[cur] = props;
            }
        };

        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            tag.clear();
            if (!(iss >> tag)) continue;

            if (tag == "newmtl") {
                flush();
                props = MtlProps{};
                iss >> cur;
            }
            else if (tag == "Kd") {
                double r,g,b; if (iss >> r >> g >> b) props.Kd = vec3(r,g,b);
            }
            else if (tag == "Ks") {
                double r,g,b; if (iss >> r >> g >> b) props.Ks = vec3(r,g,b);
            }
            else if (tag == "Ke") {
                double r,g,b; if (iss >> r >> g >> b) props.Ke = vec3(r,g,b);
            }
            else if (tag == "Ns") {
                double ns; if (iss >> ns) props.Ns = ns;
            }
            else if (tag == "d") {
                double dd; if (iss >> dd) props.d = dd;
            }
            else if (tag == "Ni") {
                double ni; if (iss >> ni) props.Ni = ni;
            }
            else if (tag == "map_Kd") {
                iss >> props.map_Kd;
            }
            else if (tag == "map_Ke") {
                iss >> props.map_Ke;
            }
            // ignore the rest
        }
        flush();
        return out;
    }

    void load_obj_from_file(const std::string& obj_path, double scale) {
        std::ifstream in(obj_path);
        if (!in) return;

        const std::string base_dir = obj_path.substr(0, obj_path.find_last_of("/\\") + 1);
        std::unordered_map<std::string, MtlProps> mtl;
        std::unordered_map<std::string, std::shared_ptr<material>> mat_cache;  // <-- ADD

        std::string line, tag, cur_mtl;

        auto parse_face_idx = [](const std::string& t)->std::tuple<int,int,int> {
            int v=0, vt=0, vn=0;
            if (std::sscanf(t.c_str(), "%d/%d/%d", &v,&vt,&vn) == 3) return {v,vt,vn};
            if (std::sscanf(t.c_str(), "%d//%d",    &v,&vn)     == 2) return {v,0,vn};
            if (std::sscanf(t.c_str(), "%d/%d",     &v,&vt)     == 2) return {v,vt,0};
            if (std::sscanf(t.c_str(), "%d",        &v)         == 1) return {v,0,0};
            return {0,0,0};
        };

        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            if (!(iss >> tag)) continue;

            if (tag == "mtllib") {
                std::string mtl_name; iss >> mtl_name;
                auto tbl = load_mtl(base_dir + mtl_name);
                for (auto& kv : tbl) mtl[kv.first] = std::move(kv.second);
            }
            else if (tag == "usemtl") {
                iss >> cur_mtl;
            }
            else if (tag == "v") {
                double x,y,z; if (iss >> x >> y >> z) verts.emplace_back(scale*x, scale*y, scale*z);
            }
            else if (tag == "vt") {
                float u,v; if (iss >> u >> v) uvs.emplace_back(u, 1.0f - v, 0.0f);
            }
            else if (tag == "f") {
                std::vector<std::string> toks;
                std::string tok;
                while (iss >> tok) toks.push_back(tok);
                if (toks.size() < 3) continue;

                // --- 🔧 FIXED MATERIAL LOOKUP ---
                std::shared_ptr<material> use_mat = fallback;
                if (!cur_mtl.empty()) {
                    auto it = mat_cache.find(cur_mtl);
                    if (it != mat_cache.end()) {
                        use_mat = it->second;                // reuse cached CPU material
                    } else if (auto mit = mtl.find(cur_mtl); mit != mtl.end()) {
                        use_mat = to_cpu_material_from_mtl(mit->second, base_dir);
                        mat_cache[cur_mtl] = use_mat;        // cache it for reuse
                    }
                }

                auto [i0, it0, in0] = parse_face_idx(toks[0]);
                if (i0 == 0) continue;
                const vec3 v0 = verts[i0-1];
                const vec3 uv0 = (it0>0 && it0 <= (int)uvs.size()) ? uvs[it0-1] : vec3(0,0,0);

                for (size_t k = 1; k + 1 < toks.size(); ++k) {
                    auto [i1, it1, in1] = parse_face_idx(toks[k]);
                    auto [i2, it2, in2] = parse_face_idx(toks[k+1]);
                    if (i1==0 || i2==0) continue;

                    const vec3 v1 = verts[i1-1];
                    const vec3 v2 = verts[i2-1];
                    const vec3 uv1 = (it1>0 && it1 <= (int)uvs.size()) ? uvs[it1-1] : vec3(0,0,0);
                    const vec3 uv2 = (it2>0 && it2 <= (int)uvs.size()) ? uvs[it2-1] : vec3(0,0,0);

                    triangles.emplace_back(v0, v1, v2, uv0, uv1, uv2, use_mat);

                    std::string tex_path;
                    if (!cur_mtl.empty()) {
                        auto it = mtl.find(cur_mtl);
                        if (it != mtl.end() && !it->second.map_Kd.empty()) {
                            tex_path = base_dir + it->second.map_Kd;
                        }
                    }
                    tri_map_Kd.push_back(tex_path);  // "" if no texture
                }
            }
        }
    }

};
