#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

#include <cstdio>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <sstream>

#include "rtweekend.h"
#include "hittable.h"
#include "hittable_list.h"
#include "bvh.h"
#include "material.h"
#include "texture.h"
#include "triangle.h"

// Minimal OBJ/MTL triangle mesh that supports:
// - v / vt (uv) / vn (vn ignored for shading; we use geometric normal)
// - f with v, v/vt, v//vn, or v/vt/vn
// - mtllib, usemtl
// - MTL: newmtl, Kd, map_Kd
//
// Two constructors:
//  (1) triangle_mesh(FILE* obj, shared_ptr<material> single_mat, double scale=1.0)
//  (2) triangle_mesh(FILE* obj, const char* obj_dir, shared_ptr<material> fallback, double scale=1.0)
//      Per-material build using .mtl in obj_dir (map_Kd + Kd).

class triangle_mesh : public hittable {
  public:
    // Single-material mesh
    triangle_mesh(FILE* fp, shared_ptr<material> single_mat, double scale = 1.0) {
        std::vector<vec3> P, UV;
        std::vector<Face> F;
        std::vector<std::string> mtllibs;

        parse_obj_from_FILE(fp, P, UV, F, mtllibs, scale);

        hittable_list tris;
        tris.objects.reserve(F.size());
        for (const auto& face : F) {
            // Triangulated already; faces are triples
            const Idx& i0 = face.idx[0];
            const Idx& i1 = face.idx[1];
            const Idx& i2 = face.idx[2];

            // positions are required
            if (!valid_v(i0.v, P) || !valid_v(i1.v, P) || !valid_v(i2.v, P)) continue;

            const bool has_uv = valid_vt(i0.vt, UV) && valid_vt(i1.vt, UV) && valid_vt(i2.vt, UV);

            if (has_uv) {
                tris.add(make_shared<triangle_uv>(
                    P[i0.v], P[i1.v], P[i2.v],
                    UV[i0.vt], UV[i1.vt], UV[i2.vt],
                    single_mat
                ));
            } else {
                tris.add(make_shared<triangle>(P[i0.v], P[i1.v], P[i2.v], single_mat));
            }
        }

        accel = make_shared<bvh_node>(tris);
        bbox_ = accel->bounding_box();
    }

    // Per-material mesh, loading .mtl(s) from obj_dir
    triangle_mesh(FILE* fp, const char* obj_dir, shared_ptr<material> fallback, double scale = 1.0) {
        std::vector<vec3> P, UV;
        std::vector<Face> F;
        std::vector<std::string> mtllibs;

        parse_obj_from_FILE(fp, P, UV, F, mtllibs, scale);

        // Load all materials referenced by mtllib
        std::unordered_map<std::string, MtlRecord> matdefs;
        for (const auto& mtlname : mtllibs) {
            parse_mtl_file(join_path(obj_dir, mtlname).c_str(), matdefs, obj_dir);
        }

        // Cache shared_ptr<material> by material name
        std::unordered_map<std::string, shared_ptr<material>> mat_cache;

        auto mat_of = [&](const std::string& name)->shared_ptr<material> {
            auto it = mat_cache.find(name);
            if (it != mat_cache.end()) return it->second;

            auto md = matdefs.find(name);
            shared_ptr<material> out;
            if (md == matdefs.end()) {
                // Material name not found in .mtl; use fallback
                out = fallback;
            } else {
                const auto& def = md->second;
                if (!def.map_Kd.empty()) {
                    out = make_shared<lambertian>( make_shared<image_texture>(def.map_Kd.c_str()) );
                } else {
                    out = make_shared<lambertian>( def.Kd );
                }
            }
            mat_cache[name] = out;
            return out;
        };

        hittable_list tris;
        tris.objects.reserve(F.size());
        for (const auto& face : F) {
            const Idx& i0 = face.idx[0];
            const Idx& i1 = face.idx[1];
            const Idx& i2 = face.idx[2];

            if (!valid_v(i0.v, P) || !valid_v(i1.v, P) || !valid_v(i2.v, P)) continue;

            const bool has_uv = valid_vt(i0.vt, UV) && valid_vt(i1.vt, UV) && valid_vt(i2.vt, UV);
            auto mat_to_use = mat_of(face.mtl);

            if (has_uv) {
                tris.add(make_shared<triangle_uv>(
                    P[i0.v], P[i1.v], P[i2.v],
                    UV[i0.vt], UV[i1.vt], UV[i2.vt],
                    mat_to_use
                ));
            } else {
                tris.add(make_shared<triangle>(P[i0.v], P[i1.v], P[i2.v], mat_to_use));
            }
        }

        accel = make_shared<bvh_node>(tris);
        bbox_ = accel->bounding_box();
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        return accel && accel->hit(r, ray_t, rec);
    }

    aabb bounding_box() const override {
        return bbox_;
    }

  private:
    // ----------------- Internal structures -----------------
    struct Idx { int v=-1, vt=-1, vn=-1; };
    struct Face {
        Idx idx[3];
        std::string mtl;
    };

    struct MtlRecord {
        color Kd = color(0.8, 0.8, 0.8);
        std::string map_Kd;  // absolute/normalized path (joined)
    };

    shared_ptr<hittable> accel;
    aabb bbox_;

    // ----------------- Parsing helpers -----------------
    static inline bool valid_v(int i, const std::vector<vec3>& P)  { return i>=0 && i<(int)P.size(); }
    static inline bool valid_vt(int i, const std::vector<vec3>& UV){ return i>=0 && i<(int)UV.size(); }

    static std::string join_path(const std::string& a, const std::string& b) {
        if (a.empty()) return b;
        if (b.empty()) return a;
        char sep = '/';
#ifdef _WIN32
        // Keep Windows drive roots intact; normalize to '/'
        if (b.size()>1 && (b[1]==':' || b[0]=='\\' || b[0]=='/')) return b;
#endif
        if (a.back()=='/' || a.back()=='\\') return a + b;
        return a + sep + b;
    }

    static void parse_obj_from_FILE(FILE* fp,
                                    std::vector<vec3>& P,
                                    std::vector<vec3>& UV,
                                    std::vector<Face>& F,
                                    std::vector<std::string>& mtllibs,
                                    double scale)
    {
        char line[4096];
        std::string cur_mtl;

        while (fgets(line, sizeof(line), fp)) {
            // skip comments/blank
            if (line[0]=='#' || std::isspace((unsigned char)line[0])) {
                // could still be tokens later; fallthrough to parse via stringstream
            }

            std::string s(line);
            std::istringstream iss(s);
            std::string tok;
            if (!(iss >> tok)) continue;

            if (tok == "v") {
                double x,y,z; iss >> x >> y >> z;
                P.emplace_back(scale*x, scale*y, scale*z);
            } else if (tok == "vt") {
                double u=0, v=0; iss >> u >> v;
                UV.emplace_back(u, v, 0.0);
            } else if (tok == "vn") {
                // we ignore vn for shading (using geometric normal)
                double nx, ny, nz; iss >> nx >> ny >> nz; (void)nx; (void)ny; (void)nz;
            } else if (tok == "f") {
                // faces: triangulate fan if >3 vertices
                std::vector<Idx> poly;
                std::string vtok;
                while (iss >> vtok) {
                    Idx idx;
                    // parse "v", "v/vt", "v//vn", or "v/vt/vn"
                    int v=-1, vt=-1, vn=-1;
                    const char* c = vtok.c_str();
                    // read v
                    v = std::strtol(c, (char**)&c, 10);
                    if (*c == '/') {
                        c++;
                        if (*c != '/') {
                            vt = std::strtol(c, (char**)&c, 10);
                        }
                        if (*c == '/') {
                            c++;
                            vn = std::strtol(c, (char**)&c, 10);
                        }
                    }
                    // OBJ indices are 1-based; handle negatives minimalistically
                    idx.v  = (v  > 0) ? (v-1)  : v;  // we won't support negative wrap robustly
                    idx.vt = (vt > 0) ? (vt-1) : -1;
                    idx.vn = (vn > 0) ? (vn-1) : -1;
                    poly.push_back(idx);
                }
                if (poly.size() >= 3) {
                    // triangle fan
                    for (size_t k=1; k+1<poly.size(); ++k) {
                        Face f; f.mtl = cur_mtl;
                        f.idx[0] = poly[0];
                        f.idx[1] = poly[k];
                        f.idx[2] = poly[k+1];
                        F.push_back(f);
                    }
                }
            } else if (tok == "usemtl") {
                iss >> cur_mtl;
            } else if (tok == "mtllib") {
                std::string mtlfile;
                iss >> mtlfile;
                if (!mtlfile.empty())
                    mtllibs.push_back(mtlfile);
            }
        }
    }

    static void parse_mtl_file(const char* path,
                               std::unordered_map<std::string, MtlRecord>& out,
                               const char* obj_dir)
    {
        FILE* f = std::fopen(path, "rb");
        if (!f) {
            // try join with obj_dir if not already
            std::string retry = join_path(obj_dir ? obj_dir : "", path);
            f = std::fopen(retry.c_str(), "rb");
            if (!f) return;
        }

        char line[4096];
        std::string cur;
        auto commit = [&]() {
            if (!cur.empty()) {
                // nothing to do at commit; out[cur] already modified by Kd/map_Kd
            }
        };

        while (fgets(line, sizeof(line), f)) {
            if (line[0]=='#') continue;
            std::string s(line);
            std::istringstream iss(s);
            std::string tok;
            if (!(iss >> tok)) continue;

            if (tok == "newmtl") {
                commit();
                iss >> cur;
                if (!cur.empty() && out.find(cur)==out.end())
                    out[cur] = MtlRecord{};
            } else if (tok == "Kd") {
                double r=0,g=0,b=0; iss >> r >> g >> b;
                if (!cur.empty()) out[cur].Kd = color(r,g,b);
            } else if (tok == "map_Kd") {
                std::string tex; iss >> tex;
                if (!cur.empty()) {
                    out[cur].map_Kd = join_path( extract_dir(path), tex );
                }
            } else {
                // ignore other params (Ka, Ks, Ns, Pr, Pm, aniso, etc.)
            }
        }
        commit();
        std::fclose(f);
    }

    static std::string extract_dir(const std::string& full) {
        size_t p = full.find_last_of("/\\");
        if (p == std::string::npos) return "";
        return full.substr(0, p);
    }
};

#endif // TRIANGLE_MESH_H
