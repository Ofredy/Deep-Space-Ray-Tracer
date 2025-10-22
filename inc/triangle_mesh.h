#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

#include "hittable.h"
#include "hittable_list.h"
#include "bvh.h"
#include "rtweekend.h"
#include "triangle.h"
#include "texture.h"    // image_texture
#include "material.h"   // uses your lambertian/diffuse_light/etc.

#include <cstdio>
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <iostream>
#include <climits>
#include <unordered_map>
#include <fstream>
#include <cctype>

class triangle_mesh : public hittable {
  public:
    // Old/simple path: apply ONE material to the whole mesh
    triangle_mesh(FILE* obj_file, shared_ptr<material> single_mat, double scale = 1.0)
    {
        std::vector<point3> P; std::vector<vec3> N; std::vector<vec3> UV;
        std::vector<Face>   F;
        if (!parse_obj_from_FILE(obj_file, P, N, UV, F /*mtllibs ignored*/)) {
            std::cerr << "[triangle_mesh] Failed to parse OBJ from FILE*\n";
            accel=nullptr; bbox=aabb(); return;
        }
        if (scale != 1.0) for (auto& p : P) p = point3(p.x()*scale, p.y()*scale, p.z()*scale);

        hittable_list tris; tris.objects.reserve(F.size());
        for (const auto& face : F) {
            const auto& i0 = face.idx[0]; const auto& i1 = face.idx[1]; const auto& i2 = face.idx[2];
            if (!valid_v(i0.v,P) || !valid_v(i1.v,P) || !valid_v(i2.v,P)) continue;
            tris.add(make_shared<triangle>(P[i0.v], P[i1.v], P[i2.v], single_mat));
        }
        if (tris.objects.empty()) { accel=nullptr; bbox=aabb(); return; }
        accel = make_shared<bvh_node>(tris);
        bbox  = accel->bounding_box();
    }

    // Full path: honor mtllib/usemtl (per-face materials). `obj_dir` is the folder of the OBJ/MTL/textures.
    triangle_mesh(FILE* obj_file, const char* obj_dir, shared_ptr<material> fallback, double scale = 1.0)
    {
        std::vector<point3> P; std::vector<vec3> N; std::vector<vec3> UV;
        std::vector<Face>   F;
        std::vector<std::string> mtllibs;
        if (!parse_obj_from_FILE(obj_file, P, N, UV, F, &mtllibs)) {
            std::cerr << "[triangle_mesh] Failed to parse OBJ from FILE*\n";
            accel=nullptr; bbox=aabb(); return;
        }
        if (scale != 1.0) for (auto& p : P) p = point3(p.x()*scale, p.y()*scale, p.z()*scale);

        // Load MTLs
        std::unordered_map<std::string, MaterialDef> matdefs;
        for (const auto& mtl : mtllibs) parse_mtl_file(join_path(obj_dir, mtl), matdefs);

        // Build lambertian materials from Kd / map_Kd
        std::unordered_map<std::string, shared_ptr<material>> mat_cache;
        auto mat_of = [&](const std::string& name)->shared_ptr<material>{
            if (name.empty()) return fallback;
            auto it = mat_cache.find(name); if (it != mat_cache.end()) return it->second;
            auto md = matdefs.find(name);
            shared_ptr<material> m;
            if (md == matdefs.end()) {
                m = fallback;
            } else if (!md->second.map_Kd.empty()) {
                m = make_shared<lambertian>( make_shared<image_texture>(md->second.map_Kd.c_str()) );
            } else {
                m = make_shared<lambertian>( md->second.Kd );
            }
            mat_cache[name] = m; return m;
        };

        // Build triangles with per-face materials
        hittable_list tris; tris.objects.reserve(F.size());
        for (const auto& face : F) {
            const auto& i0 = face.idx[0]; const auto& i1 = face.idx[1]; const auto& i2 = face.idx[2];
            if (!valid_v(i0.v,P) || !valid_v(i1.v,P) || !valid_v(i2.v,P)) continue;
            tris.add(make_shared<triangle>(P[i0.v], P[i1.v], P[i2.v], mat_of(face.mtl)));
        }
        if (tris.objects.empty()) { accel=nullptr; bbox=aabb(); return; }
        accel = make_shared<bvh_node>(tris);
        bbox  = accel->bounding_box();
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        return accel && accel->hit(r, ray_t, rec);
    }
    aabb bounding_box() const override { return bbox; }
    double pdf_value(const point3& origin, const vec3& direction) const override {
        return accel ? accel->pdf_value(origin, direction) : 0.0;
    }
    vec3 random(const point3& origin) const override {
        return accel ? accel->random(origin) : vec3(1,0,0);
    }

  private:
    // -------- data --------
    struct Idx { int v=-1, vt=-1, vn=-1; };
    struct Face { Idx idx[3]; std::string mtl; };
    struct MaterialDef { color Kd=color(0.8,0.8,0.8); std::string map_Kd; };

    shared_ptr<hittable> accel;
    aabb bbox;

    static bool valid_v(int v, const std::vector<point3>& P) { return v>=0 && v<(int)P.size(); }

    // -------- OBJ parsing --------
    static bool parse_obj_from_FILE(
        FILE* f,
        std::vector<point3>& positions,
        std::vector<vec3>& normals,
        std::vector<vec3>& uvs,
        std::vector<Face>& faces,
        std::vector<std::string>* mtllibs = nullptr
    ) {
        if (!f) return false;
        char buf[1<<15]; std::string line, current_mtl;

        while (std::fgets(buf, sizeof(buf), f)) {
            line.assign(buf); trim(line);
            if (line.empty() || line[0]=='#') continue;

            if (starts_with(line,"v ")) {
                std::istringstream iss(line.substr(2)); double x,y,z; if (!(iss>>x>>y>>z)) continue;
                positions.emplace_back(x,y,z); continue;
            }
            if (starts_with(line,"vt ")) {
                std::istringstream iss(line.substr(3)); double u=0,v=0; iss>>u>>v;
                uvs.emplace_back(u,v,0.0); continue;
            }
            if (starts_with(line,"vn ")) {
                std::istringstream iss(line.substr(3)); double nx,ny,nz; if (!(iss>>nx>>ny>>nz)) continue;
                normals.emplace_back(nx,ny,nz); continue;
            }
            if (starts_with(line,"mtllib ")) {
                if (mtllibs) { auto n=trim_copy(line.substr(7)); if (!n.empty()) mtllibs->push_back(n); }
                continue;
            }
            if (starts_with(line,"usemtl ")) { current_mtl = trim_copy(line.substr(7)); continue; }
            if (starts_with(line,"f ")) {
                std::istringstream iss(line.substr(2)); std::string t0,t1,t2;
                if (!(iss>>t0>>t1>>t2)) continue; // tris only
                Face face;
                if (!parse_face_triplet(t0, face.idx[0], (int)positions.size(), (int)uvs.size(), (int)normals.size())) continue;
                if (!parse_face_triplet(t1, face.idx[1], (int)positions.size(), (int)uvs.size(), (int)normals.size())) continue;
                if (!parse_face_triplet(t2, face.idx[2], (int)positions.size(), (int)uvs.size(), (int)normals.size())) continue;
                face.mtl = current_mtl;
                faces.push_back(face);
                continue;
            }
        }
        return !positions.empty();
    }

    static bool parse_face_triplet(const std::string& tok, Idx& out, int nv, int nut, int nno) {
        // supports: v | v/vt | v//vn | v/vt/vn (1-based; negatives allowed)
        int parts[3] = {INT_MIN, INT_MIN, INT_MIN}; int pi=0; size_t s=0;
        for (size_t i=0;i<=tok.size();++i) {
            if (i==tok.size() || tok[i]=='/') {
                std::string sub = tok.substr(s, i-s);
                if (!sub.empty()) { try { parts[pi] = std::stoi(sub); } catch(...) { parts[pi]=INT_MIN; } }
                ++pi; s=i+1; if (pi>2) break;
            }
        }
        if (parts[0]==INT_MIN) return false;
        out.v  = fix_index(parts[0], nv);
        out.vt = (parts[1]==INT_MIN) ? -1 : fix_index(parts[1], nut);
        out.vn = (parts[2]==INT_MIN) ? -1 : fix_index(parts[2], nno);
        if (out.v<0||out.v>=nv) return false;
        if (out.vt!=-1 && (out.vt<0||out.vt>=nut)) out.vt=-1;
        if (out.vn!=-1 && (out.vn<0||out.vn>=nno)) out.vn=-1;
        return true;
    }

    static inline int  fix_index(int idx,int n){ return (idx>0)?(idx-1):(n+idx); }
    static inline bool starts_with(const std::string& s,const char* p){ return s.rfind(p,0)==0; }
    static inline void trim(std::string& s){
        size_t a=0,b=s.size();
        while (a<b && std::isspace((unsigned char)s[a])) ++a;
        while (b>a && std::isspace((unsigned char)s[b-1])) --b;
        s = s.substr(a,b-a);
    }
    static inline std::string trim_copy(std::string s){ trim(s); return s; }

    // -------- MTL parsing (Kd + map_Kd) --------
    static void parse_mtl_file(const std::string& path, std::unordered_map<std::string,MaterialDef>& out) {
        std::ifstream in(path);
        if (!in) { std::cerr<<"[triangle_mesh] Could not open MTL: "<<path<<"\n"; return; }
        std::string dir=parent_dir(path), line, cur; MaterialDef md;
        auto commit=[&](){ if(!cur.empty()) out[cur]=md; md=MaterialDef{}; };
        while (std::getline(in,line)) {
            trim(line); if (line.empty()||line[0]=='#') continue;
            if (starts_with(line,"newmtl ")) { commit(); cur=trim_copy(line.substr(7)); continue; }
            if (starts_with(line,"Kd "))     { std::istringstream iss(line.substr(3)); double r=0.8,g=0.8,b=0.8; iss>>r>>g>>b; md.Kd=color(r,g,b); continue; }
            if (starts_with(line,"map_Kd ")) { md.map_Kd = join_path(dir, trim_copy(line.substr(7))); continue; }
        }
        commit();
    }

    // -------- path helpers --------
    static std::string join_path(const std::string& a, const std::string& b) {
        if (a.empty()) return b;
    #ifdef _WIN32
        const char sep='\\';
    #else
        const char sep='/';
    #endif
        if (!a.empty() && a.back()==sep) return a + b;
        return a + sep + b;
    }
    static std::string parent_dir(const std::string& p){
        size_t i=p.find_last_of("/\\"); return (i==std::string::npos)?std::string():p.substr(0,i);
    }
};

#endif // TRIANGLE_MESH_H