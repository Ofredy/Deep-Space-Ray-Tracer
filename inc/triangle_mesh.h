#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

#include "hittable.h"
#include "hittable_list.h"
#include "bvh.h"
#include "rtweekend.h"
#include "triangle.h"

#include <cstdio>
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <iostream>
#include <climits>
#include <cmath>

class triangle_mesh : public hittable {
  public:
    // Construct from an already-open OBJ FILE*. (We do NOT fclose() it.)
    // scale: optional uniform scale on positions.
    triangle_mesh(FILE* obj_file, shared_ptr<material> mat, double scale = 1.0)
    {
        std::vector<point3> P;
        std::vector<vec3>   N;
        std::vector<vec3>   UV; // store (u,v,0)
        std::vector<std::array<Idx,3>> F;

        if (!parse_obj_from_FILE(obj_file, P, N, UV, F)) {
            std::cerr << "[triangle_mesh] Failed to parse OBJ from FILE*\n";
            accel = nullptr;
            bbox  = aabb();
            return;
        }

        // Uniform scale
        if (scale != 1.0) {
            for (auto& p : P) p = point3(p.x()*scale, p.y()*scale, p.z()*scale);
        }

        // Build triangles (flat-shaded; geometric normal in triangle::hit)
        hittable_list tris;
        tris.objects.reserve(F.size());
        for (const auto& f : F) {
            const auto& i0 = f[0];
            const auto& i1 = f[1];
            const auto& i2 = f[2];

            if (i0.v < 0 || i0.v >= (int)P.size()
             || i1.v < 0 || i1.v >= (int)P.size()
             || i2.v < 0 || i2.v >= (int)P.size()) {
                continue;
            }

            const point3 a = P[i0.v];
            const point3 b = P[i1.v];
            const point3 c = P[i2.v];

            tris.add(make_shared<triangle>(a, b, c, mat));
        }

        if (tris.objects.empty()) {
            std::cerr << "[triangle_mesh] OBJ had no valid triangles.\n";
            accel = nullptr;
            bbox  = aabb();
            return;
        }

        // BVH acceleration + cache bbox
        accel = make_shared<bvh_node>(tris);
        bbox  = accel->bounding_box();
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        return accel && accel->hit(r, ray_t, rec);
    }

    aabb bounding_box() const override { return bbox; }

    // Delegates to BVH (valid for stationary geometry; matches sphere pattern)
    double pdf_value(const point3& origin, const vec3& direction) const override {
        return accel ? accel->pdf_value(origin, direction) : 0.0;
    }

    vec3 random(const point3& origin) const override {
        return accel ? accel->random(origin) : vec3(1,0,0);
    }

  private:
    struct Idx { int v=-1, vt=-1, vn=-1; }; // 0-based after fixup
    shared_ptr<hittable> accel;
    aabb bbox;

    static bool parse_obj_from_FILE(
        FILE* f,
        std::vector<point3>& positions,
        std::vector<vec3>& normals,
        std::vector<vec3>& uvs,
        std::vector<std::array<Idx,3>>& faces
    ) {
        if (!f) return false;

        char buf[1<<15]; // 32KB per line
        std::string line;

        while (std::fgets(buf, sizeof(buf), f)) {
            line.assign(buf);
            trim(line);
            if (line.empty() || line[0] == '#') continue;

            if (starts_with(line, "v ")) {
                std::istringstream iss(line.substr(2));
                double x,y,z;
                if (!(iss >> x >> y >> z)) continue;
                positions.emplace_back(x,y,z);
                continue;
            }

            if (starts_with(line, "vt ")) {
                std::istringstream iss(line.substr(3));
                double u=0.0, v=0.0;
                iss >> u >> v;
                uvs.emplace_back(u, v, 0.0);
                continue;
            }

            if (starts_with(line, "vn ")) {
                std::istringstream iss(line.substr(3));
                double nx,ny,nz;
                if (!(iss >> nx >> ny >> nz)) continue;
                normals.emplace_back(nx,ny,nz);
                continue;
            }

            if (starts_with(line, "f ")) {
                std::istringstream iss(line.substr(2));
                std::string t0, t1, t2;
                if (!(iss >> t0 >> t1 >> t2)) continue; // tri only; ignore extra verts

                Idx i0, i1, i2;
                if (!parse_face_triplet(t0, i0, (int)positions.size(), (int)uvs.size(), (int)normals.size())) continue;
                if (!parse_face_triplet(t1, i1, (int)positions.size(), (int)uvs.size(), (int)normals.size())) continue;
                if (!parse_face_triplet(t2, i2, (int)positions.size(), (int)uvs.size(), (int)normals.size())) continue;

                faces.push_back({i0, i1, i2});
                continue;
            }

            // ignore others: o, g, s, usemtl, mtllib, etc.
        }

        return !positions.empty();
    }

    static bool parse_face_triplet(
        const std::string& token, Idx& out, int nv, int nut, int nno
    ) {
        // form: v | v/vt | v//vn | v/vt/vn (OBJ 1-based; negatives allowed)
        int parts[3] = {INT_MIN, INT_MIN, INT_MIN}; // v, vt, vn
        int part_idx = 0;

        size_t start = 0;
        for (size_t i = 0; i <= token.size(); ++i) {
            if (i == token.size() || token[i] == '/') {
                std::string s = token.substr(start, i - start);
                if (!s.empty()) {
                    try {
                        parts[part_idx] = std::stoi(s);
                    } catch (...) { parts[part_idx] = INT_MIN; }
                }
                ++part_idx;
                start = i + 1;
                if (part_idx > 2) break;
            }
        }

        if (parts[0] == INT_MIN) return false;
        out.v  = fix_index(parts[0], nv);
        out.vt = (parts[1] == INT_MIN) ? -1 : fix_index(parts[1], nut);
        out.vn = (parts[2] == INT_MIN) ? -1 : fix_index(parts[2], nno);

        if (out.v  < 0 || out.v  >= nv) return false;
        if (out.vt != -1 && (out.vt < 0 || out.vt >= nut)) out.vt = -1;
        if (out.vn != -1 && (out.vn < 0 || out.vn >= nno)) out.vn = -1;

        return true;
    }

    // OBJ: positive indices are 1-based; negative are relative to end
    static inline int fix_index(int idx, int n) {
        return (idx > 0) ? (idx - 1) : (n + idx);
    }

    static inline bool starts_with(const std::string& s, const char* pfx) {
        return s.rfind(pfx, 0) == 0;
    }

    static inline void trim(std::string& s) {
        size_t a = 0, b = s.size();
        while (a < b && (s[a]==' ' || s[a]=='\t' || s[a]=='\r' || s[a]=='\n')) ++a;
        while (b > a && (s[b-1]==' ' || s[b-1]=='\t' || s[b-1]=='\r' || s[b-1]=='\n')) --b;
        s = s.substr(a, b-a);
    }
};

#endif // TRIANGLE_MESH_H