#include "gpu_scene.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>

// ----------------------
// Ray on GPU
// ----------------------
struct RayGPU {
    float3 origin;
    float3 dir;

    __device__ float3 at(float t) const {
        return make_float3(
            origin.x + t * dir.x,
            origin.y + t * dir.y,
            origin.z + t * dir.z
        );
    }
};

// ----------------------
// Helpers
// ----------------------
static void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        printf("CUDA ERROR %s: %s\n", msg, cudaGetErrorString(result));
        // hard fail on host
        // flush so we actually see the message in MSVC output
        fflush(stdout);
        fflush(stderr);
        // bail
        abort(); // or exit(EXIT_FAILURE);
    }
}

__device__ inline float3 make_float3_from_vec3(const vec3& v) {
    return make_float3((float)v.x(), (float)v.y(), (float)v.z());
}

__device__ inline float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float  dot3(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ inline float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

__device__ inline float3 normalize3(float3 v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    float inv = (len > 0.0f) ? (1.0f/len) : 0.0f;
    return make_float3(v.x*inv, v.y*inv, v.z*inv);
}

// ----------------------
// Sphere hit
// ----------------------
__device__
bool hit_sphere(const GPUSphere& s,
                const RayGPU& r,
                float tmin,
                float tmax,
                float& t_out,
                float3& normal_out,
                int&   mat_id_out)
{
    // oc = sphere_center - ray_origin
    float3 center = make_float3((float)s.center.x(), (float)s.center.y(), (float)s.center.z());
    float3 oc = make_float3(
        center.x - r.origin.x,
        center.y - r.origin.y,
        center.z - r.origin.z
    );

    float a = dot3(r.dir, r.dir);
    float h = dot3(r.dir, oc);
    float c = dot3(oc, oc) - (s.radius * s.radius);

    float disc = h*h - a*c;
    if (disc < 0.0f) {
        return false;
    }

    float sqrt_disc = sqrtf(disc);

    // try near root
    float t = (h - sqrt_disc) / a;
    if (t < tmin || t > tmax) {
        t = (h + sqrt_disc) / a;
        if (t < tmin || t > tmax) {
            return false;
        }
    }

    t_out = t;

    float3 p = r.at(t);
    float inv_r = 1.0f / s.radius;
    normal_out = make_float3(
        (p.x - center.x)*inv_r,
        (p.y - center.y)*inv_r,
        (p.z - center.z)*inv_r
    );

    mat_id_out = s.material_id;
    return true;
}

// ----------------------
// Triangle hit (Möller–Trumbore)
// ----------------------
__device__
bool hit_triangle(const GPUTriangle& tri,
                  const RayGPU& r,
                  float tmin,
                  float tmax,
                  float& t_out,
                  float3& normal_out,
                  int&   mat_id_out)
{
    // get triangle verts
    float3 v0 = make_float3((float)tri.v0.x(), (float)tri.v0.y(), (float)tri.v0.z());
    float3 v1 = make_float3((float)tri.v1.x(), (float)tri.v1.y(), (float)tri.v1.z());
    float3 v2 = make_float3((float)tri.v2.x(), (float)tri.v2.y(), (float)tri.v2.z());

    float3 edge1 = sub3(v1, v0);
    float3 edge2 = sub3(v2, v0);

    float3 pvec = cross3(r.dir, edge2);
    float det = dot3(edge1, pvec);

    // backface cull off -> allow both sides
    if (fabsf(det) < 1e-8f) {
        return false;
    }
    float invDet = 1.0f / det;

    float3 tvec = sub3(r.origin, v0);
    float u = dot3(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    float3 qvec = cross3(tvec, edge1);
    float v = dot3(r.dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    float t = dot3(edge2, qvec) * invDet;

    if (t < tmin || t > tmax) {
        return false;
    }

    t_out = t;

    // flat normal
    float3 n = cross3(edge1, edge2);
    normal_out = normalize3(n);

    mat_id_out = tri.material_id;
    return true;
}

// ----------------------
// Scene hit: find closest sphere/triangle
// ----------------------
__device__
bool scene_hit(const GPUScene& scene,
               const RayGPU& r,
               float tmin,
               float tmax,
               float3& normal_out,
               int&   mat_id_out)
{
    float closest = tmax;
    bool  hit_any = false;

    // spheres
    for (int i = 0; i < scene.num_spheres; ++i) {
        float t;
        float3 n;
        int mid;
        if (hit_sphere(scene.d_spheres[i], r, tmin, closest, t, n, mid)) {
            hit_any = true;
            closest = t;
            normal_out = n;
            mat_id_out = mid;
        }
    }

    // triangles
    for (int i = 0; i < scene.num_tris; ++i) {
        float t;
        float3 n;
        int mid;
        if (hit_triangle(scene.d_tris[i], r, tmin, closest, t, n, mid)) {
            hit_any = true;
            closest = t;
            normal_out = n;
            mat_id_out = mid;
        }
    }

    return hit_any;
}

// ----------------------
// simple color: normal -> rgb, emissive if mat is light
// for now we'll just map normal to color so we SEE SHAPES
// ----------------------
__device__
uchar3 shade_normal(const float3& n) {
    // remap [-1,1] to [0,1]
    float r = 0.5f * (n.x + 1.0f);
    float g = 0.5f * (n.y + 1.0f);
    float b = 0.5f * (n.z + 1.0f);

    // clamp and convert to 0-255
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);

    uchar3 out;
    out.x = (unsigned char)(r * 255.0f);
    out.y = (unsigned char)(g * 255.0f);
    out.z = (unsigned char)(b * 255.0f);
    return out;
}

// ----------------------
// Generate primary ray from camera
// ----------------------
__device__
RayGPU make_camera_ray(const GPUScene& scene, int px, int py) {
    const GPUCamera& cam = scene.cam;

    // normalize pixel coords to [0,1]
    float u = float(px) / float(scene.image_width  - 1);
    float v = float(py) / float(scene.image_height - 1);

    // convert the camera basis vectors (which are vec3) into float3
    float3 llc = make_float3_from_vec3(cam.lower_left_corner); // lower_left_corner
    float3 hor = make_float3_from_vec3(cam.horizontal);        // horizontal
    float3 ver = make_float3_from_vec3(cam.vertical);          // vertical
    float3 org = make_float3_from_vec3(cam.origin);            // origin

    // pixel_pos = lower_left_corner + u*horizontal + v*vertical
    float3 pixel_pos = make_float3(
        llc.x + u * hor.x + v * ver.x,
        llc.y + u * hor.y + v * ver.y,
        llc.z + u * hor.z + v * ver.z
    );

    RayGPU r;
    r.origin = org;
    r.dir    = make_float3(
        pixel_pos.x - org.x,
        pixel_pos.y - org.y,
        pixel_pos.z - org.z
    );

    return r;
}

__device__ inline float3 get_albedo_rgb(const GPUMaterial& m) {
    // vec3 in GPUMaterial stores doubles on the host
    // We downcast because the kernel shades in float.
    return make_float3(
        (float)m.albedo.x(),
        (float)m.albedo.y(),
        (float)m.albedo.z()
    );
}

__device__
uchar3 shade_lit_material(
    const GPUScene& scene,
    int mat_id,
    const float3& surf_normal
) {
    // safety check
    if (mat_id < 0 || mat_id >= scene.num_mats) {
        // hot magenta to show bad IDs
        return make_uchar3(255, 0, 255);
    }

    // fetch material
    const GPUMaterial& m = scene.d_mats[mat_id];

    // base color from material
    float3 base = get_albedo_rgb(m);

    // pick a fake directional light in world space
    // coming from +Y,+Z so it doesn't go black from straight overhead
    float3 light_dir = make_float3(0.0f, 0.7f, 0.7f);
    light_dir = normalize3(light_dir);

    // cosine term
    float ndotl = surf_normal.x * light_dir.x +
                  surf_normal.y * light_dir.y +
                  surf_normal.z * light_dir.z;
    if (ndotl < 0.0f) ndotl = 0.0f;

    // basic diffuse = albedo * ndotl
    float3 lit = make_float3(
        base.x * ndotl,
        base.y * ndotl,
        base.z * ndotl
    );

    // super crude "emissive" boost for lights:
    // if this mat is supposed to be a light, just add its albedo raw.
    if (m.type == MAT_DIFFUSE_LIGHT) {
        lit.x += base.x;
        lit.y += base.y;
        lit.z += base.z;
    }

    // clamp to [0,1]
    lit.x = fminf(fmaxf(lit.x, 0.0f), 1.0f);
    lit.y = fminf(fmaxf(lit.y, 0.0f), 1.0f);
    lit.z = fminf(fmaxf(lit.z, 0.0f), 1.0f);

    // convert to 0-255
    return make_uchar3(
        (unsigned char)(255.0f * lit.x),
        (unsigned char)(255.0f * lit.y),
        (unsigned char)(255.0f * lit.z)
    );
}

// ----------------------
// Render kernel
// ----------------------
__global__
void render_kernel(const GPUScene scene, unsigned char* out_rgb) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= scene.image_width || y >= scene.image_height) return;

    // flip Y so (0,0) is bottom-left in output image
    int py = scene.image_height - 1 - y;
    int px = x;

    RayGPU ray = make_camera_ray(scene, px, py);

    float3 n_hit;
    int mat_id;
    bool hit = scene_hit(scene, ray, 0.001f, 1e30f, n_hit, mat_id);

    uchar3 rgb;
    if (hit) {
        rgb = shade_lit_material(scene, mat_id, n_hit);
    } else {
        rgb = make_uchar3(0, 0, 0);
    }

    int idx = (y * scene.image_width + x) * 3;
    out_rgb[idx + 0] = rgb.x;
    out_rgb[idx + 1] = rgb.y;
    out_rgb[idx + 2] = rgb.z;
}

// ----------------------
// Host entrypoint
// ----------------------
extern "C"
void gpu_render_scene(const GPUScene& scene, int width, int height) {
    printf("gpu_render_scene() called with width=%d, height=%d\n", width, height);

    // sanity check: width/height should match scene
    // (your CPU caller is already passing them in from cam)
    size_t nbytes = (size_t)width * (size_t)height * 3;
    unsigned char* d_fb = nullptr;
    checkCuda(cudaMalloc(&d_fb, nbytes), "cudaMalloc d_fb");

    dim3 block(16,16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    printf("Launching render_kernel grid=(%d,%d) block=(%d,%d)\n",
        grid.x, grid.y, block.x, block.y);

    render_kernel<<<grid, block>>>(scene, d_fb);
    checkCuda(cudaGetLastError(), "render_kernel launch");
    checkCuda(cudaDeviceSynchronize(), "render_kernel sync");

    std::vector<unsigned char> host_fb(nbytes);
    checkCuda(cudaMemcpy(host_fb.data(), d_fb, nbytes, cudaMemcpyDeviceToHost), "memcpy back");

    // dump a few pixels for debug
    printf("First pixel rgb: %u %u %u\n", host_fb[0], host_fb[1], host_fb[2]);

    FILE* f = fopen("output.ppm", "wb");
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    fwrite(host_fb.data(), 1, nbytes, f);
    fclose(f);

    cudaFree(d_fb);
    printf("Done writing output.ppm\n");
}
