#pragma once
#include "camera.h"
#include "hittable_list.h"
#include "gpu_scene.h" // or wherever GPUTriangle / GPUMaterial / GPUNode / GPUCamera are declared

GPUScene build_gpu_scene(const hittable_list& world, const camera& cam);
void free_gpu_scene(GPUScene& scene);
