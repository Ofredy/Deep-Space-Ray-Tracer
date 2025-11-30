# Deep Space Ray Tracer

## 🧠 Overview

**Deep Space Ray Tracer** is a GPU based renderer built from the foundations of [Peter Shirley’s **Ray Tracing in One Weekend**, **Ray Tracing: The Next Week**, and **Ray Tracing: The Rest of Your Life**](https://raytracing.github.io/). After completing all three books in CPU form, the renderer was extended and ported to CUDA with support for **triangle meshes**, **BVH acceleration**, **double-precision deep-space coordinates**, and **directional solar lighting**.

The system renders spacecraft in **real orbital-mechanics-driven scenarios**. Camera and model poses are generated using a Kepler + SPICE pipeline, exported to a `.txt` file, and consumed by the GPU renderer frame-by-frame.

NASA’s official **[ISS 3D model](https://science.nasa.gov/resource/international-space-station-3d-model/)** serves as the primary rendering asset.

---

## ⚙️ Core Functionality

| Component | Description |
|----------|-------------|
| **CPU Orbital Simulator** | Solves Kepler’s problem for a 2-body lunar polar orbit. Uses JPL SPICE to compute the Moon’s absolute position relative to the Sun. Outputs pose `.txt` files. |
| **Pose File Parser** | Reads per-frame camera and ISS positions + yaw/pitch/roll from a `.txt` file. |
| **Double-Precision World Frame** | Handles large-scale distances (10⁶–10⁹ m) without floating-point drift. |
| **Model-Frame Transform System** | Converts world-frame coordinates to ISS model frame using double-precision rotations. |
| **Directional Sun Lighting** | Sun direction is computed from SPICE ephemerides and treated as an infinite directional emitter. |
| **Triangle Mesh Rendering** | Loads and renders NASA's ISS OBJ file with proper materials and scaling. |
| **BVH Acceleration Structure** | Efficient GPU triangle traversal for large meshes. |
| **CUDA Path Tracer** | Full GPU-based renderer adapted from the book series with triangle support. |
| **Image Exporter** | Saves PPM → PNG and optionally upscales outputs using a Python upsampler. |

---

## 🧩 Rendering Pipeline Summary

1. **Orbital Simulation**
   - A simple 2-body Kepler solver generates a lunar **polar orbit**.
   - Moon’s state is retrieved using **NASA JPL SPICE** relative to the Sun.
   - Camera and ISS positions are computed in the Sun-centered inertial frame.

2. **Pose File Generation**
   - For each frame, the simulator outputs the positions of the target and chaser vehicles in units of **meters**.

3. **Transformation into Model Frame**
   - The ray tracer reads the poses, then:
     - Computes camera → ISS relative vectors
     - Applies the frame rotation
     - Converts double → float only after constructing the local frame
     - Computes normalized Sun direction

4. **Scene Assembly**
   - ISS mesh placed at the origin  
   - Camera pointed at the ISS  
   - Directional light stored as a normalized vector  
   - GPUScene built with triangles, BVH nodes, and camera parameters

5. **CUDA Rendering**
   - Path tracing is performed entirely on the GPU:
     - Shadow rays
     - Bounce recursion
     - Material shading
     - Triangle intersection via BVH

6. **Export & Upscaling**
   - Saves PPM → PNG  
   - Optional AI upscaling via the `--upscale` flag

---

## 🛰️ Running the Orbital Simulator

The orbital simulator generates the camera/ISS trajectory and exports the pose `.txt` file used by the GPU renderer.

### 1️⃣ Create the Conda Environment for the Orbital Simulator

> The environment requirements for the orbital sim are located in:  
> `orbit_sim/environment.yml`

Create the environment:

```bash
conda env create -f orbit_sim/environment.yml
conda activate orbit_sim
```
### Run the Lunar Polar Orbit Simulation
This produces chaser and target vehicle states in meters and writes the pose file used by the ray tracer.
```bash
python .\lunar_polar_orbit_sim.py --time 1 --dt 0.01
```
The script will output rendezvous_1s_dt0_01s.txt where --time → total simulation duration in seconds & --dt → timestep resolution (smaller = more frames)

## 📊 Deep Space Ray Tracer Build & Example Usage

### Configure and Build (from top of repo)

```bash
# From the top of the repository:
mkdir build
cd build

# Configure
cmake ..

# Build (Release configuration)
cmake --build . --config Release
```
---
### Run the Ray Tracer (no upscaling)
```bash
.\Release\ray_tracer.exe ^
  --input_txt ..\orbit_sim\rendezvous_1s_dt0_01s.txt ^
  --output_dir os_1s_dt0_01s
```
### Run the Ray Tracer (with upscaling)
***NOTE***
- To use --upscale, you must:
- Build the Conda environment using the .yml file in scripts/
- Ensure the upsample command in main.cpp is updated to point to your Conda Python path and correct upsample.py location.
```bash
.\Release\ray_tracer.exe ^
  --input_txt ..\orbit_sim\rendezvous_1s_dt0_01s.txt ^
  --output_dir os_1s_dt0_01s
```
