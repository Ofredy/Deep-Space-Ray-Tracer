#Deep Space Ray Tracer

Deep Space Ray Tracer is a high-fidelity GPU-accelerated deep-space rendering engine built from the foundations of Peter Shirley’s Ray Tracing in One Weekend, The Next Week, and The Rest of Your Life. The original CPU ray tracer was completed fully through the third book, then extended and ported to CUDA to support triangle meshes, BVH acceleration, directional lighting, and deep-space coordinate transforms using double precision.

The renderer is designed to work with orbital-mechanics-driven camera and model trajectories, including real SPICE ephemerides, enabling physically meaningful spaceflight visualizations.

#Overview

Deep Space Ray Tracer renders photorealistic spacecraft in deep-space conditions by combining:

1. CPU-Generated Orbital States

A separate physics simulation computes:

A two-body polar lunar orbit using Kepler’s problem

True Moon position relative to the Sun using NASA JPL SPICE ephemerides

Camera and model positions written to a plain .txt file for each frame

#GPU Path Tracing (CUDA)

The CPU ray tracer from the books was ported to GPU, extended with:

Triangle mesh loading for complex models (ISS)

BVH acceleration structures

Directional Sun lighting

Double-precision world-frame math for deep-space distances

Float conversion only after transforming into the local model frame
