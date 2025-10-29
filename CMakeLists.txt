cmake_minimum_required(VERSION 3.16)

project(RayTracer
    VERSION 0.1.0
    LANGUAGES CXX CUDA)

# C++ config
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA config
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Collect sources
file(GLOB_RECURSE RT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

# Explicitly ensure our new files are included
list(APPEND RT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_scene_builder.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_render.cu"
)

add_executable(ray_tracer ${RT_SOURCES})

# Include headers
target_include_directories(ray_tracer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

# Threads for CPU parts
find_package(Threads REQUIRED)
target_link_libraries(ray_tracer PRIVATE Threads::Threads)

# Enable extended lambdas for nvcc
target_compile_options(ray_tracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

# --- Optional: show CUDA arch build info ---
# (If you want to make sure it's compiling for your GPU)
# set_property(TARGET ray_tracer PROPERTY CUDA_ARCHITECTURES 86)
