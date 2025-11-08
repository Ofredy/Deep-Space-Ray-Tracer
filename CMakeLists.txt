cmake_minimum_required(VERSION 3.16)

project(RayTracer
    VERSION 0.1.0
    LANGUAGES CXX CUDA)

# -----------------------------
# C++ / CUDA language settings
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ---------------------------------------
# Make MSVC use the DLL runtime (/MD[/d])
# Also make CUDA use the shared runtime
# ---------------------------------------
if(MSVC)
  # CMake 3.15+ way (preferred)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
# Ensure CUDA uses the shared runtime too
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# -----------------------------
# Sources
# -----------------------------
file(GLOB_RECURSE RT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

# Ensure the stb implementation TU is compiled exactly once
# (Create this file if you haven't: src/stb_image_impl.cpp with
#  #define STB_IMAGE_IMPLEMENTATION then #include "stb_image.h")
list(APPEND RT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/stb_image_impl.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_scene_builder.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_render.cu"
)

add_executable(ray_tracer ${RT_SOURCES})

# -----------------------------
# Include directories
# -----------------------------
target_include_directories(ray_tracer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

# -----------------------------
# Libraries
# -----------------------------
find_package(Threads REQUIRED)
target_link_libraries(ray_tracer PRIVATE Threads::Threads)

# -----------------------------
# Compiler flags / options
# -----------------------------
# Enable extended lambdas for nvcc
target_compile_options(ray_tracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

# If you're on an older CMake and the MSVC runtime setting above is ignored,
# uncomment this fallback to force /MD (/MDd for Debug):
# if(MSVC)
#   target_compile_options(ray_tracer PRIVATE
#     $<$<CONFIG:Debug>:/MDd>
#     $<$<CONFIG:Release>:/MD>
#     $<$<CONFIG:RelWithDebInfo>:/MD>
#     $<$<CONFIG:MinSizeRel>:/MD>
#   )
# endif()

# -----------------------------
# (Optional) target GPU archs
# -----------------------------
# set_property(TARGET ray_tracer PROPERTY CUDA_ARCHITECTURES 86)
