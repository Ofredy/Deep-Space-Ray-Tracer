#ifndef CUDA_COMPAT_H
#define CUDA_COMPAT_H

// If we're compiling with NVCC (CUDA compiler), __CUDACC__ is defined.
// If we're just compiling normal C++ with MSVC/Clang/etc., it's not.

#ifdef __CUDACC__
    #define CUDA_HD __host__ __device__
    #define CUDA_D  __device__
    #define CUDA_H  __host__
#else
    // On the CPU build, these should disappear so MSVC doesn't complain.
    #define CUDA_HD
    #define CUDA_D
    #define CUDA_H
#endif

#endif // CUDA_COMPAT_H