/**
 * @file cuda.cuh
 * @brief CUDA-specific helper functions and macros.
 *
 * This file contains low-level utilities for CUDA programming within the fastvol library.
 * It includes macros for error checking, memory allocation (device, host, pinned), and
 * memory copy operations. These helpers simplify CUDA API calls and ensure consistent
 * error handling across the library.
 */

#pragma once

#ifdef FASTVOL_CUDA_ENABLED
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>

namespace fastvol::detail::cuda
{
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* helpers ---------------------------------------------------------------------------------------*/
inline void check_cuda_error(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[fastvol] CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

inline void *alloc_device(size_t bytes)
{
    void *ptr = nullptr;
    check_cuda_error(cudaMalloc(&ptr, bytes), __FILE__, __LINE__);
    return ptr;
}

inline void *alloc_device_zero(size_t bytes)
{
    void *ptr = alloc_device(bytes);
    check_cuda_error(cudaMemset(ptr, 0, bytes), __FILE__, __LINE__);
    return ptr;
}

inline void *alloc_host(size_t bytes)
{
    void *ptr = nullptr;
    check_cuda_error(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault), __FILE__, __LINE__);
    return ptr;
}

inline void free_device(void *ptr)
{
    if (ptr != nullptr)
    {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[fastvol] WARNING: cudaFree failed (%s)\n", cudaGetErrorString(err));
        }
    }
}

inline void free_host(void *ptr)
{
    if (ptr != nullptr)
    {
        check_cuda_error(cudaFreeHost(ptr), __FILE__, __LINE__);
    }
}

inline void
memcpy_async(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream)
{
    check_cuda_error(cudaMemcpyAsync(dst, src, bytes, kind, stream), __FILE__, __LINE__);
}

inline void memcpy_sync(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind)
{
    check_cuda_error(cudaMemcpy(dst, src, bytes, kind), __FILE__, __LINE__);
}
/* price
 * -----------------------------------------------------------------------------------------*/

} // namespace fastvol::detail::cuda

#define CUDA_CHECK(call) ::fastvol::detail::cuda::check_cuda_error((call), __FILE__, __LINE__)

#define CUDA_ALLOC_DEVICE(type, n)                                                                 \
    ((type *)::fastvol::detail::cuda::alloc_device((n) * sizeof(type)))

#define CUDA_ALLOC_DEVICE_ZERO(type, n)                                                            \
    ((type *)::fastvol::detail::cuda::alloc_device_zero((n) * sizeof(type)))

#define CUDA_ALLOC_HOST(type, n) ((type *)::fastvol::detail::cuda::alloc_host((n) * sizeof(type)))

#define CUDA_FREE(ptr) ::fastvol::detail::cuda::free_device(ptr)

#define CUDA_FREE_HOST(ptr) ::fastvol::detail::cuda::free_host(ptr)

#define CUDA_MEMCPY_ASYNC(dst, src, bytes, dir, stream)                                            \
    ::fastvol::detail::cuda::memcpy_async(dst, src, bytes, dir, stream)

#define CUDA_MEMCPY(dst, src, bytes, dir) ::fastvol::detail::cuda::memcpy_sync(dst, src, bytes, dir)

#endif