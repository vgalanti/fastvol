/**
 * @file price.cuh
 * @brief CUDA kernel launch helpers for pricing functions.
 *
 * This file provides a set of templated helper functions for launching CUDA pricing kernels.
 * It abstracts the logic for handling different memory types (device, pinned, pageable)
 * and execution strategies (e.g., using CUDA graphs) to simplify the main pricing code.
 */

#pragma once

#include "fastvol/detail/cuda.cuh"
#include <cstddef>
#include <cuda_runtime.h>

/**
 * @namespace fastvol::detail::cuda::price
 * @brief Provides host-side helper functions for launching CUDA pricing kernels.
 *
 * This namespace contains templated functions that abstract the complexities of
 * CUDA memory management and kernel launching for different scenarios:
 * 1. `device`: For data already residing on the GPU.
 * 2. `graph`: For pinned host memory, using CUDA graphs for efficient repeated execution.
 * 3. `batch`: For standard pageable host memory, processing data in batches.
 *
 * These helpers ensure that option pricing models can be executed efficiently on the GPU
 * regardless of the initial location and type of the input data.
 */
namespace fastvol::detail::cuda::price
{
/**
 * @brief Launches a pricing kernel when all data is already on the device.
 *
 * This function is a simple wrapper that sets the device, creates a stream if necessary,
 * launches the provided kernel, and synchronizes. It is used when no data transfer
 * between host and device is needed.
 *
 * @tparam T The floating-point type (e.g., float, double).
 * @tparam LaunchFunc The type of the callable kernel launch object.
 * @param launch_kernel A callable object (e.g., a lambda) that launches the CUDA kernel.
 * @param grid_size The grid size for the kernel launch.
 * @param block_size The block size for the kernel launch.
 * @param stream The CUDA stream for execution. A new stream is created if this is 0.
 * @param device The GPU device ID to execute on.
 * @param S Device pointer to underlying asset prices.
 * @param K Device pointer to strike prices.
 * @param cp_flag Device pointer to option type flags.
 * @param ttm Device pointer to times to maturity.
 * @param iv Device pointer to implied volatilities.
 * @param r Device pointer to risk-free rates.
 * @param q Device pointer to dividend yields.
 * @param n_options The total number of options to process.
 * @param[out] result Device pointer to store the output prices.
 * @param sync If true, synchronizes the CUDA stream after the launch.
 * @param[out] delta (Optional) Device pointer for delta results.
 * @param[out] gamma (Optional) Device pointer for gamma results.
 * @param[out] theta (Optional) Device pointer for theta results.
 */
template <typename T, typename LaunchFunc>
void device(LaunchFunc   launch_kernel,
            int          grid_size,
            int          block_size,
            cudaStream_t stream,
            int          device,
            const T     *S,
            const T     *K,
            const char  *cp_flag,
            const T     *ttm,
            const T     *iv,
            const T     *r,
            const T     *q,
            size_t       n_options,
            T           *result,
            bool         sync,
            T           *delta = nullptr,
            T           *gamma = nullptr,
            T           *theta = nullptr)
{
    if (n_options <= 0) return;
    if (device >= 0) CUDA_CHECK(cudaSetDevice(device));

    bool stream_created = (stream == 0);
    if (stream_created) CUDA_CHECK(cudaStreamCreate(&stream));

    launch_kernel(grid_size,
                  block_size,
                  stream,
                  S,
                  K,
                  cp_flag,
                  ttm,
                  iv,
                  r,
                  q,
                  n_options,
                  result,
                  delta,
                  gamma,
                  theta);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[fastvol] Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (sync) CUDA_CHECK(cudaStreamSynchronize(stream));
}

/**
 * @brief Launches a pricing kernel using CUDA graphs for data in pinned host memory.
 *
 * This function is optimized for scenarios where the same kernel is launched many times
 * with different data of the same size. It uses CUDA graphs to capture the kernel launch
 * and memory copy operations, minimizing launch overhead. Data is processed in batches.
 *
 * @tparam T The floating-point type (e.g., float, double).
 * @tparam LaunchFunc The type of the callable kernel launch object.
 * @param launch_kernel A callable object that launches the CUDA kernel.
 * @param batch_size The number of options to process in a single batch.
 * @param grid_size The grid size for the kernel launch (typically same as batch_size).
 * @param block_size The block size for the kernel launch.
 * @param stream The CUDA stream for execution.
 * @param device The GPU device ID.
 * @param S Host pointer to underlying asset prices (must be pinned memory).
 * @param K Host pointer to strike prices (must be pinned memory).
 * @param cp_flag Host pointer to option type flags (must be pinned memory).
 * @param ttm Host pointer to times to maturity (must be pinned memory).
 * @param iv Host pointer to implied volatilities (must be pinned memory).
 * @param r Host pointer to risk-free rates (must be pinned memory).
 * @param q Host pointer to dividend yields (must be pinned memory).
 * @param n_options The total number of options to process.
 * @param[out] result Host pointer for the output prices (must be pinned memory).
 * @param sync If true, synchronizes the CUDA stream after all batches are processed.
 * @param[out] delta (Optional) Host pointer for delta results.
 * @param[out] gamma (Optional) Host pointer for gamma results.
 * @param[out] theta (Optional) Host pointer for theta results.
 */
template <typename T, typename LaunchFunc>
void graph(LaunchFunc   launch_kernel,
           size_t       batch_size,
           int          grid_size,
           int          block_size,
           cudaStream_t stream,
           int          device,
           const T     *S,
           const T     *K,
           const char  *cp_flag,
           const T     *ttm,
           const T     *iv,
           const T     *r,
           const T     *q,
           size_t       n_options,
           T           *result,
           bool         sync,
           T           *delta = nullptr,
           T           *gamma = nullptr,
           T           *theta = nullptr)
{
    if (n_options <= 0) return;
    if (device >= 0) CUDA_CHECK(cudaSetDevice(device));

    bool stream_created = (stream == 0);
    if (stream_created) CUDA_CHECK(cudaStreamCreate(&stream));

    size_t n_batches = (n_options + batch_size - 1) / batch_size;

    // allocate device arrays
    T    *d_S     = CUDA_ALLOC_DEVICE(T, batch_size);
    T    *d_K     = CUDA_ALLOC_DEVICE(T, batch_size);
    char *d_cp    = CUDA_ALLOC_DEVICE(char, batch_size);
    T    *d_ttm   = CUDA_ALLOC_DEVICE(T, batch_size);
    T    *d_iv    = CUDA_ALLOC_DEVICE(T, batch_size);
    T    *d_r     = CUDA_ALLOC_DEVICE(T, batch_size);
    T    *d_q     = CUDA_ALLOC_DEVICE(T, batch_size);
    T    *d_res   = CUDA_ALLOC_DEVICE(T, batch_size);
    T    *d_delta = delta ? CUDA_ALLOC_DEVICE(T, batch_size) : nullptr;
    T    *d_gamma = gamma ? CUDA_ALLOC_DEVICE(T, batch_size) : nullptr;
    T    *d_theta = theta ? CUDA_ALLOC_DEVICE(T, batch_size) : nullptr;

    // create graph
    cudaGraph_t     graph;
    cudaGraphExec_t graph_exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_kernel(grid_size,
                  block_size,
                  stream,
                  d_S,
                  d_K,
                  d_cp,
                  d_ttm,
                  d_iv,
                  d_r,
                  d_q,
                  batch_size,
                  d_res,
                  d_gamma,
                  d_delta,
                  d_theta);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    // batch over options
    for (size_t b = 0; b < n_batches; b++)
    {
        const size_t b_o  = b * batch_size;                   // batch offset
        const size_t b_n  = MIN(batch_size, n_options - b_o); // batch size/n
        const size_t b_s  = b_n * sizeof(T);                  // batch data size
        const size_t b_sc = b_n * sizeof(char);               // batch char size

        // async copy to device
        CUDA_MEMCPY_ASYNC(d_S, S + b_o, b_s, cudaMemcpyHostToDevice, stream);
        CUDA_MEMCPY_ASYNC(d_K, K + b_o, b_s, cudaMemcpyHostToDevice, stream);
        CUDA_MEMCPY_ASYNC(d_cp, cp_flag + b_o, b_sc, cudaMemcpyHostToDevice, stream);
        CUDA_MEMCPY_ASYNC(d_ttm, ttm + b_o, b_s, cudaMemcpyHostToDevice, stream);
        CUDA_MEMCPY_ASYNC(d_iv, iv + b_o, b_s, cudaMemcpyHostToDevice, stream);
        CUDA_MEMCPY_ASYNC(d_r, r + b_o, b_s, cudaMemcpyHostToDevice, stream);
        CUDA_MEMCPY_ASYNC(d_q, q + b_o, b_s, cudaMemcpyHostToDevice, stream);

        // launch graph
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));

        // copy res device -> host
        CUDA_MEMCPY_ASYNC(result + b_o, d_res, b_s, cudaMemcpyDeviceToHost, stream);
        if (delta) CUDA_MEMCPY_ASYNC(delta + b_o, d_delta, b_s, cudaMemcpyDeviceToHost, stream);
        if (gamma) CUDA_MEMCPY_ASYNC(gamma + b_o, d_gamma, b_s, cudaMemcpyDeviceToHost, stream);
        if (theta) CUDA_MEMCPY_ASYNC(theta + b_o, d_theta, b_s, cudaMemcpyDeviceToHost, stream);
    }

    // cleanup
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));

    if (stream_created)
        CUDA_CHECK(cudaStreamDestroy(stream));
    else if (sync)
        CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_FREE(d_S);
    CUDA_FREE(d_K);
    CUDA_FREE(d_cp);
    CUDA_FREE(d_ttm);
    CUDA_FREE(d_iv);
    CUDA_FREE(d_r);
    CUDA_FREE(d_q);
    CUDA_FREE(d_res);
    if (delta) CUDA_FREE(d_delta);
    if (gamma) CUDA_FREE(d_gamma);
    if (theta) CUDA_FREE(d_theta);
}

/**
 * @brief Launches a pricing kernel for standard pageable host memory.
 *
 * This function handles data from standard host memory by allocating device memory,
 * copying data in batches, launching the kernel for each batch, and copying the
 * results back. This is the most general-purpose approach but may have higher
 * overhead than the `graph` or `device` versions.
 *
 * @tparam T The floating-point type (e.g., float, double).
 * @tparam LaunchFunc The type of the callable kernel launch object.
 * @param launch_kernel A callable object that launches the CUDA kernel.
 * @param batch_size The number of options to process in a single batch.
 * @param grid_size The grid size for the kernel launch.
 * @param block_size The block size for the kernel launch.
 * @param stream The CUDA stream for execution.
 * @param device The GPU device ID.
 * @param S Host pointer to underlying asset prices.
 * @param K Host pointer to strike prices.
 * @param cp_flag Host pointer to option type flags.
 * @param ttm Host pointer to times to maturity.
 * @param iv Host pointer to implied volatilities.
 * @param r Host pointer to risk-free rates.
 * @param q Host pointer to dividend yields.
 * @param n_options The total number of options to process.
 * @param[out] result Host pointer for the output prices.
 * @param sync If true, synchronizes the CUDA stream after all batches are processed.
 * @param[out] delta (Optional) Host pointer for delta results.
 * @param[out] gamma (Optional) Host pointer for gamma results.
 * @param[out] theta (Optional) Host pointer for theta results.
 */
template <typename T, typename LaunchFunc>
void batch(LaunchFunc   launch_kernel,
           size_t       batch_size,
           int          grid_size,
           int          block_size,
           cudaStream_t stream,
           int          device,
           const T     *S,
           const T     *K,
           const char  *cp_flag,
           const T     *ttm,
           const T     *iv,
           const T     *r,
           const T     *q,
           size_t       n_options,
           T           *result,
           bool         sync,
           T           *delta = nullptr,
           T           *gamma = nullptr,
           T           *theta = nullptr)
{

    if (n_options <= 0) return;
    if (device >= 0) CUDA_CHECK(cudaSetDevice(device));

    bool stream_created = (stream == 0);
    if (stream_created) CUDA_CHECK(cudaStreamCreate(&stream));

    const size_t n_batches = (n_options + batch_size - 1) / batch_size;

    // allocate device arrays
    T    *d_S     = CUDA_ALLOC_DEVICE(T, batch_size);
    T    *d_K     = CUDA_ALLOC_DEVICE(T, batch_size);
    char *d_cp    = CUDA_ALLOC_DEVICE(char, batch_size);
    T    *d_ttm   = CUDA_ALLOC_DEVICE(T, batch_size);
    T    *d_iv    = CUDA_ALLOC_DEVICE(T, batch_size);
    T    *d_r     = CUDA_ALLOC_DEVICE(T, batch_size);
    T    *d_q     = CUDA_ALLOC_DEVICE(T, batch_size);
    T    *d_res   = CUDA_ALLOC_DEVICE(T, batch_size);
    T    *d_delta = delta ? CUDA_ALLOC_DEVICE(T, batch_size) : nullptr;
    T    *d_gamma = gamma ? CUDA_ALLOC_DEVICE(T, batch_size) : nullptr;
    T    *d_theta = theta ? CUDA_ALLOC_DEVICE(T, batch_size) : nullptr;

    // batch over options
    for (size_t b = 0; b < n_batches; b++)
    {
        const size_t b_o  = b * batch_size;                   // batch offset
        const size_t b_n  = MIN(batch_size, n_options - b_o); // batch size/n
        const size_t b_s  = b_n * sizeof(T);                  // batch data size
        const size_t b_sc = b_n * sizeof(char);               // batch char size

        // copy to device
        CUDA_MEMCPY(d_S, S + b_o, b_s, cudaMemcpyHostToDevice);
        CUDA_MEMCPY(d_K, K + b_o, b_s, cudaMemcpyHostToDevice);
        CUDA_MEMCPY(d_cp, cp_flag + b_o, b_sc, cudaMemcpyHostToDevice);
        CUDA_MEMCPY(d_ttm, ttm + b_o, b_s, cudaMemcpyHostToDevice);
        CUDA_MEMCPY(d_iv, iv + b_o, b_s, cudaMemcpyHostToDevice);
        CUDA_MEMCPY(d_r, r + b_o, b_s, cudaMemcpyHostToDevice);
        CUDA_MEMCPY(d_q, q + b_o, b_s, cudaMemcpyHostToDevice);

        // launch kernel
        launch_kernel(grid_size,
                      block_size,
                      stream,
                      d_S,
                      d_K,
                      d_cp,
                      d_ttm,
                      d_iv,
                      d_r,
                      d_q,
                      b_n,
                      d_res,
                      d_gamma,
                      d_delta,
                      d_theta);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[fastvol] Kernel launch failed: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // copy res device -> host
        CUDA_MEMCPY(result + b_o, d_res, b_s, cudaMemcpyDeviceToHost);
        if (delta) CUDA_MEMCPY(delta + b_o, d_delta, b_s, cudaMemcpyDeviceToHost);
        if (gamma) CUDA_MEMCPY(gamma + b_o, d_gamma, b_s, cudaMemcpyDeviceToHost);
        if (theta) CUDA_MEMCPY(theta + b_o, d_theta, b_s, cudaMemcpyDeviceToHost);
    }

    // cleanup
    if (sync) CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_FREE(d_S);
    CUDA_FREE(d_K);
    CUDA_FREE(d_cp);
    CUDA_FREE(d_ttm);
    CUDA_FREE(d_iv);
    CUDA_FREE(d_r);
    CUDA_FREE(d_q);
    CUDA_FREE(d_res);
    if (delta) CUDA_FREE(d_delta);
    if (gamma) CUDA_FREE(d_gamma);
    if (theta) CUDA_FREE(d_theta);
}
} // namespace fastvol::detail::cuda::price