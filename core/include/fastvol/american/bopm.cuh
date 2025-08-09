/**
 * @file bopm.cuh
 * @brief CUDA implementation of the Binomial Options Pricing Model (BOPM) for American options.
 *
 * @author Valerio Galanti
 * @date 2025
 * @version 0.1.0
 * @license MIT License
 *
 * This header defines the device functions, kernels, and host launchers required
 * to calculate the price of American options using the BOPM on a CUDA-enabled GPU.
 * It is optimized for batch processing and leverages shared memory and cooperative groups
 * for high performance.
 */

#pragma once

#include "fastvol/detail/cuda.cuh"
#include "fastvol/detail/math.hpp"
#include "fastvol/detail/price.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;
namespace cm = fastvol::detail::math::cuda;

/**
 * @namespace fastvol::american::bopm::cuda
 * @brief Provides the CUDA implementation for the Binomial Options Pricing Model (BOPM).
 *
 * This namespace contains the device functions, kernels, and host launchers required
 * to calculate the price of American options using the BOPM on a CUDA-enabled GPU.
 * It is optimized for batch processing and leverages shared memory and cooperative groups
 * for high performance.
 */
namespace fastvol::american::bopm::cuda
{

inline constexpr size_t BATCH_SIZE     = 32'768;
inline constexpr size_t MAX_BATCH_SIZE = 1'048'576;
inline constexpr size_t MIN_GRAPH_SIZE = 3 * MAX_BATCH_SIZE;

/* price =========================================================================================*/
/* device func -----------------------------------------------------------------------------------*/
/**
 * @brief Device function to calculate the price of a single American option using BOPM.
 * @tparam T The floating-point type (float or double).
 * @param S The current price of the underlying asset.
 * @param K The strike price of the option.
 * @param cp The option type flag ('c' for call, 'p' for put).
 * @param ttm The time to maturity of the option.
 * @param iv The implied volatility.
 * @param r The risk-free interest rate.
 * @param q The dividend yield.
 * @param n_steps The number of steps in the binomial tree.
 * @param v_prev Shared memory buffer for the previous step's option values.
 * @param v_curr Shared memory buffer for the current step's option values.
 * @param pe Shared memory buffer for payoff/exercise values at even steps.
 * @param po Shared memory buffer for payoff/exercise values at odd steps.
 * @return The calculated price of the option.
 */
template <typename T>
__forceinline__ __device__ T
p_func(T S, T K, char cp, T ttm, T iv, T r, T q, int n_steps, T *v_prev, T *v_curr, T *pe, T *po)
{
    cg::thread_block block = cg::this_thread_block();
    const int        tid   = threadIdx.x;

    // compute bopm consts
    const T dt    = ttm / n_steps;
    const T u     = cm::exp(iv * cm::sqrt(dt));
    const T d     = T(1.0) / u;
    const T p     = (cm::exp((r - q) * dt) - d) / (u - d);
    const T disc  = cm::exp(-r * dt);
    const T dp    = disc * p;
    const T d1p   = disc * (T(1.0) - p);
    const T log_u = cm::log(u);
    const T lsu   = cm::fma(-T(n_steps), log_u, cm::log(S));
    const T s     = T(2.0) * (cp & 1) - T(1.0);
    const T sK    = K * -s;

    // precompute exercise values w. tiling
    for (int j = tid; j <= n_steps; j += blockDim.x)
    {
        pe[j]     = cm::fmax(cm::fma(s, cm::exp(cm::fma(T(2 * j), log_u, lsu)), sK), T(0.0));
        po[j]     = cm::fmax(cm::fma(s, cm::exp(cm::fma(T(2 * j + 1), log_u, lsu)), sK), T(0.0));
        v_prev[j] = pe[j]; // init fill v
    }
    block.sync();

    // backtrack
    for (int i = 1; i <= n_steps; ++i)
    {
        const T  *ex = (i & 1) ? po : pe; // which ex arr
        const int o  = i >> 1;            // offset within ex

        for (int j = tid; j <= n_steps - i; j += blockDim.x)
        {
            const T held = cm::fma(dp, v_prev[j + 1], d1p * v_prev[j]);
            v_curr[j]    = cm::fmax(held, ex[o + j]);
        }
        block.sync();

        // swap buffers
        T *tmp = v_prev;
        v_prev = v_curr;
        v_curr = tmp;
    }

    // all threads return same value
    return v_prev[0];
}

/* kernel ----------------------------------------------------------------------------------------*/
/**
 * @brief CUDA kernel for calculating BOPM prices for a batch of options.
 *
 * Each block processes one option. Threads within the block cooperate to calculate the price.
 * @tparam T The floating-point type (float or double).
 * @param S Device pointer to underlying asset prices.
 * @param K Device pointer to strike prices.
 * @param cp Device pointer to option type flags.
 * @param ttm Device pointer to times to maturity.
 * @param iv Device pointer to implied volatilities.
 * @param r Device pointer to risk-free interest rates.
 * @param q Device pointer to dividend yields.
 * @param n_steps The number of steps in the binomial tree.
 * @param[out] result Device pointer to store the calculated prices.
 */
template <typename T>
static __global__ __launch_bounds__(512) void p_kernel(const T *__restrict__ S,
                                                       const T *__restrict__ K,
                                                       const char *__restrict__ cp,
                                                       const T *__restrict__ ttm,
                                                       const T *__restrict__ iv,
                                                       const T *__restrict__ r,
                                                       const T *__restrict__ q,
                                                       int n_steps,
                                                       T *__restrict__ result)
{
    const int i   = blockIdx.x;
    const int tid = threadIdx.x;

    // shared memory setup
    extern __shared__ char raw[];
    T                     *shared = reinterpret_cast<T *>(raw);
    T                     *v_prev = shared;
    T                     *v_curr = v_prev + (n_steps + 1);
    T                     *pe     = v_curr + (n_steps + 1);
    T                     *po     = pe + (n_steps + 1);

    T price = p_func(S[i], K[i], cp[i], ttm[i], iv[i], r[i], q[i], n_steps, v_prev, v_curr, pe, po);

    if (tid == 0) result[i] = price;
}

/* host func -------------------------------------------------------------------------------------*/
/**
 * @brief Determines the optimal number of threads per block for a given number of steps.
 * @tparam T The floating-point type (float or double).
 * @param n_steps The number of steps in the binomial tree.
 * @return The optimal number of threads.
 */
template <typename T> static inline int optimal_threads_per_option(int n_steps);

template <> inline int optimal_threads_per_option<float>(int n_steps)
{
    if (n_steps <= 512)
        return 64;
    else if (n_steps <= 1024)
        return 128;
    else if (n_steps <= 2048)
        return 256;
    else
        return 512;
}

template <> inline int optimal_threads_per_option<double>(int n_steps)
{
    if (n_steps <= 256)
        return 64;
    else if (n_steps <= 512)
        return 128;
    else if (n_steps <= 1024)
        return 256;
    else
        return 512;
}

/**
 * @brief Host function to launch the BOPM price calculation on the GPU.
 *
 * This function handles memory management (device allocation, data transfer) and kernel
 * launching. It supports different data layouts: data already on device, pinned host memory
 * (using CUDA graphs for efficiency), or standard pageable host memory.
 *
 * @tparam T The floating-point type (float or double).
 * @param S Pointer to the underlying asset prices on the host or device.
 * @param K Pointer to the strike prices on the host or device.
 * @param cp_flag Pointer to the option type flags on the host or device.
 * @param ttm Pointer to the times to maturity on the host or device.
 * @param iv Pointer to the implied volatilities on the host or device.
 * @param r Pointer to the risk-free interest rates on the host or device.
 * @param q Pointer to the dividend yields on the host or device.
 * @param n_steps The number of steps in the binomial tree.
 * @param n_options The number of options in the batch.
 * @param[out] result Pointer to the pre-allocated memory for the results on the host or device.
 * @param stream The CUDA stream for asynchronous execution.
 * @param device The ID of the GPU device to use.
 * @param on_device A flag indicating if the input/output data is already on the device.
 * @param is_pinned A flag indicating if the host memory is pinned.
 * @param sync A flag indicating whether to synchronize the stream after the kernel launch.
 */
template <typename T>
void price(const T *__restrict__ S,
           const T *__restrict__ K,
           const char *__restrict__ cp_flag,
           const T *__restrict__ ttm,
           const T *__restrict__ iv,
           const T *__restrict__ r,
           const T *__restrict__ q,
           int    n_steps,
           size_t n_options,
           T *__restrict__ result,
           cudaStream_t stream    = 0,
           int          device    = 0,
           bool         on_device = false,
           bool         is_pinned = false,
           bool         sync      = true)
{
    int    block_size = optimal_threads_per_option<T>(n_steps);
    size_t shared_mem = 4 * (n_steps + 1) * sizeof(T);

    // set maximum shared memory
    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (shared_mem > static_cast<size_t>(max_shared_mem))
    {
        fprintf(stderr,
                "[fastvol] ERROR: n_steps=%d requires %zu bytes shared memory, but max is %d\n",
                n_steps,
                shared_mem,
                max_shared_mem);
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        p_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem));

    auto launch_kernel = [shared_mem, n_steps](int          grid_size,
                                               int          block_size,
                                               cudaStream_t stream,
                                               const T     *S,
                                               const T     *K,
                                               const char  *cp,
                                               const T     *ttm,
                                               const T     *iv,
                                               const T     *r,
                                               const T     *q,
                                               size_t       n_options,
                                               T           *result,
                                               T           *delta,
                                               T           *gamma,
                                               T           *theta)
    {
        (void)delta;
        (void)gamma;
        (void)theta;
        p_kernel<T><<<grid_size, block_size, shared_mem, stream>>>(
            S, K, cp, ttm, iv, r, q, n_steps, result);
    };

    if (on_device)
    {
        int grid_size = static_cast<int>(n_options);
        fastvol::detail::cuda::price::device(launch_kernel,
                                             grid_size,
                                             block_size,
                                             stream,
                                             device,
                                             S,
                                             K,
                                             cp_flag,
                                             ttm,
                                             iv,
                                             r,
                                             q,
                                             n_options,
                                             result,
                                             sync);
    }
    else if (is_pinned && n_options > MIN_GRAPH_SIZE)
    {
        size_t batch_size = BATCH_SIZE;
        int    grid_size  = static_cast<int>(batch_size);
        fastvol::detail::cuda::price::graph(launch_kernel,
                                            batch_size,
                                            grid_size,
                                            block_size,
                                            stream,
                                            device,
                                            S,
                                            K,
                                            cp_flag,
                                            ttm,
                                            iv,
                                            r,
                                            q,
                                            n_options,
                                            result,
                                            sync);
    }
    else
    {
        size_t batch_size = std::min(n_options, MAX_BATCH_SIZE);
        int    grid_size  = static_cast<int>(batch_size);
        fastvol::detail::cuda::price::batch(launch_kernel,
                                            batch_size,
                                            grid_size,
                                            block_size,
                                            stream,
                                            device,
                                            S,
                                            K,
                                            cp_flag,
                                            ttm,
                                            iv,
                                            r,
                                            q,
                                            n_options,
                                            result,
                                            sync);
    }
}
/* iv ============================================================================================*/

} // namespace fastvol::american::bopm::cuda