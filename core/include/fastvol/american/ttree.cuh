/**
 * @file ttree.cuh
 * @brief CUDA implementation of the Trinomial Tree model for American options.
 *
 * @author Valerio Galanti
 * @date 2025
 * @version 0.1.0
 * @license MIT License
 *
 * This header defines the device functions, kernels, and host launchers required
 * to calculate the price of American options using a trinomial tree on a CUDA-enabled GPU.
 * It is optimized for batch processing and leverages shared memory and cooperative groups.
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
 * @namespace fastvol::american::ttree::cuda
 * @brief Provides the CUDA implementation for the Trinomial Tree model for American options.
 *
 * This namespace contains the device functions, kernels, and host launchers required
 * to calculate the price of American options using a trinomial tree on a CUDA-enabled GPU.
 * It is optimized for batch processing and leverages shared memory and cooperative groups.
 */
namespace fastvol::american::ttree::cuda
{

inline constexpr size_t BATCH_SIZE     = 32'768;
inline constexpr size_t MAX_BATCH_SIZE = 1'048'576;
inline constexpr size_t MIN_GRAPH_SIZE = 3 * MAX_BATCH_SIZE;

/* price =========================================================================================*/
/* device func -----------------------------------------------------------------------------------*/
/**
 * @brief Device function to calculate the price of a single American option using a trinomial tree.
 * @tparam T The floating-point type (float or double).
 * @param S The current price of the underlying asset.
 * @param K The strike price of the option.
 * @param cp The option type flag ('c' for call, 'p' for put).
 * @param ttm The time to maturity of the option.
 * @param iv The implied volatility.
 * @param r The risk-free interest rate.
 * @param q The dividend yield.
 * @param n_steps The number of steps in the trinomial tree.
 * @param v_prev Shared memory buffer for the previous step's option values.
 * @param v_curr Shared memory buffer for the current step's option values.
 * @param ex Shared memory buffer for the pre-calculated exercise values at each node.
 * @return The calculated price of the option.
 */
template <typename T>
__forceinline__ __device__ T
p_func(T S, T K, char cp, T ttm, T iv, T r, T q, int n_steps, T *v_prev, T *v_curr, T *ex)
{
    cg::thread_block block = cg::this_thread_block();
    const int        tid   = threadIdx.x;

    // compute ttree consts
    const T dt   = ttm / n_steps;
    const T nu   = r - q - T(0.5) * iv * iv;
    const T dx   = iv * cm::sqrt(T(3.0) * dt);
    const T disc = cm::exp(-r * dt);

    const T u = cm::exp(dx);
    const T p = nu * cm::sqrt(dt) / (T(2.0) * iv * cm::sqrt(T(3.0)));

    const T pu = T(1.0) / T(6.0) + p;
    const T pm = T(2.0) / T(3.0);
    const T pd = T(1.0) / T(6.0) - p;

    const T dpu = disc * pu;
    const T dpm = disc * pm;
    const T dpd = disc * pd;

    const T log_u = cm::log(u);
    const T lsu   = cm::fma(-T(n_steps), log_u, cm::log(S));
    const T s     = T(2.0) * (cp & 1) - T(1.0);
    const T sK    = K * -s;

    // precompute exercise values w. tiling
    for (int j = tid; j < 2 * n_steps + 1; j += blockDim.x)
    {
        ex[j]     = cm::fmax(cm::fma(s, cm::exp(cm::fma(T(j), log_u, lsu)), sK), T(0.0));
        v_prev[j] = ex[j]; // init fill v
    }
    block.sync();

    // backtrack
    for (int i = 1; i <= n_steps; ++i)
    {
        for (int j = tid; j <= 2 * n_steps + 1 - 2 * i; j += blockDim.x)
        {
            const T held = dpu * v_prev[j + 2] + dpm * v_prev[j + 1] + dpd * v_prev[j];
            v_curr[j]    = cm::fmax(held, ex[i + j]);
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
 * @brief CUDA kernel for calculating trinomial tree prices for a batch of options.
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
 * @param n_steps The number of steps in the trinomial tree.
 * @param[out] result Device pointer to store the calculated prices.
 */
template <typename T>
static __global__ __launch_bounds__(256) void p_kernel(const T *__restrict__ S,
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
    T                     *v_curr = v_prev + (2 * n_steps + 1);
    T                     *ex     = v_curr + (2 * n_steps + 1);

    T price = p_func(S[i], K[i], cp[i], ttm[i], iv[i], r[i], q[i], n_steps, v_prev, v_curr, ex);

    if (tid == 0) result[i] = price;
}

/* host func -------------------------------------------------------------------------------------*/
/**
 * @brief Determines the optimal number of threads per block for the trinomial tree kernel.
 * @tparam T The floating-point type (float or double).
 * @param n_steps The number of steps in the tree.
 * @return The optimal number of threads (currently fixed at 256).
 */
template <typename T> static inline int optimal_threads_per_option(int n_steps);

template <> inline int optimal_threads_per_option<float>(int n_steps) { return 256; }

template <> inline int optimal_threads_per_option<double>(int n_steps) { return 256; }

/**
 * @brief Host function to launch the trinomial tree price calculation on the GPU.
 *
 * This function handles memory management and kernel launching, dispatching to the
 * appropriate helper from `fastvol::detail::cuda::price` based on the data's memory layout.
 *
 * @tparam T The floating-point type (float or double).
 * @param S Pointer to the underlying asset prices on the host or device.
 * @param K Pointer to the strike prices on the host or device.
 * @param cp_flag Pointer to the option type flags on the host or device.
 * @param ttm Pointer to the times to maturity on the host or device.
 * @param iv Pointer to the implied volatilities on the host or device.
 * @param r Pointer to the risk-free interest rates on the host or device.
 * @param q Pointer to the dividend yields on the host or device.
 * @param n_steps The number of steps in the trinomial tree.
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
    size_t shared_mem = 3 * (2 * n_steps + 1) * sizeof(T);

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
    else if (is_pinned && n_options > MIN_BATCH_SIZE)
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

} // namespace fastvol::american::ttree::cuda
