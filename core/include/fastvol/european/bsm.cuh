/**
 * @file bsm.cuh
 * @brief CUDA implementation of the Black-Scholes-Merton (BSM) model for European options.
 *
 * @author Valerio Galanti
 * @date 2025
 * @version 0.1.1
 * @license MIT License
 *
 * This header defines the device functions, kernels, and host launchers required
 * to calculate the price of European options using the BSM model on a CUDA-enabled GPU.
 */

#pragma once

#include "fastvol/detail/math.hpp"
#include "fastvol/detail/price.cuh"
#include <cuda_runtime.h>

namespace cm = fastvol::detail::math::cuda;

/**
 * @namespace fastvol::european::bsm::cuda
 * @brief Provides the CUDA implementation for the Black-Scholes-Merton (BSM) model.
 *
 * This namespace contains the device functions, kernels, and host launchers required
 * to calculate the price of European options using the BSM model on a CUDA-enabled GPU.
 * It is optimized for batch processing and leverages the generic pricing helpers from
 * `fastvol::detail::cuda::price`.
 */
namespace fastvol::european::bsm::cuda
{
inline constexpr int    BLOCK_SIZE     = 512;
inline constexpr size_t MAX_BATCH_SIZE = 1'048'576;
inline constexpr size_t MIN_GRAPH_SIZE = 3 * MAX_BATCH_SIZE;

/* price =========================================================================================*/
/* device func -----------------------------------------------------------------------------------*/
/**
 * @brief Device function to calculate the price of a single European option using the BSM formula.
 * @tparam T The floating-point type (float or double).
 * @param S The current price of the underlying asset.
 * @param K The strike price of the option.
 * @param cp The option type flag ('c' for call, 'p' for put).
 * @param ttm The time to maturity of the option.
 * @param iv The implied volatility.
 * @param r The risk-free interest rate.
 * @param q The dividend yield.
 * @return The calculated price of the option.
 */
template <typename T> __forceinline__ __device__ T p_func(T S, T K, char cp, T ttm, T iv, T r, T q)
{
    const T iv_sqrt_t = iv * cm::sqrt(ttm);
    const T d1        = (cm::log(S / K) + (r + T(0.5) * iv * iv) * ttm) / iv_sqrt_t;
    const T d2        = d1 - iv_sqrt_t;
    const T nd1       = cm::norm_cdf(d1);
    const T nd2       = cm::norm_cdf(d2);
    const T S_        = S * cm::exp(-q * ttm);
    const T K_        = K * cm::exp(-r * ttm);
    const T c         = S_ * nd1 - K_ * nd2;
    const T p         = K_ * (T(1.0) - nd2) - S_ * (T(1.0) - nd1);

    return (cp & 1) * c + ((~cp) & 1) * p;
}
/* kernel ----------------------------------------------------------------------------------------*/
/**
 * @brief CUDA kernel for calculating BSM prices for a batch of options.
 *
 * Each thread processes one option.
 * @tparam T The floating-point type (float or double).
 * @param S Device pointer to underlying asset prices.
 * @param K Device pointer to strike prices.
 * @param cp Device pointer to option type flags.
 * @param ttm Device pointer to times to maturity.
 * @param iv Device pointer to implied volatilities.
 * @param r Device pointer to risk-free interest rates.
 * @param q Device pointer to dividend yields.
 * @param n_options The number of options in the batch.
 * @param[out] result Device pointer to store the calculated prices.
 */
template <typename T>
static __global__ void p_kernel(const T *__restrict__ S,
                                const T *__restrict__ K,
                                const char *__restrict__ cp,
                                const T *__restrict__ ttm,
                                const T *__restrict__ iv,
                                const T *__restrict__ r,
                                const T *__restrict__ q,
                                const size_t n_options,
                                T *__restrict__ result)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_options) return;

    result[i] = p_func(S[i], K[i], cp[i], ttm[i], iv[i], r[i], q[i]);
}
/* host func -------------------------------------------------------------------------------------*/
/**
 * @brief Determines an optimal batch size for processing.
 * @param n_options The total number of options.
 * @return The suggested batch size.
 */
static inline size_t optimal_batch_size(size_t n_options)
{
    return std::min(n_options, (n_options <= size_t{1'000'000}) ? size_t{32'768} : size_t{65'536});
}

/**
 * @brief Host function to launch the BSM price calculation on the GPU.
 *
 * This function acts as a dispatcher, selecting the appropriate CUDA launch helper
 * from `fastvol::detail::cuda::price` based on the memory layout of the input data
 * (on-device, pinned, or pageable).
 *
 * @tparam T The floating-point type (float or double).
 * @param S Pointer to the underlying asset prices on the host or device.
 * @param K Pointer to the strike prices on the host or device.
 * @param cp_flag Pointer to the option type flags on the host or device.
 * @param ttm Pointer to the times to maturity on the host or device.
 * @param iv Pointer to the implied volatilities on the host or device.
 * @param r Pointer to the risk-free interest rates on the host or device.
 * @param q Pointer to the dividend yields on the host or device.
 * @param n_options The number of options in the batch.
 * @param[out] result Pointer to the pre-allocated memory for the results on the host or device.
 * @param stream The CUDA stream for asynchronous execution.
 * @param device The ID of the GPU device to use.
 * @param on_device A flag indicating if the input/output data is already on the device.
 * @param is_pinned A flag indicating if the host memory is pinned.
 * @param sync A flag indicating whether to synchronize the stream after the kernel launch.
 */
template <typename T>
void price(const T     *S,
           const T     *K,
           const char  *cp_flag,
           const T     *ttm,
           const T     *iv,
           const T     *r,
           const T     *q,
           size_t       n_options,
           T           *result,
           cudaStream_t stream    = 0,
           int          device    = 0,
           bool         on_device = false,
           bool         is_pinned = false,
           bool         sync      = true)
{
    int    block_size = BLOCK_SIZE;
    size_t shared_mem = 0;

    auto launch_kernel = [shared_mem](int          grid_size,
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
            S, K, cp, ttm, iv, r, q, n_options, result);
    };

    if (on_device)
    {
        int grid_size = (n_options + block_size - 1) / block_size;
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
        size_t batch_size = optimal_batch_size(n_options);
        int    grid_size  = (batch_size + block_size - 1) / block_size;
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
        size_t batch_size = MIN(n_options, MAX_BATCH_SIZE);
        int    grid_size  = (batch_size + block_size - 1) / block_size;
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

} // namespace fastvol::european::bsm::cuda
