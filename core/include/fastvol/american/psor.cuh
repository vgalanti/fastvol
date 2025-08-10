/**
 * @file psor.cuh
 * @brief CUDA implementation of the Projected Successive Over-Relaxation (PSOR) method for American
 * options.
 *
 * @author Valerio Galanti
 * @date 2025
 * @version 0.1.1
 * @license MIT License
 *
 * This header defines the device functions, kernels, and host launchers for pricing
 * American options using a finite difference grid solved with the RB-PSOR method on a CUDA-enabled
 * GPU. The implementation uses a Red-Black PSOR scheme for parallelization and supports
 * adaptive relaxation parameter adjustment.
 */

#pragma once

#include "fastvol/detail/cuda.cuh"
#include "fastvol/detail/math.hpp"
#include "fastvol/detail/price.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;
namespace cm = fastvol::detail::math::cuda;

/**
 * @namespace fastvol::american::psor::cuda
 * @brief Provides the CUDA implementation for the Projected Successive Over-Relaxation (
PSOR)
 * method.
 *
 * This namespace contains the device functions, kernels, and host launchers for pricing
 * American options using a finite difference grid solved with the PSOR method on a CUDA-enabled
 * GPU. The implementation uses a Red-Black Gauss-Seidel scheme for parallelization and supports
 * adaptive relaxation parameter adjustment.
 */
namespace fastvol::american::psor::cuda
{
inline constexpr int    BLOCK_SIZE     = 256;
inline constexpr size_t BATCH_SIZE     = 32'768;
inline constexpr size_t MAX_BATCH_SIZE = 1'048'576;
inline constexpr size_t MIN_GRAPH_SIZE = 3 * MAX_BATCH_SIZE;

/* utility functions ============================================================================*/
/**
 * @brief Determines the optimal number of threads per block for the PSOR kernel.
 * @param n_s The number of grid points for the underlying asset price.
 * @return The optimal number of threads.
 */
static inline int optimal_threads_per_option(int n_s)
{
    if (n_s <= 256) return 32;
    if (n_s <= 512) return 64;
    if (n_s <= 1024) return 96;
    return 128;
}

/**
 * @brief Interpolates the option value on the Red-Black grid.
 * @tparam T The floating-point type.
 * @param S The underlying asset price at which to interpolate.
 * @param x_min The minimum value of the log-price grid.
 * @param dx The step size of the log-price grid.
 * @param n The number of interior grid points.
 * @param vr Device pointer to the 'red' nodes of the grid.
 * @param vb Device pointer to the 'black' nodes of the grid.
 * @return The interpolated option value.
 */
template <typename T>
static __device__ __inline__ T rb_interp(const T   S,
                                         const T   x_min,
                                         const T   dx,
                                         const int n,
                                         const T *__restrict__ vr,
                                         const T *__restrict__ vb)
{
    T temp        = (cm::log(S) - x_min - dx) / dx;
    temp          = (temp > T(0.0)) ? temp : T(0.0);
    temp          = (temp < T(n - 2)) ? temp : T(n - 2);
    const int idx = (int)temp;

    const T x0 = cm::exp(x_min + T(idx + 1) * dx);
    const T x1 = cm::exp(x_min + T(idx + 2) * dx);

    const T y = (S - x0) / (x1 - x0);

    const int idx_rb = idx / 2; // index converted to rb idx
    T         lo     = (idx & 1) ? vb[idx_rb] : vr[idx_rb];
    T         hi     = (idx & 1) ? vr[idx_rb + 1] : vb[idx_rb];

    return cm::fmax((T(1.0) - y) * lo + y * hi, T(0.0));
}

/* rb-psor device function =====================================================================*/
/**
 * @brief Core device function to calculate the price of a single American option using PSOR.
 *
 * This function implements the Red-Black Projected Successive Over-Relaxation algorithm
 * to solve the Black-Scholes PDE for an American option. It also computes Greeks (Delta, Gamma,
 * Theta) via finite differences on the final grid.
 *
 * @tparam T The floating-point type.
 * @param S The current price of the underlying asset.
 * @param K The strike price of the option.
 * @param cp The option type flag ('c' for call, 'p' for put).
 * @param ttm The time to maturity of the option.
 * @param iv The implied volatility.
 * @param r The risk-free interest rate.
 * @param q The dividend yield.
 * @param n_s The number of grid points for the underlying asset price.
 * @param n_t The number of grid points for time.
 * @param k_mul A multiplier for the strike price to set the grid boundaries.
 * @param w The relaxation parameter. If <= 0.5, an adaptive value is used.
 * @param tol The tolerance for the PSOR solver.
 * @param max_iter The maximum number of iterations for the PSOR solver.
 * @param vr Shared memory for 'red' nodes.
 * @param vb Shared memory for 'black' nodes.
 * @param xr Shared memory for temporary 'red' node values.
 * @param xb Shared memory for temporary 'black' node values.
 * @param pr Shared memory for 'red' node payoff values.
 * @param pb Shared memory for 'black' node payoff values.
 * @param corrections Shared memory for boundary condition corrections.
 * @param warp_sums Shared memory for intermediate reduction results.
 * @param converged Shared memory flag for convergence status.
 * @param[out] delta Pointer to store the calculated Delta.
 * @param[out] gamma Pointer to store the calculated Gamma.
 * @param[out] theta Pointer to store the calculated Theta.
 * @return The calculated price of the option.
 */
template <typename T>
__forceinline__ __device__ T p_func(T    S,
                                    T    K,
                                    char cp,
                                    T    ttm,
                                    T    iv,
                                    T    r,
                                    T    q,
                                    int  n_s,
                                    int  n_t,
                                    int  k_mul,
                                    T    w,
                                    T    tol,
                                    int  max_iter,
                                    T *__restrict__ vr,
                                    T *__restrict__ vb,
                                    T *__restrict__ xr,
                                    T *__restrict__ xb,
                                    T *__restrict__ pr,
                                    T *__restrict__ pb,
                                    T *__restrict__ corrections,
                                    T *__restrict__ warp_sums,
                                    bool *__restrict__ converged,
                                    T *__restrict__ delta = nullptr,
                                    T *__restrict__ gamma = nullptr,
                                    T *__restrict__ theta = nullptr)
{
    // cooperative group setup
    cg::thread_block          block = cg::this_thread_block();
    cg::thread_block_tile<32> tile  = cg::tiled_partition<32>(block);
    const int                 tid   = threadIdx.x;

    // grid specifications
    const int n    = n_s - 2;   // number of interior points
    const T   tol2 = tol * tol; // squared tolerance

    const T s_min = K / k_mul;
    const T s_max = K * k_mul;
    const T x_min = cm::log(s_min);
    const T x_max = cm::log(s_max);
    const T dx    = (x_max - x_min) / (n_s - 1);
    const T dt    = ttm / n_t;
    const T iv2   = iv * iv;
    const T dx2   = dx * dx;

    // tridiagonal matrix coefs
    const T a = T(0.5) * dt * ((r - q - T(0.5) * iv2) / dx - iv2 / dx2);
    const T b = T(1.0) + dt * (iv2 / dx2 + r);
    const T c = -T(0.5) * dt * ((r - q - T(0.5) * iv2) / dx + iv2 / dx2);

    // adaptive w logic
    const int w_adaptive = (w <= T(0.5));
    int       prev_iter  = max_iter;
    const T   rho        = cm::fabs((a + c) / b) * cm::cos(cm::m_pi<T> / (n + 1));
    const T   w_theory   = (T(2.0) / (T(1.0) + cm::sqrt(T(1.0) - rho * rho)));
    T         w_hi       = cm::fmin(T(1.9), w_theory + T(0.3));
    T         w_lo       = cm::fmax(T(0.6), w_theory - T(0.3));
    T         w_delta    = (w_hi - w_lo) / T(10.0);
    w                    = w_adaptive ? w_theory : w;
    int iter;

    // split-buffer bounds
    const int last_is_red = n & 1;           // whether the last element is red
    const int nr          = n / 2 + (n & 1); // size of red (even interior idx) array
    const int nb          = n / 2;           // size of black (odd interior idx) array

    // correction / payoff logic
    const int is_call = (cp & 1);
    const T   cf      = is_call ? c : a; // correction factor
    const T   s       = is_call ? s_max : -s_min;
    const T   k       = is_call ? -K : K;
    const T   sgn     = T(2.0) * is_call - T(1.0);

    // pre-compute boundary corrections
    for (int i = tid; i < n_t; i += blockDim.x)
    {
        const T tau    = (i + 1) * dt;
        corrections[i] = cf * cm::fmax(s + k, s * cm::exp(-q * tau) + k * cm::exp(-r * tau));
    }

    // pre-compute exercise/projection values
    for (int i = tid; i < nr; i += blockDim.x)
    {
        pr[i] = cm::fmax(cm::fma(sgn, cm::exp(cm::fma(T(2 * i + 1), dx, x_min)), k), T(0.0));
        vr[i] = pr[i];
        pb[i] = cm::fmax(cm::fma(sgn, cm::exp(cm::fma(T(2 * i + 2), dx, x_min)), k), T(0.0));
        vb[i] = pb[i];
    }
    block.sync();

    // backtrack
    for (int t = 0; t < n_t; t++)
    {
        const T one_m_w = T(1.0) - w;
        const T w_div_b = w / b;

        // correct boundary
        if (tid == 0)
        {
            if (is_call)
            {
                if (last_is_red)
                    vr[nr - 1] -= corrections[t];
                else
                    vb[nb - 1] -= corrections[t];
            }
            else
                vr[0] -= corrections[t];

            *converged = false;
        }
        block.sync();

        // first iteration to avoid copying p -> x

        // red
        for (int j = tid; j < nr; j += blockDim.x)
        {
            const T left  = (j > 0) ? a * pb[j - 1] : T(0.0);
            const T right = (j < nr - 1) ? c * pb[j] : T(0.0);

            xr[j] = cm::fmax(pr[j], one_m_w * pr[j] + w_div_b * (vr[j] - left - right));
        }
        block.sync();

        // last red or first black
        for (int j = tid; j < nb; j += blockDim.x)
        {
            const T left  = a * xr[j];
            const T right = (j < nb - 1) ? c * xr[j + 1] : T(0.0);

            xb[j] = cm::fmax(pb[j], one_m_w * pb[j] + w_div_b * (vb[j] - left - right));
        }
        block.sync();

        // remaining iterations
        for (iter = 1; iter < max_iter; iter++)
        {
            T sqdist, old, diff;
            sqdist = T(0.0);

            // red
            for (int j = tid; j < nr; j += blockDim.x)
            {
                const T left  = (j > 0) ? a * xb[j - 1] : T(0.0);
                const T right = (j < nr - 1) ? c * xb[j] : T(0.0);

                old   = xr[j];
                xr[j] = cm::fmax(pr[j], one_m_w * xr[j] + w_div_b * (vr[j] - left - right));
                diff  = xr[j] - old;
                sqdist += diff * diff;
            }
            block.sync();

            // black
            for (int j = tid; j < nb; j += blockDim.x)
            {
                const T left  = a * xr[j];
                const T right = (j < nb - 1) ? c * xr[j + 1] : T(0.0);

                old   = xb[j];
                xb[j] = cm::fmax(pb[j], one_m_w * xb[j] + w_div_b * (vb[j] - left - right));
                diff  = xb[j] - old;
                sqdist += diff * diff;
            }
            block.sync();

            // convergence check
            T tile_sum = cg::reduce(tile, sqdist, cg::plus<T>());

            if (tile.thread_rank() == 0) warp_sums[tile.meta_group_rank()] = tile_sum;
            block.sync();

            T block_sq = T(0.0);
            if (tid == 0)
            {
                for (int i = 0; i < blockDim.x / 32; i++)
                    block_sq += warp_sums[i];

                if (block_sq / T(n) < tol2) *converged = true;
            }
            block.sync();

            if (*converged) break;
        }
        // swap xr <-> vr, xb <-> vb
        T *tmp;
        tmp = xr;
        xr  = vr;
        vr  = tmp;
        tmp = xb;
        xb  = vb;
        vb  = tmp;

        // adaptive w logic
        if ((t < 100) && w_adaptive && (t > 0))
        {
            if (iter > prev_iter) w_delta = -w_delta;

            if (iter < prev_iter)
            {
                if (w_delta < T(0.0))
                    w_hi = w - w_delta;
                else
                    w_lo = w - w_delta;

                w_delta = cm::copysign((w_hi - w_lo) / T(10.0), w_delta);
            }
            w = cm::fmin(cm::fmax(w + w_delta, w_lo), w_hi);
        }
        prev_iter = iter;
    }

    if (tid == 0)
    {
        // final interpolation
        const T v = rb_interp<T>(S, x_min, dx, n, vr, vb);

        // compute greeks
        if (delta || gamma)
        {
            const T h  = S * cm::fmax(cm::exp(dx) - T(1.0), T(1e-4));
            const T vp = rb_interp<T>(S + h, x_min, dx, n, vr, vb);
            const T vm = rb_interp<T>(S - h, x_min, dx, n, vr, vb);

            if (delta) *delta = (vp - vm) / (T(2.0) * h);

            if (gamma) *gamma = (vp - T(2.0) * v + vm) / (h * h);
        }

        if (theta)
        {
            const T vp = rb_interp<T>(S, x_min, dx, n, xr, xb);
            *theta     = (vp - v) / (dt * T(365.0));
        }

        return v;
    }

    return T(0.0);
}

/* rb-psor kernel ===============================================================================*/
/**
 * @brief CUDA kernel for calculating PSOR prices for a batch of options.
 *
 * Each block processes one option. Threads within the block cooperate to solve the PDE.
 * @tparam T The floating-point type.
 * @param S Device pointer to underlying asset prices.
 * @param K Device pointer to strike prices.
 * @param cp_flag Device pointer to option type flags.
 * @param ttm Device pointer to times to maturity.
 * @param iv Device pointer to implied volatilities.
 * @param r Device pointer to risk-free interest rates.
 * @param q Device pointer to dividend yields.
 * @param n_s The number of grid points for the underlying asset price.
 * @param n_t The number of grid points for time.
 * @param k_mul A multiplier for the strike price to set the grid boundaries.
 * @param w The relaxation parameter.
 * @param tol The tolerance for the PSOR solver.
 * @param max_iter The maximum number of iterations for the PSOR solver.
 * @param[out] result Device pointer to store the calculated prices.
 * @param[out] delta Device pointer to store the calculated Deltas.
 * @param[out] gamma Device pointer to store the calculated Gammas.
 * @param[out] theta Device pointer to store the calculated Thetas.
 */
template <typename T>
static __global__ void __launch_bounds__(128) p_kernel(const T *__restrict__ S,
                                                       const T *__restrict__ K,
                                                       const char *__restrict__ cp_flag,
                                                       const T *__restrict__ ttm,
                                                       const T *__restrict__ iv,
                                                       const T *__restrict__ r,
                                                       const T *__restrict__ q,
                                                       const int n_s,
                                                       const int n_t,
                                                       const int k_mul,
                                                       T         w,
                                                       const T   tol,
                                                       const int max_iter,
                                                       T *__restrict__ result,
                                                       T *__restrict__ delta,
                                                       T *__restrict__ gamma,
                                                       T *__restrict__ theta)
{
    const int i   = blockIdx.x;
    const int tid = threadIdx.x;

    // shared memory setup
    extern __shared__ char raw[];
    T                     *shared      = reinterpret_cast<T *>(raw);
    T                     *vr          = shared;
    T                     *vb          = vr + (n_s / 2);
    T                     *xr          = vb + (n_s / 2);
    T                     *xb          = xr + (n_s / 2);
    T                     *pr          = xb + (n_s / 2);
    T                     *pb          = pr + (n_s / 2);
    T                     *corrections = pb + (n_s / 2);
    T                     *warp_sums   = corrections + n_t;
    bool                  *converged   = (bool *)(warp_sums + (blockDim.x / 32));

    T price = p_func(S[i],
                     K[i],
                     cp_flag[i],
                     ttm[i],
                     iv[i],
                     r[i],
                     q[i],
                     n_s,
                     n_t,
                     k_mul,
                     w,
                     tol,
                     max_iter,
                     vr,
                     vb,
                     xr,
                     xb,
                     pr,
                     pb,
                     corrections,
                     warp_sums,
                     converged,
                     delta ? &delta[i] : nullptr,
                     gamma ? &gamma[i] : nullptr,
                     theta ? &theta[i] : nullptr);

    if (tid == 0) result[i] = price;
}

/* host functions ==============================================================================*/
/**
 * @brief Host function to launch the PSOR price and Greeks calculation on the GPU.
 *
 * This function handles memory management and kernel launching, dispatching to the
 * appropriate helper from `fastvol::detail::cuda::price` based on the data's memory layout.
 *
 * @tparam T The floating-point type.
 * @param S Pointer to the underlying asset prices on the host or device.
 * @param K Pointer to the strike prices on the host or device.
 * @param cp_flag Pointer to the option type flags on the host or device.
 * @param ttm Pointer to the times to maturity on the host or device.
 * @param iv Pointer to the implied volatilities on the host or device.
 * @param r Pointer to the risk-free interest rates on the host or device.
 * @param q Pointer to the dividend yields on the host or device.
 * @param n_s The number of grid points for the underlying asset price.
 * @param n_t The number of grid points for time.
 * @param k_mul A multiplier for the strike price to set the grid boundaries.
 * @param w The relaxation parameter. If <= 0.5, an adaptive value is used.
 * @param tol The tolerance for the PSOR solver.
 * @param max_iter The maximum number of iterations for the PSOR solver.
 * @param n_options The number of options in the batch.
 * @param[out] result Pointer to the pre-allocated memory for the results on the host or device.
 * @param[out] delta Pointer to the pre-allocated memory for the Deltas.
 * @param[out] gamma Pointer to the pre-allocated memory for the Gammas.
 * @param[out] theta Pointer to the pre-allocated memory for the Thetas.
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
           int    n_s,
           int    n_t,
           int    k_mul,
           T      w,
           T      tol,
           int    max_iter,
           size_t n_options,
           T *__restrict__ result,
           T *__restrict__ delta  = nullptr,
           T *__restrict__ gamma  = nullptr,
           T *__restrict__ theta  = nullptr,
           cudaStream_t stream    = 0,
           int          device    = 0,
           bool         on_device = false,
           bool         is_pinned = false,
           bool         sync      = true)
{
    int    block_size = optimal_threads_per_option(n_s);
    size_t shared_mem = (3 * n_s + n_t + (block_size / 32)) * sizeof(T) + sizeof(bool);

    // set maximum shared memory
    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (shared_mem > static_cast<size_t>(max_shared_mem))
    {
        fprintf(stderr,
                "[fastvol] ERROR: n_s=%d n_t=%d requires %zu bytes shared memory, but max is %d\n",
                n_s,
                n_t,
                shared_mem,
                max_shared_mem);
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        p_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem));

    auto launch_kernel = [shared_mem, n_s, n_t, k_mul, w, tol, max_iter](int          grid_size,
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
        p_kernel<T><<<grid_size, block_size, shared_mem, stream>>>(S,
                                                                   K,
                                                                   cp,
                                                                   ttm,
                                                                   iv,
                                                                   r,
                                                                   q,
                                                                   n_s,
                                                                   n_t,
                                                                   k_mul,
                                                                   w,
                                                                   tol,
                                                                   max_iter,
                                                                   result,
                                                                   delta,
                                                                   gamma,
                                                                   theta);
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
                                             sync,
                                             delta,
                                             gamma,
                                             theta);
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
                                            sync,
                                            delta,
                                            gamma,
                                            theta);
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
                                            sync,
                                            delta,
                                            gamma,
                                            theta);
    }
}

} // namespace fastvol::american::psor::cuda