/**
 * @file ttree.hpp
 * @brief Implements the Trinomial Tree model for American options.
 *
 * @author Valerio Galanti
 * @date 2025
 * @version 0.1.1
 * @license MIT License
 *
 * This file contains the C++ implementation of the Trinomial Tree model for pricing American
 * options, calculating their Greeks, and estimating implied volatility. It supports both single
 * and double precision floating-point types and includes CPU (OpenMP-parallelized) and
 * CUDA-accelerated versions for batch operations.
 */

#pragma once

#include "fastvol/detail/fd.hpp"
#include "fastvol/detail/iv.hpp"
#include "fastvol/detail/math.hpp"
#include "fastvol/european/bsm.hpp"
#include <cstring>
#include <omp.h>
#include <stdlib.h>

#ifdef FASTVOL_CUDA_ENABLED
#include <cuda_runtime.h>
#else
typedef void *cudaStream_t;
#endif

namespace fm = fastvol::detail::math;
namespace fd = fastvol::detail::fd;

/**
 * @namespace fastvol::american::ttree
 * @brief Provides implementations of the Trinomial Tree model for American options.
 *
 * This namespace contains functions for pricing, calculating Greeks (via finite differences),
 * and finding the implied volatility of American options using a trinomial lattice.
 * It supports both single and double precision, batch processing via OpenMP, and hooks for
 * CUDA-accelerated versions.
 */
namespace fastvol::american::ttree
{
inline constexpr int MAX_STEPS_STACK = 2048;

inline constexpr int                     EURO_TTREE64_THRESHOLD = 512;
inline constexpr size_t                  INIT_MAX_ITER          = 10;
template <typename T> inline constexpr T INIT_EURO_MARGIN       = T(0.10);
template <typename T> inline constexpr T INIT_TTREE64_MARGIN    = T(0.05);
template <typename T> inline constexpr T INIT_TOL               = T(5e-2);

/* templates =====================================================================================*/
/* price -----------------------------------------------------------------------------------------*/
/**
 * @brief Calculates the price of an American option using a trinomial tree.
 * @tparam T The floating-point type (float or double).
 * @param S The current price of the underlying asset.
 * @param K The strike price of the option.
 * @param cp_flag The option type flag.
 * @param ttm The time to maturity of the option, in years.
 * @param iv The implied volatility of the underlying asset.
 * @param r The risk-free interest rate.
 * @param q The dividend yield of the underlying asset.
 * @param n_steps The number of steps in the trinomial tree.
 * @return The calculated price of the option.
 */
template <typename T> T price(T S, T K, char cp_flag, T ttm, T iv, T r, T q, int n_steps)
{
    const int n = 2 * n_steps + 1;

    alignas(64) T ex_stack[2 * MAX_STEPS_STACK + 1];
    alignas(64) T v_stack[2 * MAX_STEPS_STACK + 1];

    T *__restrict__ ex = ex_stack;
    T *__restrict__ v  = v_stack;

    const bool use_malloc = n_steps > MAX_STEPS_STACK;
    if (use_malloc)
    {
        int err1 = posix_memalign((void **)&ex, 64, n * sizeof(T));
        int err2 = posix_memalign((void **)&v, 64, n * sizeof(T));

        if (err1 || err2)
        {
            if (!err1) free(ex);
            if (!err2) free(v);
            return T(-1);
        }

        ex = (T *)__builtin_assume_aligned(ex, 64);
        v  = (T *)__builtin_assume_aligned(v, 64);
    }

    const T dt   = ttm / n_steps;
    const T nu   = r - q - T(0.5) * iv * iv;
    const T dx   = iv * fm::sqrt(T(3.0) * dt);
    const T disc = fm::exp(-r * dt);

    const T u = fm::exp(dx);
    const T p = nu * fm::sqrt(dt) / (T(2.0) * iv * fm::sqrt(T(3.0)));

    const T pu = T(1.0) / T(6.0) + p;
    const T pm = T(2.0) / T(3.0);
    const T pd = T(1.0) / T(6.0) - p;

    const T dpu = disc * pu;
    const T dpm = disc * pm;
    const T dpd = disc * pd;

    const T log_u = fm::log(u);
    const T lsu   = fm::fma(-T(n_steps), log_u, fm::log(S));
    const T s     = T(2.0) * (cp_flag & 1) - T(1.0);
    const T sK    = K * -s;

#pragma omp simd
    for (int i = 0; i < n; i++)
    {
        ex[i] = fm::fmax(fm::fma(s, fm::exp(fm::fma(T(i), log_u, lsu)), sK), T(0.0));
    }

    memcpy(v, ex, n * sizeof(T)); // init fill v

    for (int i = 1; i <= n_steps; i++) // backtrack
    {
#pragma omp simd
        for (int j = 0; j < n - 2 * i; j++)
        {
            T h  = dpu * v[j + 2] + dpm * v[j + 1] + dpd * v[j]; // held
            T e  = ex[i + j];                                    // exercised
            v[j] = fm::fmax(e, h);                               // max
        }
    }

    T result = v[0]; // value at origin

    if (use_malloc) // cleanup
    {
        free(ex);
        free(v);
    }

    return result;
}

/* greeks ----------------------------------------------------------------------------------------*/
/**
 * @brief Calculates the Greeks of an American option using finite differences.
 * @tparam T The floating-point type.
 * @param[out] delta A pointer to store the calculated Delta. Can be NULL.
 * @param[out] gamma A pointer to store the calculated Gamma. Can be NULL.
 * @param[out] theta A pointer to store the calculated Theta. Can be NULL.
 * @param[out] vega A pointer to store the calculated Vega. Can be NULL.
 * @param[out] rho A pointer to store the calculated Rho. Can be NULL.
 * @see price for other parameter details.
 */
template <typename T>
inline void greeks(T    S,
                   T    K,
                   char cp_flag,
                   T    ttm,
                   T    iv,
                   T    r,
                   T    q,
                   int  n_steps,
                   T   *delta = nullptr,
                   T   *gamma = nullptr,
                   T   *theta = nullptr,
                   T   *vega  = nullptr,
                   T   *rho   = nullptr)
{
    return fd::greeks<T>(
        S,
        K,
        cp_flag,
        ttm,
        iv,
        r,
        q,
        [n_steps](T S, T K, char cp, T ttm, T iv, T r, T q)
        { return price<T>(S, K, cp, ttm, iv, r, q, n_steps); },
        delta,
        gamma,
        theta,
        vega,
        rho);
}

/**
 * @brief Calculates the Delta of an American option.
 * @return The calculated Delta.
 * @see price for parameter details.
 */
template <typename T> inline T delta(T S, T K, char cp_flag, T ttm, T iv, T r, T q, int n_steps)
{
    return fd::delta<T>(S,
                        K,
                        cp_flag,
                        ttm,
                        iv,
                        r,
                        q,
                        [n_steps](T S, T K, char cp, T ttm, T iv, T r, T q)
                        { return price<T>(S, K, cp, ttm, iv, r, q, n_steps); });
}

/**
 * @brief Calculates the Gamma of an American option.
 * @return The calculated Gamma.
 * @see price for parameter details.
 */
template <typename T> inline T gamma(T S, T K, char cp_flag, T ttm, T iv, T r, T q, int n_steps)
{
    return fd::gamma<T>(S,
                        K,
                        cp_flag,
                        ttm,
                        iv,
                        r,
                        q,
                        [n_steps](T S, T K, char cp, T ttm, T iv, T r, T q)
                        { return price<T>(S, K, cp, ttm, iv, r, q, n_steps); });
}

/**
 * @brief Calculates the Theta of an American option.
 * @return The calculated Theta.
 * @see price for parameter details.
 */
template <typename T> inline T theta(T S, T K, char cp_flag, T ttm, T iv, T r, T q, int n_steps)
{
    return fd::theta<T>(S,
                        K,
                        cp_flag,
                        ttm,
                        iv,
                        r,
                        q,
                        [n_steps](T S, T K, char cp, T ttm, T iv, T r, T q)
                        { return price<T>(S, K, cp, ttm, iv, r, q, n_steps); });
}

/**
 * @brief Calculates the Vega of an American option.
 * @return The calculated Vega.
 * @see price for parameter details.
 */
template <typename T> inline T vega(T S, T K, char cp_flag, T ttm, T iv, T r, T q, int n_steps)
{
    return fd::vega<T>(S,
                       K,
                       cp_flag,
                       ttm,
                       iv,
                       r,
                       q,
                       [n_steps](T S, T K, char cp, T ttm, T iv, T r, T q)
                       { return price<T>(S, K, cp, ttm, iv, r, q, n_steps); });
}

/**
 * @brief Calculates the Rho of an American option.
 * @return The calculated Rho.
 * @see price for parameter details.
 */
template <typename T> inline T rho(T S, T K, char cp_flag, T ttm, T iv, T r, T q, int n_steps)
{
    return fd::rho<T>(S,
                      K,
                      cp_flag,
                      ttm,
                      iv,
                      r,
                      q,
                      [n_steps](T S, T K, char cp, T ttm, T iv, T r, T q)
                      { return price<T>(S, K, cp, ttm, iv, r, q, n_steps); });
}

/* iv --------------------------------------------------------------------------------------------*/
/**
 * @brief Calculates the implied volatility of an American option using Brent's method.
 * @tparam T The floating-point type.
 * @param P The market price of the option.
 * @param tol The tolerance for the root-finding algorithm.
 * @param max_iter The maximum number of iterations.
 * @param lo_init An optional initial lower bound for the search.
 * @param hi_init An optional initial upper bound for the search.
 * @return The calculated implied volatility.
 * @see price for other parameter details.
 */
template <typename T>
T iv(T      P,
     T      S,
     T      K,
     char   cp_flag,
     T      ttm,
     T      r,
     T      q,
     int    n_steps,
     T      tol                    = T{1e-3},
     size_t max_iter               = 100,
     const T *__restrict__ lo_init = nullptr,
     const T *__restrict__ hi_init = nullptr)
{
    T init, lo, hi;
    if (!lo_init || !hi_init)
    {
        if (n_steps <= EURO_TTREE64_THRESHOLD)
        {
            init = fastvol::european::bsm::iv(
                P, S, K, cp_flag, ttm, r, q, INIT_TOL<T>, INIT_MAX_ITER, lo_init, hi_init);
            lo = lo_init ? *lo_init : (1.0 - INIT_EURO_MARGIN<T>)*init;
            hi = hi_init ? *hi_init : (1.0 + INIT_EURO_MARGIN<T>)*init;
        }
        else
        {
            init = fastvol::detail::iv::brent<T>(
                P,
                S,
                K,
                cp_flag,
                ttm,
                r,
                q,
                [](T S, T K, char cp, T ttm, T iv, T r, T q)
                { return price<T>(S, K, cp, ttm, iv, r, q, 64); },
                INIT_TOL<T>,
                INIT_MAX_ITER,
                lo_init,
                hi_init);
            lo = lo_init ? *lo_init : (1.0 - INIT_TTREE64_MARGIN<T>)*init;
            hi = hi_init ? *hi_init : (1.0 + INIT_TTREE64_MARGIN<T>)*init;
        }
    }

    return fastvol::detail::iv::brent<T>(
        P,
        S,
        K,
        cp_flag,
        ttm,
        r,
        q,
        [n_steps](T S, T K, char cp, T ttm, T iv, T r, T q)
        { return price<T>(S, K, cp, ttm, iv, r, q, n_steps); },
        tol,
        max_iter,
        lo_init ? lo_init : &lo,
        hi_init ? hi_init : &hi);
}

/* price_batch -----------------------------------------------------------------------------------*/
/**
 * @brief Calculates the prices of a batch of American options on the CPU.
 * @param n_options The number of options in the batch.
 * @param[out] results A pre-allocated array to store the calculated option prices.
 * @see price for other parameter details.
 */
template <typename T>
void price_batch(const T *__restrict__ S,
                 const T *__restrict__ K,
                 const char *__restrict__ cp_flag,
                 const T *__restrict__ ttm,
                 const T *__restrict__ iv,
                 const T *__restrict__ r,
                 const T *__restrict__ q,
                 int    n_steps,
                 size_t n_options,
                 T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
        results[i] = price<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], n_steps);
}

/* greeks_batch ----------------------------------------------------------------------------------*/
/**
 * @brief Calculates the Greeks for a batch of American options on the CPU.
 * @see greeks for parameter details.
 * @see price_batch for other parameter details.
 */
template <typename T>
inline void greeks_batch(const T *__restrict__ S,
                         const T *__restrict__ K,
                         const char *__restrict__ cp_flag,
                         const T *__restrict__ ttm,
                         const T *__restrict__ iv,
                         const T *__restrict__ r,
                         const T *__restrict__ q,
                         int    n_steps,
                         size_t n_options,
                         T *__restrict__ delta = nullptr,
                         T *__restrict__ gamma = nullptr,
                         T *__restrict__ theta = nullptr,
                         T *__restrict__ vega  = nullptr,
                         T *__restrict__ rho   = nullptr)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        greeks<T>(S[i],
                  K[i],
                  cp_flag[i],
                  ttm[i],
                  iv[i],
                  r[i],
                  q[i],
                  n_steps,
                  delta ? &delta[i] : nullptr,
                  gamma ? &gamma[i] : nullptr,
                  theta ? &theta[i] : nullptr,
                  vega ? &vega[i] : nullptr,
                  rho ? &rho[i] : nullptr);
    }
}

/**
 * @brief Calculates the Deltas for a batch of American options on the CPU.
 * @see price_batch for parameter details.
 */
template <typename T>
inline void delta_batch(const T *__restrict__ S,
                        const T *__restrict__ K,
                        const char *__restrict__ cp_flag,
                        const T *__restrict__ ttm,
                        const T *__restrict__ iv,
                        const T *__restrict__ r,
                        const T *__restrict__ q,
                        int    n_steps,
                        size_t n_options,
                        T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        results[i] = delta<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], n_steps);
    }
}

/**
 * @brief Calculates the Gammas for a batch of American options on the CPU.
 * @see price_batch for parameter details.
 */
template <typename T>
inline void gamma_batch(const T *__restrict__ S,
                        const T *__restrict__ K,
                        const char *__restrict__ cp_flag,
                        const T *__restrict__ ttm,
                        const T *__restrict__ iv,
                        const T *__restrict__ r,
                        const T *__restrict__ q,
                        int    n_steps,
                        size_t n_options,
                        T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        results[i] = gamma<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], n_steps);
    }
}

/**
 * @brief Calculates the Thetas for a batch of American options on the CPU.
 * @see price_batch for parameter details.
 */
template <typename T>
inline void theta_batch(const T *__restrict__ S,
                        const T *__restrict__ K,
                        const char *__restrict__ cp_flag,
                        const T *__restrict__ ttm,
                        const T *__restrict__ iv,
                        const T *__restrict__ r,
                        const T *__restrict__ q,
                        int    n_steps,
                        size_t n_options,
                        T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        results[i] = theta<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], n_steps);
    }
}

/**
 * @brief Calculates the Vegas for a batch of American options on the CPU.
 * @see price_batch for parameter details.
 */
template <typename T>
inline void vega_batch(const T *__restrict__ S,
                       const T *__restrict__ K,
                       const char *__restrict__ cp_flag,
                       const T *__restrict__ ttm,
                       const T *__restrict__ iv,
                       const T *__restrict__ r,
                       const T *__restrict__ q,
                       int    n_steps,
                       size_t n_options,
                       T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        results[i] = vega<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], n_steps);
    }
}

/**
 * @brief Calculates the Rhos for a batch of American options on the CPU.
 * @see price_batch for parameter details.
 */
template <typename T>
inline void rho_batch(const T *__restrict__ S,
                      const T *__restrict__ K,
                      const char *__restrict__ cp_flag,
                      const T *__restrict__ ttm,
                      const T *__restrict__ iv,
                      const T *__restrict__ r,
                      const T *__restrict__ q,
                      int    n_steps,
                      size_t n_options,
                      T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        results[i] = rho<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], n_steps);
    }
}

/* iv_batch --------------------------------------------------------------------------------------*/
/**
 * @brief Calculates the implied volatilities for a batch of American options on the CPU.
 * @see iv for parameter details.
 * @see price_batch for other parameter details.
 */
template <typename T>
void iv_batch(const T *__restrict__ P,
              const T *__restrict__ S,
              const T *__restrict__ K,
              const char *__restrict__ cp_flag,
              const T *__restrict__ ttm,
              const T *__restrict__ r,
              const T *__restrict__ q,
              int    n_steps,
              size_t n_options,
              T *__restrict__ results,
              T      tol                    = T{1e-3},
              size_t max_iter               = 100,
              const T *__restrict__ lo_init = nullptr,
              const T *__restrict__ hi_init = nullptr)
{
    T *lo = nullptr;
    T *hi = nullptr;

    if (!lo_init || !hi_init)
    {
        T margin;
        if (n_steps <= EURO_TTREE64_THRESHOLD)
        {
            margin = INIT_EURO_MARGIN<T>;
            fastvol::european::bsm::iv_batch(P,
                                             S,
                                             K,
                                             cp_flag,
                                             ttm,
                                             r,
                                             q,
                                             n_options,
                                             results,
                                             INIT_TOL<T>,
                                             INIT_MAX_ITER,
                                             lo_init,
                                             hi_init);
        }
        else
        {
            margin = INIT_TTREE64_MARGIN<T>;
            fastvol::detail::iv::brent_batch(
                P,
                S,
                K,
                cp_flag,
                ttm,
                r,
                q,
                [](T S, T K, char cp, T ttm, T iv, T r, T q)
                { return price<T>(S, K, cp, ttm, iv, r, q, 64); },
                n_options,
                results,
                INIT_TOL<T>,
                INIT_MAX_ITER,
                lo_init,
                hi_init);
        }

        // if null lower bound, generate it
        if (!lo_init)
        {
            int err_lo = posix_memalign((void **)&lo, 64, n_options * sizeof(T));
            if (err_lo) return;
            lo = (T *)__builtin_assume_aligned(lo, 64);

#pragma omp simd
            for (size_t i = 0; i < n_options; i++)
                lo[i] = (T(1) - margin) * results[i];
        }

        // if null upper bound, generate it
        if (!hi_init)
        {
            int err_hi = posix_memalign((void **)&hi, 64, n_options * sizeof(T));
            if (err_hi)
            {
                if (!lo_init) free(lo);
                return;
            }
            hi = (T *)__builtin_assume_aligned(hi, 64);

#pragma omp simd
            for (size_t i = 0; i < n_options; i++)
                hi[i] = (T(1) + margin) * results[i];
        }
    }

    fastvol::detail::iv::brent_batch(
        P,
        S,
        K,
        cp_flag,
        ttm,
        r,
        q,
        [n_steps](T S, T K, char cp, T ttm, T iv, T r, T q)
        { return price<T>(S, K, cp, ttm, iv, r, q, n_steps); },
        n_options,
        results,
        tol,
        max_iter,
        lo_init ? lo_init : lo,
        hi_init ? hi_init : hi);

    // cleanup
    if (!lo_init) free(lo);
    if (!hi_init) free(hi);
}

/* instanciations/CUDA ===========================================================================*/
/* fp64 ------------------------------------------------------------------------------------------*/
/* price _________________________________________________________________________________________*/
inline double
price_fp64(double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
{
    return price<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
}

inline void price_fp64_batch(const double *__restrict__ S,
                             const double *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const double *__restrict__ ttm,
                             const double *__restrict__ iv,
                             const double *__restrict__ r,
                             const double *__restrict__ q,
                             int    n_steps,
                             size_t n_options,
                             double *__restrict__ results)
{
    price_batch<double>(S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
}

/**
 * @brief Calculates the prices of a batch of American options using a trinomial tree on a
 * CUDA-enabled GPU.
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
void price_fp64_cuda(const double *__restrict__ S,
                     const double *__restrict__ K,
                     const char *__restrict__ cp_flag,
                     const double *__restrict__ ttm,
                     const double *__restrict__ iv,
                     const double *__restrict__ r,
                     const double *__restrict__ q,
                     int    n_steps,
                     size_t n_options,
                     double *__restrict__ result,
                     cudaStream_t stream    = 0,
                     int          device    = 0,
                     bool         on_device = false,
                     bool         is_pinned = false,
                     bool         sync      = true);

/* greeks ________________________________________________________________________________________*/
inline void greeks_fp64(double  S,
                        double  K,
                        char    cp_flag,
                        double  ttm,
                        double  iv,
                        double  r,
                        double  q,
                        int     n_steps,
                        double *delta = nullptr,
                        double *gamma = nullptr,
                        double *theta = nullptr,
                        double *vega  = nullptr,
                        double *rho   = nullptr)
{
    greeks<double>(S, K, cp_flag, ttm, iv, r, q, n_steps, delta, gamma, theta, vega, rho);
}

inline double
delta_fp64(double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
{
    return delta<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
}

inline double
gamma_fp64(double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
{
    return gamma<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
}

inline double
theta_fp64(double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
{
    return theta<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
}

inline double
vega_fp64(double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
{
    return vega<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
}

inline double
rho_fp64(double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
{
    return rho<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
}

void greeks_fp64_batch(const double *__restrict__ S,
                       const double *__restrict__ K,
                       const char *__restrict__ cp_flag,
                       const double *__restrict__ ttm,
                       const double *__restrict__ iv,
                       const double *__restrict__ r,
                       const double *__restrict__ q,
                       int    n_steps,
                       size_t n_options,
                       double *__restrict__ delta = nullptr,
                       double *__restrict__ gamma = nullptr,
                       double *__restrict__ theta = nullptr,
                       double *__restrict__ vega  = nullptr,
                       double *__restrict__ rho   = nullptr)
{
    greeks_batch<double>(
        S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, delta, gamma, theta, vega, rho);
}

inline void delta_fp64_batch(const double *__restrict__ S,
                             const double *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const double *__restrict__ ttm,
                             const double *__restrict__ iv,
                             const double *__restrict__ r,
                             const double *__restrict__ q,
                             int    n_steps,
                             size_t n_options,
                             double *__restrict__ results)
{
    delta_batch<double>(S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
}

inline void gamma_fp64_batch(const double *__restrict__ S,
                             const double *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const double *__restrict__ ttm,
                             const double *__restrict__ iv,
                             const double *__restrict__ r,
                             const double *__restrict__ q,
                             int    n_steps,
                             size_t n_options,
                             double *__restrict__ results)
{
    gamma_batch<double>(S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
}

inline void theta_fp64_batch(const double *__restrict__ S,
                             const double *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const double *__restrict__ ttm,
                             const double *__restrict__ iv,
                             const double *__restrict__ r,
                             const double *__restrict__ q,
                             int    n_steps,
                             size_t n_options,
                             double *__restrict__ results)
{
    theta_batch<double>(S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
}

inline void vega_fp64_batch(const double *__restrict__ S,
                            const double *__restrict__ K,
                            const char *__restrict__ cp_flag,
                            const double *__restrict__ ttm,
                            const double *__restrict__ iv,
                            const double *__restrict__ r,
                            const double *__restrict__ q,
                            int    n_steps,
                            size_t n_options,
                            double *__restrict__ results)
{
    vega_batch<double>(S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
}

inline void rho_fp64_batch(const double *__restrict__ S,
                           const double *__restrict__ K,
                           const char *__restrict__ cp_flag,
                           const double *__restrict__ ttm,
                           const double *__restrict__ iv,
                           const double *__restrict__ r,
                           const double *__restrict__ q,
                           int    n_steps,
                           size_t n_options,
                           double *__restrict__ results)
{
    rho_batch<double>(S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
}

/* iv ____________________________________________________________________________________________*/
inline double iv_fp64(double P,
                      double S,
                      double K,
                      char   cp_flag,
                      double ttm,
                      double r,
                      double q,
                      int    n_steps,
                      double tol                         = 1e-3,
                      size_t max_iter                    = 100,
                      const double *__restrict__ lo_init = nullptr,
                      const double *__restrict__ hi_init = nullptr)
{
    return iv<double>(P, S, K, cp_flag, ttm, r, q, n_steps, tol, max_iter, lo_init, hi_init);
}

inline void iv_fp64_batch(const double *__restrict__ P,
                          const double *__restrict__ S,
                          const double *__restrict__ K,
                          const char *__restrict__ cp_flag,
                          const double *__restrict__ ttm,
                          const double *__restrict__ r,
                          const double *__restrict__ q,
                          int    n_steps,
                          size_t n_options,
                          double *__restrict__ results,
                          double tol                         = 1e-3,
                          size_t max_iter                    = 100,
                          const double *__restrict__ lo_init = nullptr,
                          const double *__restrict__ hi_init = nullptr)
{
    iv_batch<double>(
        P, S, K, cp_flag, ttm, r, q, n_steps, n_options, results, tol, max_iter, lo_init, hi_init);
}

/* fp32 ------------------------------------------------------------------------------------------*/
/* price _________________________________________________________________________________________*/
inline float
price_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
{
    return price<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
}

inline void price_fp32_batch(const float *__restrict__ S,
                             const float *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const float *__restrict__ ttm,
                             const float *__restrict__ iv,
                             const float *__restrict__ r,
                             const float *__restrict__ q,
                             int    n_steps,
                             size_t n_options,
                             float *__restrict__ results)
{
    price_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
}

/**
 * @brief Calculates the prices of a batch of American options using a trinomial tree on a
 * CUDA-enabled GPU (single precision).
 * @see price_fp64_cuda for parameter details.
 */
void price_fp32_cuda(const float *__restrict__ S,
                     const float *__restrict__ K,
                     const char *__restrict__ cp_flag,
                     const float *__restrict__ ttm,
                     const float *__restrict__ iv,
                     const float *__restrict__ r,
                     const float *__restrict__ q,
                     int    n_steps,
                     size_t n_options,
                     float *__restrict__ result,
                     cudaStream_t stream    = 0,
                     int          device    = 0,
                     bool         on_device = false,
                     bool         is_pinned = false,
                     bool         sync      = true);

/* greeks ________________________________________________________________________________________*/
inline void greeks_fp32(float  S,
                        float  K,
                        char   cp_flag,
                        float  ttm,
                        float  iv,
                        float  r,
                        float  q,
                        int    n_steps,
                        float *delta = nullptr,
                        float *gamma = nullptr,
                        float *theta = nullptr,
                        float *vega  = nullptr,
                        float *rho   = nullptr)
{
    greeks<float>(S, K, cp_flag, ttm, iv, r, q, n_steps, delta, gamma, theta, vega, rho);
}

inline float
delta_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
{
    return delta<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
}

inline float
gamma_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
{
    return gamma<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
}

inline float
theta_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
{
    return theta<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
}

inline float
vega_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
{
    return vega<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
}

inline float
rho_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
{
    return rho<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
}

void greeks_fp32_batch(const float *__restrict__ S,
                       const float *__restrict__ K,
                       const char *__restrict__ cp_flag,
                       const float *__restrict__ ttm,
                       const float *__restrict__ iv,
                       const float *__restrict__ r,
                       const float *__restrict__ q,
                       int    n_steps,
                       size_t n_options,
                       float *__restrict__ delta = nullptr,
                       float *__restrict__ gamma = nullptr,
                       float *__restrict__ theta = nullptr,
                       float *__restrict__ vega  = nullptr,
                       float *__restrict__ rho   = nullptr)
{
    greeks_batch<float>(
        S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, delta, gamma, theta, vega, rho);
}

inline void delta_fp32_batch(const float *__restrict__ S,
                             const float *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const float *__restrict__ ttm,
                             const float *__restrict__ iv,
                             const float *__restrict__ r,
                             const float *__restrict__ q,
                             int    n_steps,
                             size_t n_options,
                             float *__restrict__ results)
{
    delta_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
}

inline void gamma_fp32_batch(const float *__restrict__ S,
                             const float *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const float *__restrict__ ttm,
                             const float *__restrict__ iv,
                             const float *__restrict__ r,
                             const float *__restrict__ q,
                             int    n_steps,
                             size_t n_options,
                             float *__restrict__ results)
{
    gamma_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
}

inline void theta_fp32_batch(const float *__restrict__ S,
                             const float *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const float *__restrict__ ttm,
                             const float *__restrict__ iv,
                             const float *__restrict__ r,
                             const float *__restrict__ q,
                             int    n_steps,
                             size_t n_options,
                             float *__restrict__ results)
{
    theta_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
}

inline void vega_fp32_batch(const float *__restrict__ S,
                            const float *__restrict__ K,
                            const char *__restrict__ cp_flag,
                            const float *__restrict__ ttm,
                            const float *__restrict__ iv,
                            const float *__restrict__ r,
                            const float *__restrict__ q,
                            int    n_steps,
                            size_t n_options,
                            float *__restrict__ results)
{
    vega_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
}

inline void rho_fp32_batch(const float *__restrict__ S,
                           const float *__restrict__ K,
                           const char *__restrict__ cp_flag,
                           const float *__restrict__ ttm,
                           const float *__restrict__ iv,
                           const float *__restrict__ r,
                           const float *__restrict__ q,
                           int    n_steps,
                           size_t n_options,
                           float *__restrict__ results)
{
    rho_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
}

/* iv ____________________________________________________________________________________________*/
inline float iv_fp32(float  P,
                     float  S,
                     float  K,
                     char   cp_flag,
                     float  ttm,
                     float  r,
                     float  q,
                     int    n_steps,
                     float  tol                        = 1e-3f,
                     size_t max_iter                   = 100,
                     const float *__restrict__ lo_init = nullptr,
                     const float *__restrict__ hi_init = nullptr)
{
    return iv<float>(P, S, K, cp_flag, ttm, r, q, n_steps, tol, max_iter, lo_init, hi_init);
}

inline void iv_fp32_batch(const float *__restrict__ P,
                          const float *__restrict__ S,
                          const float *__restrict__ K,
                          const char *__restrict__ cp_flag,
                          const float *__restrict__ ttm,
                          const float *__restrict__ r,
                          const float *__restrict__ q,
                          int    n_steps,
                          size_t n_options,
                          float *__restrict__ results,
                          float  tol                        = 1e-3f,
                          size_t max_iter                   = 100,
                          const float *__restrict__ lo_init = nullptr,
                          const float *__restrict__ hi_init = nullptr)
{
    iv_batch<float>(
        P, S, K, cp_flag, ttm, r, q, n_steps, n_options, results, tol, max_iter, lo_init, hi_init);
}

} // namespace fastvol::american::ttree
