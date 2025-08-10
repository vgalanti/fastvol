/**
 * @file psor.hpp
 * @brief Implements the Projected Successive Over-Relaxation (PSOR) method for American options.
 *
 * @author Valerio Galanti
 * @date 2025
 * @version 0.1.1
 * @license MIT License
 *
 * This file contains the C++ implementation of the PSOR finite difference method for pricing
 * American options, calculating their Greeks, and estimating implied volatility. It supports
 * both single and double precision, and includes CPU (OpenMP) and CUDA-accelerated versions.
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
 * @namespace fastvol::american::psor
 * @brief Provides implementations of the PSOR method for American options.
 *
 * This namespace contains functions for pricing, calculating Greeks, and finding the implied
 * volatility of American options using a finite difference grid solved with the Projected
 * Successive Over-Relaxation (PSOR) method. It supports both single and double precision,
 * batch processing via OpenMP, and hooks for CUDA-accelerated versions.
 */
namespace fastvol::american::psor
{
inline constexpr int MAX_NS_STACK = 1024;
inline constexpr int MAX_NT_STACK = 4096;

inline constexpr size_t                  INIT_MAX_ITER       = 10;
inline constexpr int                     EURO_PSOR_THRESHOLD = 512;
template <typename T> inline constexpr T INIT_EURO_MARGIN    = T(0.10);
template <typename T> inline constexpr T INIT_PSOR_MARGIN    = T(0.05);
template <typename T> inline constexpr T INIT_TOL            = T(5e-2);

/* templates =====================================================================================*/
/* price -----------------------------------------------------------------------------------------*/
/**
 * @brief Interpolates the option value on the Red-Black grid.
 * @tparam T The floating-point type.
 * @param S The underlying asset price at which to interpolate.
 * @param x_min The minimum value of the log-price grid.
 * @param dx The step size of the log-price grid.
 * @param n The number of interior grid points.
 * @param vr Pointer to the 'red' nodes of the grid.
 * @param vb Pointer to the 'black' nodes of the grid.
 * @return The interpolated option value.
 */
template <typename T>
static inline T
rb_interp(T S, T x_min, T dx, int n, const T *__restrict__ vr, const T *__restrict__ vb)
{
    const int idx = (int)fm::fmin(fm::fmax((fm::log(S) - x_min - dx) / dx, T(0)), T(n - 2));
    const T   x0  = fm::exp(x_min + (idx + 1) * dx);
    const T   x1  = fm::exp(x_min + (idx + 2) * dx);
    const T   y   = (S - x0) / (x1 - x0);

    const int idx_rb = idx / 2; // idx converted to rb idx
    T         lo     = (idx & 1) ? vb[idx_rb] : vr[idx_rb];
    T         hi     = (idx & 1) ? vr[idx_rb + 1] : vb[idx_rb];

    return fm::fmax((T(1.0) - y) * lo + y * hi, T(0.0));
}

/**
 * @brief Calculates the price of an American option using the PSOR method.
 * @tparam T The floating-point type (float or double).
 * @param S The current price of the underlying asset.
 * @param K The strike price of the option.
 * @param cp_flag The option type flag. Accepted values are 1, 'c', or 'C' for a call option, and 0,
 * 'p', or 'P' for a put option.
 * @param ttm The time to maturity of the option.
 * @param iv The implied volatility of the underlying asset.
 * @param r The risk-free interest rate.
 * @param q The dividend yield of the underlying asset.
 * @param n_s The number of grid points for the underlying asset price.
 * @param n_t The number of grid points for time.
 * @param k_mul A multiplier for the strike price to set the upper boundary of the asset price grid.
 * @param w The relaxation parameter. If set to 0, the model will use an adaptive optimal value.
 * @param tol The tolerance for the PSOR solver.
 * @param max_iter The maximum number of iterations for the PSOR solver.
 * @param[out] delta A pointer to store the calculated Delta (optional). Can be NULL.
 * @param[out] gamma A pointer to store the calculated Gamma (optional). Can be NULL.
 * @param[out] theta A pointer to store the calculated Theta (optional). Can be NULL.
 * @return The calculated price of the option.
 */
template <typename T>
T price(T    S,
        T    K,
        char cp_flag,
        T    ttm,
        T    iv,
        T    r,
        T    q,
        int  n_s,
        int  n_t,
        int  k_mul            = 4,
        T    w                = T{0},
        T    tol              = T{1e-4},
        int  max_iter         = 200,
        T *__restrict__ delta = nullptr,
        T *__restrict__ gamma = nullptr,
        T *__restrict__ theta = nullptr)
{
    // grid specifications
    const int n    = n_s - 2;
    const T   tol2 = tol * tol;

    const T s_min = K / k_mul;
    const T s_max = K * k_mul;
    const T x_min = fm::log(s_min);
    const T x_max = fm::log(s_max);
    const T dx    = (x_max - x_min) / (n_s - 1);
    const T dt    = ttm / n_t;
    const T iv2   = iv * iv;
    const T dx2   = dx * dx;

    // tridiagonal matrix coefs
    const T a = T(0.5) * dt * ((r - q - T(0.5) * iv2) / dx - iv2 / dx2);
    const T b = T(1.0) + dt * (iv2 / dx2 + r);
    const T c = -T(0.5) * dt * ((r - q - T(0.5) * iv2) / dx + iv2 / dx2);

    // v/x/p arrays
    alignas(64) T vr_stack[MAX_NS_STACK / 2]; // red value stack
    alignas(64) T vb_stack[MAX_NS_STACK / 2]; // black values stack
    alignas(64) T xr_stack[MAX_NS_STACK / 2]; // red psor stack
    alignas(64) T xb_stack[MAX_NS_STACK / 2]; // black psor stack
    alignas(64) T pr_stack[MAX_NS_STACK / 2]; // red payoff stack
    alignas(64) T pb_stack[MAX_NS_STACK / 2]; // black payoff stack

    T *__restrict__ vr = vr_stack;
    T *__restrict__ vb = vb_stack;
    T *__restrict__ xr = xr_stack;
    T *__restrict__ xb = xb_stack;
    T *__restrict__ pr = pr_stack;
    T *__restrict__ pb = pb_stack;

    bool malloc_ns = n_s > MAX_NS_STACK;
    if (malloc_ns)
    {
        const size_t s = (n / 2 + 1) * sizeof(T);

        int err1 = posix_memalign((void **)&vr, 64, s);
        int err2 = posix_memalign((void **)&vb, 64, s);
        int err3 = posix_memalign((void **)&xr, 64, s);
        int err4 = posix_memalign((void **)&xb, 64, s);
        int err5 = posix_memalign((void **)&pr, 64, s);
        int err6 = posix_memalign((void **)&pb, 64, s);

        if (err1 || err2 || err3 || err4 || err5 || err6)
        {
            if (!err1) free(vr);
            if (!err2) free(vb);
            if (!err3) free(xr);
            if (!err4) free(xb);
            if (!err5) free(pr);
            if (!err6) free(pb);
            return T(-1);
        }

        vr = (T *)__builtin_assume_aligned(vr, 64);
        vb = (T *)__builtin_assume_aligned(vb, 64);
        xr = (T *)__builtin_assume_aligned(xr, 64);
        xb = (T *)__builtin_assume_aligned(xb, 64);
        pr = (T *)__builtin_assume_aligned(pr, 64);
        pb = (T *)__builtin_assume_aligned(pb, 64);
    }

    alignas(64) T corrections_stack[MAX_NT_STACK]; // boundary corrections
    T *__restrict__ corrections = corrections_stack;

    const bool malloc_nt = n_t > MAX_NT_STACK;
    if (malloc_nt)
    {
        size_t s   = n_t * sizeof(T);
        int    err = posix_memalign((void **)&corrections, 64, s);

        if (err) return T(-1);

        corrections = (T *)__builtin_assume_aligned(corrections, 64);
    }

    // adaptive w logic
    int     w_adaptive = (w <= T(0.1));
    int     prev_iter  = max_iter;
    const T rho        = fm::fabs((a + c) / b) * fm::cos(fm::m_pi<T> / (n + 1));
    const T w_theory   = (T(2.0) / (T(1.0) + fm::sqrt(T(1.0) - rho * rho))); // optimal non-P SOR w
    T       w_hi       = fm::fmin(T(1.9), w_theory + T(0.3));
    T       w_lo       = fm::fmax(T(0.6), w_theory - T(0.3));
    T       w_delta    = (w_hi - w_lo) / T(10);
    w                  = w_adaptive ? w_theory : w;
    int iter;

    // split-buffer bounds
    const int last_is_red = n & 1;           // whether the last element is red
    const int nr          = n / 2 + (n & 1); // size of red (even interior idx) array
    const int nb          = n / 2;           // size of black (odd interior idx) array
    const int nr_loop     = nb;              // max idx in nr that isn't the last element
    const int nb_loop     = nr - 1;          // max idx in nb that isn't the last element

    // correction / payoff logic
    const int is_call = (cp_flag & 1);
    const T   cf      = is_call ? c : a; // correction factor
    const T   s       = is_call ? s_max : -s_min;
    const T   k       = is_call ? -K : K;
    const T   sgn     = T(2.0) * is_call - T(1.0);

    // pre-compute boundary corrections
#pragma omp simd
    for (int i = 0; i < n_t; i++)
    {
        const T tau    = (i + 1) * dt;
        corrections[i] = cf * fm::fmax(s + k, s * fm::exp(-q * tau) + k * fm::exp(-r * tau));
    }

    // pre-compute exercise/projection values
#pragma omp simd
    for (int i = 0; i < nr; i++)
    {
        pr[i] = fm::fmax(fm::fma(sgn, fm::exp(fm::fma(T(2 * i + 1), dx, x_min)), k), T(0.0));
        vr[i] = pr[i];
        pb[i] = fm::fmax(fm::fma(sgn, fm::exp(fm::fma(T(2 * i + 2), dx, x_min)), k), T(0.0));
        vb[i] = pb[i];
    }

    // backtrack
    for (int t = 0; t < n_t; t++)
    {
        T  p_last   = last_is_red ? pr[nr - 1] : pb[nb - 1];   // guess for last x
        T *x_last   = last_is_red ? &xr[nr - 1] : &xb[nb - 1]; // last x
        T *v_last   = last_is_red ? &vr[nr - 1] : &vb[nb - 1]; // v to update last x
        T *x_b4last = last_is_red ? &xb[nb - 1] : &xr[nr - 1]; // x[n-2]
        T *x_b4_it0 = last_is_red ? &pb[nb - 1] : &xr[nr - 1]; // x[n-2] on iter 0
        T *v_corr   = is_call ? v_last : &vr[0];               // boundary to correct
        T  one_m_w  = T(1.0) - w;
        T  w_div_b  = w / b;

        *v_corr -= corrections[t];

        // manual first iteration to avoid copying p -> x
        // red
        xr[0] = fm::fmax(pr[0], one_m_w * pr[0] + w_div_b * (vr[0] - c * pb[0]));

#pragma omp simd
        for (int j = 1; j < nr_loop; j++)
            xr[j] =
                fm::fmax(pr[j], one_m_w * pr[j] + w_div_b * (vr[j] - a * pb[j - 1] - c * pb[j]));

        // last red or first black
        *x_last = fm::fmax(p_last, one_m_w * p_last + w_div_b * ((*v_last) - a * (*x_b4_it0)));

        // black
#pragma omp simd
        for (int j = 0; j < nb_loop; j++)
            xb[j] =
                fm::fmax(pb[j], one_m_w * pb[j] + w_div_b * (vb[j] - a * xr[j] - c * xr[j + 1]));

        // remaining iterations
        for (iter = 1; iter < max_iter; iter++)
        {
            T sqdist, old, diff;
            sqdist = T(0.0);

            // red
            old   = xr[0];
            xr[0] = fm::fmax(pr[0], one_m_w * xr[0] + w_div_b * (vr[0] - c * xb[0]));
            diff  = xr[0] - old;
            sqdist += diff * diff;

#pragma omp simd
            for (int j = 1; j < nr_loop; j++)
            {
                old   = xr[j];
                xr[j] = fm::fmax(pr[j],
                                 one_m_w * xr[j] + w_div_b * (vr[j] - a * xb[j - 1] - c * xb[j]));
                diff  = xr[j] - old;
                sqdist += diff * diff;
            }

            // last red or first black
            old = *x_last;
            *x_last =
                fm::fmax(p_last, one_m_w * (*x_last) + w_div_b * ((*v_last) - a * (*x_b4last)));
            diff = *x_last - old;
            sqdist += diff * diff;

            // black
#pragma omp simd
            for (int j = 0; j < nb_loop; j++)
            {
                old   = xb[j];
                xb[j] = fm::fmax(pb[j],
                                 one_m_w * xb[j] + w_div_b * (vb[j] - a * xr[j] - c * xr[j + 1]));
                diff  = xb[j] - old;
                sqdist += diff * diff;
            }

            if (sqdist / n < tol2)
            {
                break;
            }
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
                if (w_delta < T(0))
                    w_hi = w - w_delta;
                else
                    w_lo = w - w_delta;

                w_delta = fm::copysign((w_hi - w_lo) / T(10.0), w_delta);
            }
            w = fm::fmin(fm::fmax(w + w_delta, w_lo), w_hi);
        }
        prev_iter = iter;
    }

    // final interpolation
    const T v = rb_interp(S, x_min, dx, n, vr, vb);

    // compute greeks
    if (delta || gamma)
    {
        const T h  = S * fm::fmax(fm::exp(dx) - T(1.0), T(1e-4));
        const T vp = rb_interp(S + h, x_min, dx, n, vr, vb);
        const T vm = rb_interp(S - h, x_min, dx, n, vr, vb);

        if (delta) *delta = (vp - vm) / (T(2.0) * h);

        if (gamma) *gamma = (vp - T(2.0) * v + vm) / (h * h);
    }
    if (theta)
    {
        const T vp = rb_interp(S, x_min, dx, n, xr, xb);
        *theta     = (vp - v) / (dt * T(365.0));
    }

    // cleanup
    if (malloc_ns)
    {
        free(vr);
        free(vb);
        free(xr);
        free(xb);
        free(pr);
        free(pb);
    }
    if (malloc_nt) free(corrections);

    return v;
}

/* greeks ----------------------------------------------------------------------------------------*/
/**
 * @brief Calculates the Greeks of an American option using the PSOR method.
 * @tparam T The floating-point type (float or double).
 * @see price for parameter details.
 */
template <typename T>
void greeks(T    S,
            T    K,
            char cp_flag,
            T    ttm,
            T    iv,
            T    r,
            T    q,
            int  n_s,
            int  n_t,
            int  k_mul    = 4,
            T    w        = T{0},
            T    tol      = T{1e-4},
            int  max_iter = 200,
            T   *delta    = nullptr,
            T   *gamma    = nullptr,
            T   *theta    = nullptr,
            T   *vega     = nullptr,
            T   *rho      = nullptr)
{
    if (delta || gamma || theta)
        price<T>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, delta, gamma, theta);

    if (vega || rho)
        fd::greeks<T>(
            S,
            K,
            cp_flag,
            ttm,
            iv,
            r,
            q,
            [n_s, n_t, k_mul, w, tol, max_iter](T S, T K, char cp, T ttm, T iv, T r, T q)
            { return price<T>(S, K, cp, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter); },
            nullptr,
            nullptr,
            nullptr,
            vega,
            rho);
}

/**
 * @brief Calculates the Delta of an American option using the PSOR method.
 * @see greeks for parameter details.
 */
template <typename T>
T delta(T    S,
        T    K,
        char cp_flag,
        T    ttm,
        T    iv,
        T    r,
        T    q,
        int  n_s,
        int  n_t,
        int  k_mul    = 4,
        T    w        = T{0},
        T    tol      = T{1e-4},
        int  max_iter = 200)
{
    T result;
    price<T>(
        S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, &result, nullptr, nullptr);
    return result;
}

/**
 * @brief Calculates the Gamma of an American option using the PSOR method.
 * @see greeks for parameter details.
 */
template <typename T>
T gamma(T    S,
        T    K,
        char cp_flag,
        T    ttm,
        T    iv,
        T    r,
        T    q,
        int  n_s,
        int  n_t,
        int  k_mul    = 4,
        T    w        = T{0},
        T    tol      = T{1e-4},
        int  max_iter = 200)
{
    T result;
    price<T>(
        S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, nullptr, &result, nullptr);
    return result;
}

/**
 * @brief Calculates the Theta of an American option using the PSOR method.
 * @see greeks for parameter details.
 */
template <typename T>
T theta(T    S,
        T    K,
        char cp_flag,
        T    ttm,
        T    iv,
        T    r,
        T    q,
        int  n_s,
        int  n_t,
        int  k_mul    = 4,
        T    w        = T{0},
        T    tol      = T{1e-4},
        int  max_iter = 200)
{
    T result;
    price<T>(
        S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, nullptr, nullptr, &result);
    return result;
}

/**
 * @brief Calculates the Vega of an American option using the PSOR method.
 * @see greeks for parameter details.
 */
template <typename T>
T vega(T    S,
       T    K,
       char cp_flag,
       T    ttm,
       T    iv,
       T    r,
       T    q,
       int  n_s,
       int  n_t,
       int  k_mul    = 4,
       T    w        = T{0},
       T    tol      = T{1e-4},
       int  max_iter = 200)
{
    return fd::vega<T>(
        S,
        K,
        cp_flag,
        ttm,
        iv,
        r,
        q,
        [n_s, n_t, k_mul, w, tol, max_iter](T S, T K, char cp, T ttm, T iv, T r, T q)
        { return price<T>(S, K, cp, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter); });
}

/**
 * @brief Calculates the Rho of an American option using the PSOR method.
 * @see greeks for parameter details.
 */
template <typename T>
T rho(T    S,
      T    K,
      char cp_flag,
      T    ttm,
      T    iv,
      T    r,
      T    q,
      int  n_s,
      int  n_t,
      int  k_mul    = 4,
      T    w        = T{0},
      T    tol      = T{1e-4},
      int  max_iter = 200)
{
    return fd::rho<T>(
        S,
        K,
        cp_flag,
        ttm,
        iv,
        r,
        q,
        [n_s, n_t, k_mul, w, tol, max_iter](T S, T K, char cp, T ttm, T iv, T r, T q)
        { return price<T>(S, K, cp, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter); });
}

/* iv --------------------------------------------------------------------------------------------*/
/**
 * @brief Calculates the implied volatility of an American option using the PSOR method.
 * @tparam T The floating-point type (float or double).
 * @param P The market price of the option.
 * @param psor_tol The tolerance for the inner PSOR solver.
 * @param psor_max_iter The maximum number of iterations for the inner PSOR solver.
 * @param tol The tolerance for the root-finding algorithm.
 * @param max_iter The maximum number of iterations for the root-finding algorithm.
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
     int    n_s,
     int    n_t,
     int    k_mul                  = 4,
     T      w                      = T{0},
     T      psor_tol               = T{1e-4},
     int    psor_max_iter          = 200,
     T      tol                    = T{1e-3},
     size_t max_iter               = 100,
     const T *__restrict__ lo_init = nullptr,
     const T *__restrict__ hi_init = nullptr)
{
    T init, lo, hi;
    if (!lo_init || !hi_init)
    {
        if (n_s <= EURO_PSOR_THRESHOLD && n_t <= EURO_PSOR_THRESHOLD)
        {
            init = fastvol::european::bsm::iv(
                P, S, K, cp_flag, ttm, r, q, INIT_TOL<T>, INIT_MAX_ITER, lo_init, hi_init);
            lo = lo_init ? *lo_init : (T(1.0) - INIT_EURO_MARGIN<T>)*init;
            hi = hi_init ? *hi_init : (T(1.0) + INIT_EURO_MARGIN<T>)*init;
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
                { return price<T>(S, K, cp, ttm, iv, r, q, 128, 256); },
                INIT_TOL<T>,
                INIT_MAX_ITER,
                lo_init,
                hi_init);
            lo = lo_init ? *lo_init : (T(1.0) - INIT_PSOR_MARGIN<T>)*init;
            hi = hi_init ? *hi_init : (T(1.0) + INIT_PSOR_MARGIN<T>)*init;
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
        [n_s, n_t, k_mul, w, psor_tol, psor_max_iter](T S, T K, char cp, T ttm, T iv, T r, T q)
        { return price<T>(S, K, cp, ttm, iv, r, q, n_s, n_t, k_mul, w, psor_tol, psor_max_iter); },
        tol,
        max_iter,
        lo_init ? lo_init : &lo,
        hi_init ? hi_init : &hi);
}

/* price_batch -----------------------------------------------------------------------------------*/
/**
 * @brief Calculates the prices of a batch of American options using the PSOR method on the CPU.
 * @tparam T The floating-point type (float or double).
 * @see price for parameter details.
 */
template <typename T>
void price_batch(const T *__restrict__ S,
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
                 T *__restrict__ delta = nullptr,
                 T *__restrict__ gamma = nullptr,
                 T *__restrict__ theta = nullptr)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        T v = price<T>(S[i],
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
                       delta ? &delta[i] : nullptr,
                       gamma ? &gamma[i] : nullptr,
                       theta ? &theta[i] : nullptr);

        if (result) result[i] = v;
    }
}

/* greeks_batch ----------------------------------------------------------------------------------*/
/**
 * @brief Calculates the Greeks for a batch of American options using the PSOR method on the CPU.
 * @tparam T The floating-point type (float or double).
 * @see greeks for parameter details.
 */
template <typename T>
void greeks_batch(const T *__restrict__ S,
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
                  T *__restrict__ delta = nullptr,
                  T *__restrict__ gamma = nullptr,
                  T *__restrict__ theta = nullptr,
                  T *__restrict__ vega  = nullptr,
                  T *__restrict__ rho   = nullptr)
{
    if (delta || gamma || theta)
        price_batch<T>(S,
                       K,
                       cp_flag,
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
                       n_options,
                       nullptr,
                       delta,
                       gamma,
                       theta);

    if (vega || rho)
        fd::greeks_batch<T>(
            S,
            K,
            cp_flag,
            ttm,
            iv,
            r,
            q,
            [n_s, n_t, k_mul, w, tol, max_iter](T S, T K, char cp, T ttm, T iv, T r, T q)
            { return price<T>(S, K, cp, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter); },
            n_options,
            nullptr,
            nullptr,
            nullptr,
            vega,
            rho);
}

/**
 * @brief Calculates the Deltas for a batch of American options using the PSOR method on the CPU.
 * @see greeks_batch for parameter details.
 */
template <typename T>
inline void delta_batch(const T *__restrict__ S,
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
                        T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        results[i] = delta<T>(
            S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], n_s, n_t, k_mul, w, tol, max_iter);
    }
}

/**
 * @brief Calculates the Gammas for a batch of American options using the PSOR method on the CPU.
 * @see greeks_batch for parameter details.
 */
template <typename T>
inline void gamma_batch(const T *__restrict__ S,
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
                        T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        results[i] = gamma<T>(
            S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], n_s, n_t, k_mul, w, tol, max_iter);
    }
}

/**
 * @brief Calculates the Thetas for a batch of American options using the PSOR method on the CPU.
 * @see greeks_batch for parameter details.
 */
template <typename T>
inline void theta_batch(const T *__restrict__ S,
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
                        T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        results[i] = theta<T>(
            S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], n_s, n_t, k_mul, w, tol, max_iter);
    }
}

/**
 * @brief Calculates the Vegas for a batch of American options using the PSOR method on the CPU.
 * @see greeks_batch for parameter details.
 */
template <typename T>
inline void vega_batch(const T *__restrict__ S,
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
                       T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        results[i] = vega<T>(
            S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], n_s, n_t, k_mul, w, tol, max_iter);
    }
}

/**
 * @brief Calculates the Rhos for a batch of American options using the PSOR method on the CPU.
 * @see greeks_batch for parameter details.
 */
template <typename T>
inline void rho_batch(const T *__restrict__ S,
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
                      T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        results[i] = rho<T>(
            S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], n_s, n_t, k_mul, w, tol, max_iter);
    }
}

/* iv_batch --------------------------------------------------------------------------------------*/
/**
 * @brief Calculates the implied volatilities for a batch of American options using the PSOR method
 * on the CPU.
 * @tparam T The floating-point type (float or double).
 * @see iv for parameter details.
 */
template <typename T>
void iv_batch(const T *__restrict__ P,
              const T *__restrict__ S,
              const T *__restrict__ K,
              const char *__restrict__ cp_flag,
              const T *__restrict__ ttm,
              const T *__restrict__ r,
              const T *__restrict__ q,
              int    n_s,
              int    n_t,
              int    k_mul,
              T      w,
              T      psor_tol,
              int    psor_max_iter,
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
        if (n_s <= EURO_PSOR_THRESHOLD && n_t <= EURO_PSOR_THRESHOLD)
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
            margin = INIT_PSOR_MARGIN<T>;
            fastvol::detail::iv::brent_batch(
                P,
                S,
                K,
                cp_flag,
                ttm,
                r,
                q,
                [](T S, T K, char cp, T ttm, T iv, T r, T q)
                { return price<T>(S, K, cp, ttm, iv, r, q, 128, 256); },
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
        [n_s, n_t, k_mul, w, psor_tol, psor_max_iter](T S, T K, char cp, T ttm, T iv, T r, T q)
        { return price<T>(S, K, cp, ttm, iv, r, q, n_s, n_t, k_mul, w, psor_tol, psor_max_iter); },
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
inline double price_fp64(double S,
                         double K,
                         char   cp_flag,
                         double ttm,
                         double iv,
                         double r,
                         double q,
                         int    n_s,
                         int    n_t,
                         int    k_mul               = 4,
                         double w                   = 0.0,
                         double tol                 = 1e-4,
                         int    max_iter            = 200,
                         double *__restrict__ delta = nullptr,
                         double *__restrict__ gamma = nullptr,
                         double *__restrict__ theta = nullptr)
{
    return price<double>(
        S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, delta, gamma, theta);
}

inline void price_fp64_batch(const double *__restrict__ S,
                             const double *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const double *__restrict__ ttm,
                             const double *__restrict__ iv,
                             const double *__restrict__ r,
                             const double *__restrict__ q,
                             int    n_s,
                             int    n_t,
                             int    k_mul,
                             double w,
                             double tol,
                             int    max_iter,
                             size_t n_options,
                             double *__restrict__ result,
                             double *__restrict__ delta = nullptr,
                             double *__restrict__ gamma = nullptr,
                             double *__restrict__ theta = nullptr)
{
    price_batch<double>(S,
                        K,
                        cp_flag,
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
                        n_options,
                        result,
                        delta,
                        gamma,
                        theta);
}

/**
 * @brief CUDA kernel launcher for PSOR price calculation (double precision).
 * @see price for parameter details.
 */
void price_fp64_cuda(const double *__restrict__ S,
                     const double *__restrict__ K,
                     const char *__restrict__ cp_flag,
                     const double *__restrict__ ttm,
                     const double *__restrict__ iv,
                     const double *__restrict__ r,
                     const double *__restrict__ q,
                     int    n_s,
                     int    n_t,
                     int    k_mul,
                     double w,
                     double tol,
                     int    max_iter,
                     size_t n_options,
                     double *__restrict__ result,
                     double *__restrict__ delta,
                     double *__restrict__ gamma,
                     double *__restrict__ theta,
                     cudaStream_t stream,
                     int          device,
                     bool         on_device,
                     bool         is_pinned,
                     bool         sync);

/* greeks ________________________________________________________________________________________*/
inline void greeks_fp64(double  S,
                        double  K,
                        char    cp_flag,
                        double  ttm,
                        double  iv,
                        double  r,
                        double  q,
                        int     n_s,
                        int     n_t,
                        int     k_mul    = 4,
                        double  w        = 0.0,
                        double  tol      = 1e-4,
                        int     max_iter = 200,
                        double *delta    = nullptr,
                        double *gamma    = nullptr,
                        double *theta    = nullptr,
                        double *vega     = nullptr,
                        double *rho      = nullptr)
{
    return greeks<double>(S,
                          K,
                          cp_flag,
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
                          delta,
                          gamma,
                          theta,
                          vega,
                          rho);
}

inline double delta_fp64(double S,
                         double K,
                         char   cp_flag,
                         double ttm,
                         double iv,
                         double r,
                         double q,
                         int    n_s,
                         int    n_t,
                         int    k_mul    = 4,
                         double w        = 0.0,
                         double tol      = 1e-4,
                         int    max_iter = 200)
{
    return delta<double>(S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
}

inline double gamma_fp64(double S,
                         double K,
                         char   cp_flag,
                         double ttm,
                         double iv,
                         double r,
                         double q,
                         int    n_s,
                         int    n_t,
                         int    k_mul    = 4,
                         double w        = 0.0,
                         double tol      = 1e-4,
                         int    max_iter = 200)
{
    return gamma<double>(S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
}

inline double theta_fp64(double S,
                         double K,
                         char   cp_flag,
                         double ttm,
                         double iv,
                         double r,
                         double q,
                         int    n_s,
                         int    n_t,
                         int    k_mul    = 4,
                         double w        = 0.0,
                         double tol      = 1e-4,
                         int    max_iter = 200)
{
    return theta<double>(S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
}

inline double vega_fp64(double S,
                        double K,
                        char   cp_flag,
                        double ttm,
                        double iv,
                        double r,
                        double q,
                        int    n_s,
                        int    n_t,
                        int    k_mul    = 4,
                        double w        = 0.0,
                        double tol      = 1e-4,
                        int    max_iter = 200)
{
    return vega<double>(S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
}

inline double rho_fp64(double S,
                       double K,
                       char   cp_flag,
                       double ttm,
                       double iv,
                       double r,
                       double q,
                       int    n_s,
                       int    n_t,
                       int    k_mul    = 4,
                       double w        = 0.0,
                       double tol      = 1e-4,
                       int    max_iter = 200)
{
    return rho<double>(S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
}

void greeks_fp64_batch(const double *__restrict__ S,
                       const double *__restrict__ K,
                       const char *__restrict__ cp_flag,
                       const double *__restrict__ ttm,
                       const double *__restrict__ iv,
                       const double *__restrict__ r,
                       const double *__restrict__ q,
                       int    n_s,
                       int    n_t,
                       int    k_mul,
                       double w,
                       double tol,
                       int    max_iter,
                       size_t n_options,
                       double *__restrict__ delta = nullptr,
                       double *__restrict__ gamma = nullptr,
                       double *__restrict__ theta = nullptr,
                       double *__restrict__ vega  = nullptr,
                       double *__restrict__ rho   = nullptr)
{
    greeks_batch<double>(S,
                         K,
                         cp_flag,
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
                         n_options,
                         delta,
                         gamma,
                         theta,
                         vega,
                         rho);
}

inline void delta_fp64_batch(const double *__restrict__ S,
                             const double *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const double *__restrict__ ttm,
                             const double *__restrict__ iv,
                             const double *__restrict__ r,
                             const double *__restrict__ q,
                             int    n_s,
                             int    n_t,
                             int    k_mul,
                             double w,
                             double tol,
                             int    max_iter,
                             size_t n_options,
                             double *__restrict__ results)
{
    delta_batch<double>(
        S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
}

inline void gamma_fp64_batch(const double *__restrict__ S,
                             const double *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const double *__restrict__ ttm,
                             const double *__restrict__ iv,
                             const double *__restrict__ r,
                             const double *__restrict__ q,
                             int    n_s,
                             int    n_t,
                             int    k_mul,
                             double w,
                             double tol,
                             int    max_iter,
                             size_t n_options,
                             double *__restrict__ results)
{
    gamma_batch<double>(
        S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
}

inline void theta_fp64_batch(const double *__restrict__ S,
                             const double *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const double *__restrict__ ttm,
                             const double *__restrict__ iv,
                             const double *__restrict__ r,
                             const double *__restrict__ q,
                             int    n_s,
                             int    n_t,
                             int    k_mul,
                             double w,
                             double tol,
                             int    max_iter,
                             size_t n_options,
                             double *__restrict__ results)
{
    theta_batch<double>(
        S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
}

inline void vega_fp64_batch(const double *__restrict__ S,
                            const double *__restrict__ K,
                            const char *__restrict__ cp_flag,
                            const double *__restrict__ ttm,
                            const double *__restrict__ iv,
                            const double *__restrict__ r,
                            const double *__restrict__ q,
                            int    n_s,
                            int    n_t,
                            int    k_mul,
                            double w,
                            double tol,
                            int    max_iter,
                            size_t n_options,
                            double *__restrict__ results)
{
    vega_batch<double>(
        S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
}

inline void rho_fp64_batch(const double *__restrict__ S,
                           const double *__restrict__ K,
                           const char *__restrict__ cp_flag,
                           const double *__restrict__ ttm,
                           const double *__restrict__ iv,
                           const double *__restrict__ r,
                           const double *__restrict__ q,
                           int    n_s,
                           int    n_t,
                           int    k_mul,
                           double w,
                           double tol,
                           int    max_iter,
                           size_t n_options,
                           double *__restrict__ results)
{
    rho_batch<double>(
        S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
}

/* iv ____________________________________________________________________________________________*/
inline double iv_fp64(double P,
                      double S,
                      double K,
                      char   cp_flag,
                      double ttm,
                      double r,
                      double q,
                      int    n_s,
                      int    n_t,
                      int    k_mul                       = 4,
                      double w                           = 0.0,
                      double psor_tol                    = 1e-4,
                      int    psor_max_iter               = 200,
                      double tol                         = 1e-3,
                      size_t max_iter                    = 100,
                      const double *__restrict__ lo_init = nullptr,
                      const double *__restrict__ hi_init = nullptr)
{
    return iv<double>(P,
                      S,
                      K,
                      cp_flag,
                      ttm,
                      r,
                      q,
                      n_s,
                      n_t,
                      k_mul,
                      w,
                      psor_tol,
                      psor_max_iter,
                      tol,
                      max_iter,
                      lo_init,
                      hi_init);
}

inline void iv_fp64_batch(const double *__restrict__ P,
                          const double *__restrict__ S,
                          const double *__restrict__ K,
                          const char *__restrict__ cp_flag,
                          const double *__restrict__ ttm,
                          const double *__restrict__ r,
                          const double *__restrict__ q,
                          int    n_s,
                          int    n_t,
                          int    k_mul,
                          double w,
                          double psor_tol,
                          int    psor_max_iter,
                          size_t n_options,
                          double *__restrict__ results,
                          double tol                         = 1e-3,
                          size_t max_iter                    = 100,
                          const double *__restrict__ lo_init = nullptr,
                          const double *__restrict__ hi_init = nullptr)
{
    iv_batch<double>(P,
                     S,
                     K,
                     cp_flag,
                     ttm,
                     r,
                     q,
                     n_s,
                     n_t,
                     k_mul,
                     w,
                     psor_tol,
                     psor_max_iter,
                     n_options,
                     results,
                     tol,
                     max_iter,
                     lo_init,
                     hi_init);
}

/* fp32 ------------------------------------------------------------------------------------------*/
/* price _________________________________________________________________________________________*/
inline float price_fp32(float S,
                        float K,
                        char  cp_flag,
                        float ttm,
                        float iv,
                        float r,
                        float q,
                        int   n_s,
                        int   n_t,
                        int   k_mul               = 4,
                        float w                   = 0.0f,
                        float tol                 = 1e-4f,
                        int   max_iter            = 200,
                        float *__restrict__ delta = nullptr,
                        float *__restrict__ gamma = nullptr,
                        float *__restrict__ theta = nullptr)
{
    return price<float>(
        S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, delta, gamma, theta);
}

inline void price_fp32_batch(const float *__restrict__ S,
                             const float *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const float *__restrict__ ttm,
                             const float *__restrict__ iv,
                             const float *__restrict__ r,
                             const float *__restrict__ q,
                             int    n_s,
                             int    n_t,
                             int    k_mul,
                             float  w,
                             float  tol,
                             int    max_iter,
                             size_t n_options,
                             float *__restrict__ result,
                             float *__restrict__ delta = nullptr,
                             float *__restrict__ gamma = nullptr,
                             float *__restrict__ theta = nullptr)
{
    price_batch<float>(S,
                       K,
                       cp_flag,
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
                       n_options,
                       result,
                       delta,
                       gamma,
                       theta);
}

/**
 * @brief CUDA kernel launcher for PSOR price calculation (single precision).
 * @see price for parameter details.
 */
void price_fp32_cuda(const float *__restrict__ S,
                     const float *__restrict__ K,
                     const char *__restrict__ cp_flag,
                     const float *__restrict__ ttm,
                     const float *__restrict__ iv,
                     const float *__restrict__ r,
                     const float *__restrict__ q,
                     int    n_s,
                     int    n_t,
                     int    k_mul,
                     float  w,
                     float  tol,
                     int    max_iter,
                     size_t n_options,
                     float *__restrict__ result,
                     float *__restrict__ delta,
                     float *__restrict__ gamma,
                     float *__restrict__ theta,
                     cudaStream_t stream,
                     int          device,
                     bool         on_device,
                     bool         is_pinned,
                     bool         sync);

/* greeks ________________________________________________________________________________________*/
inline void greeks_fp32(float  S,
                        float  K,
                        char   cp_flag,
                        float  ttm,
                        float  iv,
                        float  r,
                        float  q,
                        int    n_s,
                        int    n_t,
                        int    k_mul    = 4,
                        float  w        = 0.0f,
                        float  tol      = 1e-4f,
                        int    max_iter = 200,
                        float *delta    = nullptr,
                        float *gamma    = nullptr,
                        float *theta    = nullptr,
                        float *vega     = nullptr,
                        float *rho      = nullptr)
{
    return greeks<float>(S,
                         K,
                         cp_flag,
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
                         delta,
                         gamma,
                         theta,
                         vega,
                         rho);
}

inline float delta_fp32(float S,
                        float K,
                        char  cp_flag,
                        float ttm,
                        float iv,
                        float r,
                        float q,
                        int   n_s,
                        int   n_t,
                        int   k_mul    = 4,
                        float w        = 0.0f,
                        float tol      = 1e-4f,
                        int   max_iter = 200)
{
    return delta<float>(S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
}

inline float gamma_fp32(float S,
                        float K,
                        char  cp_flag,
                        float ttm,
                        float iv,
                        float r,
                        float q,
                        int   n_s,
                        int   n_t,
                        int   k_mul    = 4,
                        float w        = 0.0f,
                        float tol      = 1e-4f,
                        int   max_iter = 200)
{
    return gamma<float>(S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
}

inline float theta_fp32(float S,
                        float K,
                        char  cp_flag,
                        float ttm,
                        float iv,
                        float r,
                        float q,
                        int   n_s,
                        int   n_t,
                        int   k_mul    = 4,
                        float w        = 0.0f,
                        float tol      = 1e-4f,
                        int   max_iter = 200)
{
    return theta<float>(S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
}

inline float vega_fp32(float S,
                       float K,
                       char  cp_flag,
                       float ttm,
                       float iv,
                       float r,
                       float q,
                       int   n_s,
                       int   n_t,
                       int   k_mul    = 4,
                       float w        = 0.0f,
                       float tol      = 1e-4f,
                       int   max_iter = 200)
{
    return vega<float>(S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
}

inline float rho_fp32(float S,
                      float K,
                      char  cp_flag,
                      float ttm,
                      float iv,
                      float r,
                      float q,
                      int   n_s,
                      int   n_t,
                      int   k_mul    = 4,
                      float w        = 0.0f,
                      float tol      = 1e-4f,
                      int   max_iter = 200)
{
    return rho<float>(S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
}

void greeks_fp32_batch(const float *__restrict__ S,
                       const float *__restrict__ K,
                       const char *__restrict__ cp_flag,
                       const float *__restrict__ ttm,
                       const float *__restrict__ iv,
                       const float *__restrict__ r,
                       const float *__restrict__ q,
                       int    n_s,
                       int    n_t,
                       int    k_mul,
                       float  w,
                       float  tol,
                       int    max_iter,
                       size_t n_options,
                       float *__restrict__ delta = nullptr,
                       float *__restrict__ gamma = nullptr,
                       float *__restrict__ theta = nullptr,
                       float *__restrict__ vega  = nullptr,
                       float *__restrict__ rho   = nullptr)
{
    greeks_batch<float>(S,
                        K,
                        cp_flag,
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
                        n_options,
                        delta,
                        gamma,
                        theta,
                        vega,
                        rho);
}

inline void delta_fp32_batch(const float *__restrict__ S,
                             const float *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const float *__restrict__ ttm,
                             const float *__restrict__ iv,
                             const float *__restrict__ r,
                             const float *__restrict__ q,
                             int    n_s,
                             int    n_t,
                             int    k_mul,
                             float  w,
                             float  tol,
                             int    max_iter,
                             size_t n_options,
                             float *__restrict__ results)
{
    delta_batch<float>(
        S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
}

inline void gamma_fp32_batch(const float *__restrict__ S,
                             const float *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const float *__restrict__ ttm,
                             const float *__restrict__ iv,
                             const float *__restrict__ r,
                             const float *__restrict__ q,
                             int    n_s,
                             int    n_t,
                             int    k_mul,
                             float  w,
                             float  tol,
                             int    max_iter,
                             size_t n_options,
                             float *__restrict__ results)
{
    gamma_batch<float>(
        S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
}

inline void theta_fp32_batch(const float *__restrict__ S,
                             const float *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const float *__restrict__ ttm,
                             const float *__restrict__ iv,
                             const float *__restrict__ r,
                             const float *__restrict__ q,
                             int    n_s,
                             int    n_t,
                             int    k_mul,
                             float  w,
                             float  tol,
                             int    max_iter,
                             size_t n_options,
                             float *__restrict__ results)
{
    theta_batch<float>(
        S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
}

inline void vega_fp32_batch(const float *__restrict__ S,
                            const float *__restrict__ K,
                            const char *__restrict__ cp_flag,
                            const float *__restrict__ ttm,
                            const float *__restrict__ iv,
                            const float *__restrict__ r,
                            const float *__restrict__ q,
                            int    n_s,
                            int    n_t,
                            int    k_mul,
                            float  w,
                            float  tol,
                            int    max_iter,
                            size_t n_options,
                            float *__restrict__ results)
{
    vega_batch<float>(
        S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
}

inline void rho_fp32_batch(const float *__restrict__ S,
                           const float *__restrict__ K,
                           const char *__restrict__ cp_flag,
                           const float *__restrict__ ttm,
                           const float *__restrict__ iv,
                           const float *__restrict__ r,
                           const float *__restrict__ q,
                           int    n_s,
                           int    n_t,
                           int    k_mul,
                           float  w,
                           float  tol,
                           int    max_iter,
                           size_t n_options,
                           float *__restrict__ results)
{
    rho_batch<float>(
        S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
}

/* iv ____________________________________________________________________________________________*/
inline float iv_fp32(float  P,
                     float  S,
                     float  K,
                     char   cp_flag,
                     float  ttm,
                     float  r,
                     float  q,
                     int    n_s,
                     int    n_t,
                     int    k_mul                      = 4,
                     float  w                          = 0.0f,
                     float  psor_tol                   = 1e-4f,
                     int    psor_max_iter              = 200,
                     float  tol                        = 1e-3f,
                     size_t max_iter                   = 100,
                     const float *__restrict__ lo_init = nullptr,
                     const float *__restrict__ hi_init = nullptr)
{
    return iv<float>(P,
                     S,
                     K,
                     cp_flag,
                     ttm,
                     r,
                     q,
                     n_s,
                     n_t,
                     k_mul,
                     w,
                     psor_tol,
                     psor_max_iter,
                     tol,
                     max_iter,
                     lo_init,
                     hi_init);
}

inline void iv_fp32_batch(const float *__restrict__ P,
                          const float *__restrict__ S,
                          const float *__restrict__ K,
                          const char *__restrict__ cp_flag,
                          const float *__restrict__ ttm,
                          const float *__restrict__ r,
                          const float *__restrict__ q,
                          int    n_s,
                          int    n_t,
                          int    k_mul,
                          float  w,
                          float  psor_tol,
                          int    psor_max_iter,
                          size_t n_options,
                          float *__restrict__ results,
                          float  tol                        = 1e-3f,
                          size_t max_iter                   = 100,
                          const float *__restrict__ lo_init = nullptr,
                          const float *__restrict__ hi_init = nullptr)
{
    iv_batch<float>(P,
                    S,
                    K,
                    cp_flag,
                    ttm,
                    r,
                    q,
                    n_s,
                    n_t,
                    k_mul,
                    w,
                    psor_tol,
                    psor_max_iter,
                    n_options,
                    results,
                    tol,
                    max_iter,
                    lo_init,
                    hi_init);
}

} // namespace fastvol::american::psor
