/**
 * @file iv.hpp
 * @brief Implied volatility solvers.
 *
 * This file contains the implementations of numerical methods for finding the implied
 * volatility of an option. It includes the Bisection, Brent's, and Newton's methods,
 * which are used by the higher-level option pricing models.
 */

#pragma once

#include "fastvol/detail/math.hpp"
#include <cmath>
#include <cstddef>
#include <limits>
#include <omp.h>

namespace fm = fastvol::detail::math;

/**
 * @namespace fastvol::detail::iv
 * @brief Provides generic root-finding algorithms for implied volatility calculations.
 *
 * This namespace contains implementations of several numerical methods (Bisection, Brent's, Newton's)
 * used to find the implied volatility that matches a given market price. These functions are
 * templated to work with any pricing function and are provided in both single-option and
 * batch (OpenMP-parallelized) versions.
 */
namespace fastvol::detail::iv
{
// constants
template <typename T> inline constexpr T IV_NAN = T(-1.0);
template <typename T> inline constexpr T IV_ARB = T(-2.0);

template <typename T> inline constexpr T IV_MIN = T(1e-4);
template <typename T> inline constexpr T IV_MAX = T(20.0);

template <typename T> inline constexpr T LO_MUL = T(0.8);
template <typename T> inline constexpr T HI_MUL = T(1.2);

template <typename T> constexpr T VEGA_MIN() { return static_cast<T>(1e-12); }
template <> constexpr float       VEGA_MIN<float>() { return 1e-6f; }

/* bisection -------------------------------------------------------------------------------------*/
/**
 * @brief Calculates implied volatility using the bisection method.
 * @tparam T The floating-point type.
 * @tparam PriceFunc The type of the pricing function.
 * @param P The market price of the option.
 * @param S The current price of the underlying asset.
 * @param K The strike price of the option.
 * @param cp_flag The option type flag.
 * @param ttm The time to maturity.
 * @param r The risk-free interest rate.
 * @param q The dividend yield.
 * @param price_fn A callable object that computes the option price for a given IV.
 * @param tol The tolerance for the root-finding algorithm.
 * @param max_iter The maximum number of iterations.
 * @param lo_init An optional initial lower bound for the search.
 * @param hi_init An optional initial upper bound for the search.
 * @return The calculated implied volatility, or a special value (IV_NAN, IV_ARB) on failure.
 */
template <typename T, typename PriceFunc>
T bisection(T         P,
            T         S,
            T         K,
            char      cp_flag,
            T         ttm,
            T         r,
            T         q,
            PriceFunc price_fn,
            T         tol      = T{1e-3},
            size_t    max_iter = 100,
            const T  *lo_init  = nullptr,
            const T  *hi_init  = nullptr)
{
    const T tol_p     = fm::fmax(tol, T(2.0) * std::numeric_limits<T>::epsilon() * fm::fabs(P));
    const T intrinsic = fm::fmax(((cp_flag & 1) ? S - K : K - S), T(0));

    // edge cases
    if (ttm <= T(0) || fm::fabs(P - intrinsic) < tol_p) return IV_NAN<T>;
    if (P < intrinsic) return IV_ARB<T>;

    // initial IV bounds
    T lo = lo_init ? *lo_init : T(0.05);
    T hi = hi_init ? *hi_init : T(1.0);

    T price_lo = price_fn(S, K, cp_flag, ttm, lo, r, q);
    T price_hi = price_fn(S, K, cp_flag, ttm, hi, r, q);

    // check bounds
    while (price_lo > P && lo > IV_MIN<T>)
    {
        lo *= LO_MUL<T>;
        price_lo = price_fn(S, K, cp_flag, ttm, lo, r, q);
    }
    while (price_hi < P && hi < IV_MAX<T>)
    {
        hi *= HI_MUL<T>;
        price_hi = price_fn(S, K, cp_flag, ttm, hi, r, q);
    }

    // no valid bracket
    if (price_lo > P || P > price_hi) return IV_NAN<T>;

    // bisection loop
    for (size_t i = 0; i < max_iter; i++)
    {
        T mid       = T(0.5) * (lo + hi);
        T price_mid = price_fn(S, K, cp_flag, ttm, mid, r, q);
        T tol_iv    = T(2.0) * std::numeric_limits<T>::epsilon() * fm::fabs(mid);

        if (fm::fabs(P - price_mid) < tol_p || fm::fabs(hi - lo) < tol_iv) return mid;

        if (price_mid > P)
            hi = mid;
        else
            lo = mid;
    }

    return T(0.5) * (lo + hi);
}

/* brent -----------------------------------------------------------------------------------------*/
/**
 * @brief Calculates implied volatility using Brent's method.
 * @see bisection for parameter details.
 */
template <typename T, typename PriceFunc>
T brent(T         P,
        T         S,
        T         K,
        char      cp_flag,
        T         ttm,
        T         r,
        T         q,
        PriceFunc price_fn,
        T         tol      = T{1e-3},
        size_t    max_iter = 100,
        const T  *lo_init  = nullptr,
        const T  *hi_init  = nullptr)
{
    const T tol_p     = fm::fmax(tol, T(2.0) * std::numeric_limits<T>::epsilon() * fm::fabs(P));
    const T intrinsic = fm::fmax(((cp_flag & 1) ? S - K : K - S), T(0));

    // edge cases
    if (ttm <= T(0) || fm::fabs(P - intrinsic) < tol_p) return IV_NAN<T>;
    if (P < intrinsic) return IV_ARB<T>;

    // initial IV bounds
    T a  = lo_init ? *lo_init : T(0.05);
    T b  = hi_init ? *hi_init : T(1.0);
    T fa = price_fn(S, K, cp_flag, ttm, a, r, q) - P;
    T fb = price_fn(S, K, cp_flag, ttm, b, r, q) - P;

    // check bounds
    while (fa > T(0) && a > IV_MIN<T>)
    {
        a *= LO_MUL<T>;
        fa = price_fn(S, K, cp_flag, ttm, a, r, q) - P;
    }
    while (fb < T(0) && b < IV_MAX<T>)
    {
        b *= HI_MUL<T>;
        fb = price_fn(S, K, cp_flag, ttm, b, r, q) - P;
    }

    // no valid bracket
    if (fa * fb > T(0)) return IV_NAN<T>;

    // brent variables
    T c = a, fc = fa; // third point c
    T d = b - a;      // step size
    T e = d;          // previous step size

    for (size_t iter = 0; iter < max_iter; ++iter)
    {
        // ensure |fb| <= |fa|
        if (fm::fabs(fb) > fm::fabs(fa))
        {
            c  = b;
            fc = fb;
            b  = a;
            fb = fa;
            a  = c;
            fa = fc;
        }

        // convergence tolerance
        T m      = T(0.5) * (c - b);                                         // midpoint
        T tol_iv = T(2.0) * std::numeric_limits<T>::epsilon() * fm::fabs(b); // iv-space num. tol
        if (fm::fabs(fb) <= tol_p || fm::fabs(m) <= tol_iv) return b;

        // interpolation
        T p = T(0), q_ = T(1);
        if (fm::fabs(e) >= tol_iv && fm::fabs(fa) > fm::fabs(fb))
        {
            T s = fb / fa;
            if (a == c) // secant 2-point interp
            {
                p  = T(2.0) * m * s;
                q_ = T(1.0) - s;
            }
            else // quadratic 3-point interp
            {
                T t = fb / fc;
                T u = fa / fc;
                p   = s * (T(2.0) * m * u * (u - t) - (b - a) * (t - T(1.0)));
                q_  = (u - T(1.0)) * (t - T(1.0)) * (s - T(1.0));
            }

            // accept if within [b - m, b + m]
            if (q_ != T(0))
            {
                if (p > T(0)) q_ = -q_; // force step direction toward root
                p = fm::fabs(p);
                if (T(2.0) * p <
                    fm::fmin(T(3.0) * m * q_ - fm::fabs(tol_iv * q_), fm::fabs(e * q_)))
                {
                    e = d; // accept interpolation
                    d = p / q_;
                }
                else
                {
                    d = m; // bisection fallback
                    e = d;
                }
            }
            else
            {
                d = m;
                e = d;
            }
        }
        else
        {
            // bisection fallback
            d = m;
            e = d;
        }

        // update bracket
        a  = b;
        fa = fb;

        // take step
        b = (fm::fabs(d) > tol_iv) ? b + d : b + (m > T(0) ? tol_iv : -tol_iv);

        // eval new point
        fb = price_fn(S, K, cp_flag, ttm, b, r, q) - P;

        // maintain bracketing condition
        if ((fb > T(0) && fc > T(0)) || (fb < T(0) && fc < T(0)))
        {
            c  = a;
            fc = fa;
            d  = b - a;
            e  = d;
        }
    }

    return IV_NAN<T>;
}

/* newton ----------------------------------------------------------------------------------------*/
/**
 * @brief Calculates implied volatility using Newton's method.
 * @tparam VegaFunc The type of the vega function.
 * @param vega_fn A callable object that computes the option's vega for a given IV.
 * @see bisection for other parameter details.
 */
template <typename T, typename PriceFunc, typename VegaFunc>
T newton(T         P,
         T         S,
         T         K,
         char      cp_flag,
         T         ttm,
         T         r,
         T         q,
         PriceFunc price_fn,
         VegaFunc  vega_fn,
         T         tol      = T{1e-3},
         size_t    max_iter = 100,
         const T  *lo_init  = nullptr,
         const T  *hi_init  = nullptr)
{
    const T tol_p     = fm::fmax(tol, T(2.0) * std::numeric_limits<T>::epsilon() * fm::fabs(P));
    const T intrinsic = fm::fmax(((cp_flag & 1) ? S - K : K - S), T(0));

    // edge cases
    if (ttm <= T(0) || fm::fabs(P - intrinsic) < tol_p) return IV_NAN<T>;
    if (P < intrinsic) return IV_ARB<T>;

    // initial IV guess
    T lo    = lo_init ? *lo_init : T(0.05);
    T hi    = hi_init ? *hi_init : T(0.5);
    T sigma = fm::fmax(T(0.5) * (lo + hi), T(0));

    for (size_t iter = 0; iter < max_iter; ++iter)
    {
        // compute price and vega
        T price = price_fn(S, K, cp_flag, ttm, sigma, r, q);
        T vega  = vega_fn(S, K, cp_flag, ttm, sigma, r, q);
        T f     = price - P;

        if (fm::fabs(f) <= tol_p) return sigma;               // convergence
        if (fm::fabs(vega) < VEGA_MIN<T>()) return IV_NAN<T>; // zero/small vega

        T sigma_new = sigma - f / vega; // newton step

        // bounds check
        if (sigma_new <= IV_MIN<T>)
            sigma_new = T(0.5) * (sigma + IV_MIN<T>);
        else if (sigma_new > IV_MAX<T>)
            sigma_new = T(0.5) * (sigma + IV_MAX<T>);

        // convergence in sigma space
        const T tol_iv = T(2.0) * std::numeric_limits<T>::epsilon() * fm::fabs(sigma);
        if (fm::fabs(sigma_new - sigma) <= tol_iv) return sigma_new;

        sigma = sigma_new;
    }
    return IV_NAN<T>;
}
/* bisection_batch -------------------------------------------------------------------------------*/
/**
 * @brief Calculates implied volatilities for a batch of options using the bisection method.
 * @param n_options The number of options in the batch.
 * @param[out] results A pre-allocated array to store the calculated implied volatilities.
 * @see bisection for other parameter details.
 */
template <typename T, typename PriceFunc>
void bisection_batch(const T *__restrict__ P,
                     const T *__restrict__ S,
                     const T *__restrict__ K,
                     const char *__restrict__ cp_flag,
                     const T *__restrict__ ttm,
                     const T *__restrict__ r,
                     const T *__restrict__ q,
                     PriceFunc price_fn,
                     size_t    n_options,
                     T *__restrict__ results,
                     T      tol                    = T{1e-3},
                     size_t max_iter               = 100,
                     const T *__restrict__ lo_init = nullptr,
                     const T *__restrict__ hi_init = nullptr)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
        results[i] = bisection(P[i],
                               S[i],
                               K[i],
                               cp_flag[i],
                               ttm[i],
                               r[i],
                               q[i],
                               price_fn,
                               tol,
                               max_iter,
                               lo_init ? &lo_init[i] : nullptr,
                               hi_init ? &hi_init[i] : nullptr);
}

/* brent_batch -----------------------------------------------------------------------------------*/
/**
 * @brief Calculates implied volatilities for a batch of options using Brent's method.
 * @see bisection_batch for parameter details.
 */
template <typename T, typename PriceFunc>
void brent_batch(const T *__restrict__ P,
                 const T *__restrict__ S,
                 const T *__restrict__ K,
                 const char *__restrict__ cp_flag,
                 const T *__restrict__ ttm,
                 const T *__restrict__ r,
                 const T *__restrict__ q,
                 PriceFunc price_fn,
                 size_t    n_options,
                 T *__restrict__ results,
                 T      tol                    = T{1e-3},
                 size_t max_iter               = 100,
                 const T *__restrict__ lo_init = nullptr,
                 const T *__restrict__ hi_init = nullptr)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
        results[i] = brent(P[i],
                           S[i],
                           K[i],
                           cp_flag[i],
                           ttm[i],
                           r[i],
                           q[i],
                           price_fn,
                           tol,
                           max_iter,
                           lo_init ? &lo_init[i] : nullptr,
                           hi_init ? &hi_init[i] : nullptr);
}
/* newton_batch ----------------------------------------------------------------------------------*/
/**
 * @brief Calculates implied volatilities for a batch of options using Newton's method.
 * @see newton for parameter details.
 * @see bisection_batch for other parameter details.
 */
template <typename T, typename PriceFunc, typename VegaFunc>
void newton_batch(const T *__restrict__ P,
                  const T *__restrict__ S,
                  const T *__restrict__ K,
                  const char *__restrict__ cp_flag,
                  const T *__restrict__ ttm,
                  const T *__restrict__ r,
                  const T *__restrict__ q,
                  PriceFunc price_fn,
                  VegaFunc  vega_fn,
                  size_t    n_options,
                  T *__restrict__ results,
                  T      tol                    = T{1e-3},
                  size_t max_iter               = 100,
                  const T *__restrict__ lo_init = nullptr,
                  const T *__restrict__ hi_init = nullptr)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
        results[i] = newton(P[i],
                            S[i],
                            K[i],
                            cp_flag[i],
                            ttm[i],
                            r[i],
                            q[i],
                            price_fn,
                            vega_fn,
                            tol,
                            max_iter,
                            lo_init ? &lo_init[i] : nullptr,
                            hi_init ? &hi_init[i] : nullptr);
}
} // namespace fastvol::detail::iv
