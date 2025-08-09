/**
 * @file fd.hpp
 * @brief Finite difference method implementations for calculating option Greeks.
 *
 * This file provides generic, templated functions for calculating the Greeks (Delta, Gamma,
 * Theta, Vega, Rho) of an option using numerical finite differences. These functions are
 * designed to work with any pricing function, allowing for easy calculation of sensitivities
 * for various option models.
 */

#pragma once

#include "fastvol/detail/math.hpp"
#include <cstddef>
#include <omp.h>

namespace fm = fastvol::detail::math;

/**
 * @namespace fastvol::detail::fd
 * @brief Provides generic finite difference calculations for option Greeks.
 *
 * This namespace contains templated functions for calculating first and second-order
 * sensitivities (Greeks) of an option's price with respect to various market
 * parameters. The calculations are performed using central, forward, or backward
 * difference formulas. The functions are generic and can be used with any pricing model
 * by providing a callable pricing function. Both single and batch (OpenMP-parallelized)
 * versions are available.
 */
namespace fastvol::detail::fd
{
/**
 * @brief Calculates all Greeks of an option using finite differences.
 *
 * This function computes Delta, Gamma, Theta, Vega, and Rho by repeatedly calling the
 * provided pricing function with slightly perturbed inputs. It uses a combination of
 * central, forward, and backward difference formulas for efficiency.
 *
 * @tparam T The floating-point type (e.g., float, double).
 * @tparam PriceFunc The type of the callable pricing function.
 * @param S The current price of the underlying asset.
 * @param K The strike price of the option.
 * @param cp_flag The option type flag ('c' for call, 'p' for put).
 * @param ttm The time to maturity in years.
 * @param iv The implied volatility.
 * @param r The risk-free interest rate.
 * @param q The dividend yield.
 * @param price_fn A callable that takes (S, K, cp_flag, ttm, iv, r, q) and returns the price.
 * @param[out] delta Pointer to store the calculated Delta. Can be NULL.
 * @param[out] gamma Pointer to store the calculated Gamma. Can be NULL.
 * @param[out] theta Pointer to store the calculated Theta. Can be NULL.
 * @param[out] vega Pointer to store the calculated Vega. Can be NULL.
 * @param[out] rho Pointer to store the calculated Rho. Can be NULL.
 */
template <typename T, typename PriceFunc>
void greeks(T         S,
            T         K,
            char      cp_flag,
            T         ttm,
            T         iv,
            T         r,
            T         q,
            PriceFunc price_fn,
            T        *delta = nullptr,
            T        *gamma = nullptr,
            T        *theta = nullptr,
            T        *vega  = nullptr,
            T        *rho   = nullptr)
{
    T p = T(0);
    if (gamma || theta) p = price_fn(S, K, cp_flag, ttm, iv, r, q);

    if (delta || gamma)
    {
        const T h  = S * T(0.05);
        const T pu = price_fn(S + h, K, cp_flag, ttm, iv, r, q);
        const T pd = price_fn(S - h, K, cp_flag, ttm, iv, r, q);

        if (delta) *delta = (pu - pd) / (T(2) * h);
        if (gamma) *gamma = (pu - T(2) * p + pd) / (h * h);
    }

    if (theta)
    {
        const T h    = fm::fmin(ttm * T(0.01), T(0.1 / 365.0));
        const T ttm_ = fm::fmax(ttm - h, T(1e-6));
        const T pd   = price_fn(S, K, cp_flag, ttm_, iv, r, q);
        *theta       = (pd - p) / (h * T(365.0));
    }

    if (vega)
    {
        const T h  = fm::fmax(iv * T(0.005), T(0.0005));
        const T pu = price_fn(S, K, cp_flag, ttm, iv + h, r, q);
        const T pd = price_fn(S, K, cp_flag, ttm, iv - h, r, q);
        *vega      = (pu - pd) / (T(2) * h * T(100));
    }

    if (rho)
    {
        const T h  = fmax(r * T(0.01), T(0.0001));
        const T pu = price_fn(S, K, cp_flag, ttm, iv, r + h, q);
        const T pd = price_fn(S, K, cp_flag, ttm, iv, r - h, q);
        *rho       = (pu - pd) / (T(2) * h * T(100));
    }
}

/**
 * @brief Calculates the Delta of an option.
 * @return The calculated Delta.
 * @see greeks for parameter details.
 */
template <typename T, typename PriceFunc>
T delta(T S, T K, char cp_flag, T ttm, T iv, T r, T q, PriceFunc price_fn)
{
    const T h  = S * T(0.0005);
    const T pu = price_fn(S + h, K, cp_flag, ttm, iv, r, q);
    const T pd = price_fn(S - h, K, cp_flag, ttm, iv, r, q);
    return (pu - pd) / (T(2) * h);
}

/**
 * @brief Calculates the Gamma of an option.
 * @return The calculated Gamma.
 * @see greeks for parameter details.
 */
template <typename T, typename PriceFunc>
T gamma(T S, T K, char cp_flag, T ttm, T iv, T r, T q, PriceFunc price_fn)
{
    const T h  = S * T(0.0005);
    const T p  = price_fn(S, K, cp_flag, ttm, iv, r, q);
    const T pu = price_fn(S + h, K, cp_flag, ttm, iv, r, q);
    const T pd = price_fn(S - h, K, cp_flag, ttm, iv, r, q);
    return (pu - T(2) * p + pd) / (h * h);
}

/**
 * @brief Calculates the Theta of an option.
 * @return The calculated Theta.
 * @see greeks for parameter details.
 */
template <typename T, typename PriceFunc>
T theta(T S, T K, char cp_flag, T ttm, T iv, T r, T q, PriceFunc price_fn)
{
    const T p    = price_fn(S, K, cp_flag, ttm, iv, r, q);
    const T h    = fm::fmin(ttm * T(0.01), T(1.0 / 3650.0));
    const T ttm_ = fm::fmax(ttm - h, T(1e-6));
    const T pd   = price_fn(S, K, cp_flag, ttm_, iv, r, q);
    return (pd - p) / h;
}

/**
 * @brief Calculates the Vega of an option.
 * @return The calculated Vega.
 * @see greeks for parameter details.
 */
template <typename T, typename PriceFunc>
T vega(T S, T K, char cp_flag, T ttm, T iv, T r, T q, PriceFunc price_fn)
{
    const T h  = fm::fmax(iv * T(0.005), T(0.0005));
    const T pu = price_fn(S, K, cp_flag, ttm, iv + h, r, q);
    const T pd = price_fn(S, K, cp_flag, ttm, iv - h, r, q);
    return (pu - pd) / (T(2) * h * T(100));
}

/**
 * @brief Calculates the Rho of an option.
 * @return The calculated Rho.
 * @see greeks for parameter details.
 */
template <typename T, typename PriceFunc>
T rho(T S, T K, char cp_flag, T ttm, T iv, T r, T q, PriceFunc price_fn)
{
    const T h  = fmax(r * T(0.01), T(0.0001));
    const T pu = price_fn(S, K, cp_flag, ttm, iv, r + h, q);
    const T pd = price_fn(S, K, cp_flag, ttm, iv, r - h, q);
    return (pu - pd) / (T(2) * h * T(100));
}

/**
 * @brief Calculates all Greeks for a batch of options using OpenMP for parallelization.
 * @param n_options The number of options in the batch.
 * @see greeks for other parameter details.
 */
template <typename T, typename PriceFunc>
void greeks_batch(const T *__restrict__ S,
                  const T *__restrict__ K,
                  const char *__restrict__ cp_flag,
                  const T *__restrict__ ttm,
                  const T *__restrict__ iv,
                  const T *__restrict__ r,
                  const T *__restrict__ q,
                  PriceFunc price_fn,
                  size_t    n_options,
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
                  price_fn,
                  delta ? &delta[i] : nullptr,
                  gamma ? &gamma[i] : nullptr,
                  theta ? &theta[i] : nullptr,
                  vega ? &vega[i] : nullptr,
                  rho ? &rho[i] : nullptr);
    }
}

/**
 * @brief Calculates the Deltas for a batch of options.
 * @param[out] results A pre-allocated array to store the calculated Deltas.
 * @see greeks_batch for other parameter details.
 */
template <typename T, typename PriceFunc>
void delta_batch(const T *__restrict__ S,
                 const T *__restrict__ K,
                 const char *__restrict__ cp_flag,
                 const T *__restrict__ ttm,
                 const T *__restrict__ iv,
                 const T *__restrict__ r,
                 const T *__restrict__ q,
                 PriceFunc price_fn,
                 size_t    n_options,
                 T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        results[i] = delta<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], price_fn);
    }
}

/**
 * @brief Calculates the Gammas for a batch of options.
 * @param[out] results A pre-allocated array to store the calculated Gammas.
 * @see greeks_batch for other parameter details.
 */
template <typename T, typename PriceFunc>
void gamma_batch(const T *__restrict__ S,
                 const T *__restrict__ K,
                 const char *__restrict__ cp_flag,
                 const T *__restrict__ ttm,
                 const T *__restrict__ iv,
                 const T *__restrict__ r,
                 const T *__restrict__ q,
                 PriceFunc price_fn,
                 size_t    n_options,
                 T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        results[i] = gamma<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], price_fn);
    }
}

/**
 * @brief Calculates the Thetas for a batch of options.
 * @param[out] results A pre-allocated array to store the calculated Thetas.
 * @see greeks_batch for other parameter details.
 */
template <typename T, typename PriceFunc>
void theta_batch(const T *__restrict__ S,
                 const T *__restrict__ K,
                 const char *__restrict__ cp_flag,
                 const T *__restrict__ ttm,
                 const T *__restrict__ iv,
                 const T *__restrict__ r,
                 const T *__restrict__ q,
                 PriceFunc price_fn,
                 size_t    n_options,
                 T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        results[i] = theta<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], price_fn);
    }
}

/**
 * @brief Calculates the Vegas for a batch of options.
 * @param[out] results A pre-allocated array to store the calculated Vegas.
 * @see greeks_batch for other parameter details.
 */
template <typename T, typename PriceFunc>
void vega_batch(const T *__restrict__ S,
                const T *__restrict__ K,
                const char *__restrict__ cp_flag,
                const T *__restrict__ ttm,
                const T *__restrict__ iv,
                const T *__restrict__ r,
                const T *__restrict__ q,
                PriceFunc price_fn,
                size_t    n_options,
                T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        results[i] = vega<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], price_fn);
    }
}

/**
 * @brief Calculates the Rhos for a batch of options.
 * @param[out] results A pre-allocated array to store the calculated Rhos.
 * @see greeks_batch for other parameter details.
 */
template <typename T, typename PriceFunc>
void rho_batch(const T *__restrict__ S,
               const T *__restrict__ K,
               const char *__restrict__ cp_flag,
               const T *__restrict__ ttm,
               const T *__restrict__ iv,
               const T *__restrict__ r,
               const T *__restrict__ q,
               PriceFunc price_fn,
               size_t    n_options,
               T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
    {
        results[i] = rho<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i], price_fn);
    }
}

} // namespace fastvol::detail::fd
