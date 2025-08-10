/**
 * @file bsm.hpp
 * @brief Implements the Black-Scholes-Merton (BSM) model for European options.
 *
 * @author Valerio Galanti
 * @date 2025
 * @version 0.1.1
 * @license MIT License
 *
 * This file contains the C++ implementation of the BSM model for pricing European options,
 * calculating their Greeks, and estimating implied volatility. It supports both single and
 * double precision floating-point types and includes CPU (OpenMP-parallelized) and
 * CUDA-accelerated versions for batch operations.
 */

#pragma once

#include "fastvol/detail/iv.hpp"
#include "fastvol/detail/math.hpp"
#include <cstddef>
#include <omp.h>

#ifdef FASTVOL_CUDA_ENABLED
#include <cuda_runtime.h>
#else
typedef void *cudaStream_t;
#endif

namespace fm = fastvol::detail::math;

/**
 * @namespace fastvol::european::bsm
 * @brief Provides implementations of the Black-Scholes-Merton (BSM) model for European options.
 *
 * This namespace contains functions for pricing, calculating Greeks, and finding the implied
 * volatility of European options. It includes templated functions for single and double precision,
 * batch processing capabilities using OpenMP, and hooks for CUDA-accelerated versions.
 */
namespace fastvol::european::bsm
{
/* templates =====================================================================================*/
/* price -----------------------------------------------------------------------------------------*/
/**
 * @brief Calculates the price of a European option using the BSM formula.
 * @tparam T The floating-point type (float or double).
 * @param S The current price of the underlying asset.
 * @param K The strike price of the option.
 * @param cp_flag The option type flag. Accepted values are 1, 'c', or 'C' for a call option, and 0,
 * 'p', or 'P' for a put option.
 * @param ttm The time to maturity of the option, in years.
 * @param iv The implied volatility of the underlying asset.
 * @param r The risk-free interest rate.
 * @param q The dividend yield of the underlying asset.
 * @return The calculated price of the option.
 */
template <typename T> T price(T S, T K, char cp_flag, T ttm, T iv, T r, T q)
{
    const T iv_sqrt_t = iv * fm::sqrt(ttm);
    const T d1        = (fm::log(S / K) + (r - q + T(0.5) * iv * iv) * ttm) / iv_sqrt_t;
    const T d2        = d1 - iv_sqrt_t;
    const T nd1       = fm::norm_cdf(d1);
    const T nd2       = fm::norm_cdf(d2);
    const T S_        = S * fm::exp(-q * ttm);
    const T K_        = K * fm::exp(-r * ttm);
    return (cp_flag & 1) ? S_ * nd1 - K_ * nd2 : K_ * (T(1.0) - nd2) - S_ * (T(1.0) - nd1);
}

/* greeks ----------------------------------------------------------------------------------------*/
/**
 * @brief Calculates all Greeks of a European option.
 * @tparam T The floating-point type.
 * @param[out] delta A pointer to store the calculated Delta. Can be NULL.
 * @param[out] gamma A pointer to store the calculated Gamma. Can be NULL.
 * @param[out] theta A pointer to store the calculated Theta. Can be NULL.
 * @param[out] vega A pointer to store the calculated Vega. Can be NULL.
 * @param[out] rho A pointer to store the calculated Rho. Can be NULL.
 * @see price for other parameter details.
 */
template <typename T>
void greeks(T    S,
            T    K,
            char cp_flag,
            T    ttm,
            T    iv,
            T    r,
            T    q,
            T   *delta = nullptr,
            T   *gamma = nullptr,
            T   *theta = nullptr,
            T   *vega  = nullptr,
            T   *rho   = nullptr)
{
    T is_call  = T(cp_flag & 1);
    T is_put   = T(1) - is_call;
    T sqrt_ttm = fm::sqrt(ttm);
    T d1       = (fm::log(S / K) + (r - q + T(0.5) * iv * iv) * ttm) / (iv * sqrt_ttm);
    T d2       = d1 - iv * sqrt_ttm;

    T nd1 = T(0), nd2 = T(0), pd1 = T(0), exp_q_ttm = T(0), exp_r_ttm = T(0);

    if (delta || theta) nd1 = fm::norm_cdf(d1);
    if (theta || rho) nd2 = fm::norm_cdf(d2);
    if (gamma || theta || vega) pd1 = fm::norm_pdf(d1);
    if (delta || gamma || theta || vega) exp_q_ttm = fm::exp(-q * ttm);
    if (theta || rho) exp_r_ttm = fm::exp(-r * ttm);

    if (delta) *delta = exp_q_ttm * (is_call * nd1 + is_put * (nd1 - T(1)));

    if (gamma) *gamma = exp_q_ttm * pd1 / (S * iv * sqrt_ttm);

    if (theta)
    {
        T term1 = -S * pd1 * iv * exp_q_ttm / (T(2) * sqrt_ttm);
        T term2 = r * K * exp_r_ttm * (is_call * nd2 + is_put * (T(1) - nd2));
        T term3 = q * S * exp_q_ttm * (is_call * nd1 - is_put * (T(1) - nd1));
        *theta  = (term1 - term2 + term3) / T(365);
    }

    if (vega) *vega = (S * exp_q_ttm * pd1 * sqrt_ttm) / T(100);

    if (rho) *rho = (K * ttm * exp_r_ttm * (is_call * nd2 - is_put * (T(1) - nd2))) / T(100);
}

/**
 * @brief Calculates the Delta of a European option.
 * @return The calculated Delta.
 * @see price for parameter details.
 */
template <typename T> T delta(T S, T K, char cp_flag, T ttm, T iv, T r, T q)
{
    T is_call   = T(cp_flag & 1);
    T is_put    = T(1) - is_call;
    T sqrt_ttm  = fm::sqrt(ttm);
    T d1        = (fm::log(S / K) + (r - q + T(0.5) * iv * iv) * ttm) / (iv * sqrt_ttm);
    T nd1       = fm::norm_cdf(d1);
    T exp_q_ttm = fm::exp(-q * ttm);
    return exp_q_ttm * (is_call * nd1 + is_put * (nd1 - T(1)));
}

/**
 * @brief Calculates the Gamma of a European option.
 * @return The calculated Gamma.
 * @see price for parameter details.
 */
template <typename T> T gamma(T S, T K, char cp_flag, T ttm, T iv, T r, T q)
{
    (void)cp_flag;
    T sqrt_ttm  = fm::sqrt(ttm);
    T d1        = (fm::log(S / K) + (r - q + T(0.5) * iv * iv) * ttm) / (iv * sqrt_ttm);
    T pd1       = fm::norm_pdf(d1);
    T exp_q_ttm = fm::exp(-q * ttm);
    return exp_q_ttm * pd1 / (S * iv * sqrt_ttm);
}

/**
 * @brief Calculates the Theta of a European option.
 * @return The calculated Theta.
 * @see price for parameter details.
 */
template <typename T> T theta(T S, T K, char cp_flag, T ttm, T iv, T r, T q)
{
    T is_call   = T(cp_flag & 1);
    T is_put    = T(1) - is_call;
    T sqrt_ttm  = fm::sqrt(ttm);
    T d1        = (fm::log(S / K) + (r - q + T(0.5) * iv * iv) * ttm) / (iv * sqrt_ttm);
    T d2        = d1 - iv * sqrt_ttm;
    T nd1       = fm::norm_cdf(d1);
    T nd2       = fm::norm_cdf(d2);
    T pd1       = fm::norm_pdf(d1);
    T exp_q_ttm = fm::exp(-q * ttm);
    T exp_r_ttm = fm::exp(-r * ttm);

    T term1 = -S * pd1 * iv * exp_q_ttm / (T(2) * sqrt_ttm);
    T term2 = r * K * exp_r_ttm * (is_call * nd2 + is_put * (T(1) - nd2));
    T term3 = q * S * exp_q_ttm * (is_call * nd1 - is_put * (T(1) - nd1));

    return (term1 - term2 + term3) / T(365);
}

/**
 * @brief Calculates the Vega of a European option.
 * @return The calculated Vega.
 * @see price for parameter details.
 */
template <typename T> T vega(T S, T K, char cp_flag, T ttm, T iv, T r, T q)
{
    (void)cp_flag;
    T sqrt_ttm  = fm::sqrt(ttm);
    T d1        = (fm::log(S / K) + (r - q + T(0.5) * iv * iv) * ttm) / (iv * sqrt_ttm);
    T pd1       = fm::norm_pdf(d1);
    T exp_q_ttm = fm::exp(-q * ttm);

    return (S * exp_q_ttm * pd1 * sqrt_ttm) / T(100);
}

/**
 * @brief Calculates the Rho of a European option.
 * @return The calculated Rho.
 * @see price for parameter details.
 */
template <typename T> T rho(T S, T K, char cp_flag, T ttm, T iv, T r, T q)
{
    T is_call   = T(cp_flag & 1);
    T is_put    = T(1) - is_call;
    T sqrt_ttm  = fm::sqrt(ttm);
    T d1        = (fm::log(S / K) + (r - q + T(0.5) * iv * iv) * ttm) / (iv * sqrt_ttm);
    T d2        = d1 - iv * sqrt_ttm;
    T nd2       = fm::norm_cdf(d2);
    T exp_r_ttm = fm::exp(-r * ttm);
    return (K * ttm * exp_r_ttm * (is_call * nd2 - is_put * (T(1) - nd2))) / T(100);
}

/* iv --------------------------------------------------------------------------------------------*/
/**
 * @brief Internal helper function to calculate Vega for the IV solver.
 */
template <typename T> T _vega(T S, T K, char cp_flag, T ttm, T iv, T r, T q)
{
    (void)cp_flag;
    const T sqrt_ttm = fm::sqrt(ttm);
    const T d1       = (fm::log(S / K) + (r - q + T(0.5) * iv * iv) * ttm) / (iv * sqrt_ttm);
    return S * fm::exp(-q * ttm) * fm::norm_pdf(d1) * sqrt_ttm;
}

/**
 * @brief Calculates the implied volatility of a European option using Newton's method.
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
     T      tol                    = T{1e-3},
     size_t max_iter               = 100,
     const T *__restrict__ lo_init = nullptr,
     const T *__restrict__ hi_init = nullptr)
{
    return fastvol::detail::iv::newton<T>(
        P,
        S,
        K,
        cp_flag,
        ttm,
        r,
        q,
        [](T S, T K, char cp, T ttm, T iv, T r, T q) { return price<T>(S, K, cp, ttm, iv, r, q); },
        [](T S, T K, char cp, T ttm, T iv, T r, T q) { return _vega<T>(S, K, cp, ttm, iv, r, q); },
        tol,
        max_iter,
        lo_init,
        hi_init);
}

/* price_batch -----------------------------------------------------------------------------------*/
/**
 * @brief Calculates the prices of a batch of European options on the CPU.
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
                 size_t n_options,
                 T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
        results[i] = price<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i]);
}

/* greeks_batch ----------------------------------------------------------------------------------*/
/**
 * @brief Calculates the Greeks for a batch of European options on the CPU.
 * @see greeks for parameter details.
 * @see price_batch for other parameter details.
 */
template <typename T>
void greeks_batch(const T *__restrict__ S,
                  const T *__restrict__ K,
                  const char *__restrict__ cp_flag,
                  const T *__restrict__ ttm,
                  const T *__restrict__ iv,
                  const T *__restrict__ r,
                  const T *__restrict__ q,
                  size_t n_options,
                  T *__restrict__ delta = nullptr,
                  T *__restrict__ gamma = nullptr,
                  T *__restrict__ theta = nullptr,
                  T *__restrict__ vega  = nullptr,
                  T *__restrict__ rho   = nullptr)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
        greeks<T>(S[i],
                  K[i],
                  cp_flag[i],
                  ttm[i],
                  iv[i],
                  r[i],
                  q[i],
                  delta ? &delta[i] : nullptr,
                  gamma ? &gamma[i] : nullptr,
                  theta ? &theta[i] : nullptr,
                  vega ? &vega[i] : nullptr,
                  rho ? &rho[i] : nullptr);
}

/**
 * @brief Calculates the Deltas for a batch of European options on the CPU.
 * @see price_batch for parameter details.
 */
template <typename T>
void delta_batch(const T *__restrict__ S,
                 const T *__restrict__ K,
                 const char *__restrict__ cp_flag,
                 const T *__restrict__ ttm,
                 const T *__restrict__ iv,
                 const T *__restrict__ r,
                 const T *__restrict__ q,
                 size_t n_options,
                 T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
        results[i] = delta<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i]);
}

/**
 * @brief Calculates the Gammas for a batch of European options on the CPU.
 * @see price_batch for parameter details.
 */
template <typename T>
void gamma_batch(const T *__restrict__ S,
                 const T *__restrict__ K,
                 const char *__restrict__ cp_flag,
                 const T *__restrict__ ttm,
                 const T *__restrict__ iv,
                 const T *__restrict__ r,
                 const T *__restrict__ q,
                 size_t n_options,
                 T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
        results[i] = gamma<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i]);
}

/**
 * @brief Calculates the Thetas for a batch of European options on the CPU.
 * @see price_batch for parameter details.
 */
template <typename T>
void theta_batch(const T *__restrict__ S,
                 const T *__restrict__ K,
                 const char *__restrict__ cp_flag,
                 const T *__restrict__ ttm,
                 const T *__restrict__ iv,
                 const T *__restrict__ r,
                 const T *__restrict__ q,
                 size_t n_options,
                 T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
        results[i] = theta<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i]);
}

/**
 * @brief Calculates the Vegas for a batch of European options on the CPU.
 * @see price_batch for parameter details.
 */
template <typename T>
void vega_batch(const T *__restrict__ S,
                const T *__restrict__ K,
                const char *__restrict__ cp_flag,
                const T *__restrict__ ttm,
                const T *__restrict__ iv,
                const T *__restrict__ r,
                const T *__restrict__ q,
                size_t n_options,
                T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
        results[i] = vega<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i]);
}

/**
 * @brief Calculates the Rhos for a batch of European options on the CPU.
 * @see price_batch for parameter details.
 */
template <typename T>
void rho_batch(const T *__restrict__ S,
               const T *__restrict__ K,
               const char *__restrict__ cp_flag,
               const T *__restrict__ ttm,
               const T *__restrict__ iv,
               const T *__restrict__ r,
               const T *__restrict__ q,
               size_t n_options,
               T *__restrict__ results)
{
#pragma omp parallel for
    for (size_t i = 0; i < n_options; i++)
        results[i] = rho<T>(S[i], K[i], cp_flag[i], ttm[i], iv[i], r[i], q[i]);
}

/* iv_batch --------------------------------------------------------------------------------------*/
/**
 * @brief Calculates the implied volatilities for a batch of European options on the CPU.
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
              size_t n_options,
              T *__restrict__ results,
              T      tol                    = T{1e-3},
              size_t max_iter               = 100,
              const T *__restrict__ lo_init = nullptr,
              const T *__restrict__ hi_init = nullptr)
{
    fastvol::detail::iv::newton_batch<T>(
        P,
        S,
        K,
        cp_flag,
        ttm,
        r,
        q,
        [](T S, T K, char cp, T ttm, T iv, T r, T q) { return price<T>(S, K, cp, ttm, iv, r, q); },
        [](T S, T K, char cp, T ttm, T iv, T r, T q) { return _vega<T>(S, K, cp, ttm, iv, r, q); },
        n_options,
        results,
        tol,
        max_iter,
        lo_init,
        hi_init);
}

/* instanciations/CUDA ===========================================================================*/
/* fp64 ------------------------------------------------------------------------------------------*/
/* price _________________________________________________________________________________________*/
inline double
price_fp64(double S, double K, char cp_flag, double ttm, double iv, double r, double q)
{
    return price<double>(S, K, cp_flag, ttm, iv, r, q);
}

inline void price_fp64_batch(const double *__restrict__ S,
                             const double *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const double *__restrict__ ttm,
                             const double *__restrict__ iv,
                             const double *__restrict__ r,
                             const double *__restrict__ q,
                             size_t n_options,
                             double *__restrict__ result)
{
    price_batch<double>(S, K, cp_flag, ttm, iv, r, q, n_options, result);
}

/**
 * @brief Calculates the prices of a batch of European options on a CUDA-enabled GPU.
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
void price_fp64_cuda(const double *__restrict__ S,
                     const double *__restrict__ K,
                     const char *__restrict__ cp_flag,
                     const double *__restrict__ ttm,
                     const double *__restrict__ iv,
                     const double *__restrict__ r,
                     const double *__restrict__ q,
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
                        double *delta = nullptr,
                        double *gamma = nullptr,
                        double *theta = nullptr,
                        double *vega  = nullptr,
                        double *rho   = nullptr)
{
    greeks<double>(S, K, cp_flag, ttm, iv, r, q, delta, gamma, theta, vega, rho);
}

inline double
delta_fp64(double S, double K, char cp_flag, double ttm, double iv, double r, double q)
{
    return delta<double>(S, K, cp_flag, ttm, iv, r, q);
}

inline double
gamma_fp64(double S, double K, char cp_flag, double ttm, double iv, double r, double q)
{
    return gamma<double>(S, K, cp_flag, ttm, iv, r, q);
}

inline double
theta_fp64(double S, double K, char cp_flag, double ttm, double iv, double r, double q)
{
    return theta<double>(S, K, cp_flag, ttm, iv, r, q);
}

inline double vega_fp64(double S, double K, char cp_flag, double ttm, double iv, double r, double q)
{
    return vega<double>(S, K, cp_flag, ttm, iv, r, q);
}

inline double rho_fp64(double S, double K, char cp_flag, double ttm, double iv, double r, double q)
{
    return rho<double>(S, K, cp_flag, ttm, iv, r, q);
}

inline void greeks_fp64_batch(const double *__restrict__ S,
                              const double *__restrict__ K,
                              const char *__restrict__ cp_flag,
                              const double *__restrict__ ttm,
                              const double *__restrict__ iv,
                              const double *__restrict__ r,
                              const double *__restrict__ q,
                              size_t n_options,
                              double *__restrict__ delta = nullptr,
                              double *__restrict__ gamma = nullptr,
                              double *__restrict__ theta = nullptr,
                              double *__restrict__ vega  = nullptr,
                              double *__restrict__ rho   = nullptr)
{
    greeks_batch<double>(S, K, cp_flag, ttm, iv, r, q, n_options, delta, gamma, theta, vega, rho);
}

inline void delta_fp64_batch(const double *__restrict__ S,
                             const double *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const double *__restrict__ ttm,
                             const double *__restrict__ iv,
                             const double *__restrict__ r,
                             const double *__restrict__ q,
                             size_t n_options,
                             double *__restrict__ results)
{
    delta_batch<double>(S, K, cp_flag, ttm, iv, r, q, n_options, results);
}

inline void gamma_fp64_batch(const double *__restrict__ S,
                             const double *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const double *__restrict__ ttm,
                             const double *__restrict__ iv,
                             const double *__restrict__ r,
                             const double *__restrict__ q,
                             size_t n_options,
                             double *__restrict__ results)
{
    gamma_batch<double>(S, K, cp_flag, ttm, iv, r, q, n_options, results);
}

inline void theta_fp64_batch(const double *__restrict__ S,
                             const double *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const double *__restrict__ ttm,
                             const double *__restrict__ iv,
                             const double *__restrict__ r,
                             const double *__restrict__ q,
                             size_t n_options,
                             double *__restrict__ results)
{
    theta_batch<double>(S, K, cp_flag, ttm, iv, r, q, n_options, results);
}

inline void vega_fp64_batch(const double *__restrict__ S,
                            const double *__restrict__ K,
                            const char *__restrict__ cp_flag,
                            const double *__restrict__ ttm,
                            const double *__restrict__ iv,
                            const double *__restrict__ r,
                            const double *__restrict__ q,
                            size_t n_options,
                            double *__restrict__ results)
{
    vega_batch<double>(S, K, cp_flag, ttm, iv, r, q, n_options, results);
}

inline void rho_fp64_batch(const double *__restrict__ S,
                           const double *__restrict__ K,
                           const char *__restrict__ cp_flag,
                           const double *__restrict__ ttm,
                           const double *__restrict__ iv,
                           const double *__restrict__ r,
                           const double *__restrict__ q,
                           size_t n_options,
                           double *__restrict__ results)
{
    rho_batch<double>(S, K, cp_flag, ttm, iv, r, q, n_options, results);
}

/* iv ____________________________________________________________________________________________*/
inline double iv_fp64(double P,
                      double S,
                      double K,
                      char   cp_flag,
                      double ttm,
                      double r,
                      double q,
                      double tol                         = 1e-3,
                      size_t max_iter                    = 100,
                      const double *__restrict__ lo_init = nullptr,
                      const double *__restrict__ hi_init = nullptr)
{
    return iv<double>(P, S, K, cp_flag, ttm, r, q, tol, max_iter, lo_init, hi_init);
}

inline void iv_fp64_batch(const double *__restrict__ P,
                          const double *__restrict__ S,
                          const double *__restrict__ K,
                          const char *__restrict__ cp_flag,
                          const double *__restrict__ ttm,
                          const double *__restrict__ r,
                          const double *__restrict__ q,
                          size_t n_options,
                          double *__restrict__ results,
                          double tol                         = 1e-3,
                          size_t max_iter                    = 100,
                          const double *__restrict__ lo_init = nullptr,
                          const double *__restrict__ hi_init = nullptr)
{
    iv_batch<double>(
        P, S, K, cp_flag, ttm, r, q, n_options, results, tol, max_iter, lo_init, hi_init);
}

/* fp32 ------------------------------------------------------------------------------------------*/
/* price _________________________________________________________________________________________*/
inline float price_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q)
{
    return price<float>(S, K, cp_flag, ttm, iv, r, q);
}

inline void price_fp32_batch(const float *__restrict__ S,
                             const float *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const float *__restrict__ ttm,
                             const float *__restrict__ iv,
                             const float *__restrict__ r,
                             const float *__restrict__ q,
                             size_t n_options,
                             float *__restrict__ result)
{
    price_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_options, result);
}

/**
 * @brief Calculates the prices of a batch of European options on a CUDA-enabled GPU (single
 * precision).
 * @see price_fp64_cuda for parameter details.
 */
void price_fp32_cuda(const float *__restrict__ S,
                     const float *__restrict__ K,
                     const char *__restrict__ cp_flag,
                     const float *__restrict__ ttm,
                     const float *__restrict__ iv,
                     const float *__restrict__ r,
                     const float *__restrict__ q,
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
                        float *delta = nullptr,
                        float *gamma = nullptr,
                        float *theta = nullptr,
                        float *vega  = nullptr,
                        float *rho   = nullptr)
{
    greeks<float>(S, K, cp_flag, ttm, iv, r, q, delta, gamma, theta, vega, rho);
}

inline float delta_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q)
{
    return delta<float>(S, K, cp_flag, ttm, iv, r, q);
}

inline float gamma_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q)
{
    return gamma<float>(S, K, cp_flag, ttm, iv, r, q);
}

inline float theta_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q)
{
    return theta<float>(S, K, cp_flag, ttm, iv, r, q);
}

inline float vega_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q)
{
    return vega<float>(S, K, cp_flag, ttm, iv, r, q);
}

inline float rho_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q)
{
    return rho<float>(S, K, cp_flag, ttm, iv, r, q);
}

inline void greeks_fp32_batch(const float *__restrict__ S,
                              const float *__restrict__ K,
                              const char *__restrict__ cp_flag,
                              const float *__restrict__ ttm,
                              const float *__restrict__ iv,
                              const float *__restrict__ r,
                              const float *__restrict__ q,
                              size_t n_options,
                              float *__restrict__ delta = nullptr,
                              float *__restrict__ gamma = nullptr,
                              float *__restrict__ theta = nullptr,
                              float *__restrict__ vega  = nullptr,
                              float *__restrict__ rho   = nullptr)
{
    greeks_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_options, delta, gamma, theta, vega, rho);
}

inline void delta_fp32_batch(const float *__restrict__ S,
                             const float *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const float *__restrict__ ttm,
                             const float *__restrict__ iv,
                             const float *__restrict__ r,
                             const float *__restrict__ q,
                             size_t n_options,
                             float *__restrict__ results)
{
    delta_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_options, results);
}

inline void gamma_fp32_batch(const float *__restrict__ S,
                             const float *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const float *__restrict__ ttm,
                             const float *__restrict__ iv,
                             const float *__restrict__ r,
                             const float *__restrict__ q,
                             size_t n_options,
                             float *__restrict__ results)
{
    gamma_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_options, results);
}

inline void theta_fp32_batch(const float *__restrict__ S,
                             const float *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const float *__restrict__ ttm,
                             const float *__restrict__ iv,
                             const float *__restrict__ r,
                             const float *__restrict__ q,
                             size_t n_options,
                             float *__restrict__ results)
{
    theta_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_options, results);
}

inline void vega_fp32_batch(const float *__restrict__ S,
                            const float *__restrict__ K,
                            const char *__restrict__ cp_flag,
                            const float *__restrict__ ttm,
                            const float *__restrict__ iv,
                            const float *__restrict__ r,
                            const float *__restrict__ q,
                            size_t n_options,
                            float *__restrict__ results)
{
    vega_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_options, results);
}

inline void rho_fp32_batch(const float *__restrict__ S,
                           const float *__restrict__ K,
                           const char *__restrict__ cp_flag,
                           const float *__restrict__ ttm,
                           const float *__restrict__ iv,
                           const float *__restrict__ r,
                           const float *__restrict__ q,
                           size_t n_options,
                           float *__restrict__ results)
{
    rho_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_options, results);
}

/* iv ____________________________________________________________________________________________*/
inline float iv_fp32(float  P,
                     float  S,
                     float  K,
                     char   cp_flag,
                     float  ttm,
                     float  r,
                     float  q,
                     float  tol                        = 1e-3f,
                     size_t max_iter                   = 100,
                     const float *__restrict__ lo_init = nullptr,
                     const float *__restrict__ hi_init = nullptr)
{
    return iv<float>(P, S, K, cp_flag, ttm, r, q, tol, max_iter, lo_init, hi_init);
}

inline void iv_fp32_batch(const float *__restrict__ P,
                          const float *__restrict__ S,
                          const float *__restrict__ K,
                          const char *__restrict__ cp_flag,
                          const float *__restrict__ ttm,
                          const float *__restrict__ r,
                          const float *__restrict__ q,
                          size_t n_options,
                          float *__restrict__ results,
                          float  tol                        = 1e-3f,
                          size_t max_iter                   = 100,
                          const float *__restrict__ lo_init = nullptr,
                          const float *__restrict__ hi_init = nullptr)
{
    iv_batch<float>(
        P, S, K, cp_flag, ttm, r, q, n_options, results, tol, max_iter, lo_init, hi_init);
}

} // namespace fastvol::european::bsm