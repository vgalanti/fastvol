/**
 * @file ffi.h
 * @brief C-style Foreign Function Interface (FFI) for the fastvol library.
 *
 * @author Valerio Galanti
 * @date 2025
 * @version 0.1.1
 * @license MIT License
 *
 * This header provides a pure C interface to the core functionalities of the fastvol library,
 * enabling its use in C, Python (via ctypes/cffi), and other languages that can link against C
 * libraries. It exposes functions for option pricing, greeks calculation, and implied volatility
 * estimation for various models, supporting both single and double precision, as well as CPU and
 * CUDA execution.
 */

#pragma once

#include <stdbool.h>
#include <stdlib.h>

#ifdef FASTVOL_CUDA_ENABLED
#include <cuda_runtime.h>
#else
/**
 * @brief A placeholder for the CUDA stream type when CUDA is not enabled.
 */
typedef void *cudaStream_t;
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /* american::bopm ============================================================================*/
    /* fp64 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
    /**
     * @brief Calculates the price of an American option using the Binomial Options Pricing Model
     * (BOPM).
     * @param S The current price of the underlying asset.
     * @param K The strike price of the option.
     * @param cp_flag The option type flag. Accepted values are 1, 'c', or 'C' for a call option,
     * and 0, 'p', or 'P' for a put option.
     * @param ttm The time to maturity of the option, in years.
     * @param iv The implied volatility of the underlying asset.
     * @param r The risk-free interest rate.
     * @param q The dividend yield of the underlying asset.
     * @param n_steps The number of steps in the binomial tree.
     * @return The calculated price of the option.
     */
    double american_bopm_price_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps);

    /**
     * @brief Calculates the prices of a batch of American options using the BOPM on the CPU.
     * @param S An array of underlying asset prices.
     * @param K An array of strike prices.
     * @param cp_flag An array of option type flags. Accepted values are 1, 'c', or 'C' for a call
     * option, and 0, 'p', or 'P' for a put option.
     * @param ttm An array of times to maturity.
     * @param iv An array of implied volatilities.
     * @param r An array of risk-free interest rates.
     * @param q An array of dividend yields.
     * @param n_steps The number of steps in the binomial tree for all options.
     * @param n_options The number of options in the batch.
     * @param results A pre-allocated array to store the calculated option prices.
     */
    void american_bopm_price_fp64_batch(const double *__restrict__ S,
                                        const double *__restrict__ K,
                                        const char *__restrict__ cp_flag,
                                        const double *__restrict__ ttm,
                                        const double *__restrict__ iv,
                                        const double *__restrict__ r,
                                        const double *__restrict__ q,
                                        int    n_steps,
                                        size_t n_options,
                                        double *__restrict__ results);

    /**
     * @brief Calculates the prices of a batch of American options using the BOPM on a CUDA-enabled
     * GPU.
     * @param S Pointer to the underlying asset prices on the host or device.
     * @param K Pointer to the strike prices on the host or device.
     * @param cp_flag Pointer to the option type flags on the host or device. Accepted values are 1,
     * 'c', or 'C' for a call option, and 0, 'p', or 'P' for a put option.
     * @param ttm Pointer to the times to maturity on the host or device.
     * @param iv Pointer to the implied volatilities on the host or device.
     * @param r Pointer to the risk-free interest rates on the host or device.
     * @param q Pointer to the dividend yields on the host or device.
     * @param n_steps The number of steps in the binomial tree.
     * @param n_options The number of options in the batch.
     * @param result Pointer to the pre-allocated memory for the results on the host or device.
     * @param stream The CUDA stream for asynchronous execution.
     * @param device The ID of the GPU device to use.
     * @param on_device A flag indicating if the input/output data is already on the device.
     * @param is_pinned A flag indicating if the host memory is pinned.
     * @param sync A flag indicating whether to synchronize the stream after the kernel launch.
     */
    void american_bopm_price_fp64_cuda(const double *__restrict__ S,
                                       const double *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const double *__restrict__ ttm,
                                       const double *__restrict__ iv,
                                       const double *__restrict__ r,
                                       const double *__restrict__ q,
                                       int    n_steps,
                                       size_t n_options,
                                       double *__restrict__ result,
                                       cudaStream_t stream,
                                       int          device,
                                       bool         on_device,
                                       bool         is_pinned,
                                       bool         sync);

    /* greeks ____________________________________________________________________________________*/
    /**
     * @brief Calculates the Greeks (Delta, Gamma, Theta, Vega, Rho) of an American option using the
     * BOPM.
     * @param S The current price of the underlying asset.
     * @param K The strike price of the option.
     * @param cp_flag The option type flag. Accepted values are 1, 'c', or 'C' for a call option,
     * and 0, 'p', or 'P' for a put option.
     * @param ttm The time to maturity of the option.
     * @param iv The implied volatility of the underlying asset.
     * @param r The risk-free interest rate.
     * @param q The dividend yield of the underlying asset.
     * @param n_steps The number of steps in the binomial tree.
     * @param[out] delta A pointer to store the calculated Delta. Can be NULL.
     * @param[out] gamma A pointer to store the calculated Gamma. Can be NULL.
     * @param[out] theta A pointer to store the calculated Theta. Can be NULL.
     * @param[out] vega A pointer to store the calculated Vega. Can be NULL.
     * @param[out] rho A pointer to store the calculated Rho. Can be NULL.
     */
    void american_bopm_greeks_fp64(double  S,
                                   double  K,
                                   char    cp_flag,
                                   double  ttm,
                                   double  iv,
                                   double  r,
                                   double  q,
                                   int     n_steps,
                                   double *delta,
                                   double *gamma,
                                   double *theta,
                                   double *vega,
                                   double *rho);

    /**
     * @brief Calculates the Delta of an American option using the BOPM.
     * @return The calculated Delta of the option.
     * @see american_bopm_greeks_fp64 for parameter details.
     */
    double american_bopm_delta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps);

    /**
     * @brief Calculates the Gamma of an American option using the BOPM.
     * @return The calculated Gamma of the option.
     * @see american_bopm_greeks_fp64 for parameter details.
     */
    double american_bopm_gamma_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps);

    /**
     * @brief Calculates the Theta of an American option using the BOPM.
     * @return The calculated Theta of the option.
     * @see american_bopm_greeks_fp64 for parameter details.
     */
    double american_bopm_theta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps);

    /**
     * @brief Calculates the Vega of an American option using the BOPM.
     * @return The calculated Vega of the option.
     * @see american_bopm_greeks_fp64 for parameter details.
     */
    double american_bopm_vega_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps);

    /**
     * @brief Calculates the Rho of an American option using the BOPM.
     * @return The calculated Rho of the option.
     * @see american_bopm_greeks_fp64 for parameter details.
     */
    double american_bopm_rho_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps);

    /**
     * @brief Calculates the Greeks for a batch of American options using the BOPM on the CPU.
     * @param S An array of underlying asset prices.
     * @param K An array of strike prices.
     * @param cp_flag An array of option type flags. Accepted values are 1, 'c', or 'C' for a call
     * option, and 0, 'p', or 'P' for a put option.
     * @param ttm An array of times to maturity.
     * @param iv An array of implied volatilities.
     * @param r An array of risk-free interest rates.
     * @param q An array of dividend yields.
     * @param n_steps The number of steps in the binomial tree for all options.
     * @param n_options The number of options in the batch.
     * @param[out] delta A pre-allocated array to store the calculated Deltas. Can be NULL.
     * @param[out] gamma A pre-allocated array to store the calculated Gammas. Can be NULL.
     * @param[out] theta A pre-allocated array to store the calculated Thetas. Can be NULL.
     * @param[out] vega A pre-allocated array to store the calculated Vegas. Can be NULL.
     * @param[out] rho A pre-allocated array to store the calculated Rhos. Can be NULL.
     */
    void american_bopm_greeks_fp64_batch(const double *__restrict__ S,
                                         const double *__restrict__ K,
                                         const char *__restrict__ cp_flag,
                                         const double *__restrict__ ttm,
                                         const double *__restrict__ iv,
                                         const double *__restrict__ r,
                                         const double *__restrict__ q,
                                         int    n_steps,
                                         size_t n_options,
                                         double *__restrict__ delta,
                                         double *__restrict__ gamma,
                                         double *__restrict__ theta,
                                         double *__restrict__ vega,
                                         double *__restrict__ rho);

    /**
     * @brief Calculates the Deltas for a batch of American options using the BOPM on the CPU.
     * @param results A pre-allocated array to store the calculated Deltas.
     * @see american_bopm_greeks_fp64_batch for other parameter details.
     */
    void american_bopm_delta_fp64_batch(const double *__restrict__ S,
                                        const double *__restrict__ K,
                                        const char *__restrict__ cp_flag,
                                        const double *__restrict__ ttm,
                                        const double *__restrict__ iv,
                                        const double *__restrict__ r,
                                        const double *__restrict__ q,
                                        int    n_steps,
                                        size_t n_options,
                                        double *__restrict__ results);

    /**
     * @brief Calculates the Gammas for a batch of American options using the BOPM on the CPU.
     * @param results A pre-allocated array to store the calculated Gammas.
     * @see american_bopm_greeks_fp64_batch for other parameter details.
     */
    void american_bopm_gamma_fp64_batch(const double *__restrict__ S,
                                        const double *__restrict__ K,
                                        const char *__restrict__ cp_flag,
                                        const double *__restrict__ ttm,
                                        const double *__restrict__ iv,
                                        const double *__restrict__ r,
                                        const double *__restrict__ q,
                                        int    n_steps,
                                        size_t n_options,
                                        double *__restrict__ results);

    /**
     * @brief Calculates the Thetas for a batch of American options using the BOPM on the CPU.
     * @param results A pre-allocated array to store the calculated Thetas.
     * @see american_bopm_greeks_fp64_batch for other parameter details.
     */
    void american_bopm_theta_fp64_batch(const double *__restrict__ S,
                                        const double *__restrict__ K,
                                        const char *__restrict__ cp_flag,
                                        const double *__restrict__ ttm,
                                        const double *__restrict__ iv,
                                        const double *__restrict__ r,
                                        const double *__restrict__ q,
                                        int    n_steps,
                                        size_t n_options,
                                        double *__restrict__ results);

    /**
     * @brief Calculates the Vegas for a batch of American options using the BOPM on the CPU.
     * @param results A pre-allocated array to store the calculated Vegas.
     * @see american_bopm_greeks_fp64_batch for other parameter details.
     */
    void american_bopm_vega_fp64_batch(const double *__restrict__ S,
                                       const double *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const double *__restrict__ ttm,
                                       const double *__restrict__ iv,
                                       const double *__restrict__ r,
                                       const double *__restrict__ q,
                                       int    n_steps,
                                       size_t n_options,
                                       double *__restrict__ results);

    /**
     * @brief Calculates the Rhos for a batch of American options using the BOPM on the CPU.
     * @param results A pre-allocated array to store the calculated Rhos.
     * @see american_bopm_greeks_fp64_batch for other parameter details.
     */
    void american_bopm_rho_fp64_batch(const double *__restrict__ S,
                                      const double *__restrict__ K,
                                      const char *__restrict__ cp_flag,
                                      const double *__restrict__ ttm,
                                      const double *__restrict__ iv,
                                      const double *__restrict__ r,
                                      const double *__restrict__ q,
                                      int    n_steps,
                                      size_t n_options,
                                      double *__restrict__ results);

    /* iv ________________________________________________________________________________________*/
    /**
     * @brief Calculates the implied volatility of an American option using the BOPM.
     * @param P The market price of the option.
     * @param S The current price of the underlying asset.
     * @param K The strike price of the option.
     * @param cp_flag The option type flag. Accepted values are 1, 'c', or 'C' for a call option,
     * and 0, 'p', or 'P' for a put option.
     * @param ttm The time to maturity of the option.
     * @param r The risk-free interest rate.
     * @param q The dividend yield of the underlying asset.
     * @param n_steps The number of steps in the binomial tree.
     * @param tol The tolerance for the root-finding algorithm.
     * @param max_iter The maximum number of iterations for the root-finding algorithm.
     * @param lo_init An optional initial lower bound for the search. Can be NULL.
     * @param hi_init An optional initial upper bound for the search. Can be NULL.
     * @return The calculated implied volatility.
     */
    double american_bopm_iv_fp64(double P,
                                 double S,
                                 double K,
                                 char   cp_flag,
                                 double ttm,
                                 double r,
                                 double q,
                                 int    n_steps,
                                 double tol,
                                 size_t max_iter,
                                 const double *__restrict__ lo_init,
                                 const double *__restrict__ hi_init);

    /**
     * @brief Calculates the implied volatilities for a batch of American options using the BOPM on
     * the CPU.
     * @param P An array of market prices.
     * @param S An array of underlying asset prices.
     * @param K An array of strike prices.
     * @param cp_flag An array of option type flags. Accepted values are 1, 'c', or 'C' for a call
     * option, and 0, 'p', or 'P' for a put option.
     * @param ttm An array of times to maturity.
     * @param r An array of risk-free interest rates.
     * @param q An array of dividend yields.
     * @param n_steps The number of steps in the binomial tree for all options.
     * @param n_options The number of options in the batch.
     * @param[out] results A pre-allocated array to store the calculated implied volatilities.
     * @param tol The tolerance for the root-finding algorithm.
     * @param max_iter The maximum number of iterations for the root-finding algorithm.
     * @param lo_init An optional array of initial lower bounds for the search. Can be NULL.
     * @param hi_init An optional array of initial upper bounds for the search. Can be NULL.
     */
    void american_bopm_iv_fp64_batch(const double *__restrict__ P,
                                     const double *__restrict__ S,
                                     const double *__restrict__ K,
                                     const char *__restrict__ cp_flag,
                                     const double *__restrict__ ttm,
                                     const double *__restrict__ r,
                                     const double *__restrict__ q,
                                     int    n_steps,
                                     size_t n_options,
                                     double *__restrict__ results,
                                     double tol,
                                     size_t max_iter,
                                     const double *__restrict__ lo_init,
                                     const double *__restrict__ hi_init);

    /* fp32 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
    /**
     * @brief Calculates the price of an American option using the BOPM (single precision).
     * @see american_bopm_price_fp64 for parameter details.
     */
    float american_bopm_price_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps);

    /**
     * @brief Calculates the prices of a batch of American options using the BOPM on the CPU (single
     * precision).
     * @see american_bopm_price_fp64_batch for parameter details.
     */
    void american_bopm_price_fp32_batch(const float *__restrict__ S,
                                        const float *__restrict__ K,
                                        const char *__restrict__ cp_flag,
                                        const float *__restrict__ ttm,
                                        const float *__restrict__ iv,
                                        const float *__restrict__ r,
                                        const float *__restrict__ q,
                                        int    n_steps,
                                        size_t n_options,
                                        float *__restrict__ results);

    /**
     * @brief Calculates the prices of a batch of American options using the BOPM on a CUDA-enabled
     * GPU (single precision).
     * @see american_bopm_price_fp64_cuda for parameter details.
     */
    void american_bopm_price_fp32_cuda(const float *__restrict__ S,
                                       const float *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const float *__restrict__ ttm,
                                       const float *__restrict__ iv,
                                       const float *__restrict__ r,
                                       const float *__restrict__ q,
                                       int    n_steps,
                                       size_t n_options,
                                       float *__restrict__ result,
                                       cudaStream_t stream,
                                       int          device,
                                       bool         on_device,
                                       bool         is_pinned,
                                       bool         sync);

    /* greeks ____________________________________________________________________________________*/
    /**
     * @brief Calculates the Greeks of an American option using the BOPM (single precision).
     * @see american_bopm_greeks_fp64 for parameter details.
     */
    void american_bopm_greeks_fp32(float  S,
                                   float  K,
                                   char   cp_flag,
                                   float  ttm,
                                   float  iv,
                                   float  r,
                                   float  q,
                                   int    n_steps,
                                   float *delta,
                                   float *gamma,
                                   float *theta,
                                   float *vega,
                                   float *rho);

    /**
     * @brief Calculates the Delta of an American option using the BOPM (single precision).
     * @see american_bopm_greeks_fp64 for parameter details.
     */
    float american_bopm_delta_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps);

    /**
     * @brief Calculates the Gamma of an American option using the BOPM (single precision).
     * @see american_bopm_greeks_fp64 for parameter details.
     */
    float american_bopm_gamma_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps);

    /**
     * @brief Calculates the Theta of an American option using the BOPM (single precision).
     * @see american_bopm_greeks_fp64 for parameter details.
     */
    float american_bopm_theta_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps);

    /**
     * @brief Calculates the Vega of an American option using the BOPM (single precision).
     * @see american_bopm_greeks_fp64 for parameter details.
     */
    float american_bopm_vega_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps);

    /**
     * @brief Calculates the Rho of an American option using the BOPM (single precision).
     * @see american_bopm_greeks_fp64 for parameter details.
     */
    float american_bopm_rho_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps);

    /**
     * @brief Calculates the Greeks for a batch of American options using the BOPM on the CPU
     * (single precision).
     * @see american_bopm_greeks_fp64_batch for parameter details.
     */
    void american_bopm_greeks_fp32_batch(const float *__restrict__ S,
                                         const float *__restrict__ K,
                                         const char *__restrict__ cp_flag,
                                         const float *__restrict__ ttm,
                                         const float *__restrict__ iv,
                                         const float *__restrict__ r,
                                         const float *__restrict__ q,
                                         int    n_steps,
                                         size_t n_options,
                                         float *__restrict__ delta,
                                         float *__restrict__ gamma,
                                         float *__restrict__ theta,
                                         float *__restrict__ vega,
                                         float *__restrict__ rho);

    /**
     * @brief Calculates the Deltas for a batch of American options using the BOPM on the CPU
     * (single precision).
     * @see american_bopm_greeks_fp64_batch for parameter details.
     */
    void american_bopm_delta_fp32_batch(const float *__restrict__ S,
                                        const float *__restrict__ K,
                                        const char *__restrict__ cp_flag,
                                        const float *__restrict__ ttm,
                                        const float *__restrict__ iv,
                                        const float *__restrict__ r,
                                        const float *__restrict__ q,
                                        int    n_steps,
                                        size_t n_options,
                                        float *__restrict__ results);

    /**
     * @brief Calculates the Gammas for a batch of American options using the BOPM on the CPU
     * (single precision).
     * @see american_bopm_greeks_fp64_batch for parameter details.
     */
    void american_bopm_gamma_fp32_batch(const float *__restrict__ S,
                                        const float *__restrict__ K,
                                        const char *__restrict__ cp_flag,
                                        const float *__restrict__ ttm,
                                        const float *__restrict__ iv,
                                        const float *__restrict__ r,
                                        const float *__restrict__ q,
                                        int    n_steps,
                                        size_t n_options,
                                        float *__restrict__ results);

    /**
     * @brief Calculates the Thetas for a batch of American options using the BOPM on the CPU
     * (single precision).
     * @see american_bopm_greeks_fp64_batch for parameter details.
     */
    void american_bopm_theta_fp32_batch(const float *__restrict__ S,
                                        const float *__restrict__ K,
                                        const char *__restrict__ cp_flag,
                                        const float *__restrict__ ttm,
                                        const float *__restrict__ iv,
                                        const float *__restrict__ r,
                                        const float *__restrict__ q,
                                        int    n_steps,
                                        size_t n_options,
                                        float *__restrict__ results);

    /**
     * @brief Calculates the Vegas for a batch of American options using the BOPM on the CPU (single
     * precision).
     * @see american_bopm_greeks_fp64_batch for parameter details.
     */
    void american_bopm_vega_fp32_batch(const float *__restrict__ S,
                                       const float *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const float *__restrict__ ttm,
                                       const float *__restrict__ iv,
                                       const float *__restrict__ r,
                                       const float *__restrict__ q,
                                       int    n_steps,
                                       size_t n_options,
                                       float *__restrict__ results);

    /**
     * @brief Calculates the Rhos for a batch of American options using the BOPM on the CPU (single
     * precision).
     * @see american_bopm_greeks_fp64_batch for parameter details.
     */
    void american_bopm_rho_fp32_batch(const float *__restrict__ S,
                                      const float *__restrict__ K,
                                      const char *__restrict__ cp_flag,
                                      const float *__restrict__ ttm,
                                      const float *__restrict__ iv,
                                      const float *__restrict__ r,
                                      const float *__restrict__ q,
                                      int    n_steps,
                                      size_t n_options,
                                      float *__restrict__ results);

    /* iv ________________________________________________________________________________________*/
    /**
     * @brief Calculates the implied volatility of an American option using the BOPM (single
     * precision).
     * @see american_bopm_iv_fp64 for parameter details.
     */
    float american_bopm_iv_fp32(float  P,
                                float  S,
                                float  K,
                                char   cp_flag,
                                float  ttm,
                                float  r,
                                float  q,
                                int    n_steps,
                                float  tol,
                                size_t max_iter,
                                const float *__restrict__ lo_init,
                                const float *__restrict__ hi_init);

    /**
     * @brief Calculates the implied volatilities for a batch of American options using the BOPM on
     * the CPU (single precision).
     * @see american_bopm_iv_fp64_batch for parameter details.
     */
    void american_bopm_iv_fp32_batch(const float *__restrict__ P,
                                     const float *__restrict__ S,
                                     const float *__restrict__ K,
                                     const char *__restrict__ cp_flag,
                                     const float *__restrict__ ttm,
                                     const float *__restrict__ r,
                                     const float *__restrict__ q,
                                     int    n_steps,
                                     size_t n_options,
                                     float *__restrict__ results,
                                     float  tol,
                                     size_t max_iter,
                                     const float *__restrict__ lo_init,
                                     const float *__restrict__ hi_init);

    /* american::psor ============================================================================*/
    /* fp64 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
    /**
     * @brief Calculates the price of an American option using the Projected Successive
     * Over-Relaxation (PSOR) method.
     * @param S The current price of the underlying asset.
     * @param K The strike price of the option.
     * @param cp_flag The option type flag. Accepted values are 1, 'c', or 'C' for a call option,
     * and 0, 'p', or 'P' for a put option.
     * @param ttm The time to maturity of the option.
     * @param iv The implied volatility of the underlying asset.
     * @param r The risk-free interest rate.
     * @param q The dividend yield of the underlying asset.
     * @param n_s The number of grid points for the underlying asset price.
     * @param n_t The number of grid points for time.
     * @param k_mul A multiplier for the strike price to set the upper boundary of the asset price
     * grid.
     * @param w The relaxation parameter. If set to 0, the model will use an adaptive optimal value.
     * @param tol The tolerance for the PSOR solver.
     * @param max_iter The maximum number of iterations for the PSOR solver.
     * @param[out] delta A pointer to store the calculated Delta (optional). Can be NULL.
     * @param[out] gamma A pointer to store the calculated Gamma (optional). Can be NULL.
     * @param[out] theta A pointer to store the calculated Theta (optional). Can be NULL.
     * @return The calculated price of the option.
     */
    double american_psor_price_fp64(double S,
                                    double K,
                                    char   cp_flag,
                                    double ttm,
                                    double iv,
                                    double r,
                                    double q,
                                    int    n_s,
                                    int    n_t,
                                    int    k_mul,
                                    double w,
                                    double tol,
                                    int    max_iter,
                                    double *__restrict__ delta,
                                    double *__restrict__ gamma,
                                    double *__restrict__ theta);

    /**
     * @brief Calculates the prices of a batch of American options using the PSOR method on the CPU.
     * @param S An array of underlying asset prices.
     * @param K An array of strike prices.
     * @param cp_flag An array of option type flags. Accepted values are 1, 'c', or 'C' for a call
     * option, and 0, 'p', or 'P' for a put option.
     * @param ttm An array of times to maturity.
     * @param iv An array of implied volatilities.
     * @param r An array of risk-free interest rates.
     * @param q An array of dividend yields.
     * @param n_s The number of grid points for the underlying asset price.
     * @param n_t The number of grid points for time.
     * @param k_mul A multiplier for the strike price to set the upper boundary of the asset price
     * grid.
     * @param w The relaxation parameter. If set to 0, the model will use an adaptive optimal value.
     * @param tol The tolerance for the PSOR solver.
     * @param max_iter The maximum number of iterations for the PSOR solver.
     * @param n_options The number of options in the batch.
     * @param[out] result A pre-allocated array to store the calculated option prices.
     * @param[out] delta A pre-allocated array to store the calculated Deltas. Can be NULL.
     * @param[out] gamma A pre-allocated array to store the calculated Gammas. Can be NULL.
     * @param[out] theta A pre-allocated array to store the calculated Thetas. Can be NULL.
     */
    void american_psor_price_fp64_batch(const double *__restrict__ S,
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
                                        double *__restrict__ theta);

    /**
     * @brief Calculates the prices of a batch of American options using the PSOR method on a
     * CUDA-enabled GPU.
     * @param S Pointer to the underlying asset prices on the host or device.
     * @param K Pointer to the strike prices on the host or device.
     * @param cp_flag Pointer to the option type flags on the host or device. Accepted values are 1,
     * 'c', or 'C' for a call option, and 0, 'p', or 'P' for a put option.
     * @param ttm Pointer to the times to maturity on the host or device.
     * @param iv Pointer to the implied volatilities on the host or device.
     * @param r Pointer to the risk-free interest rates on the host or device.
     * @param q Pointer to the dividend yields on the host or device.
     * @param n_s The number of grid points for the underlying asset price.
     * @param n_t The number of grid points for time.
     * @param k_mul A multiplier for the strike price to set the upper boundary of the asset price
     * grid.
     * @param w The relaxation parameter. If set to 0, the model will use an adaptive optimal value.
     * @param tol The tolerance for the PSOR solver.
     * @param max_iter The maximum number of iterations for the PSOR solver.
     * @param n_options The number of options in the batch.
     * @param[out] result Pointer to the pre-allocated memory for the results on the host or device.
     * @param[out] delta Pointer to the pre-allocated memory for the Deltas on the host or device.
     * Can be NULL.
     * @param[out] gamma Pointer to the pre-allocated memory for the Gammas on the host or device.
     * Can be NULL.
     * @param[out] theta Pointer to the pre-allocated memory for the Thetas on the host or device.
     * Can be NULL.
     * @param stream The CUDA stream for asynchronous execution.
     * @param device The ID of the GPU device to use.
     * @param on_device A flag indicating if the input/output data is already on the device.
     * @param is_pinned A flag indicating if the host memory is pinned.
     * @param sync A flag indicating whether to synchronize the stream after the kernel launch.
     */
    void american_psor_price_fp64_cuda(const double *__restrict__ S,
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
    /* greeks ____________________________________________________________________________________*/
    /**
     * @brief Calculates the Greeks of an American option using the PSOR method.
     * @param S The current price of the underlying asset.
     * @param K The strike price of the option.
     * @param cp_flag The option type flag. Accepted values are 1, 'c', or 'C' for a call option,
     * and 0, 'p', or 'P' for a put option.
     * @param ttm The time to maturity of the option.
     * @param iv The implied volatility of the underlying asset.
     * @param r The risk-free interest rate.
     * @param q The dividend yield of the underlying asset.
     * @param n_s The number of grid points for the underlying asset price.
     * @param n_t The number of grid points for time.
     * @param k_mul A multiplier for the strike price to set the upper boundary of the asset price
     * grid.
     * @param w The relaxation parameter. If set to 0, the model will use an adaptive optimal value.
     * @param tol The tolerance for the PSOR solver.
     * @param max_iter The maximum number of iterations for the PSOR solver.
     * @param[out] delta A pointer to store the calculated Delta. Can be NULL.
     * @param[out] gamma A pointer to store the calculated Gamma. Can be NULL.
     * @param[out] theta A pointer to store the calculated Theta. Can be NULL.
     * @param[out] vega A pointer to store the calculated Vega. Can be NULL.
     * @param[out] rho A pointer to store the calculated Rho. Can be NULL.
     */
    void american_psor_greeks_fp64(double  S,
                                   double  K,
                                   char    cp_flag,
                                   double  ttm,
                                   double  iv,
                                   double  r,
                                   double  q,
                                   int     n_s,
                                   int     n_t,
                                   int     k_mul,
                                   double  w,
                                   double  tol,
                                   int     max_iter,
                                   double *delta,
                                   double *gamma,
                                   double *theta,
                                   double *vega,
                                   double *rho);

    /**
     * @brief Calculates the Delta of an American option using the PSOR method.
     * @return The calculated Delta.
     * @see american_psor_greeks_fp64 for parameter details.
     */
    double american_psor_delta_fp64(double S,
                                    double K,
                                    char   cp_flag,
                                    double ttm,
                                    double iv,
                                    double r,
                                    double q,
                                    int    n_s,
                                    int    n_t,
                                    int    k_mul,
                                    double w,
                                    double tol,
                                    int    max_iter);

    /**
     * @brief Calculates the Gamma of an American option using the PSOR method.
     * @return The calculated Gamma.
     * @see american_psor_greeks_fp64 for parameter details.
     */
    double american_psor_gamma_fp64(double S,
                                    double K,
                                    char   cp_flag,
                                    double ttm,
                                    double iv,
                                    double r,
                                    double q,
                                    int    n_s,
                                    int    n_t,
                                    int    k_mul,
                                    double w,
                                    double tol,
                                    int    max_iter);

    /**
     * @brief Calculates the Theta of an American option using the PSOR method.
     * @return The calculated Theta.
     * @see american_psor_greeks_fp64 for parameter details.
     */
    double american_psor_theta_fp64(double S,
                                    double K,
                                    char   cp_flag,
                                    double ttm,
                                    double iv,
                                    double r,
                                    double q,
                                    int    n_s,
                                    int    n_t,
                                    int    k_mul,
                                    double w,
                                    double tol,
                                    int    max_iter);

    /**
     * @brief Calculates the Vega of an American option using the PSOR method.
     * @return The calculated Vega.
     * @see american_psor_greeks_fp64 for parameter details.
     */
    double american_psor_vega_fp64(double S,
                                   double K,
                                   char   cp_flag,
                                   double ttm,
                                   double iv,
                                   double r,
                                   double q,
                                   int    n_s,
                                   int    n_t,
                                   int    k_mul,
                                   double w,
                                   double tol,
                                   int    max_iter);

    /**
     * @brief Calculates the Rho of an American option using the PSOR method.
     * @return The calculated Rho.
     * @see american_psor_greeks_fp64 for parameter details.
     */
    double american_psor_rho_fp64(double S,
                                  double K,
                                  char   cp_flag,
                                  double ttm,
                                  double iv,
                                  double r,
                                  double q,
                                  int    n_s,
                                  int    n_t,
                                  int    k_mul,
                                  double w,
                                  double tol,
                                  int    max_iter);

    /**
     * @brief Calculates the Greeks for a batch of American options using the PSOR method on the
     * CPU.
     * @param S An array of underlying asset prices.
     * @param K An array of strike prices.
     * @param cp_flag An array of option type flags. Accepted values are 1, 'c', or 'C' for a call
     * option, and 0, 'p', or 'P' for a put option.
     * @param ttm An array of times to maturity.
     * @param iv An array of implied volatilities.
     * @param r An array of risk-free interest rates.
     * @param q An array of dividend yields.
     * @param n_s The number of grid points for the underlying asset price.
     * @param n_t The number of grid points for time.
     * @param k_mul A multiplier for the strike price to set the upper boundary of the asset price
     * grid.
     * @param w The relaxation parameter. If set to 0, the model will use an adaptive optimal value.
     * @param tol The tolerance for the PSOR solver.
     * @param max_iter The maximum number of iterations for the PSOR solver.
     * @param n_options The number of options in the batch.
     * @param[out] delta A pre-allocated array to store the calculated Deltas. Can be NULL.
     * @param[out] gamma A pre-allocated array to store the calculated Gammas. Can be NULL.
     * @param[out] theta A pre-allocated array to store the calculated Thetas. Can be NULL.
     * @param[out] vega A pre-allocated array to store the calculated Vegas. Can be NULL.
     * @param[out] rho A pre-allocated array to store the calculated Rhos. Can be NULL.
     */
    void american_psor_greeks_fp64_batch(const double *__restrict__ S,
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
                                         double *__restrict__ delta,
                                         double *__restrict__ gamma,
                                         double *__restrict__ theta,
                                         double *__restrict__ vega,
                                         double *__restrict__ rho);

    /**
     * @brief Calculates the Deltas for a batch of American options using the PSOR method on the
     * CPU.
     * @param results A pre-allocated array to store the calculated Deltas.
     * @see american_psor_greeks_fp64_batch for other parameter details.
     */
    void american_psor_delta_fp64_batch(const double *__restrict__ S,
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
                                        double *__restrict__ results);

    /**
     * @brief Calculates the Gammas for a batch of American options using the PSOR method on the
     * CPU.
     * @param results A pre-allocated array to store the calculated Gammas.
     * @see american_psor_greeks_fp64_batch for other parameter details.
     */
    void american_psor_gamma_fp64_batch(const double *__restrict__ S,
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
                                        double *__restrict__ results);

    /**
     * @brief Calculates the Thetas for a batch of American options using the PSOR method on the
     * CPU.
     * @param results A pre-allocated array to store the calculated Thetas.
     * @see american_psor_greeks_fp64_batch for other parameter details.
     */
    void american_psor_theta_fp64_batch(const double *__restrict__ S,
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
                                        double *__restrict__ results);

    /**
     * @brief Calculates the Vegas for a batch of American options using the PSOR method on the CPU.
     * @param results A pre-allocated array to store the calculated Vegas.
     * @see american_psor_greeks_fp64_batch for other parameter details.
     */
    void american_psor_vega_fp64_batch(const double *__restrict__ S,
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
                                       double *__restrict__ results);

    /**
     * @brief Calculates the Rhos for a batch of American options using the PSOR method on the CPU.
     * @param results A pre-allocated array to store the calculated Rhos.
     * @see american_psor_greeks_fp64_batch for other parameter details.
     */
    void american_psor_rho_fp64_batch(const double *__restrict__ S,
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
                                      double *__restrict__ results);

    /* iv ________________________________________________________________________________________*/
    /**
     * @brief Calculates the implied volatility of an American option using the PSOR method.
     * @param P The market price of the option.
     * @param S The current price of the underlying asset.
     * @param K The strike price of the option.
     * @param cp_flag The option type flag. Accepted values are 1, 'c', or 'C' for a call option,
     * and 0, 'p', or 'P' for a put option.
     * @param ttm The time to maturity of the option.
     * @param r The risk-free interest rate.
     * @param q The dividend yield of the underlying asset.
     * @param n_s The number of grid points for the underlying asset price.
     * @param n_t The number of grid points for time.
     * @param k_mul A multiplier for the strike price to set the upper boundary of the asset price
     * grid.
     * @param w The relaxation parameter. If set to 0, the model will use an adaptive optimal value.
     * @param psor_tol The tolerance for the inner PSOR solver.
     * @param psor_max_iter The maximum number of iterations for the inner PSOR solver.
     * @param tol The tolerance for the root-finding algorithm.
     * @param max_iter The maximum number of iterations for the root-finding algorithm.
     * @param lo_init An optional initial lower bound for the search. Can be NULL.
     * @param hi_init An optional initial upper bound for the search. Can be NULL.
     * @return The calculated implied volatility.
     */
    double american_psor_iv_fp64(double P,
                                 double S,
                                 double K,
                                 char   cp_flag,
                                 double ttm,
                                 double r,
                                 double q,
                                 int    n_s,
                                 int    n_t,
                                 int    k_mul,
                                 double w,
                                 double psor_tol,
                                 int    psor_max_iter,
                                 double tol,
                                 size_t max_iter,
                                 const double *__restrict__ lo_init,
                                 const double *__restrict__ hi_init);

    /**
     * @brief Calculates the implied volatilities for a batch of American options using the PSOR
     * method on the CPU.
     * @param P An array of market prices.
     * @param S An array of underlying asset prices.
     * @param K An array of strike prices.
     * @param cp_flag An array of option type flags. Accepted values are 1, 'c', or 'C' for a call
     * option, and 0, 'p', or 'P' for a put option.
     * @param ttm An array of times to maturity.
     * @param r An array of risk-free interest rates.
     * @param q An array of dividend yields.
     * @param n_s The number of grid points for the underlying asset price.
     * @param n_t The number of grid points for time.
     * @param k_mul A multiplier for the strike price to set the upper boundary of the asset price
     * grid.
     * @param w The relaxation parameter. If set to 0, the model will use an adaptive optimal value.
     * @param psor_tol The tolerance for the inner PSOR solver.
     * @param psor_max_iter The maximum number of iterations for the inner PSOR solver.
     * @param n_options The number of options in the batch.
     * @param[out] results A pre-allocated array to store the calculated implied volatilities.
     * @param tol The tolerance for the root-finding algorithm.
     * @param max_iter The maximum number of iterations for the root-finding algorithm.
     * @param lo_init An optional array of initial lower bounds for the search. Can be NULL.
     * @param hi_init An optional array of initial upper bounds for the search. Can be NULL.
     */
    void american_psor_iv_fp64_batch(const double *__restrict__ P,
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
                                     double tol,
                                     size_t max_iter,
                                     const double *__restrict__ lo_init,
                                     const double *__restrict__ hi_init);

    /* fp32 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
    /**
     * @brief Calculates the price of an American option using the PSOR method (single precision).
     * @see american_psor_price_fp64 for parameter details.
     */
    float american_psor_price_fp32(float S,
                                   float K,
                                   char  cp_flag,
                                   float ttm,
                                   float iv,
                                   float r,
                                   float q,
                                   int   n_s,
                                   int   n_t,
                                   int   k_mul,
                                   float w,
                                   float tol,
                                   int   max_iter,
                                   float *__restrict__ delta,
                                   float *__restrict__ gamma,
                                   float *__restrict__ theta);

    /**
     * @brief Calculates the prices of a batch of American options using the PSOR method on the CPU
     * (single precision).
     * @see american_psor_price_fp64_batch for parameter details.
     */
    void american_psor_price_fp32_batch(const float *__restrict__ S,
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
                                        float *__restrict__ theta);

    /**
     * @brief Calculates the prices of a batch of American options using the PSOR method on a
     * CUDA-enabled GPU (single precision).
     * @see american_psor_price_fp64_cuda for parameter details.
     */
    void american_psor_price_fp32_cuda(const float *__restrict__ S,
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

    /* greeks ____________________________________________________________________________________*/
    /**
     * @brief Calculates the Greeks of an American option using the PSOR method (single precision).
     * @see american_psor_greeks_fp64 for parameter details.
     */
    void american_psor_greeks_fp32(float  S,
                                   float  K,
                                   char   cp_flag,
                                   float  ttm,
                                   float  iv,
                                   float  r,
                                   float  q,
                                   int    n_s,
                                   int    n_t,
                                   int    k_mul,
                                   float  w,
                                   float  tol,
                                   int    max_iter,
                                   float *delta,
                                   float *gamma,
                                   float *theta,
                                   float *vega,
                                   float *rho);

    /**
     * @brief Calculates the Delta of an American option using the PSOR method (single precision).
     * @see american_psor_greeks_fp64 for parameter details.
     */
    float american_psor_delta_fp32(float S,
                                   float K,
                                   char  cp_flag,
                                   float ttm,
                                   float iv,
                                   float r,
                                   float q,
                                   int   n_s,
                                   int   n_t,
                                   int   k_mul,
                                   float w,
                                   float tol,
                                   int   max_iter);

    /**
     * @brief Calculates the Gamma of an American option using the PSOR method (single precision).
     * @see american_psor_greeks_fp64 for parameter details.
     */
    float american_psor_gamma_fp32(float S,
                                   float K,
                                   char  cp_flag,
                                   float ttm,
                                   float iv,
                                   float r,
                                   float q,
                                   int   n_s,
                                   int   n_t,
                                   int   k_mul,
                                   float w,
                                   float tol,
                                   int   max_iter);

    /**
     * @brief Calculates the Theta of an American option using the PSOR method (single precision).
     * @see american_psor_greeks_fp64 for parameter details.
     */
    float american_psor_theta_fp32(float S,
                                   float K,
                                   char  cp_flag,
                                   float ttm,
                                   float iv,
                                   float r,
                                   float q,
                                   int   n_s,
                                   int   n_t,
                                   int   k_mul,
                                   float w,
                                   float tol,
                                   int   max_iter);

    /**
     * @brief Calculates the Vega of an American option using the PSOR method (single precision).
     * @see american_psor_greeks_fp64 for parameter details.
     */
    float american_psor_vega_fp32(float S,
                                  float K,
                                  char  cp_flag,
                                  float ttm,
                                  float iv,
                                  float r,
                                  float q,
                                  int   n_s,
                                  int   n_t,
                                  int   k_mul,
                                  float w,
                                  float tol,
                                  int   max_iter);

    /**
     * @brief Calculates the Rho of an American option using the PSOR method (single precision).
     * @see american_psor_greeks_fp64 for parameter details.
     */
    float american_psor_rho_fp32(float S,
                                 float K,
                                 char  cp_flag,
                                 float ttm,
                                 float iv,
                                 float r,
                                 float q,
                                 int   n_s,
                                 int   n_t,
                                 int   k_mul,
                                 float w,
                                 float tol,
                                 int   max_iter);

    /**
     * @brief Calculates the Greeks for a batch of American options using the PSOR method on the CPU
     * (single precision).
     * @see american_psor_greeks_fp64_batch for parameter details.
     */
    void american_psor_greeks_fp32_batch(const float *__restrict__ S,
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
                                         float *__restrict__ delta,
                                         float *__restrict__ gamma,
                                         float *__restrict__ theta,
                                         float *__restrict__ vega,
                                         float *__restrict__ rho);

    /**
     * @brief Calculates the Deltas for a batch of American options using the PSOR method on the CPU
     * (single precision).
     * @see american_psor_greeks_fp64_batch for parameter details.
     */
    void american_psor_delta_fp32_batch(const float *__restrict__ S,
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
                                        float *__restrict__ results);

    /**
     * @brief Calculates the Gammas for a batch of American options using the PSOR method on the CPU
     * (single precision).
     * @see american_psor_greeks_fp64_batch for parameter details.
     */
    void american_psor_gamma_fp32_batch(const float *__restrict__ S,
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
                                        float *__restrict__ results);

    /**
     * @brief Calculates the Thetas for a batch of American options using the PSOR method on the CPU
     * (single precision).
     * @see american_psor_greeks_fp64_batch for parameter details.
     */
    void american_psor_theta_fp32_batch(const float *__restrict__ S,
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
                                        float *__restrict__ results);

    /**
     * @brief Calculates the Vegas for a batch of American options using the PSOR method on the CPU
     * (single precision).
     * @see american_psor_greeks_fp64_batch for parameter details.
     */
    void american_psor_vega_fp32_batch(const float *__restrict__ S,
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
                                       float *__restrict__ results);

    /**
     * @brief Calculates the Rhos for a batch of American options using the PSOR method on the CPU
     * (single precision).
     * @see american_psor_greeks_fp64_batch for parameter details.
     */
    void american_psor_rho_fp32_batch(const float *__restrict__ S,
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
                                      float *__restrict__ results);

    /* iv ________________________________________________________________________________________*/
    /**
     * @brief Calculates the implied volatility of an American option using the PSOR method (single
     * precision).
     * @see american_psor_iv_fp64 for parameter details.
     */
    float american_psor_iv_fp32(float  P,
                                float  S,
                                float  K,
                                char   cp_flag,
                                float  ttm,
                                float  r,
                                float  q,
                                int    n_s,
                                int    n_t,
                                int    k_mul,
                                float  w,
                                float  psor_tol,
                                int    psor_max_iter,
                                float  tol,
                                size_t max_iter,
                                const float *__restrict__ lo_init,
                                const float *__restrict__ hi_init);

    /**
     * @brief Calculates the implied volatilities for a batch of American options using the PSOR
     * method on the CPU (single precision).
     * @see american_psor_iv_fp64_batch for parameter details.
     */
    void american_psor_iv_fp32_batch(const float *__restrict__ P,
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
                                     float  tol,
                                     size_t max_iter,
                                     const float *__restrict__ lo_init,
                                     const float *__restrict__ hi_init);

    /* american::ttree ===========================================================================*/
    /* fp64 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
    /**
     * @brief Calculates the price of an American option using a trinomial tree model.
     * @param S The current price of the underlying asset.
     * @param K The strike price of the option.
     * @param cp_flag The option type flag. Accepted values are 1, 'c', or 'C' for a call option,
     * and 0, 'p', or 'P' for a put option.
     * @param ttm The time to maturity of the option, in years.
     * @param iv The implied volatility of the underlying asset.
     * @param r The risk-free interest rate.
     * @param q The dividend yield of the underlying asset.
     * @param n_steps The number of steps in the trinomial tree.
     * @return The calculated price of the option.
     */
    double american_ttree_price_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps);

    /**
     * @brief Calculates the prices of a batch of American options using a trinomial tree on the
     * CPU.
     * @param S An array of underlying asset prices.
     * @param K An array of strike prices.
     * @param cp_flag An array of option type flags. Accepted values are 1, 'c', or 'C' for a call
     * option, and 0, 'p', or 'P' for a put option.
     * @param ttm An array of times to maturity.
     * @param iv An array of implied volatilities.
     * @param r An array of risk-free interest rates.
     * @param q An array of dividend yields.
     * @param n_steps The number of steps in the trinomial tree for all options.
     * @param n_options The number of options in the batch.
     * @param[out] results A pre-allocated array to store the calculated option prices.
     */
    void american_ttree_price_fp64_batch(const double *__restrict__ S,
                                         const double *__restrict__ K,
                                         const char *__restrict__ cp_flag,
                                         const double *__restrict__ ttm,
                                         const double *__restrict__ iv,
                                         const double *__restrict__ r,
                                         const double *__restrict__ q,
                                         int    n_steps,
                                         size_t n_options,
                                         double *__restrict__ results);

    /**
     * @brief Calculates the prices of a batch of American options using a trinomial tree on a
     * CUDA-enabled GPU.
     * @param S Pointer to the underlying asset prices on the host or device.
     * @param K Pointer to the strike prices on the host or device.
     * @param cp_flag Pointer to the option type flags on the host or device. Accepted values are 1,
     * 'c', or 'C' for a call option, and 0, 'p', or 'P' for a put option.
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
    void american_ttree_price_fp64_cuda(const double *__restrict__ S,
                                        const double *__restrict__ K,
                                        const char *__restrict__ cp_flag,
                                        const double *__restrict__ ttm,
                                        const double *__restrict__ iv,
                                        const double *__restrict__ r,
                                        const double *__restrict__ q,
                                        int    n_steps,
                                        size_t n_options,
                                        double *__restrict__ result,
                                        cudaStream_t stream,
                                        int          device,
                                        bool         on_device,
                                        bool         is_pinned,
                                        bool         sync);

    /* greeks ____________________________________________________________________________________*/
    /**
     * @brief Calculates the Greeks of an American option using a trinomial tree.
     * @param S The current price of the underlying asset.
     * @param K The strike price of the option.
     * @param cp_flag The option type flag. Accepted values are 1, 'c', or 'C' for a call option,
     * and 0, 'p', or 'P' for a put option.
     * @param ttm The time to maturity of the option.
     * @param iv The implied volatility of the underlying asset.
     * @param r The risk-free interest rate.
     * @param q The dividend yield of the underlying asset.
     * @param n_steps The number of steps in the trinomial tree.
     * @param[out] delta A pointer to store the calculated Delta. Can be NULL.
     * @param[out] gamma A pointer to store the calculated Gamma. Can be NULL.
     * @param[out] theta A pointer to store the calculated Theta. Can be NULL.
     * @param[out] vega A pointer to store the calculated Vega. Can be NULL.
     * @param[out] rho A pointer to store the calculated Rho. Can be NULL.
     */
    void american_ttree_greeks_fp64(double  S,
                                    double  K,
                                    char    cp_flag,
                                    double  ttm,
                                    double  iv,
                                    double  r,
                                    double  q,
                                    int     n_steps,
                                    double *delta,
                                    double *gamma,
                                    double *theta,
                                    double *vega,
                                    double *rho);

    /**
     * @brief Calculates the Delta of an American option using a trinomial tree.
     * @return The calculated Delta.
     * @see american_ttree_greeks_fp64 for parameter details.
     */
    double american_ttree_delta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps);

    /**
     * @brief Calculates the Gamma of an American option using a trinomial tree.
     * @return The calculated Gamma.
     * @see american_ttree_greeks_fp64 for parameter details.
     */
    double american_ttree_gamma_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps);

    /**
     * @brief Calculates the Theta of an American option using a trinomial tree.
     * @return The calculated Theta.
     * @see american_ttree_greeks_fp64 for parameter details.
     */
    double american_ttree_theta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps);

    /**
     * @brief Calculates the Vega of an American option using a trinomial tree.
     * @return The calculated Vega.
     * @see american_ttree_greeks_fp64 for parameter details.
     */
    double american_ttree_vega_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps);

    /**
     * @brief Calculates the Rho of an American option using a trinomial tree.
     * @return The calculated Rho.
     * @see american_ttree_greeks_fp64 for parameter details.
     */
    double american_ttree_rho_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps);

    /**
     * @brief Calculates the Greeks for a batch of American options using a trinomial tree on the
     * CPU.
     * @param S An array of underlying asset prices.
     * @param K An array of strike prices.
     * @param cp_flag An array of option type flags. Accepted values are 1, 'c', or 'C' for a call
     * option, and 0, 'p', or 'P' for a put option.
     * @param ttm An array of times to maturity.
     * @param iv An array of implied volatilities.
     * @param r An array of risk-free interest rates.
     * @param q An array of dividend yields.
     * @param n_steps The number of steps in the trinomial tree for all options.
     * @param n_options The number of options in the batch.
     * @param[out] delta A pre-allocated array to store the calculated Deltas. Can be NULL.
     * @param[out] gamma A pre-allocated array to store the calculated Gammas. Can be NULL.
     * @param[out] theta A pre-allocated array to store the calculated Thetas. Can be NULL.
     * @param[out] vega A pre-allocated array to store the calculated Vegas. Can be NULL.
     * @param[out] rho A pre-allocated array to store the calculated Rhos. Can be NULL.
     */
    void american_ttree_greeks_fp64_batch(const double *__restrict__ S,
                                          const double *__restrict__ K,
                                          const char *__restrict__ cp_flag,
                                          const double *__restrict__ ttm,
                                          const double *__restrict__ iv,
                                          const double *__restrict__ r,
                                          const double *__restrict__ q,
                                          int    n_steps,
                                          size_t n_options,
                                          double *__restrict__ delta,
                                          double *__restrict__ gamma,
                                          double *__restrict__ theta,
                                          double *__restrict__ vega,
                                          double *__restrict__ rho);

    /**
     * @brief Calculates the Deltas for a batch of American options using a trinomial tree on the
     * CPU.
     * @param results A pre-allocated array to store the calculated Deltas.
     * @see american_ttree_greeks_fp64_batch for other parameter details.
     */
    void american_ttree_delta_fp64_batch(const double *__restrict__ S,
                                         const double *__restrict__ K,
                                         const char *__restrict__ cp_flag,
                                         const double *__restrict__ ttm,
                                         const double *__restrict__ iv,
                                         const double *__restrict__ r,
                                         const double *__restrict__ q,
                                         int    n_steps,
                                         size_t n_options,
                                         double *__restrict__ results);

    /**
     * @brief Calculates the Gammas for a batch of American options using a trinomial tree on the
     * CPU.
     * @param results A pre-allocated array to store the calculated Gammas.
     * @see american_ttree_greeks_fp64_batch for other parameter details.
     */
    void american_ttree_gamma_fp64_batch(const double *__restrict__ S,
                                         const double *__restrict__ K,
                                         const char *__restrict__ cp_flag,
                                         const double *__restrict__ ttm,
                                         const double *__restrict__ iv,
                                         const double *__restrict__ r,
                                         const double *__restrict__ q,
                                         int    n_steps,
                                         size_t n_options,
                                         double *__restrict__ results);

    /**
     * @brief Calculates the Thetas for a batch of American options using a trinomial tree on the
     * CPU.
     * @param results A pre-allocated array to store the calculated Thetas.
     * @see american_ttree_greeks_fp64_batch for other parameter details.
     */
    void american_ttree_theta_fp64_batch(const double *__restrict__ S,
                                         const double *__restrict__ K,
                                         const char *__restrict__ cp_flag,
                                         const double *__restrict__ ttm,
                                         const double *__restrict__ iv,
                                         const double *__restrict__ r,
                                         const double *__restrict__ q,
                                         int    n_steps,
                                         size_t n_options,
                                         double *__restrict__ results);

    /**
     * @brief Calculates the Vegas for a batch of American options using a trinomial tree on the
     * CPU.
     * @param results A pre-allocated array to store the calculated Vegas.
     * @see american_ttree_greeks_fp64_batch for other parameter details.
     */
    void american_ttree_vega_fp64_batch(const double *__restrict__ S,
                                        const double *__restrict__ K,
                                        const char *__restrict__ cp_flag,
                                        const double *__restrict__ ttm,
                                        const double *__restrict__ iv,
                                        const double *__restrict__ r,
                                        const double *__restrict__ q,
                                        int    n_steps,
                                        size_t n_options,
                                        double *__restrict__ results);

    /**
     * @brief Calculates the Rhos for a batch of American options using a trinomial tree on the CPU.
     * @param results A pre-allocated array to store the calculated Rhos.
     * @see american_ttree_greeks_fp64_batch for other parameter details.
     */
    void american_ttree_rho_fp64_batch(const double *__restrict__ S,
                                       const double *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const double *__restrict__ ttm,
                                       const double *__restrict__ iv,
                                       const double *__restrict__ r,
                                       const double *__restrict__ q,
                                       int    n_steps,
                                       size_t n_options,
                                       double *__restrict__ results);

    /* iv ________________________________________________________________________________________*/
    /**
     * @brief Calculates the implied volatility of an American option using a trinomial tree.
     * @param P The market price of the option.
     * @param S The current price of the underlying asset.
     * @param K The strike price of the option.
     * @param cp_flag The option type flag. Accepted values are 1, 'c', or 'C' for a call option,
     * and 0, 'p', or 'P' for a put option.
     * @param ttm The time to maturity of the option.
     * @param r The risk-free interest rate.
     * @param q The dividend yield of the underlying asset.
     * @param n_steps The number of steps in the trinomial tree.
     * @param tol The tolerance for the root-finding algorithm.
     * @param max_iter The maximum number of iterations for the root-finding algorithm.
     * @param lo_init An optional initial lower bound for the search. Can be NULL.
     * @param hi_init An optional initial upper bound for the search. Can be NULL.
     * @return The calculated implied volatility.
     */
    double american_ttree_iv_fp64(double P,
                                  double S,
                                  double K,
                                  char   cp_flag,
                                  double ttm,
                                  double r,
                                  double q,
                                  int    n_steps,
                                  double tol,
                                  size_t max_iter,
                                  const double *__restrict__ lo_init,
                                  const double *__restrict__ hi_init);

    /**
     * @brief Calculates the implied volatilities for a batch of American options using a trinomial
     * tree on the CPU.
     * @param P An array of market prices.
     * @param S An array of underlying asset prices.
     * @param K An array of strike prices.
     * @param cp_flag An array of option type flags. Accepted values are 1, 'c', or 'C' for a call
     * option, and 0, 'p', or 'P' for a put option.
     * @param ttm An array of times to maturity.
     * @param r An array of risk-free interest rates.
     * @param q An array of dividend yields.
     * @param n_steps The number of steps in the trinomial tree for all options.
     * @param n_options The number of options in the batch.
     * @param[out] results A pre-allocated array to store the calculated implied volatilities.
     * @param tol The tolerance for the root-finding algorithm.
     * @param max_iter The maximum number of iterations for the root-finding algorithm.
     * @param lo_init An optional array of initial lower bounds for the search. Can be NULL.
     * @param hi_init An optional array of initial upper bounds for the search. Can be NULL.
     */
    void american_ttree_iv_fp64_batch(const double *__restrict__ P,
                                      const double *__restrict__ S,
                                      const double *__restrict__ K,
                                      const char *__restrict__ cp_flag,
                                      const double *__restrict__ ttm,
                                      const double *__restrict__ r,
                                      const double *__restrict__ q,
                                      int    n_steps,
                                      size_t n_options,
                                      double *__restrict__ results,
                                      double tol,
                                      size_t max_iter,
                                      const double *__restrict__ lo_init,
                                      const double *__restrict__ hi_init);

    /* fp32 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
    /**
     * @brief Calculates the price of an American option using a trinomial tree (single precision).
     * @see american_ttree_price_fp64 for parameter details.
     */
    float american_ttree_price_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps);

    /**
     * @brief Calculates the prices of a batch of American options using a trinomial tree on the CPU
     * (single precision).
     * @see american_ttree_price_fp64_batch for parameter details.
     */
    void american_ttree_price_fp32_batch(const float *__restrict__ S,
                                         const float *__restrict__ K,
                                         const char *__restrict__ cp_flag,
                                         const float *__restrict__ ttm,
                                         const float *__restrict__ iv,
                                         const float *__restrict__ r,
                                         const float *__restrict__ q,
                                         int    n_steps,
                                         size_t n_options,
                                         float *__restrict__ results);

    /**
     * @brief Calculates the prices of a batch of American options using a trinomial tree on a
     * CUDA-enabled GPU (single precision).
     * @see american_ttree_price_fp64_cuda for parameter details.
     */
    void american_ttree_price_fp32_cuda(const float *__restrict__ S,
                                        const float *__restrict__ K,
                                        const char *__restrict__ cp_flag,
                                        const float *__restrict__ ttm,
                                        const float *__restrict__ iv,
                                        const float *__restrict__ r,
                                        const float *__restrict__ q,
                                        int    n_steps,
                                        size_t n_options,
                                        float *__restrict__ result,
                                        cudaStream_t stream,
                                        int          device,
                                        bool         on_device,
                                        bool         is_pinned,
                                        bool         sync);

    /* greeks ____________________________________________________________________________________*/
    /**
     * @brief Calculates the Greeks of an American option using a trinomial tree (single precision).
     * @see american_ttree_greeks_fp64 for parameter details.
     */
    void american_ttree_greeks_fp32(float  S,
                                    float  K,
                                    char   cp_flag,
                                    float  ttm,
                                    float  iv,
                                    float  r,
                                    float  q,
                                    int    n_steps,
                                    float *delta,
                                    float *gamma,
                                    float *theta,
                                    float *vega,
                                    float *rho);

    /**
     * @brief Calculates the Delta of an American option using a trinomial tree (single precision).
     * @see american_ttree_greeks_fp64 for parameter details.
     */
    float american_ttree_delta_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps);

    /**
     * @brief Calculates the Gamma of an American option using a trinomial tree (single precision).
     * @see american_ttree_greeks_fp64 for parameter details.
     */
    float american_ttree_gamma_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps);

    /**
     * @brief Calculates the Theta of an American option using a trinomial tree (single precision).
     * @see american_ttree_greeks_fp64 for parameter details.
     */
    float american_ttree_theta_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps);

    /**
     * @brief Calculates the Vega of an American option using a trinomial tree (single precision).
     * @see american_ttree_greeks_fp64 for parameter details.
     */
    float american_ttree_vega_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps);

    /**
     * @brief Calculates the Rho of an American option using a trinomial tree (single precision).
     * @see american_ttree_greeks_fp64 for parameter details.
     */
    float american_ttree_rho_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps);

    /**
     * @brief Calculates the Greeks for a batch of American options using a trinomial tree on the
     * CPU (single precision).
     * @see american_ttree_greeks_fp64_batch for parameter details.
     */
    void american_ttree_greeks_fp32_batch(const float *__restrict__ S,
                                          const float *__restrict__ K,
                                          const char *__restrict__ cp_flag,
                                          const float *__restrict__ ttm,
                                          const float *__restrict__ iv,
                                          const float *__restrict__ r,
                                          const float *__restrict__ q,
                                          int    n_steps,
                                          size_t n_options,
                                          float *__restrict__ delta,
                                          float *__restrict__ gamma,
                                          float *__restrict__ theta,
                                          float *__restrict__ vega,
                                          float *__restrict__ rho);

    /**
     * @brief Calculates the Deltas for a batch of American options using a trinomial tree on the
     * CPU (single precision).
     * @see american_ttree_greeks_fp64_batch for parameter details.
     */
    void american_ttree_delta_fp32_batch(const float *__restrict__ S,
                                         const float *__restrict__ K,
                                         const char *__restrict__ cp_flag,
                                         const float *__restrict__ ttm,
                                         const float *__restrict__ iv,
                                         const float *__restrict__ r,
                                         const float *__restrict__ q,
                                         int    n_steps,
                                         size_t n_options,
                                         float *__restrict__ results);

    /**
     * @brief Calculates the Gammas for a batch of American options using a trinomial tree on the
     * CPU (single precision).
     * @see american_ttree_greeks_fp64_batch for parameter details.
     */
    void american_ttree_gamma_fp32_batch(const float *__restrict__ S,
                                         const float *__restrict__ K,
                                         const char *__restrict__ cp_flag,
                                         const float *__restrict__ ttm,
                                         const float *__restrict__ iv,
                                         const float *__restrict__ r,
                                         const float *__restrict__ q,
                                         int    n_steps,
                                         size_t n_options,
                                         float *__restrict__ results);

    /**
     * @brief Calculates the Thetas for a batch of American options using a trinomial tree on the
     * CPU (single precision).
     * @see american_ttree_greeks_fp64_batch for parameter details.
     */
    void american_ttree_theta_fp32_batch(const float *__restrict__ S,
                                         const float *__restrict__ K,
                                         const char *__restrict__ cp_flag,
                                         const float *__restrict__ ttm,
                                         const float *__restrict__ iv,
                                         const float *__restrict__ r,
                                         const float *__restrict__ q,
                                         int    n_steps,
                                         size_t n_options,
                                         float *__restrict__ results);

    /**
     * @brief Calculates the Vegas for a batch of American options using a trinomial tree on the CPU
     * (single precision).
     * @see american_ttree_greeks_fp64_batch for parameter details.
     */
    void american_ttree_vega_fp32_batch(const float *__restrict__ S,
                                        const float *__restrict__ K,
                                        const char *__restrict__ cp_flag,
                                        const float *__restrict__ ttm,
                                        const float *__restrict__ iv,
                                        const float *__restrict__ r,
                                        const float *__restrict__ q,
                                        int    n_steps,
                                        size_t n_options,
                                        float *__restrict__ results);

    /**
     * @brief Calculates the Rhos for a batch of American options using a trinomial tree on the CPU
     * (single precision).
     * @see american_ttree_greeks_fp64_batch for parameter details.
     */
    void american_ttree_rho_fp32_batch(const float *__restrict__ S,
                                       const float *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const float *__restrict__ ttm,
                                       const float *__restrict__ iv,
                                       const float *__restrict__ r,
                                       const float *__restrict__ q,
                                       int    n_steps,
                                       size_t n_options,
                                       float *__restrict__ results);

    /* iv ________________________________________________________________________________________*/
    /**
     * @brief Calculates the implied volatility of an American option using a trinomial tree (single
     * precision).
     * @see american_ttree_iv_fp64 for parameter details.
     */
    float american_ttree_iv_fp32(float  P,
                                 float  S,
                                 float  K,
                                 char   cp_flag,
                                 float  ttm,
                                 float  r,
                                 float  q,
                                 int    n_steps,
                                 float  tol,
                                 size_t max_iter,
                                 const float *__restrict__ lo_init,
                                 const float *__restrict__ hi_init);

    /**
     * @brief Calculates the implied volatilities for a batch of American options using a trinomial
     * tree on the CPU (single precision).
     * @see american_ttree_iv_fp64_batch for parameter details.
     */
    void american_ttree_iv_fp32_batch(const float *__restrict__ P,
                                      const float *__restrict__ S,
                                      const float *__restrict__ K,
                                      const char *__restrict__ cp_flag,
                                      const float *__restrict__ ttm,
                                      const float *__restrict__ r,
                                      const float *__restrict__ q,
                                      int    n_steps,
                                      size_t n_options,
                                      float *__restrict__ results,
                                      float  tol,
                                      size_t max_iter,
                                      const float *__restrict__ lo_init,
                                      const float *__restrict__ hi_init);

    /* european::bsm =============================================================================*/
    /* fp64 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
    /**
     * @brief Calculates the price of a European option using the Black-Scholes-Merton (BSM) model.
     * @param S The current price of the underlying asset.
     * @param K The strike price of the option.
     * @param cp_flag The option type flag. Accepted values are 1, 'c', or 'C' for a call option,
     * and 0, 'p', or 'P' for a put option.
     * @param ttm The time to maturity of the option, in years.
     * @param iv The implied volatility of the underlying asset.
     * @param r The risk-free interest rate.
     * @param q The dividend yield of the underlying asset.
     * @return The calculated price of the option.
     */
    double european_bsm_price_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q);

    /**
     * @brief Calculates the prices of a batch of European options using the BSM model on the CPU.
     * @param S An array of underlying asset prices.
     * @param K An array of strike prices.
     * @param cp_flag An array of option type flags. Accepted values are 1, 'c', or 'C' for a call
     * option, and 0, 'p', or 'P' for a put option.
     * @param ttm An array of times to maturity.
     * @param iv An array of implied volatilities.
     * @param r An array of risk-free interest rates.
     * @param q An array of dividend yields.
     * @param n_options The number of options in the batch.
     * @param[out] result A pre-allocated array to store the calculated option prices.
     */
    void european_bsm_price_fp64_batch(const double *__restrict__ S,
                                       const double *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const double *__restrict__ ttm,
                                       const double *__restrict__ iv,
                                       const double *__restrict__ r,
                                       const double *__restrict__ q,
                                       size_t n_options,
                                       double *__restrict__ result);

    /**
     * @brief Calculates the prices of a batch of European options using the BSM model on a
     * CUDA-enabled GPU.
     * @param S Pointer to the underlying asset prices on the host or device.
     * @param K Pointer to the strike prices on the host or device.
     * @param cp_flag Pointer to the option type flags on the host or device. Accepted values are 1,
     * 'c', or 'C' for a call option, and 0, 'p', or 'P' for a put option.
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
    void european_bsm_price_fp64_cuda(const double *__restrict__ S,
                                      const double *__restrict__ K,
                                      const char *__restrict__ cp_flag,
                                      const double *__restrict__ ttm,
                                      const double *__restrict__ iv,
                                      const double *__restrict__ r,
                                      const double *__restrict__ q,
                                      size_t n_options,
                                      double *__restrict__ result,
                                      cudaStream_t stream,
                                      int          device,
                                      bool         on_device,
                                      bool         is_pinned,
                                      bool         sync);

    /* greeks ____________________________________________________________________________________*/
    /**
     * @brief Calculates the Greeks of a European option using the BSM model.
     * @param S The current price of the underlying asset.
     * @param K The strike price of the option.
     * @param cp_flag The option type flag. Accepted values are 1, 'c', or 'C' for a call option,
     * and 0, 'p', or 'P' for a put option.
     * @param ttm The time to maturity of the option.
     * @param iv The implied volatility of the underlying asset.
     * @param r The risk-free interest rate.
     * @param q The dividend yield of the underlying asset.
     * @param[out] delta A pointer to store the calculated Delta. Can be NULL.
     * @param[out] gamma A pointer to store the calculated Gamma. Can be NULL.
     * @param[out] theta A pointer to store the calculated Theta. Can be NULL.
     * @param[out] vega A pointer to store the calculated Vega. Can be NULL.
     * @param[out] rho A pointer to store the calculated Rho. Can be NULL.
     */
    void european_bsm_greeks_fp64(double  S,
                                  double  K,
                                  char    cp_flag,
                                  double  ttm,
                                  double  iv,
                                  double  r,
                                  double  q,
                                  double *delta,
                                  double *gamma,
                                  double *theta,
                                  double *vega,
                                  double *rho);

    /**
     * @brief Calculates the Delta of a European option using the BSM model.
     * @return The calculated Delta.
     * @see european_bsm_greeks_fp64 for parameter details.
     */
    double european_bsm_delta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q);
    /**
     * @brief Calculates the Gamma of a European option using the BSM model.
     * @return The calculated Gamma.
     * @see european_bsm_greeks_fp64 for parameter details.
     */
    double european_bsm_gamma_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q);
    /**
     * @brief Calculates the Theta of a European option using the BSM model.
     * @return The calculated Theta.
     * @see european_bsm_greeks_fp64 for parameter details.
     */
    double european_bsm_theta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q);
    /**
     * @brief Calculates the Vega of a European option using the BSM model.
     * @return The calculated Vega.
     * @see european_bsm_greeks_fp64 for parameter details.
     */
    double european_bsm_vega_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q);
    /**
     * @brief Calculates the Rho of a European option using the BSM model.
     * @return The calculated Rho.
     * @see european_bsm_greeks_fp64 for parameter details.
     */
    double european_bsm_rho_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q);

    /**
     * @brief Calculates the Greeks for a batch of European options using the BSM model on the CPU.
     * @param S An array of underlying asset prices.
     * @param K An array of strike prices.
     * @param cp_flag An array of option type flags. Accepted values are 1, 'c', or 'C' for a call
     * option, and 0, 'p', or 'P' for a put option.
     * @param ttm An array of times to maturity.
     * @param iv An array of implied volatilities.
     * @param r An array of risk-free interest rates.
     * @param q An array of dividend yields.
     * @param n_options The number of options in the batch.
     * @param[out] delta A pre-allocated array to store the calculated Deltas. Can be NULL.
     * @param[out] gamma A pre-allocated array to store the calculated Gammas. Can be NULL.
     * @param[out] theta A pre-allocated array to store the calculated Thetas. Can be NULL.
     * @param[out] vega A pre-allocated array to store the calculated Vegas. Can be NULL.
     * @param[out] rho A pre-allocated array to store the calculated Rhos. Can be NULL.
     */
    void european_bsm_greeks_fp64_batch(const double *__restrict__ S,
                                        const double *__restrict__ K,
                                        const char *__restrict__ cp_flag,
                                        const double *__restrict__ ttm,
                                        const double *__restrict__ iv,
                                        const double *__restrict__ r,
                                        const double *__restrict__ q,
                                        size_t n_options,
                                        double *__restrict__ delta,
                                        double *__restrict__ gamma,
                                        double *__restrict__ theta,
                                        double *__restrict__ vega,
                                        double *__restrict__ rho);

    /**
     * @brief Calculates the Deltas for a batch of European options using the BSM model on the CPU.
     * @param results A pre-allocated array to store the calculated Deltas.
     * @see european_bsm_greeks_fp64_batch for other parameter details.
     */
    void european_bsm_delta_fp64_batch(const double *__restrict__ S,
                                       const double *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const double *__restrict__ ttm,
                                       const double *__restrict__ iv,
                                       const double *__restrict__ r,
                                       const double *__restrict__ q,
                                       size_t n_options,
                                       double *__restrict__ results);

    /**
     * @brief Calculates the Gammas for a batch of European options using the BSM model on the CPU.
     * @param results A pre-allocated array to store the calculated Gammas.
     * @see european_bsm_greeks_fp64_batch for other parameter details.
     */
    void european_bsm_gamma_fp64_batch(const double *__restrict__ S,
                                       const double *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const double *__restrict__ ttm,
                                       const double *__restrict__ iv,
                                       const double *__restrict__ r,
                                       const double *__restrict__ q,
                                       size_t n_options,
                                       double *__restrict__ results);

    /**
     * @brief Calculates the Thetas for a batch of European options using the BSM model on the CPU.
     * @param results A pre-allocated array to store the calculated Thetas.
     * @see european_bsm_greeks_fp64_batch for other parameter details.
     */
    void european_bsm_theta_fp64_batch(const double *__restrict__ S,
                                       const double *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const double *__restrict__ ttm,
                                       const double *__restrict__ iv,
                                       const double *__restrict__ r,
                                       const double *__restrict__ q,
                                       size_t n_options,
                                       double *__restrict__ results);

    /**
     * @brief Calculates the Vegas for a batch of European options using the BSM model on the CPU.
     * @param results A pre-allocated array to store the calculated Vegas.
     * @see european_bsm_greeks_fp64_batch for other parameter details.
     */
    void european_bsm_vega_fp64_batch(const double *__restrict__ S,
                                      const double *__restrict__ K,
                                      const char *__restrict__ cp_flag,
                                      const double *__restrict__ ttm,
                                      const double *__restrict__ iv,
                                      const double *__restrict__ r,
                                      const double *__restrict__ q,
                                      size_t n_options,
                                      double *__restrict__ results);

    /**
     * @brief Calculates the Rhos for a batch of European options using the BSM model on the CPU.
     * @param results A pre-allocated array to store the calculated Rhos.
     * @see european_bsm_greeks_fp64_batch for other parameter details.
     */
    void european_bsm_rho_fp64_batch(const double *__restrict__ S,
                                     const double *__restrict__ K,
                                     const char *__restrict__ cp_flag,
                                     const double *__restrict__ ttm,
                                     const double *__restrict__ iv,
                                     const double *__restrict__ r,
                                     const double *__restrict__ q,
                                     size_t n_options,
                                     double *__restrict__ results);

    /* iv ________________________________________________________________________________________*/
    /**
     * @brief Calculates the implied volatility of a European option using the BSM model.
     * @param P The market price of the option.
     * @param S The current price of the underlying asset.
     * @param K The strike price of the option.
     * @param cp_flag The option type flag. Accepted values are 1, 'c', or 'C' for a call option,
     * and 0, 'p', or 'P' for a put option.
     * @param ttm The time to maturity of the option.
     * @param r The risk-free interest rate.
     * @param q The dividend yield of the underlying asset.
     * @param tol The tolerance for the root-finding algorithm.
     * @param max_iter The maximum number of iterations for the root-finding algorithm.
     * @param lo_init An optional initial lower bound for the search. Can be NULL.
     * @param hi_init An optional initial upper bound for the search. Can be NULL.
     * @return The calculated implied volatility.
     */
    double european_bsm_iv_fp64(double P,
                                double S,
                                double K,
                                char   cp_flag,
                                double ttm,
                                double r,
                                double q,
                                double tol,
                                size_t max_iter,
                                const double *__restrict__ lo_init,
                                const double *__restrict__ hi_init);

    /**
     * @brief Calculates the implied volatilities for a batch of European options using the BSM
     * model on the CPU.
     * @param P An array of market prices.
     * @param S An array of underlying asset prices.
     * @param K An array of strike prices.
     * @param cp_flag An array of option type flags. Accepted values are 1, 'c', or 'C' for a call
     * option, and 0, 'p', or 'P' for a put option.
     * @param ttm An array of times to maturity.
     * @param r An array of risk-free interest rates.
     * @param q An array of dividend yields.
     * @param n_options The number of options in the batch.
     * @param[out] results A pre-allocated array to store the calculated implied volatilities.
     * @param tol The tolerance for the root-finding algorithm.
     * @param max_iter The maximum number of iterations for the root-finding algorithm.
     * @param lo_init An optional array of initial lower bounds for the search. Can be NULL.
     * @param hi_init An optional array of initial upper bounds for the search. Can be NULL.
     */
    void european_bsm_iv_fp64_batch(const double *__restrict__ P,
                                    const double *__restrict__ S,
                                    const double *__restrict__ K,
                                    const char *__restrict__ cp_flag,
                                    const double *__restrict__ ttm,
                                    const double *__restrict__ r,
                                    const double *__restrict__ q,
                                    size_t n_options,
                                    double *__restrict__ results,
                                    double tol,
                                    size_t max_iter,
                                    const double *__restrict__ lo_init,
                                    const double *__restrict__ hi_init);

    /* fp32 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
    /**
     * @brief Calculates the price of a European option using the BSM model (single precision).
     * @see european_bsm_price_fp64 for parameter details.
     */
    float
    european_bsm_price_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q);

    /**
     * @brief Calculates the prices of a batch of European options using the BSM model on the CPU
     * (single precision).
     * @see european_bsm_price_fp64_batch for parameter details.
     */
    void european_bsm_price_fp32_batch(const float *__restrict__ S,
                                       const float *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const float *__restrict__ ttm,
                                       const float *__restrict__ iv,
                                       const float *__restrict__ r,
                                       const float *__restrict__ q,
                                       size_t n_options,
                                       float *__restrict__ result);

    /**
     * @brief Calculates the prices of a batch of European options using the BSM model on a
     * CUDA-enabled GPU (single precision).
     * @see european_bsm_price_fp64_cuda for parameter details.
     */
    void european_bsm_price_fp32_cuda(const float *__restrict__ S,
                                      const float *__restrict__ K,
                                      const char *__restrict__ cp_flag,
                                      const float *__restrict__ ttm,
                                      const float *__restrict__ iv,
                                      const float *__restrict__ r,
                                      const float *__restrict__ q,
                                      size_t n_options,
                                      float *__restrict__ result,
                                      cudaStream_t stream,
                                      int          device,
                                      bool         on_device,
                                      bool         is_pinned,
                                      bool         sync);

    /* greeks ____________________________________________________________________________________*/
    /**
     * @brief Calculates the Greeks of a European option using the BSM model (single precision).
     * @see european_bsm_greeks_fp64 for parameter details.
     */
    void european_bsm_greeks_fp32(float  S,
                                  float  K,
                                  char   cp_flag,
                                  float  ttm,
                                  float  iv,
                                  float  r,
                                  float  q,
                                  float *delta,
                                  float *gamma,
                                  float *theta,
                                  float *vega,
                                  float *rho);

    /**
     * @brief Calculates the Delta of a European option using the BSM model (single precision).
     * @see european_bsm_greeks_fp64 for parameter details.
     */
    float
    european_bsm_delta_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q);
    /**
     * @brief Calculates the Gamma of a European option using the BSM model (single precision).
     * @see european_bsm_greeks_fp64 for parameter details.
     */
    float
    european_bsm_gamma_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q);
    /**
     * @brief Calculates the Theta of a European option using the BSM model (single precision).
     * @see european_bsm_greeks_fp64 for parameter details.
     */
    float
    european_bsm_theta_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q);
    /**
     * @brief Calculates the Vega of a European option using the BSM model (single precision).
     * @see european_bsm_greeks_fp64 for parameter details.
     */
    float
    european_bsm_vega_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q);
    /**
     * @brief Calculates the Rho of a European option using the BSM model (single precision).
     * @see european_bsm_greeks_fp64 for parameter details.
     */
    float
    european_bsm_rho_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q);

    /**
     * @brief Calculates the Greeks for a batch of European options using the BSM model on the CPU
     * (single precision).
     * @see european_bsm_greeks_fp64_batch for parameter details.
     */
    void european_bsm_greeks_fp32_batch(const float *__restrict__ S,
                                        const float *__restrict__ K,
                                        const char *__restrict__ cp_flag,
                                        const float *__restrict__ ttm,
                                        const float *__restrict__ iv,
                                        const float *__restrict__ r,
                                        const float *__restrict__ q,
                                        size_t n_options,
                                        float *__restrict__ delta,
                                        float *__restrict__ gamma,
                                        float *__restrict__ theta,
                                        float *__restrict__ vega,
                                        float *__restrict__ rho);

    /**
     * @brief Calculates the Deltas for a batch of European options using the BSM model on the CPU
     * (single precision).
     * @see european_bsm_greeks_fp64_batch for parameter details.
     */
    void european_bsm_delta_fp32_batch(const float *__restrict__ S,
                                       const float *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const float *__restrict__ ttm,
                                       const float *__restrict__ iv,
                                       const float *__restrict__ r,
                                       const float *__restrict__ q,
                                       size_t n_options,
                                       float *__restrict__ results);

    /**
     * @brief Calculates the Gammas for a batch of European options using the BSM model on the CPU
     * (single precision).
     * @see european_bsm_greeks_fp64_batch for parameter details.
     */
    void european_bsm_gamma_fp32_batch(const float *__restrict__ S,
                                       const float *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const float *__restrict__ ttm,
                                       const float *__restrict__ iv,
                                       const float *__restrict__ r,
                                       const float *__restrict__ q,
                                       size_t n_options,
                                       float *__restrict__ results);

    /**
     * @brief Calculates the Thetas for a batch of European options using the BSM model on the CPU
     * (single precision).
     * @see european_bsm_greeks_fp64_batch for parameter details.
     */
    void european_bsm_theta_fp32_batch(const float *__restrict__ S,
                                       const float *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const float *__restrict__ ttm,
                                       const float *__restrict__ iv,
                                       const float *__restrict__ r,
                                       const float *__restrict__ q,
                                       size_t n_options,
                                       float *__restrict__ results);

    /**
     * @brief Calculates the Vegas for a batch of European options using the BSM model on the CPU
     * (single precision).
     * @see european_bsm_greeks_fp64_batch for parameter details.
     */
    void european_bsm_vega_fp32_batch(const float *__restrict__ S,
                                      const float *__restrict__ K,
                                      const char *__restrict__ cp_flag,
                                      const float *__restrict__ ttm,
                                      const float *__restrict__ iv,
                                      const float *__restrict__ r,
                                      const float *__restrict__ q,
                                      size_t n_options,
                                      float *__restrict__ results);

    /**
     * @brief Calculates the Rhos for a batch of European options using the BSM model on the CPU
     * (single precision).
     * @see european_bsm_greeks_fp64_batch for parameter details.
     */
    void european_bsm_rho_fp32_batch(const float *__restrict__ S,
                                     const float *__restrict__ K,
                                     const char *__restrict__ cp_flag,
                                     const float *__restrict__ ttm,
                                     const float *__restrict__ iv,
                                     const float *__restrict__ r,
                                     const float *__restrict__ q,
                                     size_t n_options,
                                     float *__restrict__ results);
    /* iv ________________________________________________________________________________________*/
    /**
     * @brief Calculates the implied volatility of a European option using the BSM model (single
     * precision).
     * @see european_bsm_iv_fp64 for parameter details.
     */
    float european_bsm_iv_fp32(float  P,
                               float  S,
                               float  K,
                               char   cp_flag,
                               float  ttm,
                               float  r,
                               float  q,
                               float  tol,
                               size_t max_iter,
                               const float *__restrict__ lo_init,
                               const float *__restrict__ hi_init);

    /**
     * @brief Calculates the implied volatilities for a batch of European options using the BSM
     * model on the CPU (single precision).
     * @see european_bsm_iv_fp64_batch for parameter details.
     */
    void european_bsm_iv_fp32_batch(const float *__restrict__ P,
                                    const float *__restrict__ S,
                                    const float *__restrict__ K,
                                    const char *__restrict__ cp_flag,
                                    const float *__restrict__ ttm,
                                    const float *__restrict__ r,
                                    const float *__restrict__ q,
                                    size_t n_options,
                                    float *__restrict__ results,
                                    float  tol,
                                    size_t max_iter,
                                    const float *__restrict__ lo_init,
                                    const float *__restrict__ hi_init);

    /* ===========================================================================================*/
    /**
     * @brief Gets the version of the fastvol library.
     * @return A string representing the library version.
     */
    const char *fastvol_version(void);

    /**
     * @brief Checks if the fastvol library was compiled with CUDA support and a GPU is available.
     * @return True if CUDA is available, false otherwise.
     */
    bool fastvol_cuda_available(void);
#ifdef __cplusplus
}
#endif
