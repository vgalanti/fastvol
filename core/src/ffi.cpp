#include "fastvol/ffi.h"

#include "fastvol/american/bopm.hpp"
#include "fastvol/american/psor.hpp"
#include "fastvol/american/ttree.hpp"
#include "fastvol/european/bsm.hpp"

#include <cstddef>

extern "C"
{
    /* american::bopm ============================================================================*/
    /* fp64 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
    double american_bopm_price_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
    {
        return fastvol::american::bopm::price<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    void american_bopm_price_fp64_batch(const double *__restrict__ S,
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
        fastvol::american::bopm::price_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    /* greeks ____________________________________________________________________________________*/
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
                                   double *rho)
    {
        fastvol::american::bopm::greeks<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, delta, gamma, theta, vega, rho);
    }

    double american_bopm_delta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
    {
        return fastvol::american::bopm::delta<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    double american_bopm_gamma_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
    {
        return fastvol::american::bopm::gamma<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    double american_bopm_theta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
    {
        return fastvol::american::bopm::theta<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    double american_bopm_vega_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
    {
        return fastvol::american::bopm::vega<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    double american_bopm_rho_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
    {
        return fastvol::american::bopm::rho<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

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
                                         double *__restrict__ rho)
    {
        fastvol::american::bopm::greeks_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, delta, gamma, theta, vega, rho);
    }

    void american_bopm_delta_fp64_batch(const double *__restrict__ S,
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
        fastvol::american::bopm::delta_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_bopm_gamma_fp64_batch(const double *__restrict__ S,
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
        fastvol::american::bopm::gamma_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_bopm_theta_fp64_batch(const double *__restrict__ S,
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
        fastvol::american::bopm::theta_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_bopm_vega_fp64_batch(const double *__restrict__ S,
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
        fastvol::american::bopm::vega_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_bopm_rho_fp64_batch(const double *__restrict__ S,
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
        fastvol::american::bopm::rho_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    /* iv ________________________________________________________________________________________*/
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
                                 const double *__restrict__ hi_init)
    {
        return fastvol::american::bopm::iv<double>(
            P, S, K, cp_flag, ttm, r, q, n_steps, tol, max_iter, lo_init, hi_init);
    }

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
                                     const double *__restrict__ hi_init)
    {
        fastvol::american::bopm::iv_batch<double>(P,
                                                  S,
                                                  K,
                                                  cp_flag,
                                                  ttm,
                                                  r,
                                                  q,
                                                  n_steps,
                                                  n_options,
                                                  results,
                                                  tol,
                                                  max_iter,
                                                  lo_init,
                                                  hi_init);
    }

    /* fp32 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
    float american_bopm_price_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
    {
        return fastvol::american::bopm::price<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    void american_bopm_price_fp32_batch(const float *__restrict__ S,
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
        fastvol::american::bopm::price_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    /* greeks ____________________________________________________________________________________*/
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
                                   float *rho)
    {
        fastvol::american::bopm::greeks<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, delta, gamma, theta, vega, rho);
    }

    float american_bopm_delta_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
    {
        return fastvol::american::bopm::delta<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    float american_bopm_gamma_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
    {
        return fastvol::american::bopm::gamma<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    float american_bopm_theta_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
    {
        return fastvol::american::bopm::theta<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    float american_bopm_vega_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
    {
        return fastvol::american::bopm::vega<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    float american_bopm_rho_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
    {
        return fastvol::american::bopm::rho<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

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
                                         float *__restrict__ rho)
    {
        fastvol::american::bopm::greeks_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, delta, gamma, theta, vega, rho);
    }

    void american_bopm_delta_fp32_batch(const float *__restrict__ S,
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
        fastvol::american::bopm::delta_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_bopm_gamma_fp32_batch(const float *__restrict__ S,
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
        fastvol::american::bopm::gamma_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_bopm_theta_fp32_batch(const float *__restrict__ S,
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
        fastvol::american::bopm::theta_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_bopm_vega_fp32_batch(const float *__restrict__ S,
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
        fastvol::american::bopm::vega_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_bopm_rho_fp32_batch(const float *__restrict__ S,
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
        fastvol::american::bopm::rho_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    /* iv ________________________________________________________________________________________*/
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
                                const float *__restrict__ hi_init)
    {
        return fastvol::american::bopm::iv<float>(
            P, S, K, cp_flag, ttm, r, q, n_steps, tol, max_iter, lo_init, hi_init);
    }

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
                                     const float *__restrict__ hi_init)
    {
        fastvol::american::bopm::iv_batch<float>(P,
                                                 S,
                                                 K,
                                                 cp_flag,
                                                 ttm,
                                                 r,
                                                 q,
                                                 n_steps,
                                                 n_options,
                                                 results,
                                                 tol,
                                                 max_iter,
                                                 lo_init,
                                                 hi_init);
    }

    /* american::psor ============================================================================*/
    /* fp64 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
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
                                    double *__restrict__ theta)
    {
        return fastvol::american::psor::price<double>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, delta, gamma, theta);
    }
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
                                        double *__restrict__ theta)
    {
        fastvol::american::psor::price_batch<double>(S,
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

    /* greeks ____________________________________________________________________________________*/
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
                                   double *rho)
    {
        fastvol::american::psor::greeks<double>(S,
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
                                    int    max_iter)
    {
        return fastvol::american::psor::delta<double>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
    }

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
                                    int    max_iter)
    {
        return fastvol::american::psor::gamma<double>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
    }

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
                                    int    max_iter)
    {
        return fastvol::american::psor::theta<double>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
    }

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
                                   int    max_iter)
    {
        return fastvol::american::psor::vega<double>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
    }

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
                                  int    max_iter)
    {
        return fastvol::american::psor::rho<double>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
    }

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
                                         double *__restrict__ rho)
    {
        fastvol::american::psor::greeks_batch<double>(S,
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
                                        double *__restrict__ results)
    {
        fastvol::american::psor::delta_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
    }

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
                                        double *__restrict__ results)
    {
        fastvol::american::psor::gamma_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
    }

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
                                        double *__restrict__ results)
    {
        fastvol::american::psor::theta_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
    }

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
                                       double *__restrict__ results)
    {
        fastvol::american::psor::vega_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
    }

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
                                      double *__restrict__ results)
    {
        fastvol::american::psor::rho_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
    }

    /* iv ________________________________________________________________________________________*/
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
                                 const double *__restrict__ hi_init)
    {
        return fastvol::american::psor::iv<double>(P,
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
                                     const double *__restrict__ hi_init)
    {
        fastvol::american::psor::iv_batch<double>(P,
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

    /* fp32 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
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
                                   float *__restrict__ theta)
    {
        return fastvol::american::psor::price<float>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, delta, gamma, theta);
    }

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
                                        float *__restrict__ theta)
    {
        fastvol::american::psor::price_batch<float>(S,
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

    /* greeks ____________________________________________________________________________________*/
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
                                   float *rho)
    {
        fastvol::american::psor::greeks<float>(S,
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
                                   int   max_iter)
    {
        return fastvol::american::psor::delta<float>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
    }

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
                                   int   max_iter)
    {
        return fastvol::american::psor::gamma<float>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
    }

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
                                   int   max_iter)
    {
        return fastvol::american::psor::theta<float>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
    }

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
                                  int   max_iter)
    {
        return fastvol::american::psor::vega<float>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
    }

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
                                 int   max_iter)
    {
        return fastvol::american::psor::rho<float>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter);
    }

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
                                         float *__restrict__ rho)
    {
        fastvol::american::psor::greeks_batch<float>(S,
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
                                        float *__restrict__ results)
    {
        fastvol::american::psor::delta_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
    }

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
                                        float *__restrict__ results)
    {
        fastvol::american::psor::gamma_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
    }

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
                                        float *__restrict__ results)
    {
        fastvol::american::psor::theta_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
    }

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
                                       float *__restrict__ results)
    {
        fastvol::american::psor::vega_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
    }

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
                                      float *__restrict__ results)
    {
        fastvol::american::psor::rho_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter, n_options, results);
    }

    /* iv ________________________________________________________________________________________*/
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
                                const float *__restrict__ hi_init)
    {
        return fastvol::american::psor::iv<float>(P,
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
                                     const float *__restrict__ hi_init)
    {
        fastvol::american::psor::iv_batch<float>(P,
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

    /* american::ttree ===========================================================================*/
    /* fp64 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
    double american_ttree_price_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
    {
        return fastvol::american::ttree::price<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    void american_ttree_price_fp64_batch(const double *__restrict__ S,
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
        fastvol::american::ttree::price_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    /* greeks ____________________________________________________________________________________*/
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
                                    double *rho)
    {
        fastvol::american::ttree::greeks<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, delta, gamma, theta, vega, rho);
    }

    double american_ttree_delta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
    {
        return fastvol::american::ttree::delta<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    double american_ttree_gamma_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
    {
        return fastvol::american::ttree::gamma<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    double american_ttree_theta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
    {
        return fastvol::american::ttree::theta<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    double american_ttree_vega_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
    {
        return fastvol::american::ttree::vega<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    double american_ttree_rho_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps)
    {
        return fastvol::american::ttree::rho<double>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

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
                                          double *__restrict__ rho)
    {
        fastvol::american::ttree::greeks_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, delta, gamma, theta, vega, rho);
    }

    void american_ttree_delta_fp64_batch(const double *__restrict__ S,
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
        fastvol::american::ttree::delta_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_ttree_gamma_fp64_batch(const double *__restrict__ S,
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
        fastvol::american::ttree::gamma_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_ttree_theta_fp64_batch(const double *__restrict__ S,
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
        fastvol::american::ttree::theta_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_ttree_vega_fp64_batch(const double *__restrict__ S,
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
        fastvol::american::ttree::vega_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_ttree_rho_fp64_batch(const double *__restrict__ S,
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
        fastvol::american::ttree::rho_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    /* iv ________________________________________________________________________________________*/
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
                                  const double *__restrict__ hi_init)
    {
        return fastvol::american::ttree::iv<double>(
            P, S, K, cp_flag, ttm, r, q, n_steps, tol, max_iter, lo_init, hi_init);
    }

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
                                      const double *__restrict__ hi_init)
    {
        fastvol::american::ttree::iv_batch<double>(P,
                                                   S,
                                                   K,
                                                   cp_flag,
                                                   ttm,
                                                   r,
                                                   q,
                                                   n_steps,
                                                   n_options,
                                                   results,
                                                   tol,
                                                   max_iter,
                                                   lo_init,
                                                   hi_init);
    }

    /* fp32 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
    float american_ttree_price_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
    {
        return fastvol::american::ttree::price<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    void american_ttree_price_fp32_batch(const float *__restrict__ S,
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
        fastvol::american::ttree::price_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    /* greeks ____________________________________________________________________________________*/
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
                                    float *rho)
    {
        fastvol::american::ttree::greeks<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, delta, gamma, theta, vega, rho);
    }

    float american_ttree_delta_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
    {
        return fastvol::american::ttree::delta<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    float american_ttree_gamma_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
    {
        return fastvol::american::ttree::gamma<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    float american_ttree_theta_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
    {
        return fastvol::american::ttree::theta<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    float american_ttree_vega_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
    {
        return fastvol::american::ttree::vega<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

    float american_ttree_rho_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps)
    {
        return fastvol::american::ttree::rho<float>(S, K, cp_flag, ttm, iv, r, q, n_steps);
    }

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
                                          float *__restrict__ rho)
    {
        fastvol::american::ttree::greeks_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, delta, gamma, theta, vega, rho);
    }

    void american_ttree_delta_fp32_batch(const float *__restrict__ S,
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
        fastvol::american::ttree::delta_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_ttree_gamma_fp32_batch(const float *__restrict__ S,
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
        fastvol::american::ttree::gamma_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_ttree_theta_fp32_batch(const float *__restrict__ S,
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
        fastvol::american::ttree::theta_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_ttree_vega_fp32_batch(const float *__restrict__ S,
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
        fastvol::american::ttree::vega_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    void american_ttree_rho_fp32_batch(const float *__restrict__ S,
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
        fastvol::american::ttree::rho_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_steps, n_options, results);
    }

    /* iv ________________________________________________________________________________________*/
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
                                 const float *__restrict__ hi_init)
    {
        return fastvol::american::ttree::iv<float>(
            P, S, K, cp_flag, ttm, r, q, n_steps, tol, max_iter, lo_init, hi_init);
    }

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
                                      const float *__restrict__ hi_init)
    {
        fastvol::american::ttree::iv_batch<float>(P,
                                                  S,
                                                  K,
                                                  cp_flag,
                                                  ttm,
                                                  r,
                                                  q,
                                                  n_steps,
                                                  n_options,
                                                  results,
                                                  tol,
                                                  max_iter,
                                                  lo_init,
                                                  hi_init);
    }

    /* european::bsm =============================================================================*/
    /* fp64 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
    double european_bsm_price_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q)
    {
        return fastvol::european::bsm::price<double>(S, K, cp_flag, ttm, iv, r, q);
    }

    void european_bsm_price_fp64_batch(const double *__restrict__ S,
                                       const double *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const double *__restrict__ ttm,
                                       const double *__restrict__ iv,
                                       const double *__restrict__ r,
                                       const double *__restrict__ q,
                                       size_t n_options,
                                       double *__restrict__ result)
    {
        fastvol::european::bsm::price_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_options, result);
    }

    /* greeks ____________________________________________________________________________________*/
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
                                  double *rho)
    {
        fastvol::european::bsm::greeks<double>(
            S, K, cp_flag, ttm, iv, r, q, delta, gamma, theta, vega, rho);
    }

    double european_bsm_delta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q)
    {
        return fastvol::european::bsm::delta<double>(S, K, cp_flag, ttm, iv, r, q);
    }

    double european_bsm_gamma_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q)
    {
        return fastvol::european::bsm::gamma<double>(S, K, cp_flag, ttm, iv, r, q);
    }

    double european_bsm_theta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q)
    {
        return fastvol::european::bsm::theta<double>(S, K, cp_flag, ttm, iv, r, q);
    }

    double european_bsm_vega_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q)
    {
        return fastvol::european::bsm::vega<double>(S, K, cp_flag, ttm, iv, r, q);
    }

    double european_bsm_rho_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q)
    {
        return fastvol::european::bsm::rho<double>(S, K, cp_flag, ttm, iv, r, q);
    }

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
                                        double *__restrict__ rho)
    {
        fastvol::european::bsm::greeks_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_options, delta, gamma, theta, vega, rho);
    }

    void european_bsm_delta_fp64_batch(const double *__restrict__ S,
                                       const double *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const double *__restrict__ ttm,
                                       const double *__restrict__ iv,
                                       const double *__restrict__ r,
                                       const double *__restrict__ q,
                                       size_t n_options,
                                       double *__restrict__ results)
    {
        fastvol::european::bsm::delta_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_options, results);
    }

    void european_bsm_gamma_fp64_batch(const double *__restrict__ S,
                                       const double *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const double *__restrict__ ttm,
                                       const double *__restrict__ iv,
                                       const double *__restrict__ r,
                                       const double *__restrict__ q,
                                       size_t n_options,
                                       double *__restrict__ results)
    {
        fastvol::european::bsm::gamma_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_options, results);
    }

    void european_bsm_theta_fp64_batch(const double *__restrict__ S,
                                       const double *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const double *__restrict__ ttm,
                                       const double *__restrict__ iv,
                                       const double *__restrict__ r,
                                       const double *__restrict__ q,
                                       size_t n_options,
                                       double *__restrict__ results)
    {
        fastvol::european::bsm::theta_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_options, results);
    }

    void european_bsm_vega_fp64_batch(const double *__restrict__ S,
                                      const double *__restrict__ K,
                                      const char *__restrict__ cp_flag,
                                      const double *__restrict__ ttm,
                                      const double *__restrict__ iv,
                                      const double *__restrict__ r,
                                      const double *__restrict__ q,
                                      size_t n_options,
                                      double *__restrict__ results)
    {
        fastvol::european::bsm::vega_batch<double>(
            S, K, cp_flag, ttm, iv, r, q, n_options, results);
    }

    void european_bsm_rho_fp64_batch(const double *__restrict__ S,
                                     const double *__restrict__ K,
                                     const char *__restrict__ cp_flag,
                                     const double *__restrict__ ttm,
                                     const double *__restrict__ iv,
                                     const double *__restrict__ r,
                                     const double *__restrict__ q,
                                     size_t n_options,
                                     double *__restrict__ results)
    {
        fastvol::european::bsm::rho_batch<double>(S, K, cp_flag, ttm, iv, r, q, n_options, results);
    }
    /* iv ________________________________________________________________________________________*/
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
                                const double *__restrict__ hi_init)
    {
        return fastvol::european::bsm::iv<double>(
            P, S, K, cp_flag, ttm, r, q, tol, max_iter, lo_init, hi_init);
    }

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
                                    const double *__restrict__ hi_init)
    {
        fastvol::european::bsm::iv_batch<double>(
            P, S, K, cp_flag, ttm, r, q, n_options, results, tol, max_iter, lo_init, hi_init);
    }

    /* fp32 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
    float
    european_bsm_price_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q)
    {
        return fastvol::european::bsm::price<float>(S, K, cp_flag, ttm, iv, r, q);
    }

    void european_bsm_price_fp32_batch(const float *__restrict__ S,
                                       const float *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const float *__restrict__ ttm,
                                       const float *__restrict__ iv,
                                       const float *__restrict__ r,
                                       const float *__restrict__ q,
                                       size_t n_options,
                                       float *__restrict__ result)
    {
        fastvol::european::bsm::price_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_options, result);
    }

    /* greeks ____________________________________________________________________________________*/
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
                                  float *rho)
    {
        fastvol::european::bsm::greeks<float>(
            S, K, cp_flag, ttm, iv, r, q, delta, gamma, theta, vega, rho);
    }

    float
    european_bsm_delta_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q)
    {
        return fastvol::european::bsm::delta<float>(S, K, cp_flag, ttm, iv, r, q);
    }

    float
    european_bsm_gamma_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q)
    {
        return fastvol::european::bsm::gamma<float>(S, K, cp_flag, ttm, iv, r, q);
    }

    float
    european_bsm_theta_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q)
    {
        return fastvol::european::bsm::theta<float>(S, K, cp_flag, ttm, iv, r, q);
    }

    float
    european_bsm_vega_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q)
    {
        return fastvol::european::bsm::vega<float>(S, K, cp_flag, ttm, iv, r, q);
    }

    float
    european_bsm_rho_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q)
    {
        return fastvol::european::bsm::rho<float>(S, K, cp_flag, ttm, iv, r, q);
    }

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
                                        float *__restrict__ rho)
    {
        fastvol::european::bsm::greeks_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_options, delta, gamma, theta, vega, rho);
    }

    void european_bsm_delta_fp32_batch(const float *__restrict__ S,
                                       const float *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const float *__restrict__ ttm,
                                       const float *__restrict__ iv,
                                       const float *__restrict__ r,
                                       const float *__restrict__ q,
                                       size_t n_options,
                                       float *__restrict__ results)
    {
        fastvol::european::bsm::delta_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_options, results);
    }

    void european_bsm_gamma_fp32_batch(const float *__restrict__ S,
                                       const float *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const float *__restrict__ ttm,
                                       const float *__restrict__ iv,
                                       const float *__restrict__ r,
                                       const float *__restrict__ q,
                                       size_t n_options,
                                       float *__restrict__ results)
    {
        fastvol::european::bsm::gamma_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_options, results);
    }

    void european_bsm_theta_fp32_batch(const float *__restrict__ S,
                                       const float *__restrict__ K,
                                       const char *__restrict__ cp_flag,
                                       const float *__restrict__ ttm,
                                       const float *__restrict__ iv,
                                       const float *__restrict__ r,
                                       const float *__restrict__ q,
                                       size_t n_options,
                                       float *__restrict__ results)
    {
        fastvol::european::bsm::theta_batch<float>(
            S, K, cp_flag, ttm, iv, r, q, n_options, results);
    }

    void european_bsm_vega_fp32_batch(const float *__restrict__ S,
                                      const float *__restrict__ K,
                                      const char *__restrict__ cp_flag,
                                      const float *__restrict__ ttm,
                                      const float *__restrict__ iv,
                                      const float *__restrict__ r,
                                      const float *__restrict__ q,
                                      size_t n_options,
                                      float *__restrict__ results)
    {
        fastvol::european::bsm::vega_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_options, results);
    }

    void european_bsm_rho_fp32_batch(const float *__restrict__ S,
                                     const float *__restrict__ K,
                                     const char *__restrict__ cp_flag,
                                     const float *__restrict__ ttm,
                                     const float *__restrict__ iv,
                                     const float *__restrict__ r,
                                     const float *__restrict__ q,
                                     size_t n_options,
                                     float *__restrict__ results)
    {
        fastvol::european::bsm::rho_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_options, results);
    }

    /* iv ________________________________________________________________________________________*/
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
                               const float *__restrict__ hi_init)
    {
        return fastvol::european::bsm::iv<float>(
            P, S, K, cp_flag, ttm, r, q, tol, max_iter, lo_init, hi_init);
    }

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
                                    const float *__restrict__ hi_init)
    {
        fastvol::european::bsm::iv_batch<float>(
            P, S, K, cp_flag, ttm, r, q, n_options, results, tol, max_iter, lo_init, hi_init);
    }

    const char *fastvol_version(void) { return "0.1.1"; }
    bool        fastvol_cuda_available(void)
    {
#ifdef FASTVOL_CUDA_ENABLED
        return true;
#else
        return false;
#endif
    }
}
