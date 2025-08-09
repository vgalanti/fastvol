from libc.stdint cimport uint8_t,  uintptr_t
from libc.stddef cimport size_t
ctypedef bint bool
ctypedef void* cudaStream_t

cdef extern from "fastvol/ffi.h":

    # american::bopm ===============================================================================
    # fp64 -----------------------------------------------------------------------------------------
    # price ________________________________________________________________________________________
    double american_bopm_price_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps) nogil

    void american_bopm_price_fp64_batch(const double *S,
                                        const double *K,
                                        const char *cp_flag,
                                        const double *ttm,
                                        const double *iv,
                                        const double *r,
                                        const double *q,
                                        int    n_steps,
                                        size_t n_options,
                                        double *results) nogil

    void american_bopm_price_fp64_cuda(const double *S,
                                       const double *K,
                                       const char *cp_flag,
                                       const double *ttm,
                                       const double *iv,
                                       const double *r,
                                       const double *q,
                                       int    n_steps,
                                       size_t n_options,
                                       double *result,
                                       cudaStream_t stream,
                                       int          device,
                                       bool         on_device,
                                       bool         is_pinned,
                                       bool         sync) nogil

    # greeks _______________________________________________________________________________________
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
                                   double *rho) nogil

    double american_bopm_delta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps) nogil

    double american_bopm_gamma_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps) nogil

    double american_bopm_theta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps) nogil

    double american_bopm_vega_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps) nogil

    double american_bopm_rho_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps) nogil

    void american_bopm_greeks_fp64_batch(const double *S,
                                         const double *K,
                                         const char *cp_flag,
                                         const double *ttm,
                                         const double *iv,
                                         const double *r,
                                         const double *q,
                                         int    n_steps,
                                         size_t n_options,
                                         double *delta,
                                         double *gamma,
                                         double *theta,
                                         double *vega,
                                         double *rho) nogil

    void american_bopm_delta_fp64_batch(const double *S,
                                        const double *K,
                                        const char *cp_flag,
                                        const double *ttm,
                                        const double *iv,
                                        const double *r,
                                        const double *q,
                                        int    n_steps,
                                        size_t n_options,
                                        double *results) nogil

    void american_bopm_gamma_fp64_batch(const double *S,
                                        const double *K,
                                        const char *cp_flag,
                                        const double *ttm,
                                        const double *iv,
                                        const double *r,
                                        const double *q,
                                        int    n_steps,
                                        size_t n_options,
                                        double *results) nogil

    void american_bopm_theta_fp64_batch(const double *S,
                                        const double *K,
                                        const char *cp_flag,
                                        const double *ttm,
                                        const double *iv,
                                        const double *r,
                                        const double *q,
                                        int    n_steps,
                                        size_t n_options,
                                        double *results) nogil

    void american_bopm_vega_fp64_batch(const double *S,
                                       const double *K,
                                       const char *cp_flag,
                                       const double *ttm,
                                       const double *iv,
                                       const double *r,
                                       const double *q,
                                       int    n_steps,
                                       size_t n_options,
                                       double *results) nogil

    void american_bopm_rho_fp64_batch(const double *S,
                                      const double *K,
                                      const char *cp_flag,
                                      const double *ttm,
                                      const double *iv,
                                      const double *r,
                                      const double *q,
                                      int    n_steps,
                                      size_t n_options,
                                      double *results) nogil

    # iv ___________________________________________________________________________________________
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
                                 const double *lo_init,
                                 const double *hi_init) nogil

    void american_bopm_iv_fp64_batch(const double *P,
                                     const double *S,
                                     const double *K,
                                     const char *cp_flag,
                                     const double *ttm,
                                     const double *r,
                                     const double *q,
                                     int    n_steps,
                                     size_t n_options,
                                     double *results,
                                     double tol,
                                     size_t max_iter,
                                     const double *lo_init,
                                     const double *hi_init) nogil

    # fp32 -----------------------------------------------------------------------------------------
    # price ________________________________________________________________________________________
    float american_bopm_price_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps) nogil

    void american_bopm_price_fp32_batch(const float *S,
                                        const float *K,
                                        const char *cp_flag,
                                        const float *ttm,
                                        const float *iv,
                                        const float *r,
                                        const float *q,
                                        int    n_steps,
                                        size_t n_options,
                                        float *results) nogil

    void american_bopm_price_fp32_cuda(const float *S,
                                       const float *K,
                                       const char *cp_flag,
                                       const float *ttm,
                                       const float *iv,
                                       const float *r,
                                       const float *q,
                                       int    n_steps,
                                       size_t n_options,
                                       float *result,
                                       cudaStream_t stream,
                                       int          device,
                                       bool         on_device,
                                       bool         is_pinned,
                                       bool         sync) nogil

    # greeks _______________________________________________________________________________________
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
                                   float *rho) nogil

    float american_bopm_delta_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps) nogil

    float american_bopm_gamma_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps) nogil

    float american_bopm_theta_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps) nogil

    float american_bopm_vega_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps) nogil

    float american_bopm_rho_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps) nogil

    void american_bopm_greeks_fp32_batch(const float *S,
                                         const float *K,
                                         const char *cp_flag,
                                         const float *ttm,
                                         const float *iv,
                                         const float *r,
                                         const float *q,
                                         int    n_steps,
                                         size_t n_options,
                                         float *delta,
                                         float *gamma,
                                         float *theta,
                                         float *vega,
                                         float *rho) nogil

    void american_bopm_delta_fp32_batch(const float *S,
                                        const float *K,
                                        const char *cp_flag,
                                        const float *ttm,
                                        const float *iv,
                                        const float *r,
                                        const float *q,
                                        int    n_steps,
                                        size_t n_options,
                                        float *results) nogil

    void american_bopm_gamma_fp32_batch(const float *S,
                                        const float *K,
                                        const char *cp_flag,
                                        const float *ttm,
                                        const float *iv,
                                        const float *r,
                                        const float *q,
                                        int    n_steps,
                                        size_t n_options,
                                        float *results) nogil

    void american_bopm_theta_fp32_batch(const float *S,
                                        const float *K,
                                        const char *cp_flag,
                                        const float *ttm,
                                        const float *iv,
                                        const float *r,
                                        const float *q,
                                        int    n_steps,
                                        size_t n_options,
                                        float *results) nogil

    void american_bopm_vega_fp32_batch(const float *S,
                                       const float *K,
                                       const char *cp_flag,
                                       const float *ttm,
                                       const float *iv,
                                       const float *r,
                                       const float *q,
                                       int    n_steps,
                                       size_t n_options,
                                       float *results) nogil

    void american_bopm_rho_fp32_batch(const float *S,
                                      const float *K,
                                      const char *cp_flag,
                                      const float *ttm,
                                      const float *iv,
                                      const float *r,
                                      const float *q,
                                      int    n_steps,
                                      size_t n_options,
                                      float *results) nogil

    # iv ___________________________________________________________________________________________
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
                                const float *lo_init,
                                const float *hi_init) nogil

    void american_bopm_iv_fp32_batch(const float *P,
                                     const float *S,
                                     const float *K,
                                     const char *cp_flag,
                                     const float *ttm,
                                     const float *r,
                                     const float *q,
                                     int    n_steps,
                                     size_t n_options,
                                     float *results,
                                     float  tol,
                                     size_t max_iter,
                                     const float *lo_init,
                                     const float *hi_init) nogil

    #  american::psor ==============================================================================
    #  fp64 ----------------------------------------------------------------------------------------
    #  price _______________________________________________________________________________________
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
                                    double *delta,
                                    double *gamma,
                                    double *theta) nogil

    void american_psor_price_fp64_batch(const double *S,
                                        const double *K,
                                        const char *cp_flag,
                                        const double *ttm,
                                        const double *iv,
                                        const double *r,
                                        const double *q,
                                        int    n_s,
                                        int    n_t,
                                        int    k_mul,
                                        double w,
                                        double tol,
                                        int    max_iter,
                                        size_t n_options,
                                        double *result,
                                        double *delta,
                                        double *gamma,
                                        double *theta) nogil

    void american_psor_price_fp64_cuda(const double *S,
                                       const double *K,
                                       const char *cp_flag,
                                       const double *ttm,
                                       const double *iv,
                                       const double *r,
                                       const double *q,
                                       int    n_s,
                                       int    n_t,
                                       int    k_mul,
                                       double w,
                                       double tol,
                                       int    max_iter,
                                       size_t n_options,
                                       double *result,
                                       double *delta,
                                       double *gamma,
                                       double *theta,
                                       cudaStream_t stream,
                                       int          device,
                                       bool         on_device,
                                       bool         is_pinned,
                                       bool         sync) nogil
    #  greeks ______________________________________________________________________________________
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
                                   double *rho) nogil

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
                                    int    max_iter) nogil

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
                                    int    max_iter) nogil

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
                                    int    max_iter) nogil

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
                                   int    max_iter) nogil

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
                                  int    max_iter) nogil

    void american_psor_greeks_fp64_batch(const double *S,
                                         const double *K,
                                         const char *cp_flag,
                                         const double *ttm,
                                         const double *iv,
                                         const double *r,
                                         const double *q,
                                         int    n_s,
                                         int    n_t,
                                         int    k_mul,
                                         double w,
                                         double tol,
                                         int    max_iter,
                                         size_t n_options,
                                         double *delta,
                                         double *gamma,
                                         double *theta,
                                         double *vega,
                                         double *rho) nogil

    void american_psor_delta_fp64_batch(const double *S,
                                        const double *K,
                                        const char *cp_flag,
                                        const double *ttm,
                                        const double *iv,
                                        const double *r,
                                        const double *q,
                                        int    n_s,
                                        int    n_t,
                                        int    k_mul,
                                        double w,
                                        double tol,
                                        int    max_iter,
                                        size_t n_options,
                                        double *results) nogil

    void american_psor_gamma_fp64_batch(const double *S,
                                        const double *K,
                                        const char *cp_flag,
                                        const double *ttm,
                                        const double *iv,
                                        const double *r,
                                        const double *q,
                                        int    n_s,
                                        int    n_t,
                                        int    k_mul,
                                        double w,
                                        double tol,
                                        int    max_iter,
                                        size_t n_options,
                                        double *results) nogil

    void american_psor_theta_fp64_batch(const double *S,
                                        const double *K,
                                        const char *cp_flag,
                                        const double *ttm,
                                        const double *iv,
                                        const double *r,
                                        const double *q,
                                        int    n_s,
                                        int    n_t,
                                        int    k_mul,
                                        double w,
                                        double tol,
                                        int    max_iter,
                                        size_t n_options,
                                        double *results) nogil

    void american_psor_vega_fp64_batch(const double *S,
                                       const double *K,
                                       const char *cp_flag,
                                       const double *ttm,
                                       const double *iv,
                                       const double *r,
                                       const double *q,
                                       int    n_s,
                                       int    n_t,
                                       int    k_mul,
                                       double w,
                                       double tol,
                                       int    max_iter,
                                       size_t n_options,
                                       double *results) nogil

    void american_psor_rho_fp64_batch(const double *S,
                                      const double *K,
                                      const char *cp_flag,
                                      const double *ttm,
                                      const double *iv,
                                      const double *r,
                                      const double *q,
                                      int    n_s,
                                      int    n_t,
                                      int    k_mul,
                                      double w,
                                      double tol,
                                      int    max_iter,
                                      size_t n_options,
                                      double *results) nogil

    #  iv __________________________________________________________________________________________
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
                                 const double *lo_init,
                                 const double *hi_init) nogil

    void american_psor_iv_fp64_batch(const double *P,
                                     const double *S,
                                     const double *K,
                                     const char *cp_flag,
                                     const double *ttm,
                                     const double *r,
                                     const double *q,
                                     int    n_s,
                                     int    n_t,
                                     int    k_mul,
                                     double w,
                                     double psor_tol,
                                     int    psor_max_iter,
                                     size_t n_options,
                                     double *results,
                                     double tol,
                                     size_t max_iter,
                                     const double *lo_init,
                                     const double *hi_init) nogil

    #  fp32 ----------------------------------------------------------------------------------------
    #  price _______________________________________________________________________________________
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
                                   float *delta,
                                   float *gamma,
                                   float *theta) nogil

    void american_psor_price_fp32_batch(const float *S,
                                        const float *K,
                                        const char *cp_flag,
                                        const float *ttm,
                                        const float *iv,
                                        const float *r,
                                        const float *q,
                                        int    n_s,
                                        int    n_t,
                                        int    k_mul,
                                        float  w,
                                        float  tol,
                                        int    max_iter,
                                        size_t n_options,
                                        float *result,
                                        float *delta,
                                        float *gamma,
                                        float *theta) nogil

    void american_psor_price_fp32_cuda(const float *S,
                                       const float *K,
                                       const char *cp_flag,
                                       const float *ttm,
                                       const float *iv,
                                       const float *r,
                                       const float *q,
                                       int    n_s,
                                       int    n_t,
                                       int    k_mul,
                                       float  w,
                                       float  tol,
                                       int    max_iter,
                                       size_t n_options,
                                       float *result,
                                       float *delta,
                                       float *gamma,
                                       float *theta,
                                       cudaStream_t stream,
                                       int          device,
                                       bool         on_device,
                                       bool         is_pinned,
                                       bool         sync) nogil

    #  greeks ______________________________________________________________________________________
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
                                   float *rho) nogil

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
                                   int   max_iter) nogil

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
                                   int   max_iter) nogil

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
                                   int   max_iter) nogil

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
                                  int   max_iter) nogil

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
                                 int   max_iter) nogil

    void american_psor_greeks_fp32_batch(const float *S,
                                         const float *K,
                                         const char *cp_flag,
                                         const float *ttm,
                                         const float *iv,
                                         const float *r,
                                         const float *q,
                                         int    n_s,
                                         int    n_t,
                                         int    k_mul,
                                         float  w,
                                         float  tol,
                                         int    max_iter,
                                         size_t n_options,
                                         float *delta,
                                         float *gamma,
                                         float *theta,
                                         float *vega,
                                         float *rho) nogil

    void american_psor_delta_fp32_batch(const float *S,
                                        const float *K,
                                        const char *cp_flag,
                                        const float *ttm,
                                        const float *iv,
                                        const float *r,
                                        const float *q,
                                        int    n_s,
                                        int    n_t,
                                        int    k_mul,
                                        float  w,
                                        float  tol,
                                        int    max_iter,
                                        size_t n_options,
                                        float *results) nogil

    void american_psor_gamma_fp32_batch(const float *S,
                                        const float *K,
                                        const char *cp_flag,
                                        const float *ttm,
                                        const float *iv,
                                        const float *r,
                                        const float *q,
                                        int    n_s,
                                        int    n_t,
                                        int    k_mul,
                                        float  w,
                                        float  tol,
                                        int    max_iter,
                                        size_t n_options,
                                        float *results) nogil

    void american_psor_theta_fp32_batch(const float *S,
                                        const float *K,
                                        const char *cp_flag,
                                        const float *ttm,
                                        const float *iv,
                                        const float *r,
                                        const float *q,
                                        int    n_s,
                                        int    n_t,
                                        int    k_mul,
                                        float  w,
                                        float  tol,
                                        int    max_iter,
                                        size_t n_options,
                                        float *results) nogil

    void american_psor_vega_fp32_batch(const float *S,
                                       const float *K,
                                       const char *cp_flag,
                                       const float *ttm,
                                       const float *iv,
                                       const float *r,
                                       const float *q,
                                       int    n_s,
                                       int    n_t,
                                       int    k_mul,
                                       float  w,
                                       float  tol,
                                       int    max_iter,
                                       size_t n_options,
                                       float *results) nogil

    void american_psor_rho_fp32_batch(const float *S,
                                      const float *K,
                                      const char *cp_flag,
                                      const float *ttm,
                                      const float *iv,
                                      const float *r,
                                      const float *q,
                                      int    n_s,
                                      int    n_t,
                                      int    k_mul,
                                      float  w,
                                      float  tol,
                                      int    max_iter,
                                      size_t n_options,
                                      float *results) nogil

    #  iv __________________________________________________________________________________________
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
                                const float *lo_init,
                                const float *hi_init) nogil

    void american_psor_iv_fp32_batch(const float *P,
                                     const float *S,
                                     const float *K,
                                     const char *cp_flag,
                                     const float *ttm,
                                     const float *r,
                                     const float *q,
                                     int    n_s,
                                     int    n_t,
                                     int    k_mul,
                                     float  w,
                                     float  psor_tol,
                                     int    psor_max_iter,
                                     size_t n_options,
                                     float *results,
                                     float  tol,
                                     size_t max_iter,
                                     const float *lo_init,
                                     const float *hi_init) nogil

    #  american::ttree =============================================================================
    #  fp64 ----------------------------------------------------------------------------------------
    #  price _______________________________________________________________________________________
    double american_ttree_price_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps) nogil

    void american_ttree_price_fp64_batch(const double *S,
                                         const double *K,
                                         const char *cp_flag,
                                         const double *ttm,
                                         const double *iv,
                                         const double *r,
                                         const double *q,
                                         int    n_steps,
                                         size_t n_options,
                                         double *results) nogil

    void american_ttree_price_fp64_cuda(const double *S,
                                        const double *K,
                                        const char *cp_flag,
                                        const double *ttm,
                                        const double *iv,
                                        const double *r,
                                        const double *q,
                                        int    n_steps,
                                        size_t n_options,
                                        double *result,
                                        cudaStream_t stream,
                                        int          device,
                                        bool         on_device,
                                        bool         is_pinned,
                                        bool         sync) nogil

    #  greeks ______________________________________________________________________________________
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
                                    double *rho) nogil

    double american_ttree_delta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps) nogil

    double american_ttree_gamma_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps) nogil

    double american_ttree_theta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps) nogil

    double american_ttree_vega_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps) nogil

    double american_ttree_rho_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q, int n_steps) nogil

    void american_ttree_greeks_fp64_batch(const double *S,
                                          const double *K,
                                          const char *cp_flag,
                                          const double *ttm,
                                          const double *iv,
                                          const double *r,
                                          const double *q,
                                          int    n_steps,
                                          size_t n_options,
                                          double *delta,
                                          double *gamma,
                                          double *theta,
                                          double *vega,
                                          double *rho) nogil

    void american_ttree_delta_fp64_batch(const double *S,
                                         const double *K,
                                         const char *cp_flag,
                                         const double *ttm,
                                         const double *iv,
                                         const double *r,
                                         const double *q,
                                         int    n_steps,
                                         size_t n_options,
                                         double *results) nogil

    void american_ttree_gamma_fp64_batch(const double *S,
                                         const double *K,
                                         const char *cp_flag,
                                         const double *ttm,
                                         const double *iv,
                                         const double *r,
                                         const double *q,
                                         int    n_steps,
                                         size_t n_options,
                                         double *results) nogil

    void american_ttree_theta_fp64_batch(const double *S,
                                         const double *K,
                                         const char *cp_flag,
                                         const double *ttm,
                                         const double *iv,
                                         const double *r,
                                         const double *q,
                                         int    n_steps,
                                         size_t n_options,
                                         double *results) nogil

    void american_ttree_vega_fp64_batch(const double *S,
                                        const double *K,
                                        const char *cp_flag,
                                        const double *ttm,
                                        const double *iv,
                                        const double *r,
                                        const double *q,
                                        int    n_steps,
                                        size_t n_options,
                                        double *results) nogil

    void american_ttree_rho_fp64_batch(const double *S,
                                       const double *K,
                                       const char *cp_flag,
                                       const double *ttm,
                                       const double *iv,
                                       const double *r,
                                       const double *q,
                                       int    n_steps,
                                       size_t n_options,
                                       double *results) nogil

    #  iv __________________________________________________________________________________________
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
                                  const double *lo_init,
                                  const double *hi_init) nogil

    void american_ttree_iv_fp64_batch(const double *P,
                                      const double *S,
                                      const double *K,
                                      const char *cp_flag,
                                      const double *ttm,
                                      const double *r,
                                      const double *q,
                                      int    n_steps,
                                      size_t n_options,
                                      double *results,
                                      double tol,
                                      size_t max_iter,
                                      const double *lo_init,
                                      const double *hi_init) nogil

    #  fp32 ----------------------------------------------------------------------------------------
    #  price _______________________________________________________________________________________
    float american_ttree_price_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps) nogil

    void american_ttree_price_fp32_batch(const float *S,
                                         const float *K,
                                         const char *cp_flag,
                                         const float *ttm,
                                         const float *iv,
                                         const float *r,
                                         const float *q,
                                         int    n_steps,
                                         size_t n_options,
                                         float *results) nogil

    void american_ttree_price_fp32_cuda(const float *S,
                                        const float *K,
                                        const char *cp_flag,
                                        const float *ttm,
                                        const float *iv,
                                        const float *r,
                                        const float *q,
                                        int    n_steps,
                                        size_t n_options,
                                        float *result,
                                        cudaStream_t stream,
                                        int          device,
                                        bool         on_device,
                                        bool         is_pinned,
                                        bool         sync) nogil

    #  greeks ______________________________________________________________________________________
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
                                    float *rho) nogil

    float american_ttree_delta_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps) nogil

    float american_ttree_gamma_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps) nogil

    float american_ttree_theta_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps) nogil

    float american_ttree_vega_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps) nogil

    float american_ttree_rho_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q, int n_steps) nogil

    void american_ttree_greeks_fp32_batch(const float *S,
                                          const float *K,
                                          const char *cp_flag,
                                          const float *ttm,
                                          const float *iv,
                                          const float *r,
                                          const float *q,
                                          int    n_steps,
                                          size_t n_options,
                                          float *delta,
                                          float *gamma,
                                          float *theta,
                                          float *vega,
                                          float *rho) nogil

    void american_ttree_delta_fp32_batch(const float *S,
                                         const float *K,
                                         const char *cp_flag,
                                         const float *ttm,
                                         const float *iv,
                                         const float *r,
                                         const float *q,
                                         int    n_steps,
                                         size_t n_options,
                                         float *results) nogil

    void american_ttree_gamma_fp32_batch(const float *S,
                                         const float *K,
                                         const char *cp_flag,
                                         const float *ttm,
                                         const float *iv,
                                         const float *r,
                                         const float *q,
                                         int    n_steps,
                                         size_t n_options,
                                         float *results) nogil

    void american_ttree_theta_fp32_batch(const float *S,
                                         const float *K,
                                         const char *cp_flag,
                                         const float *ttm,
                                         const float *iv,
                                         const float *r,
                                         const float *q,
                                         int    n_steps,
                                         size_t n_options,
                                         float *results) nogil

    void american_ttree_vega_fp32_batch(const float *S,
                                        const float *K,
                                        const char *cp_flag,
                                        const float *ttm,
                                        const float *iv,
                                        const float *r,
                                        const float *q,
                                        int    n_steps,
                                        size_t n_options,
                                        float *results) nogil

    void american_ttree_rho_fp32_batch(const float *S,
                                       const float *K,
                                       const char *cp_flag,
                                       const float *ttm,
                                       const float *iv,
                                       const float *r,
                                       const float *q,
                                       int    n_steps,
                                       size_t n_options,
                                       float *results) nogil

    #  iv __________________________________________________________________________________________
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
                                 const float *lo_init,
                                 const float *hi_init) nogil

    void american_ttree_iv_fp32_batch(const float *P,
                                      const float *S,
                                      const float *K,
                                      const char *cp_flag,
                                      const float *ttm,
                                      const float *r,
                                      const float *q,
                                      int    n_steps,
                                      size_t n_options,
                                      float *results,
                                      float  tol,
                                      size_t max_iter,
                                      const float *lo_init,
                                      const float *hi_init) nogil

    #  european::bsm ===============================================================================
    #  fp64 ----------------------------------------------------------------------------------------
    #  price _______________________________________________________________________________________
    double european_bsm_price_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q) nogil

    void european_bsm_price_fp64_batch(const double *S,
                                       const double *K,
                                       const char *cp_flag,
                                       const double *ttm,
                                       const double *iv,
                                       const double *r,
                                       const double *q,
                                       size_t n_options,
                                       double *result) nogil

    void european_bsm_price_fp64_cuda(const double *S,
                                      const double *K,
                                      const char *cp_flag,
                                      const double *ttm,
                                      const double *iv,
                                      const double *r,
                                      const double *q,
                                      size_t n_options,
                                      double *result,
                                      cudaStream_t stream,
                                      int          device,
                                      bool         on_device,
                                      bool         is_pinned,
                                      bool         sync) nogil

    #  greeks ______________________________________________________________________________________
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
                                  double *rho) nogil

    double european_bsm_delta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q) nogil
    double european_bsm_gamma_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q) nogil
    double european_bsm_theta_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q) nogil
    double european_bsm_vega_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q) nogil
    double european_bsm_rho_fp64(
        double S, double K, char cp_flag, double ttm, double iv, double r, double q) nogil

    void european_bsm_greeks_fp64_batch(const double *S,
                                        const double *K,
                                        const char *cp_flag,
                                        const double *ttm,
                                        const double *iv,
                                        const double *r,
                                        const double *q,
                                        size_t n_options,
                                        double *delta,
                                        double *gamma,
                                        double *theta,
                                        double *vega,
                                        double *rho) nogil

    void european_bsm_delta_fp64_batch(const double *S,
                                       const double *K,
                                       const char *cp_flag,
                                       const double *ttm,
                                       const double *iv,
                                       const double *r,
                                       const double *q,
                                       size_t n_options,
                                       double *results) nogil

    void european_bsm_gamma_fp64_batch(const double *S,
                                       const double *K,
                                       const char *cp_flag,
                                       const double *ttm,
                                       const double *iv,
                                       const double *r,
                                       const double *q,
                                       size_t n_options,
                                       double *results) nogil

    void european_bsm_theta_fp64_batch(const double *S,
                                       const double *K,
                                       const char *cp_flag,
                                       const double *ttm,
                                       const double *iv,
                                       const double *r,
                                       const double *q,
                                       size_t n_options,
                                       double *results) nogil

    void european_bsm_vega_fp64_batch(const double *S,
                                      const double *K,
                                      const char *cp_flag,
                                      const double *ttm,
                                      const double *iv,
                                      const double *r,
                                      const double *q,
                                      size_t n_options,
                                      double *results) nogil

    void european_bsm_rho_fp64_batch(const double *S,
                                     const double *K,
                                     const char *cp_flag,
                                     const double *ttm,
                                     const double *iv,
                                     const double *r,
                                     const double *q,
                                     size_t n_options,
                                     double *results) nogil

    #  iv __________________________________________________________________________________________
    double european_bsm_iv_fp64(double P,
                                double S,
                                double K,
                                char   cp_flag,
                                double ttm,
                                double r,
                                double q,
                                double tol,
                                size_t max_iter,
                                const double *lo_init,
                                const double *hi_init) nogil

    void european_bsm_iv_fp64_batch(const double *P,
                                    const double *S,
                                    const double *K,
                                    const char *cp_flag,
                                    const double *ttm,
                                    const double *r,
                                    const double *q,
                                    size_t n_options,
                                    double *results,
                                    double tol,
                                    size_t max_iter,
                                    const double *lo_init,
                                    const double *hi_init) nogil

    #  fp32 ----------------------------------------------------------------------------------------
    #  price _______________________________________________________________________________________
    float european_bsm_price_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q) nogil

    void european_bsm_price_fp32_batch(const float *S,
                                       const float *K,
                                       const char *cp_flag,
                                       const float *ttm,
                                       const float *iv,
                                       const float *r,
                                       const float *q,
                                       size_t n_options,
                                       float *result) nogil

    void european_bsm_price_fp32_cuda(const float *S,
                                      const float *K,
                                      const char *cp_flag,
                                      const float *ttm,
                                      const float *iv,
                                      const float *r,
                                      const float *q,
                                      size_t n_options,
                                      float *result,
                                      cudaStream_t stream,
                                      int          device,
                                      bool         on_device,
                                      bool         is_pinned,
                                      bool         sync) nogil

    #  greeks ______________________________________________________________________________________
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
                                  float *rho) nogil

    float european_bsm_delta_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q) nogil
    float european_bsm_gamma_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q) nogil
    float european_bsm_theta_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q) nogil
    float european_bsm_vega_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q) nogil
    float european_bsm_rho_fp32(
        float S, float K, char cp_flag, float ttm, float iv, float r, float q) nogil

    void european_bsm_greeks_fp32_batch(const float *S,
                                        const float *K,
                                        const char *cp_flag,
                                        const float *ttm,
                                        const float *iv,
                                        const float *r,
                                        const float *q,
                                        size_t n_options,
                                        float *delta,
                                        float *gamma,
                                        float *theta,
                                        float *vega,
                                        float *rho) nogil

    void european_bsm_delta_fp32_batch(const float *S,
                                       const float *K,
                                       const char *cp_flag,
                                       const float *ttm,
                                       const float *iv,
                                       const float *r,
                                       const float *q,
                                       size_t n_options,
                                       float *results) nogil

    void european_bsm_gamma_fp32_batch(const float *S,
                                       const float *K,
                                       const char *cp_flag,
                                       const float *ttm,
                                       const float *iv,
                                       const float *r,
                                       const float *q,
                                       size_t n_options,
                                       float *results) nogil

    void european_bsm_theta_fp32_batch(const float *S,
                                       const float *K,
                                       const char *cp_flag,
                                       const float *ttm,
                                       const float *iv,
                                       const float *r,
                                       const float *q,
                                       size_t n_options,
                                       float *results) nogil

    void european_bsm_vega_fp32_batch(const float *S,
                                      const float *K,
                                      const char *cp_flag,
                                      const float *ttm,
                                      const float *iv,
                                      const float *r,
                                      const float *q,
                                      size_t n_options,
                                      float *results) nogil

    void european_bsm_rho_fp32_batch(const float *S,
                                     const float *K,
                                     const char *cp_flag,
                                     const float *ttm,
                                     const float *iv,
                                     const float *r,
                                     const float *q,
                                     size_t n_options,
                                     float *results) nogil
    #  iv __________________________________________________________________________________________
    float european_bsm_iv_fp32(float  P,
                               float  S,
                               float  K,
                               char   cp_flag,
                               float  ttm,
                               float  r,
                               float  q,
                               float  tol,
                               size_t max_iter,
                               const float *lo_init,
                               const float *hi_init) nogil

    void european_bsm_iv_fp32_batch(const float *P,
                                    const float *S,
                                    const float *K,
                                    const char *cp_flag,
                                    const float *ttm,
                                    const float *r,
                                    const float *q,
                                    size_t n_options,
                                    float *results,
                                    float  tol,
                                    size_t max_iter,
                                    const float *lo_init,
                                    const float *hi_init) nogil


    # ==============================================================================================
    const char* fastvol_version()
    bint fastvol_cuda_available()
