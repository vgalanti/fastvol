#include "fastvol/ffi.h"

#include "fastvol/american/bopm.cuh"
#include "fastvol/american/psor.cuh"
#include "fastvol/american/ttree.cuh"
#include "fastvol/european/bsm.cuh"

#include <cstddef>

extern "C"
{
    /* american::bopm ============================================================================*/
    /* fp64 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
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
                                       bool         sync)
    {
        fastvol::american::bopm::cuda::price<double>(S,
                                                     K,
                                                     cp_flag,
                                                     ttm,
                                                     iv,
                                                     r,
                                                     q,
                                                     n_steps,
                                                     n_options,
                                                     result,
                                                     stream,
                                                     device,
                                                     on_device,
                                                     is_pinned,
                                                     sync);
    }

    /* fp32 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
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
                                       bool         sync)
    {
        fastvol::american::bopm::cuda::price<float>(S,
                                                    K,
                                                    cp_flag,
                                                    ttm,
                                                    iv,
                                                    r,
                                                    q,
                                                    n_steps,
                                                    n_options,
                                                    result,
                                                    stream,
                                                    device,
                                                    on_device,
                                                    is_pinned,
                                                    sync);
    }

    /* american::psor ============================================================================*/
    /* fp64 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
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
                                       bool         sync)
    {
        fastvol::american::psor::cuda::price<double>(S,
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
                                                     theta,
                                                     stream,
                                                     device,
                                                     on_device,
                                                     is_pinned,
                                                     sync);
    }

    /* fp32 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
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
                                       bool         sync)
    {
        fastvol::american::psor::cuda::price<float>(S,
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
                                                    theta,
                                                    stream,
                                                    device,
                                                    on_device,
                                                    is_pinned,
                                                    sync);
    }

    /* american::ttree ===========================================================================*/
    /* fp64 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
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
                                        bool         sync)
    {
        fastvol::american::ttree::cuda::price<double>(S,
                                                      K,
                                                      cp_flag,
                                                      ttm,
                                                      iv,
                                                      r,
                                                      q,
                                                      n_steps,
                                                      n_options,
                                                      result,
                                                      stream,
                                                      device,
                                                      on_device,
                                                      is_pinned,
                                                      sync);
    }

    /* fp32 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
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
                                        bool         sync)
    {
        fastvol::american::ttree::cuda::price<float>(S,
                                                     K,
                                                     cp_flag,
                                                     ttm,
                                                     iv,
                                                     r,
                                                     q,
                                                     n_steps,
                                                     n_options,
                                                     result,
                                                     stream,
                                                     device,
                                                     on_device,
                                                     is_pinned,
                                                     sync);
    }

    /* european::bsm =============================================================================*/
    /* fp64 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
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
                                      bool         sync)
    {
        fastvol::european::bsm::cuda::price<double>(S,
                                                    K,
                                                    cp_flag,
                                                    ttm,
                                                    iv,
                                                    r,
                                                    q,
                                                    n_options,
                                                    result,
                                                    stream,
                                                    device,
                                                    on_device,
                                                    is_pinned,
                                                    sync);
    }

    /* fp32 --------------------------------------------------------------------------------------*/
    /* price _____________________________________________________________________________________*/
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
                                      bool         sync)
    {
        fastvol::european::bsm::cuda::price<float>(S,
                                                   K,
                                                   cp_flag,
                                                   ttm,
                                                   iv,
                                                   r,
                                                   q,
                                                   n_options,
                                                   result,
                                                   stream,
                                                   device,
                                                   on_device,
                                                   is_pinned,
                                                   sync);
    }
}
