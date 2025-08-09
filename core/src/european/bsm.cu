#include "fastvol/european/bsm.cuh"

namespace fastvol::european::bsm
{
void price_fp64_cuda(const double *__restrict__ S,
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
    cuda::price<double>(S,
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

void price_fp32_cuda(const float *__restrict__ S,
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
    cuda::price<float>(S,
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

} // namespace fastvol::european::bsm
