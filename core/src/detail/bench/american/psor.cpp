#include "fastvol/american/psor.hpp"
#include "fastvol/detail/bench.hpp"
#include <cstddef>

using namespace fastvol::detail::bench;
using namespace fastvol::american::psor;

namespace fastvol::detail::bench::american::psor
{

template <typename T> inline constexpr T TOL      = T(1e-3);
constexpr size_t                         MAX_ITER = 100;

constexpr int ns[3] = {128, 256, 512};
constexpr int nt[3] = {256, 512, 1024};
constexpr int ns_n  = 3;
constexpr int nt_n  = 3;

constexpr int                            k_mul         = 4;
template <typename T> inline constexpr T w             = T(0);
constexpr int                            psor_max_iter = 200;
constexpr double                         psor_tol      = 1e-6;
constexpr float                          psor_tol_f    = 1e-3f;

/* price -----------------------------------------------------------------------------------------*/
void price_cpu()
{
    // setup
    char   label[64];
    size_t n = 200;

    TestData<double> h;
    TestData<float>  h_f;

    size_t mult = get_batch_multiplier();
    generate_data<double>(h, n * mult);
    generate_data<float>(h_f, n * mult);

    // benchmark
    print_section("[fp64] (cpu) price ", '-', true);
    print_header_cpu();

    for (int j = 0; j < ns_n; j++)
    {
        for (int k = 0; k < nt_n; k++)
        {
            if (k == 0)
                snprintf(label, sizeof(label), "> n_s: %3d | n_t: %4d", ns[j], nt[k]);
            else
                snprintf(label, sizeof(label), ">          | n_t: %4d", nt[k]);

            benchmark_cpu(
                label,
                h,
                n,
                [&](size_t i)
                {
                    return price_fp64(h.S[i],
                                      h.K[i],
                                      h.cp[i],
                                      h.ttm[i],
                                      h.iv[i],
                                      h.r[i],
                                      h.q[i],
                                      ns[j],
                                      nt[k],
                                      k_mul,
                                      w<double>,
                                      psor_tol,
                                      psor_max_iter);
                },
                [&](size_t i)
                {
                    price_fp64_batch(h.S,
                                     h.K,
                                     h.cp,
                                     h.ttm,
                                     h.iv,
                                     h.r,
                                     h.q,
                                     ns[j],
                                     nt[k],
                                     k_mul,
                                     w<double>,
                                     psor_tol,
                                     psor_max_iter,
                                     i,
                                     h.P);
                });
        }
    }

    print_section("[fp32] (cpu) price ", '-', false);
    print_header_cpu();

    for (int j = 0; j < ns_n; j++)
    {
        for (int k = 0; k < nt_n; k++)
        {
            if (k == 0)
                snprintf(label, sizeof(label), "> n_s: %3d | n_t: %4d", ns[j], nt[k]);
            else
                snprintf(label, sizeof(label), ">          | n_t: %4d", nt[k]);

            benchmark_cpu(
                label,
                h_f,
                n,
                [&](size_t i)
                {
                    return price_fp32(h_f.S[i],
                                      h_f.K[i],
                                      h_f.cp[i],
                                      h_f.ttm[i],
                                      h_f.iv[i],
                                      h_f.r[i],
                                      h_f.q[i],
                                      ns[j],
                                      nt[k],
                                      k_mul,
                                      w<float>,
                                      psor_tol_f,
                                      psor_max_iter);
                },
                [&](size_t i)
                {
                    price_fp32_batch(h_f.S,
                                     h_f.K,
                                     h_f.cp,
                                     h_f.ttm,
                                     h_f.iv,
                                     h_f.r,
                                     h_f.q,
                                     ns[j],
                                     nt[k],
                                     k_mul,
                                     w<float>,
                                     psor_tol_f,
                                     psor_max_iter,
                                     i,
                                     h_f.P);
                });
        }
    }

    // cleanup
    free_data(h);
    free_data(h_f);
}

#ifdef FASTVOL_CUDA_ENABLED
void price_cuda()
{
    // setup
    size_t n    = 30'000;
    int    ks[] = {5'000, 15'000, 30'000};
    int    n_ks = 3;
    char   label[64];

    TestData<double> h;
    TestData<float>  h_f;

    generate_data<double>(h, n);
    generate_data<float>(h_f, n);

    TestData<double> p = generate_pinned_copy<double>(h);
    TestData<double> d = generate_device_copy<double>(h);

    TestData<float> p_f = generate_pinned_copy<float>(h_f);
    TestData<float> d_f = generate_device_copy<float>(h_f);

    // benchmark
    print_section("[fp64] (cuda) price ", '-', true);
    print_header_cuda();

    for (int j = 0; j < ns_n; j++)
    {
        for (int k = 0; k < nt_n; k++)
        {
            for (int l = 0; l < n_ks; l++)
            {
                std::string k_buf = kformat(ks[l]);

                if (k == 0 && l == 0)
                    snprintf(label,
                             sizeof(label),
                             "> n_s: %3d | n_t: %4d | n: %s",
                             ns[j],
                             nt[k],
                             k_buf.c_str());

                else if (l == 0)
                    snprintf(label,
                             sizeof(label),
                             ">          | n_t: %4d | n: %s",
                             nt[k],
                             k_buf.c_str());
                else
                    snprintf(label, sizeof(label), ">          |           | n: %s", k_buf.c_str());

                benchmark_cuda(label,
                               h,
                               p,
                               d,
                               ks[l],
                               [j, k](const double *S,
                                      const double *K,
                                      const char   *cp,
                                      const double *ttm,
                                      const double *iv,
                                      const double *r,
                                      const double *q,
                                      size_t        n,
                                      double       *P,
                                      cudaStream_t  stream,
                                      int           device,
                                      bool          on_device,
                                      bool          is_pinned,
                                      bool          sync)
                               {
                                   fastvol::american::psor::price_fp64_cuda(S,
                                                                            K,
                                                                            cp,
                                                                            ttm,
                                                                            iv,
                                                                            r,
                                                                            q,
                                                                            ns[j],
                                                                            nt[k],
                                                                            k_mul,
                                                                            w<double>,
                                                                            psor_tol,
                                                                            psor_max_iter,
                                                                            n,
                                                                            P,
                                                                            nullptr,
                                                                            nullptr,
                                                                            nullptr,
                                                                            stream,
                                                                            device,
                                                                            on_device,
                                                                            is_pinned,
                                                                            sync);
                               });
            }
        }
    }

    print_section("[fp32] (cuda) price ", '-', false);
    print_header_cuda();

    for (int j = 0; j < ns_n; j++)
    {
        for (int k = 0; k < nt_n; k++)
        {
            for (int l = 0; l < n_ks; l++)
            {
                std::string k_buf = kformat(ks[l]);

                if (k == 0 && l == 0)
                    snprintf(label,
                             sizeof(label),
                             "> n_s: %3d | n_t: %4d | n: %s",
                             ns[j],
                             nt[k],
                             k_buf.c_str());

                else if (l == 0)
                    snprintf(label,
                             sizeof(label),
                             ">          | n_t: %4d | n: %s",
                             nt[k],
                             k_buf.c_str());
                else
                    snprintf(label, sizeof(label), ">          |           | n: %s", k_buf.c_str());

                benchmark_cuda(label,
                               h_f,
                               p_f,
                               d_f,
                               ks[l],
                               [j, k](const float *S,
                                      const float *K,
                                      const char  *cp,
                                      const float *ttm,
                                      const float *iv,
                                      const float *r,
                                      const float *q,
                                      size_t       n,
                                      float       *P,
                                      cudaStream_t stream,
                                      int          device,
                                      bool         on_device,
                                      bool         is_pinned,
                                      bool         sync)
                               {
                                   fastvol::american::psor::price_fp32_cuda(S,
                                                                            K,
                                                                            cp,
                                                                            ttm,
                                                                            iv,
                                                                            r,
                                                                            q,
                                                                            ns[j],
                                                                            nt[k],
                                                                            k_mul,
                                                                            w<float>,
                                                                            psor_tol_f,
                                                                            psor_max_iter,
                                                                            n,
                                                                            P,
                                                                            nullptr,
                                                                            nullptr,
                                                                            nullptr,
                                                                            stream,
                                                                            device,
                                                                            on_device,
                                                                            is_pinned,
                                                                            sync);
                               });
            }
        }
    }

    // cleanup
    free_data(h);
    free_data(h_f);
}
#endif

/* price -----------------------------------------------------------------------------------------*/
void greeks_cpu()
{
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#endif
    // setup
    char   label[64];
    size_t n = 10;

    TestData<double> h;
    TestData<float>  h_f;

    size_t mult = get_batch_multiplier();
    generate_data<double>(h, n * mult);
    generate_data<float>(h_f, n * mult);

    // benchmark
    print_section("[fp64] (cpu) greeks ", '-', true);
    print_header_cpu();

    for (int j = 0; j < ns_n; j++)
    {
        for (int k = 0; k < nt_n; k++)
        {
            if (k == 0)
                snprintf(label, sizeof(label), "> n_s: %3d | n_t: %4d | *", ns[j], nt[k]);
            else
                snprintf(label, sizeof(label), "           | n_t: %4d | *", nt[k]);

            benchmark_cpu(
                label,
                h,
                n,
                [&](size_t i)
                {
                    greeks_fp64(h.S[i],
                                h.K[i],
                                h.cp[i],
                                h.ttm[i],
                                h.iv[i],
                                h.r[i],
                                h.q[i],
                                ns[j],
                                nt[k],
                                k_mul,
                                w<double>,
                                psor_tol,
                                psor_max_iter,
                                &h.P[i],
                                &h.P[i],
                                &h.P[i],
                                &h.P[i],
                                &h.P[i]);
                    return h.P[i];
                },
                [&](size_t i)
                {
                    greeks_fp64_batch(h.S,
                                      h.K,
                                      h.cp,
                                      h.ttm,
                                      h.iv,
                                      h.r,
                                      h.q,
                                      ns[j],
                                      nt[k],
                                      k_mul,
                                      w<double>,
                                      psor_tol,
                                      psor_max_iter,
                                      i,
                                      h.P,
                                      h.P,
                                      h.P,
                                      h.P,
                                      h.P);
                });

            benchmark_cpu(
                "                       | delta",
                h,
                n,
                [&](size_t i)
                {
                    return delta_fp64(h.S[i],
                                      h.K[i],
                                      h.cp[i],
                                      h.ttm[i],
                                      h.iv[i],
                                      h.r[i],
                                      h.q[i],
                                      ns[j],
                                      nt[k],
                                      k_mul,
                                      w<double>,
                                      psor_tol,
                                      psor_max_iter);
                },
                [&](size_t i)
                {
                    delta_fp64_batch(h.S,
                                     h.K,
                                     h.cp,
                                     h.ttm,
                                     h.iv,
                                     h.r,
                                     h.q,
                                     ns[j],
                                     nt[k],
                                     k_mul,
                                     w<double>,
                                     psor_tol,
                                     psor_max_iter,
                                     i,
                                     h.P);
                });

            benchmark_cpu(
                "                       | gamma",
                h,
                n,
                [&](size_t i)
                {
                    return gamma_fp64(h.S[i],
                                      h.K[i],
                                      h.cp[i],
                                      h.ttm[i],
                                      h.iv[i],
                                      h.r[i],
                                      h.q[i],
                                      ns[j],
                                      nt[k],
                                      k_mul,
                                      w<double>,
                                      psor_tol,
                                      psor_max_iter);
                },
                [&](size_t i)
                {
                    gamma_fp64_batch(h.S,
                                     h.K,
                                     h.cp,
                                     h.ttm,
                                     h.iv,
                                     h.r,
                                     h.q,
                                     ns[j],
                                     nt[k],
                                     k_mul,
                                     w<double>,
                                     psor_tol,
                                     psor_max_iter,
                                     i,
                                     h.P);
                });

            benchmark_cpu(
                "                       | theta",
                h,
                n,
                [&](size_t i)
                {
                    return theta_fp64(h.S[i],
                                      h.K[i],
                                      h.cp[i],
                                      h.ttm[i],
                                      h.iv[i],
                                      h.r[i],
                                      h.q[i],
                                      ns[j],
                                      nt[k],
                                      k_mul,
                                      w<double>,
                                      psor_tol,
                                      psor_max_iter);
                },
                [&](size_t i)
                {
                    theta_fp64_batch(h.S,
                                     h.K,
                                     h.cp,
                                     h.ttm,
                                     h.iv,
                                     h.r,
                                     h.q,
                                     ns[j],
                                     nt[k],
                                     k_mul,
                                     w<double>,
                                     psor_tol,
                                     psor_max_iter,
                                     i,
                                     h.P);
                });

            benchmark_cpu(
                "                       | vega",
                h,
                n,
                [&](size_t i)
                {
                    return vega_fp64(h.S[i],
                                     h.K[i],
                                     h.cp[i],
                                     h.ttm[i],
                                     h.iv[i],
                                     h.r[i],
                                     h.q[i],
                                     ns[j],
                                     nt[k],
                                     k_mul,
                                     w<double>,
                                     psor_tol,
                                     psor_max_iter);
                },
                [&](size_t i)
                {
                    vega_fp64_batch(h.S,
                                    h.K,
                                    h.cp,
                                    h.ttm,
                                    h.iv,
                                    h.r,
                                    h.q,
                                    ns[j],
                                    nt[k],
                                    k_mul,
                                    w<double>,
                                    psor_tol,
                                    psor_max_iter,
                                    i,
                                    h.P);
                });

            benchmark_cpu(
                "                       | rho",
                h,
                n,
                [&](size_t i)
                {
                    return rho_fp64(h.S[i],
                                    h.K[i],
                                    h.cp[i],
                                    h.ttm[i],
                                    h.iv[i],
                                    h.r[i],
                                    h.q[i],
                                    ns[j],
                                    nt[k],
                                    k_mul,
                                    w<double>,
                                    psor_tol,
                                    psor_max_iter);
                },
                [&](size_t i)
                {
                    rho_fp64_batch(h.S,
                                   h.K,
                                   h.cp,
                                   h.ttm,
                                   h.iv,
                                   h.r,
                                   h.q,
                                   ns[j],
                                   nt[k],
                                   k_mul,
                                   w<double>,
                                   psor_tol,
                                   psor_max_iter,
                                   i,
                                   h.P);
                });
        }
    }

    print_section("[fp32] (cpu) greeks ", '-', false);
    print_header_cpu();

    for (int j = 0; j < ns_n; j++)
    {
        for (int k = 0; k < nt_n; k++)
        {
            if (k == 0)
                snprintf(label, sizeof(label), "> n_s: %3d | n_t: %4d | *", ns[j], nt[k]);
            else
                snprintf(label, sizeof(label), "           | n_t: %4d | *", nt[k]);

            benchmark_cpu(
                label,
                h_f,
                n,
                [&](size_t i)
                {
                    greeks_fp32(h_f.S[i],
                                h_f.K[i],
                                h_f.cp[i],
                                h_f.ttm[i],
                                h_f.iv[i],
                                h_f.r[i],
                                h_f.q[i],
                                ns[j],
                                nt[k],
                                k_mul,
                                w<float>,
                                psor_tol_f,
                                psor_max_iter,
                                &h_f.P[i],
                                &h_f.P[i],
                                &h_f.P[i],
                                &h_f.P[i],
                                &h_f.P[i]);
                    return h_f.P[i];
                },
                [&](size_t i)
                {
                    greeks_fp32_batch(h_f.S,
                                      h_f.K,
                                      h_f.cp,
                                      h_f.ttm,
                                      h_f.iv,
                                      h_f.r,
                                      h_f.q,
                                      ns[j],
                                      nt[k],
                                      k_mul,
                                      w<float>,
                                      psor_tol_f,
                                      psor_max_iter,
                                      i,
                                      h_f.P,
                                      h_f.P,
                                      h_f.P,
                                      h_f.P,
                                      h_f.P);
                });

            benchmark_cpu(
                "                       | delta",
                h_f,
                n,
                [&](size_t i)
                {
                    return delta_fp32(h_f.S[i],
                                      h_f.K[i],
                                      h_f.cp[i],
                                      h_f.ttm[i],
                                      h_f.iv[i],
                                      h_f.r[i],
                                      h_f.q[i],
                                      ns[j],
                                      nt[k],
                                      k_mul,
                                      w<float>,
                                      psor_tol_f,
                                      psor_max_iter);
                },
                [&](size_t i)
                {
                    delta_fp32_batch(h_f.S,
                                     h_f.K,
                                     h_f.cp,
                                     h_f.ttm,
                                     h_f.iv,
                                     h_f.r,
                                     h_f.q,
                                     ns[j],
                                     nt[k],
                                     k_mul,
                                     w<float>,
                                     psor_tol_f,
                                     psor_max_iter,
                                     i,
                                     h_f.P);
                });

            benchmark_cpu(
                "                       | gamma",
                h_f,
                n,
                [&](size_t i)
                {
                    return gamma_fp32(h_f.S[i],
                                      h_f.K[i],
                                      h_f.cp[i],
                                      h_f.ttm[i],
                                      h_f.iv[i],
                                      h_f.r[i],
                                      h_f.q[i],
                                      ns[j],
                                      nt[k],
                                      k_mul,
                                      w<float>,
                                      psor_tol_f,
                                      psor_max_iter);
                },
                [&](size_t i)
                {
                    gamma_fp32_batch(h_f.S,
                                     h_f.K,
                                     h_f.cp,
                                     h_f.ttm,
                                     h_f.iv,
                                     h_f.r,
                                     h_f.q,
                                     ns[j],
                                     nt[k],
                                     k_mul,
                                     w<float>,
                                     psor_tol_f,
                                     psor_max_iter,
                                     i,
                                     h_f.P);
                });

            benchmark_cpu(
                "                       | theta",
                h_f,
                n,
                [&](size_t i)
                {
                    return theta_fp32(h_f.S[i],
                                      h_f.K[i],
                                      h_f.cp[i],
                                      h_f.ttm[i],
                                      h_f.iv[i],
                                      h_f.r[i],
                                      h_f.q[i],
                                      ns[j],
                                      nt[k],
                                      k_mul,
                                      w<float>,
                                      psor_tol_f,
                                      psor_max_iter);
                },
                [&](size_t i)
                {
                    theta_fp32_batch(h_f.S,
                                     h_f.K,
                                     h_f.cp,
                                     h_f.ttm,
                                     h_f.iv,
                                     h_f.r,
                                     h_f.q,
                                     ns[j],
                                     nt[k],
                                     k_mul,
                                     w<float>,
                                     psor_tol_f,
                                     psor_max_iter,
                                     i,
                                     h_f.P);
                });

            benchmark_cpu(
                "                       | vega",
                h_f,
                n,
                [&](size_t i)
                {
                    return vega_fp32(h_f.S[i],
                                     h_f.K[i],
                                     h_f.cp[i],
                                     h_f.ttm[i],
                                     h_f.iv[i],
                                     h_f.r[i],
                                     h_f.q[i],
                                     ns[j],
                                     nt[k],
                                     k_mul,
                                     w<float>,
                                     psor_tol_f,
                                     psor_max_iter);
                },
                [&](size_t i)
                {
                    vega_fp32_batch(h_f.S,
                                    h_f.K,
                                    h_f.cp,
                                    h_f.ttm,
                                    h_f.iv,
                                    h_f.r,
                                    h_f.q,
                                    ns[j],
                                    nt[k],
                                    k_mul,
                                    w<float>,
                                    psor_tol_f,
                                    psor_max_iter,
                                    i,
                                    h_f.P);
                });

            benchmark_cpu(
                "                       | rho",
                h_f,
                n,
                [&](size_t i)
                {
                    return rho_fp32(h_f.S[i],
                                    h_f.K[i],
                                    h_f.cp[i],
                                    h_f.ttm[i],
                                    h_f.iv[i],
                                    h_f.r[i],
                                    h_f.q[i],
                                    ns[j],
                                    nt[k],
                                    k_mul,
                                    w<float>,
                                    psor_tol_f,
                                    psor_max_iter);
                },
                [&](size_t i)
                {
                    rho_fp32_batch(h_f.S,
                                   h_f.K,
                                   h_f.cp,
                                   h_f.ttm,
                                   h_f.iv,
                                   h_f.r,
                                   h_f.q,
                                   ns[j],
                                   nt[k],
                                   k_mul,
                                   w<float>,
                                   psor_tol_f,
                                   psor_max_iter,
                                   i,
                                   h_f.P);
                });
        }
    }

    // cleanup
    free_data(h);
    free_data(h_f);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

/* iv --------------------------------------------------------------------------------------------*/
void iv_cpu()
{
    // setup
    char   label[64];
    size_t n = 20;
    size_t k = n * get_batch_multiplier();

    TestData<double> h;
    TestData<float>  h_f;

    generate_data<double>(h, k);
    generate_data<float>(h_f, k);

    price_fp64_batch(h.S,
                     h.K,
                     h.cp,
                     h.ttm,
                     h.iv,
                     h.r,
                     h.q,
                     128,
                     256,
                     k_mul,
                     w<double>,
                     psor_tol,
                     psor_max_iter,
                     k,
                     h.P);
    price_fp32_batch(h_f.S,
                     h_f.K,
                     h_f.cp,
                     h_f.ttm,
                     h_f.iv,
                     h_f.r,
                     h_f.q,
                     128,
                     256,
                     k_mul,
                     w<float>,
                     psor_tol,
                     psor_max_iter,
                     k,
                     h_f.P);

    // benchmark
    print_section("[fp64] (cpu) iv ", '-', true);
    print_header_cpu();

    for (int j = 0; j < ns_n; j++)
    {
        for (int k = 0; k < nt_n; k++)
        {
            if (k == 0)
                snprintf(label, sizeof(label), "> n_s: %3d | n_t: %4d", ns[j], nt[k]);
            else
                snprintf(label, sizeof(label), ">          | n_t: %4d", nt[k]);

            benchmark_cpu(
                label,
                h,
                n,
                [&](size_t i)
                {
                    return iv_fp64(h.P[i],
                                   h.S[i],
                                   h.K[i],
                                   h.cp[i],
                                   h.ttm[i],
                                   h.r[i],
                                   h.q[i],
                                   ns[j],
                                   nt[k],
                                   k_mul,
                                   w<double>,
                                   psor_tol,
                                   psor_max_iter,
                                   TOL<double>,
                                   MAX_ITER);
                },
                [&](size_t i)
                {
                    iv_fp64_batch(h.P,
                                  h.S,
                                  h.K,
                                  h.cp,
                                  h.ttm,
                                  h.r,
                                  h.q,
                                  ns[j],
                                  nt[k],
                                  k_mul,
                                  w<double>,
                                  psor_tol,
                                  psor_max_iter,
                                  i,
                                  h.iv,
                                  TOL<double>,
                                  MAX_ITER);
                });
        }
    }

    print_section("[fp32] (cpu) iv ", '-', false);
    print_header_cpu();

    for (int j = 0; j < ns_n; j++)
    {
        for (int k = 0; k < nt_n; k++)
        {
            if (k == 0)
                snprintf(label, sizeof(label), "> n_s: %3d | n_t: %4d", ns[j], nt[k]);
            else
                snprintf(label, sizeof(label), ">          | n_t: %4d", nt[k]);

            benchmark_cpu(
                label,
                h_f,
                n,
                [&](size_t i)
                {
                    return iv_fp32(h_f.P[i],
                                   h_f.S[i],
                                   h_f.K[i],
                                   h_f.cp[i],
                                   h_f.ttm[i],
                                   h_f.r[i],
                                   h_f.q[i],
                                   ns[j],
                                   nt[k],
                                   k_mul,
                                   w<float>,
                                   psor_tol_f,
                                   psor_max_iter,
                                   TOL<float>,
                                   MAX_ITER);
                },
                [&](size_t i)
                {
                    iv_fp32_batch(h_f.P,
                                  h_f.S,
                                  h_f.K,
                                  h_f.cp,
                                  h_f.ttm,
                                  h_f.r,
                                  h_f.q,
                                  ns[j],
                                  nt[k],
                                  k_mul,
                                  w<float>,
                                  psor_tol_f,
                                  psor_max_iter,
                                  i,
                                  h_f.iv,
                                  TOL<float>,
                                  MAX_ITER);
                });
        }
    }

    // cleanup
    free_data(h);
    free_data(h_f);
}

/* cpu
 * -------------------------------------------------------------------------------------------*/
void cpu()
{
    print_section(". american::psor ", '=', true);
    price_cpu();
    greeks_cpu();
    iv_cpu();
    // print_section("", '-', false);
}

/* cuda ------------------------------------------------------------------------------------------*/
#ifdef FASTVOL_CUDA_ENABLED
void cuda()
{
    print_section(". american::psor ", '=', true);
    price_cuda();
    // print_section("", '-', false);
}
#endif

/* all
 * -------------------------------------------------------------------------------------------*/
void all()
{
    print_section(". american::psor ", '=', true);
    price_cpu();
    greeks_cpu();
    iv_cpu();
#ifdef FASTVOL_CUDA_ENABLED
    price_cuda();
#endif
    // print_section("", '-', false);
}
} // namespace fastvol::detail::bench::american::psor
