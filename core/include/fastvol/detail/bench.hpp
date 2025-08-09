/**
 * @file bench.hpp
 * @brief Benchmarking utilities for the fastvol library.
 *
 * This file provides a collection of helper functions and data structures for benchmarking
 * the performance of the option pricing models. It includes utilities for generating random
 * test data, timing code execution on both CPU and CUDA devices, and formatting the output
 * in a readable tabular format.
 */

#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <omp.h>
#include <random>
#include <string>
#include <sys/utsname.h>

#ifdef FASTVOL_CUDA_ENABLED
#include "fastvol/detail/cuda.cuh"
#include <cuda_runtime.h>
using namespace fastvol::detail::cuda;
#endif

namespace fastvol::detail::bench
{

/* rand ------------------------------------------------------------------------------------------*/
template <typename T> inline T runiform(T min, T max)
{
    static std::default_random_engine engine(std::random_device{}());
    std::uniform_real_distribution<T> dist(min, max);
    return dist(engine);
}

/* data gen --------------------------------------------------------------------------------------*/
template <typename T> struct TestData
{
    T     *P, *S, *K, *ttm, *iv, *r, *q;
    char  *cp;
    size_t n;
};

template <typename T> void generate_data(TestData<T> &data, size_t n)
{
    data.P   = static_cast<T *>(malloc(n * sizeof(T)));
    data.S   = static_cast<T *>(malloc(n * sizeof(T)));
    data.K   = static_cast<T *>(malloc(n * sizeof(T)));
    data.ttm = static_cast<T *>(malloc(n * sizeof(T)));
    data.iv  = static_cast<T *>(malloc(n * sizeof(T)));
    data.r   = static_cast<T *>(malloc(n * sizeof(T)));
    data.q   = static_cast<T *>(malloc(n * sizeof(T)));
    data.cp  = static_cast<char *>(malloc(n * sizeof(char)));
    data.n   = n;

    assert(data.P && data.S && data.K && data.ttm && data.iv && data.r && data.q && data.cp &&
           "malloc failed");

    for (size_t i = 0; i < n; i++)
    {
        data.S[i]   = runiform<T>(80.0, 120.0);
        data.K[i]   = runiform<T>(50.0, 200.0);
        data.ttm[i] = runiform<T>(0.001, 2.0);
        data.iv[i]  = runiform<T>(0.05, 1.5);
        data.r[i]   = runiform<T>(0.0, 0.10);
        data.q[i]   = runiform<T>(0.0, 0.05);
        data.cp[i]  = rand() & 1;
    }
}

template <typename T> void free_data(TestData<T> &data)
{
    free(data.P);
    free(data.S);
    free(data.K);
    free(data.ttm);
    free(data.iv);
    free(data.r);
    free(data.q);
    free(data.cp);
}

#ifdef FASTVOL_CUDA_ENABLED
template <typename T> TestData<T> generate_pinned_copy(const TestData<T> &src)
{
    TestData<T> dst;
    dst.n = src.n;

    dst.P   = CUDA_ALLOC_HOST(T, dst.n);
    dst.S   = CUDA_ALLOC_HOST(T, dst.n);
    dst.K   = CUDA_ALLOC_HOST(T, dst.n);
    dst.ttm = CUDA_ALLOC_HOST(T, dst.n);
    dst.iv  = CUDA_ALLOC_HOST(T, dst.n);
    dst.r   = CUDA_ALLOC_HOST(T, dst.n);
    dst.q   = CUDA_ALLOC_HOST(T, dst.n);
    dst.cp  = CUDA_ALLOC_HOST(char, dst.n);

    memcpy(dst.P, src.P, dst.n * sizeof(T));
    memcpy(dst.S, src.S, dst.n * sizeof(T));
    memcpy(dst.K, src.K, dst.n * sizeof(T));
    memcpy(dst.ttm, src.ttm, dst.n * sizeof(T));
    memcpy(dst.iv, src.iv, dst.n * sizeof(T));
    memcpy(dst.r, src.r, dst.n * sizeof(T));
    memcpy(dst.q, src.q, dst.n * sizeof(T));
    memcpy(dst.cp, src.cp, dst.n * sizeof(char));

    return dst;
}

template <typename T> TestData<T> generate_device_copy(const TestData<T> &src)
{
    TestData<T> dst;
    dst.n = src.n;

    dst.P   = CUDA_ALLOC_DEVICE(T, dst.n);
    dst.S   = CUDA_ALLOC_DEVICE(T, dst.n);
    dst.K   = CUDA_ALLOC_DEVICE(T, dst.n);
    dst.ttm = CUDA_ALLOC_DEVICE(T, dst.n);
    dst.iv  = CUDA_ALLOC_DEVICE(T, dst.n);
    dst.r   = CUDA_ALLOC_DEVICE(T, dst.n);
    dst.q   = CUDA_ALLOC_DEVICE(T, dst.n);
    dst.cp  = CUDA_ALLOC_DEVICE(char, dst.n);

    CUDA_MEMCPY(dst.P, src.P, dst.n * sizeof(T), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(dst.S, src.S, dst.n * sizeof(T), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(dst.K, src.K, dst.n * sizeof(T), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(dst.ttm, src.ttm, dst.n * sizeof(T), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(dst.iv, src.iv, dst.n * sizeof(T), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(dst.r, src.r, dst.n * sizeof(T), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(dst.q, src.q, dst.n * sizeof(T), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(dst.cp, src.cp, dst.n * sizeof(char), cudaMemcpyHostToDevice);

    return dst;
}

template <typename T> void free_pinned_data(TestData<T> &data)
{
    CUDA_FREE_HOST(data.P);
    CUDA_FREE_HOST(data.S);
    CUDA_FREE_HOST(data.K);
    CUDA_FREE_HOST(data.ttm);
    CUDA_FREE_HOST(data.iv);
    CUDA_FREE_HOST(data.r);
    CUDA_FREE_HOST(data.q);
    CUDA_FREE_HOST(data.cp);
}

template <typename T> void free_device_data(TestData<T> &data)
{
    CUDA_FREE(data.P);
    CUDA_FREE(data.S);
    CUDA_FREE(data.K);
    CUDA_FREE(data.ttm);
    CUDA_FREE(data.iv);
    CUDA_FREE(data.r);
    CUDA_FREE(data.q);
    CUDA_FREE(data.cp);
}
#endif

/* timing ----------------------------------------------------------------------------------------*/
inline double tns(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
}

inline void tstmp(struct timespec *wall_time, struct timespec *cpu_time)
{
    clock_gettime(CLOCK_MONOTONIC, wall_time);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, cpu_time);
}

#ifdef FASTVOL_CUDA_ENABLED
inline void tstart_cuda(cudaEvent_t *start, cudaEvent_t *stop)
{
    CUDA_CHECK(cudaEventCreate(start));
    CUDA_CHECK(cudaEventCreate(stop));
    cudaEventRecord(*start, 0);
}

inline double tstop_cuda(cudaEvent_t *start, cudaEvent_t *stop)
{
    CUDA_CHECK(cudaEventRecord(*stop, 0));
    CUDA_CHECK(cudaEventSynchronize(*stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, *start, *stop));

    CUDA_CHECK(cudaEventDestroy(*start));
    CUDA_CHECK(cudaEventDestroy(*stop));

    return (double)ms * 1e6;
}
#endif

/* formatting ------------------------------------------------------------------------------------*/
inline constexpr int TW  = 90;     // table width
inline constexpr int LW  = 45;     // label width
inline constexpr int CW  = 15;     // column width
inline constexpr int LCW = LW - 2; // label content width
inline constexpr int CCW = CW - 3; // column content width

inline std::string tformat(double ns)
{
    const char *unit;
    double      val;

    if (ns < 1.0)
    {
        val  = ns * 1e3;
        unit = "ps";
    }
    else if (ns < 1e3)
    {
        val  = ns;
        unit = "ns";
    }
    else if (ns < 1e6)
    {
        val  = ns / 1e3;
        unit = "us";
    }
    else if (ns < 1e9)
    {
        val  = ns / 1e6;
        unit = "ms";
    }
    else
    {
        val  = ns / 1e9;
        unit = "s ";
    }

    char buf[32];
    std::snprintf(buf, sizeof(buf), "%6.1f %s", val, unit);
    return std::string(buf);
}

inline std::string kformat(int k)
{
    char buf[32];
    if (k < 1000)
    {
        std::snprintf(buf, sizeof(buf), "%7d", k);
    }
    else if (k < 1000000)
    {
        double val = k / 1e3;
        std::snprintf(buf, sizeof(buf), "%5.1f k", val);
    }
    else
    {
        double val = k / 1e6;
        std::snprintf(buf, sizeof(buf), "%5.1f M", val);
    }
    return std::string(buf);
}

inline void print_section(const char *name, char pad_char, bool pad_above)
{
    std::printf("\n");
    if (pad_above)
    {
        for (int i = 0; i < TW; i++)
            std::putchar(pad_char);
        std::printf("\n");
    }

    int padding = TW - std::strlen(name);
    if (padding < 0) padding = 0;

    std::printf("%s", name);
    for (int i = 0; i < padding; i++)
        std::putchar(pad_char);

    std::printf("\n");
}

static void print_delimiter()
{
    std::printf("%-*s+", LW - 1, "");

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < CW - 1; ++j)
            std::putchar('-');
        std::putchar(i < 2 ? '+' : '|');
    }
    std::putchar('\n');
}

inline void print_header_cpu()
{
    std::printf("%-*s   %*s | %*s | %*s |\n",
                LCW,
                "",
                CCW,
                "scalar",
                CCW,
                "batch (cpu)",
                CCW,
                "batch (wall)");
    print_delimiter();
}

inline void print_bench_cpu(const char *label, double ns, double batch_cpu_ns, double batch_wall_ns)
{
    std::string sbuf  = tformat(ns);
    std::string bcpu  = tformat(batch_cpu_ns);
    std::string bwall = tformat(batch_wall_ns);
    std::printf("%-*s | %*s | %*s | %*s |\n",
                LCW,
                label,
                CCW,
                sbuf.c_str(),
                CCW,
                bcpu.c_str(),
                CCW,
                bwall.c_str());
}

#ifdef FASTVOL_CUDA_ENABLED
inline void print_header_cuda()
{
    std::printf("%-*s   %*s | %*s | %*s |\n", LCW, "", CCW, "host", CCW, "pinned", CCW, "device");
    print_delimiter();
}

inline void print_bench_cuda(const char *label, double host_ns, double pinned_ns, double device_ns)
{
    std::string host   = tformat(host_ns);
    std::string pinned = tformat(pinned_ns);
    std::string device = tformat(device_ns);
    std::printf("%-*s | %*s | %*s | %*s |\n",
                LCW,
                label,
                CCW,
                host.c_str(),
                CCW,
                pinned.c_str(),
                CCW,
                device.c_str());
}
#endif

inline void print_hardware_info()
{
    // CPU info
    struct utsname sys_info;
    if (uname(&sys_info) == 0)
    {
        std::printf("[host] OS   : %s %s\n", sys_info.sysname, sys_info.release);
        std::printf("[host] Arch : %s\n", sys_info.machine);
    }

    std::FILE *cpuinfo = std::fopen("/proc/cpuinfo", "r");
    if (cpuinfo)
    {
        char line[512];
        while (std::fgets(line, sizeof(line), cpuinfo))
        {
            if (std::strncmp(line, "model name", 10) == 0)
            {
                char *model = std::strchr(line, ':');
                if (model)
                {
                    model += 2; // skip ": "
                    std::printf("[host] CPU : %s", model);
                }
                break;
            }
        }
        std::fclose(cpuinfo);
    }

#ifdef FASTVOL_CUDA_ENABLED
    // GPU info
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; ++i)
    {
        struct cudaDeviceProp prop;
        int                   sharedMemPerSM, maxThreadsPerSM, warpSize, maxWarpsPerSM;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        CUDA_CHECK(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, i));
        CUDA_CHECK(
            cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, i));
        CUDA_CHECK(cudaDeviceGetAttribute(
            &sharedMemPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, i));
        maxWarpsPerSM = maxThreadsPerSM / warpSize;

        std::printf("[gpu%d] Name            : %s\n", i, prop.name);
        std::printf(
            "[gpu%d] Memory          : %.1f GB\n", i, prop.totalGlobalMem / (1024.0 * 1024 * 1024));
        std::printf("[gpu%d] SM [count ]     : %d\n", i, prop.multiProcessorCount);
        std::printf("[gpu%d] SM [shared_mem] : %d KB\n", i, sharedMemPerSM / 1024);
        std::printf("[gpu%d] SM [n_threads ] : %d\n", i, maxThreadsPerSM);
        std::printf("[gpu%d] SM [n_warps ]   : %d\n", i, maxWarpsPerSM);
    }
#endif
}

/* benchmarking ----------------------------------------------------------------------------------*/
inline constexpr size_t MAX_OMP_N_MULTIPLIER = 32;

inline size_t get_batch_multiplier()
{
    return std::min(static_cast<size_t>(omp_get_max_threads()), MAX_OMP_N_MULTIPLIER);
}

inline size_t get_batch_k(size_t scalar_k, size_t n)
{
    return std::min(n, scalar_k * get_batch_multiplier());
}

template <typename T, typename ScalarFunc, typename BatchFunc>
void benchmark_cpu(
    const char *label, TestData<T> &data, size_t k, ScalarFunc scalar_fn, BatchFunc batch_fn)
{
    struct timespec wall_s, wall_e, cpu_s, cpu_e;
    double          scalar_ns, batch_cpu_ns, batch_wall_ns;
    size_t          scalar_k = data.n < k ? data.n : k;
    size_t          batch_k  = get_batch_k(scalar_k, data.n);
    volatile T      sink;

    tstmp(&wall_s, &cpu_s);
    for (size_t i = 0; i < scalar_k; i++)
        sink = scalar_fn(i);
    tstmp(&wall_e, &cpu_e);
    scalar_ns = tns(cpu_s, cpu_e) / scalar_k;
    (void)sink;

    tstmp(&wall_s, &cpu_s);
    batch_fn(batch_k);
    tstmp(&wall_e, &cpu_e);
    batch_wall_ns = tns(wall_s, wall_e) / batch_k;
    batch_cpu_ns  = tns(cpu_s, cpu_e) / batch_k;

    print_bench_cpu(label, scalar_ns, batch_cpu_ns, batch_wall_ns);
}

#ifdef FASTVOL_CUDA_ENABLED
template <typename T, typename Func>
void benchmark_cuda(const char  *label,
                    TestData<T> &host,
                    TestData<T> &pinned,
                    TestData<T> &device,
                    size_t       n,
                    Func         kernel_launcher)
{
    cudaEvent_t start, stop;
    size_t      k = host.n < n ? host.n : n;
    double      host_ns, pinned_ns, device_ns;

    // host
    tstart_cuda(&start, &stop);
    kernel_launcher(host.S,
                    host.K,
                    host.cp,
                    host.ttm,
                    host.iv,
                    host.r,
                    host.q,
                    k,
                    host.P,
                    0,
                    0,
                    false,
                    false,
                    true);
    host_ns = tstop_cuda(&start, &stop) / n;

    // pinned
    tstart_cuda(&start, &stop);
    kernel_launcher(pinned.S,
                    pinned.K,
                    pinned.cp,
                    pinned.ttm,
                    pinned.iv,
                    pinned.r,
                    pinned.q,
                    k,
                    pinned.P,
                    0,
                    0,
                    false,
                    true,
                    true);
    pinned_ns = tstop_cuda(&start, &stop) / n;

    // device
    tstart_cuda(&start, &stop);
    kernel_launcher(device.S,
                    device.K,
                    device.cp,
                    device.ttm,
                    device.iv,
                    device.r,
                    device.q,
                    k,
                    device.P,
                    0,
                    0,
                    true,
                    false,
                    true);
    device_ns = tstop_cuda(&start, &stop) / n;

    // print
    print_bench_cuda(label, host_ns, pinned_ns, device_ns);
}
#endif

} // namespace fastvol::detail::bench

/* european::bsm =================================================================================*/
namespace fastvol::detail::bench::european::bsm
{
void price_cpu();
void price_cuda();
void greeks_cpu();
void iv_cpu();
void cpu();
void cuda();
void all();
} // namespace fastvol::detail::bench::european::bsm

/* american::bopm ================================================================================*/
namespace fastvol::detail::bench::american::bopm
{
void price_cpu();
void price_cuda();
void greeks_cpu();
void iv_cpu();
void cpu();
void cuda();
void all();
} // namespace fastvol::detail::bench::american::bopm

/* american::ttree ===============================================================================*/
namespace fastvol::detail::bench::american::ttree
{
void price_cpu();
void price_cuda();
void greeks_cpu();
void iv_cpu();
void cpu();
void cuda();
void all();
} // namespace fastvol::detail::bench::american::ttree

/* american::psor ================================================================================*/
namespace fastvol::detail::bench::american::psor
{
void price_cpu();
void price_cuda();
void greeks_cpu();
void iv_cpu();
void cpu();
void cuda();
void all();
} // namespace fastvol::detail::bench::american::psor