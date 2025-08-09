/**
 * @file math.hpp
 * @brief Mathematical constants and functions for the fastvol library.
 *
 * This file provides a collection of mathematical utilities, including constants and
 * functions for standard mathematical operations, as well as normal distribution
 * calculations (CDF and PDF). It also includes a `cuda` namespace for device-specific
 * implementations when compiled with CUDA.
 */

#pragma once
#include <cmath>

namespace fastvol::detail::math
{

// math funcs -- SLEEF support coming
template <typename T> inline T exp(T x) { return std::exp(x); }
template <typename T> inline T log(T x) { return std::log(x); }
template <typename T> inline T sqrt(T x) { return std::sqrt(x); }
template <typename T> inline T erf(T x) { return std::erf(x); }
template <typename T> inline T cos(T x) { return std::cos(x); }
template <typename T> inline T fmax(T x, T y) { return std::fmax(x, y); }
template <typename T> inline T fmin(T x, T y) { return std::fmin(x, y); }
template <typename T> inline T fabs(T x) { return std::fabs(x); }
template <typename T> inline T fma(T x, T y, T z) { return std::fma(x, y, z); }
template <typename T> inline T copysign(T x, T y) { return std::copysign(x, y); }

// constants
template <typename T> inline constexpr T inv_sqrt2 = T(0.70710678118654752440);
template <typename T> inline constexpr T m_pi      = T(M_PI);

// normal dist
template <typename T> inline T norm_cdf(T x) { return T(0.5) * (T(1.0) + erf(x * inv_sqrt2<T>)); }
template <typename T> inline T norm_pdf(T x)
{
    return (T(1.0) / sqrt(T(2.0) * m_pi<T>)) * exp(-T(0.5) * x * x);
}

namespace cuda
{
#ifdef __CUDACC__
// math funcs
template <typename T> __device__ T exp(T x) { return ::exp(x); }
template <typename T> __device__ T log(T x) { return ::log(x); }
template <typename T> __device__ T sqrt(T x) { return ::sqrt(x); }
template <typename T> __device__ T erf(T x) { return ::erf(x); }
template <typename T> __device__ T cos(T x) { return ::cos(x); }
template <typename T> __device__ T fmax(T x, T y) { return ::fmax(x, y); }
template <typename T> __device__ T fmin(T x, T y) { return ::fmin(x, y); }
template <typename T> __device__ T fabs(T x) { return ::fabs(x); }
template <typename T> __device__ T fma(T x, T y, T z) { return ::fma(x, y, z); }
template <typename T> __device__ T copysign(T x, T y) { return ::copysign(x, y); }

// constants
template <typename T> __device__ constexpr T inv_sqrt2 = T(0.70710678118654752440);
template <typename T> __device__ constexpr T m_pi      = T(M_PI);

// normal dist
template <typename T> __device__ T norm_cdf(T x)
{
    return T(0.5) * (T(1.0) + erf(x * inv_sqrt2<T>));
}
template <typename T> __device__ T norm_pdf(T x)
{
    return (T(1.0) / sqrt(T(2.0) * m_pi<T>)) * exp(-T(0.5) * x * x);
}
#endif
} // namespace cuda

} // namespace fastvol::detail::math