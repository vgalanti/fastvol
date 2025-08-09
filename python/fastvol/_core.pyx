# distutils: language = c
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

from ._ffi cimport *

from libc.stdint cimport uint8_t, uintptr_t
from libc.stddef cimport size_t

# american::bopm ===================================================================================
# fp64 ---------------------------------------------------------------------------------------------
# price ____________________________________________________________________________________________
def c_american_bopm_price_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q, int n_steps) -> float:
    return american_bopm_price_fp64(S, K, cp_flag, ttm, iv, r, q, n_steps)

def c_american_bopm_price_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out,
    bool         cuda,
    uintptr_t    stream,
    int          device,
    bool         on_device,
    bool         is_pinned,
    bool         sync
) -> None:

    if FASTVOL_CUDA_ENABLED and cuda:
        with nogil:
            american_bopm_price_fp64_cuda(
                <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
                <double*>r, <double*>q, n_steps, n_options, <double*>out, 
                <cudaStream_t>stream, device, on_device, is_pinned, sync
            )
    else:
        with nogil:
            american_bopm_price_fp64_batch(
                <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
                <double*>r, <double*>q, n_steps, n_options, <double*>out 
            )

# greeks ___________________________________________________________________________________________
def c_american_bopm_greeks_fp64(
    double S, double K, char cp_flag, double ttm, double iv, double r, double q, 
    int n_steps, bool delta, bool gamma, bool theta, bool vega, bool rho) -> float | tuple:

    cdef double d, g, t, v, rh

    american_bopm_greeks_fp64(
        S, K, cp_flag, ttm, iv, r, q, n_steps,
        &d if delta else NULL,
        &g if gamma else NULL,
        &t if theta else NULL,
        &v if vega else NULL,
        &rh if rho else NULL
    )

    res = []
    if delta: 
        res.append(d)
    if gamma: 
        res.append(g)
    if theta: 
        res.append(t)
    if vega: 
        res.append(v)
    if rho: 
        res.append(rh)

    return res[0] if len(res) == 1 else tuple(res)

def c_american_bopm_delta_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q, int n_steps) -> float:
    return american_bopm_delta_fp64(S, K, cp_flag, ttm, iv, r, q, n_steps)

def c_american_bopm_gamma_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q, int n_steps) -> float:
    return american_bopm_gamma_fp64(S, K, cp_flag, ttm, iv, r, q, n_steps)

def c_american_bopm_theta_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q, int n_steps) -> float:
    return american_bopm_theta_fp64(S, K, cp_flag, ttm, iv, r, q, n_steps)

def c_american_bopm_vega_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q, int n_steps) -> float:
    return american_bopm_vega_fp64(S, K, cp_flag, ttm, iv, r, q, n_steps)

def c_american_bopm_rho_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q, int n_steps) -> float:
    return american_bopm_rho_fp64(S, K, cp_flag, ttm, iv, r, q, n_steps)

def c_american_bopm_greeks_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    delta,
    uintptr_t    gamma,
    uintptr_t    theta,
    uintptr_t    vega,
    uintptr_t    rho
    ) -> None:
    with nogil:
        american_bopm_greeks_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_steps, n_options, 
            <double*>delta, <double*>gamma, <double*>theta, <double*>vega, <double*>rho
        )

def c_american_bopm_delta_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_bopm_delta_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_steps, n_options, <double*> out
        )

def c_american_bopm_gamma_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_bopm_gamma_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_steps, n_options, <double*> out
        )

def c_american_bopm_theta_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_bopm_theta_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_steps, n_options, <double*> out
        )

def c_american_bopm_vega_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_bopm_vega_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_steps, n_options, <double*> out
        )

def c_american_bopm_rho_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_bopm_rho_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_steps, n_options, <double*> out
        )

# iv _______________________________________________________________________________________________
def c_american_bopm_iv_fp64(
    double P, double S, double K, char cp_flag, double ttm, double iv, double r, double q, 
    int n_steps, double tol, size_t max_iter, uintptr_t lo_init, uintptr_t hi_init
    ) -> float:

    american_bopm_iv_fp64(
        P, S, K, cp_flag, ttm, r, q, n_steps, 
        tol, max_iter, <double*> lo_init, <double*> hi_init
    )

def c_american_bopm_iv_fp64_batch(
    uintptr_t    P,
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out,
    double       tol,
    size_t       max_iter,
    uintptr_t    lo_init,
    uintptr_t    hi_init
    ) -> None:

    with nogil:
        american_bopm_iv_fp64_batch(
            <double*>P, <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, 
            <double*>r, <double*>q, n_steps, n_options, <double*> out, 
            tol, max_iter, <double*>lo_init, <double*>hi_init
        )

# fp32 ---------------------------------------------------------------------------------------------
# price ____________________________________________________________________________________________
def c_american_bopm_price_fp32(float S, float K, bool cp_flag, float ttm, 
                                float iv, float r, float q, int n_steps) -> float:
    return american_bopm_price_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, n_steps
    )

def c_american_bopm_price_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out,
    bool         cuda,
    uintptr_t    stream,
    int          device,
    bool         on_device,
    bool         is_pinned,
    bool         sync
) -> None:

    if FASTVOL_CUDA_ENABLED and cuda:
        with nogil:
            american_bopm_price_fp32_cuda(
                <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
                <float*>r, <float*>q, n_steps, n_options, <float*>out, 
                <cudaStream_t>stream, device, on_device, is_pinned, sync
            )
    else:
        with nogil:
            american_bopm_price_fp32_batch(
                <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
                <float*>r, <float*>q, n_steps, n_options, <float*>out 
            )

# greeks ___________________________________________________________________________________________
def c_american_bopm_greeks_fp32(
    float S, float K, char cp_flag, float ttm, float iv, float r, float q, 
    int n_steps, bool delta, bool gamma, bool theta, bool vega, bool rho) -> float | tuple:

    cdef float d, g, t, v, rh

    american_bopm_greeks_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, n_steps,
        &d if delta else NULL,
        &g if gamma else NULL,
        &t if theta else NULL,
        &v if vega else NULL,
        &rh if rho else NULL
    )

    res = []
    if delta: 
        res.append(d)
    if gamma: 
        res.append(g)
    if theta: 
        res.append(t)
    if vega: 
        res.append(v)
    if rho: 
        res.append(rh)

    return res[0] if len(res) == 1 else tuple(res)

def c_american_bopm_delta_fp32(float S, float K, char cp_flag, float ttm, 
                                float iv, float r, float q, int n_steps) -> float:
    return american_bopm_delta_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, n_steps)

def c_american_bopm_gamma_fp32(float S, float K, char cp_flag, float ttm, 
                                float iv, float r, float q, int n_steps) -> float:
    return american_bopm_gamma_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, n_steps)

def c_american_bopm_theta_fp32(float S, float K, char cp_flag, float ttm, 
                                float iv, float r, float q, int n_steps) -> float:
    return american_bopm_theta_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, n_steps)

def c_american_bopm_vega_fp32(float S, float K, char cp_flag, float ttm, 
                                float iv, float r, float q, int n_steps) -> float:
    return american_bopm_vega_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, n_steps)

def c_american_bopm_rho_fp32(float S, float K, char cp_flag, float ttm, 
                                float iv, float r, float q, int n_steps) -> float:
    return american_bopm_rho_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, n_steps)

def c_american_bopm_greeks_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    delta,
    uintptr_t    gamma,
    uintptr_t    theta,
    uintptr_t    vega,
    uintptr_t    rho
    ) -> None:
    with nogil:
        american_bopm_greeks_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_steps, n_options, 
            <float*>delta, <float*>gamma, <float*>theta, <float*>vega, <float*>rho
        )

def c_american_bopm_delta_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_bopm_delta_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_steps, n_options, <float*> out
        )

def c_american_bopm_gamma_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_bopm_gamma_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_steps, n_options, <float*> out
        )

def c_american_bopm_theta_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_bopm_theta_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_steps, n_options, <float*> out
        )

def c_american_bopm_vega_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_bopm_vega_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_steps, n_options, <float*> out
        )

def c_american_bopm_rho_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_bopm_rho_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_steps, n_options, <float*> out
        )

# iv _______________________________________________________________________________________________
def c_american_bopm_iv_fp32(
    float P, float S, float K, char cp_flag, float ttm, float iv, float r, float q, 
    int n_steps, float tol, size_t max_iter, uintptr_t lo_init, uintptr_t hi_init
    ) -> float:

    return american_bopm_iv_fp32(
        <float>P, <float>S, <float>K, cp_flag, <float>ttm, <float>r, <float>q, n_steps, 
        <float>tol, max_iter, <float*>lo_init, <float*>hi_init
    )

def c_american_bopm_iv_fp32_batch(
    uintptr_t    P,
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out,
    float        tol,
    size_t       max_iter,
    uintptr_t    lo_init,
    uintptr_t    hi_init
    ) -> None:

    with nogil:
        american_bopm_iv_fp32_batch(
            <float*>P, <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, 
            <float*>r, <float*>q, n_steps, n_options, <float*>out, 
            <float>tol, max_iter, <float*>lo_init, <float*>hi_init
        )

# american::psor ===================================================================================
# fp64 ---------------------------------------------------------------------------------------------
# price ____________________________________________________________________________________________
def c_american_psor_price_fp64(
    S: float, K: float, cp_flag: bool, ttm: float, iv: float, r: float, q: float, 
    n_s: int, n_t: int, k_mul: int, w: float, tol: float, max_iter: int) -> float:
    
    return american_psor_price_fp64(
        S, K, cp_flag, ttm, iv, r, q, 
        n_s, n_t, k_mul, w, tol, max_iter, NULL, NULL, NULL
    )

def c_american_psor_price_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    float        w, 
    float        tol, 
    int          max_iter,
    size_t       n_options,
    uintptr_t    out,
    bool         cuda,
    uintptr_t    stream,
    int          device,
    bool         on_device,
    bool         is_pinned,
    bool         sync
    ) -> None:

    if FASTVOL_CUDA_ENABLED and cuda:
        with nogil:
            american_psor_price_fp64_cuda(
                <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
                <double*>r, <double*>q, n_s, n_t, k_mul, w, tol, max_iter, n_options, 
                <double*>out, NULL, NULL, NULL, 
                <cudaStream_t>stream, device, on_device, is_pinned, sync
            )
    else:
        with nogil:
            american_psor_price_fp64_batch(
                <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
                <double*>r, <double*>q, n_s, n_t, k_mul, w, tol, max_iter, n_options, 
                <double*>out, NULL, NULL, NULL
            )

# greeks ___________________________________________________________________________________________
def c_american_psor_greeks_fp64(
    double S, double K, char cp_flag, double ttm, double iv, double r, double q,
    int n_s, int n_t, int k_mul, double w, double tol, int max_iter,
    bool delta, bool gamma, bool theta, bool vega, bool rho
    ) -> float | tuple:

    cdef double d, g, t, v, rh

    american_psor_greeks_fp64(
        S, K, cp_flag, ttm, iv, r, q,
        n_s, n_t, k_mul, w, tol, max_iter,
        &d if delta else NULL,
        &g if gamma else NULL,
        &t if theta else NULL,
        &v if vega else NULL,
        &rh if rho else NULL
    )

    res = []
    if delta: 
        res.append(d)
    if gamma: 
        res.append(g)
    if theta: 
        res.append(t)
    if vega:  
        res.append(v)
    if rho:   
        res.append(rh)

    return res[0] if len(res) == 1 else tuple(res)

def c_american_psor_delta_fp64(double S, double K, char cp_flag, double ttm,
                                double iv, double r, double q, int n_s, int n_t,
                                int k_mul, double w, double tol, int max_iter) -> float:
    return american_psor_delta_fp64(S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter)

def c_american_psor_gamma_fp64(double S, double K, char cp_flag, double ttm,
                                double iv, double r, double q, int n_s, int n_t,
                                int k_mul, double w, double tol, int max_iter) -> float:
    return american_psor_gamma_fp64(S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter)

def c_american_psor_theta_fp64(double S, double K, char cp_flag, double ttm,
                                double iv, double r, double q, int n_s, int n_t,
                                int k_mul, double w, double tol, int max_iter) -> float:
    return american_psor_theta_fp64(S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter)

def c_american_psor_vega_fp64(double S, double K, char cp_flag, double ttm,
                               double iv, double r, double q, int n_s, int n_t,
                               int k_mul, double w, double tol, int max_iter) -> float:
    return american_psor_vega_fp64(S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter)

def c_american_psor_rho_fp64(double S, double K, char cp_flag, double ttm,
                              double iv, double r, double q, int n_s, int n_t,
                              int k_mul, double w, double tol, int max_iter) -> float:
    return american_psor_rho_fp64(S, K, cp_flag, ttm, iv, r, q, n_s, n_t, k_mul, w, tol, max_iter)

def c_american_psor_greeks_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    double       w, 
    double       tol, 
    int          max_iter,
    size_t       n_options,
    uintptr_t    delta,
    uintptr_t    gamma,
    uintptr_t    theta,
    uintptr_t    vega,
    uintptr_t    rho
    ) -> None:

    with nogil:
        american_psor_greeks_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv,
            <double*>r, <double*>q,
            n_s, n_t, k_mul, w, tol, max_iter, n_options,
            <double*>delta, <double*>gamma, <double*>theta, <double*>vega, <double*>rho
        )

def c_american_psor_delta_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    double       w, 
    double       tol, 
    int          max_iter,
    size_t       n_options,
    uintptr_t    out
    ) -> None:

    with nogil:
        american_psor_delta_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv,
            <double*>r, <double*>q,
            n_s, n_t, k_mul, w, tol, max_iter, n_options, <double*>out
        )

def c_american_psor_gamma_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    double       w, 
    double       tol, 
    int          max_iter,
    size_t       n_options,
    uintptr_t    out
    ) -> None:

    with nogil:
        american_psor_gamma_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv,
            <double*>r, <double*>q,
            n_s, n_t, k_mul, w, tol, max_iter, n_options, <double*>out
        )

def c_american_psor_theta_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    double       w, 
    double       tol, 
    int          max_iter,
    size_t       n_options,
    uintptr_t    out
    ) -> None:

    with nogil:
        american_psor_theta_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv,
            <double*>r, <double*>q,
            n_s, n_t, k_mul, w, tol, max_iter, n_options, <double*>out
        )

def c_american_psor_vega_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    double       w, 
    double       tol, 
    int          max_iter,
    size_t       n_options,
    uintptr_t    out
    ) -> None:

    with nogil:
        american_psor_vega_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv,
            <double*>r, <double*>q,
            n_s, n_t, k_mul, w, tol, max_iter, n_options, <double*>out
        )

def c_american_psor_rho_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    double       w, 
    double       tol, 
    int          max_iter,
    size_t       n_options,
    uintptr_t    out
    ) -> None:

    with nogil:
        american_psor_rho_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv,
            <double*>r, <double*>q,
            n_s, n_t, k_mul, w, tol, max_iter, n_options, <double*>out
        )

# iv _______________________________________________________________________________________________
def c_american_psor_iv_fp64(
    double       P,
    double       S,
    double       K,
    bool         cp_flag,
    double       ttm,
    double       r,
    double       q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    double       w, 
    double       psor_tol, 
    int          psor_max_iter,
    double       tol,
    size_t       max_iter,
    uintptr_t    lo_init,
    uintptr_t    hi_init
    ) -> float:

    return american_psor_iv_fp64(
        P, S, K, cp_flag, ttm, r, q,
        n_s, n_t, k_mul, w, psor_tol, psor_max_iter,
        tol, max_iter, <double*>lo_init, <double*>hi_init
    )

def c_american_psor_iv_fp64_batch(
    uintptr_t    P,
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    double       w, 
    double       psor_tol, 
    int          psor_max_iter,
    size_t       n_options,
    uintptr_t    out,
    double       tol,
    size_t       max_iter,
    uintptr_t    lo_init,
    uintptr_t    hi_init
    ) -> None:

    with nogil:
        american_psor_iv_fp64_batch(
            <double*>P, <double*>S, <double*>K, <char*>cp_flag,
            <double*>ttm, <double*>r, <double*>q,
            n_s, n_t, k_mul, w, psor_tol, psor_max_iter,
            n_options, <double*>out, tol, max_iter,
            <double*>lo_init, <double*>hi_init
        )

# fp32 ---------------------------------------------------------------------------------------------
# price ____________________________________________________________________________________________
def c_american_psor_price_fp32(
    S: float, K: float, cp_flag: bool, ttm: float, iv: float, r: float, q: float, 
    n_s: int, n_t: int, k_mul: int, w: float, tol: float, max_iter: int) -> float:
    
    return american_psor_price_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, 
        n_s, n_t, k_mul, <float>w, <float>tol, max_iter, NULL, NULL, NULL
    )

def c_american_psor_price_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    float        w, 
    float        tol, 
    int          max_iter,
    size_t       n_options,
    uintptr_t    out,
    bool         cuda,
    uintptr_t    stream,
    int          device,
    bool         on_device,
    bool         is_pinned,
    bool         sync
    ) -> None:

    if FASTVOL_CUDA_ENABLED and cuda:
        with nogil:
            american_psor_price_fp32_cuda(
                <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
                <float*>r, <float*>q, n_s, n_t, k_mul, w, tol, max_iter, n_options, 
                <float*>out, NULL, NULL, NULL, 
                <cudaStream_t>stream, device, on_device, is_pinned, sync
            )
    else:
        with nogil:
            american_psor_price_fp32_batch(
                <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
                <float*>r, <float*>q, n_s, n_t, k_mul, w, tol, max_iter, n_options, 
                <float*>out, NULL, NULL, NULL
            )

# greeks ___________________________________________________________________________________________
def c_american_psor_greeks_fp32(
    float S, float K, char cp_flag, float ttm, float iv, float r, float q,
    int n_s, int n_t, int k_mul, float w, float tol, int max_iter,
    bool delta, bool gamma, bool theta, bool vega, bool rho
    ) -> float | tuple:

    cdef float d, g, t, v, rh

    american_psor_greeks_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q,
        n_s, n_t, k_mul, <float>w, <float>tol, max_iter,
        &d if delta else NULL,
        &g if gamma else NULL,
        &t if theta else NULL,
        &v if vega else NULL,
        &rh if rho else NULL
    )

    res = []
    if delta: 
        res.append(d)
    if gamma: 
        res.append(g)
    if theta: 
        res.append(t)
    if vega:  
        res.append(v)
    if rho:   
        res.append(rh)

    return res[0] if len(res) == 1 else tuple(res)

def c_american_psor_delta_fp32(float S, float K, char cp_flag, float ttm,
                                float iv, float r, float q, int n_s, int n_t,
                                int k_mul, float w, float tol, int max_iter) -> float:
    return american_psor_delta_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, 
        n_s, n_t, k_mul, <float>w, <float>tol, max_iter)

def c_american_psor_gamma_fp32(float S, float K, char cp_flag, float ttm,
                                float iv, float r, float q, int n_s, int n_t,
                                int k_mul, float w, float tol, int max_iter) -> float:
    return american_psor_gamma_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, 
        n_s, n_t, k_mul, <float>w, <float>tol, max_iter)

def c_american_psor_theta_fp32(float S, float K, char cp_flag, float ttm,
                                float iv, float r, float q, int n_s, int n_t,
                                int k_mul, float w, float tol, int max_iter) -> float:
    return american_psor_theta_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, 
        n_s, n_t, k_mul, <float>w, <float>tol, max_iter)

def c_american_psor_vega_fp32(float S, float K, char cp_flag, float ttm,
                               float iv, float r, float q, int n_s, int n_t,
                               int k_mul, float w, float tol, int max_iter) -> float:
    return american_psor_vega_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, 
        n_s, n_t, k_mul, <float>w, <float>tol, max_iter)

def c_american_psor_rho_fp32(float S, float K, char cp_flag, float ttm,
                              float iv, float r, float q, int n_s, int n_t,
                              int k_mul, float w, float tol, int max_iter) -> float:
    return american_psor_rho_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, 
        n_s, n_t, k_mul, <float>w, <float>tol, max_iter)

def c_american_psor_greeks_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    float        w, 
    float        tol, 
    int          max_iter,
    size_t       n_options,
    uintptr_t    delta,
    uintptr_t    gamma,
    uintptr_t    theta,
    uintptr_t    vega,
    uintptr_t    rho
    ) -> None:

    with nogil:
        american_psor_greeks_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv,
            <float*>r, <float*>q,
            n_s, n_t, k_mul, <float>w, <float>tol, max_iter, n_options,
            <float*>delta, <float*>gamma, <float*>theta, <float*>vega, <float*>rho
        )

def c_american_psor_delta_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    float        w, 
    float        tol, 
    int          max_iter,
    size_t       n_options,
    uintptr_t    out
    ) -> None:

    with nogil:
        american_psor_delta_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv,
            <float*>r, <float*>q,
            n_s, n_t, k_mul, <float>w, <float>tol, max_iter, n_options, <float*>out
        )

def c_american_psor_gamma_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    float        w, 
    float        tol, 
    int          max_iter,
    size_t       n_options,
    uintptr_t    out
    ) -> None:

    with nogil:
        american_psor_gamma_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv,
            <float*>r, <float*>q,
            n_s, n_t, k_mul, <float>w, <float>tol, max_iter, n_options, <float*>out
        )

def c_american_psor_theta_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    float        w, 
    float        tol, 
    int          max_iter,
    size_t       n_options,
    uintptr_t    out
    ) -> None:

    with nogil:
        american_psor_theta_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv,
            <float*>r, <float*>q,
            n_s, n_t, k_mul, <float>w, <float>tol, max_iter, n_options, <float*>out
        )

def c_american_psor_vega_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    float        w, 
    float        tol, 
    int          max_iter,
    size_t       n_options,
    uintptr_t    out
    ) -> None:

    with nogil:
        american_psor_vega_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv,
            <float*>r, <float*>q,
            n_s, n_t, k_mul, <float>w, <float>tol, max_iter, n_options, <float*>out
        )

def c_american_psor_rho_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    float        w, 
    float        tol, 
    int          max_iter,
    size_t       n_options,
    uintptr_t    out
    ) -> None:

    with nogil:
        american_psor_rho_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv,
            <float*>r, <float*>q,
            n_s, n_t, k_mul, <float>w, <float>tol, max_iter, n_options, <float*>out
        )



# iv _______________________________________________________________________________________________
def c_american_psor_iv_fp32(
    float        P,
    float        S,
    float        K,
    bool         cp_flag,
    float        ttm,
    float        r,
    float        q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    float        w, 
    float        psor_tol, 
    int          psor_max_iter,
    float        tol,
    size_t       max_iter,
    uintptr_t    lo_init,
    uintptr_t    hi_init
    ) -> float:

    return american_psor_iv_fp32(
        <float>P, <float>S, <float>K, cp_flag, <float>ttm, <float>r, <float>q,
        n_s, n_t, k_mul, <float>w, <float>psor_tol, psor_max_iter,
        <float>tol, max_iter, <float*>lo_init, <float*>hi_init
    )

def c_american_psor_iv_fp32_batch(
    uintptr_t    P,
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    r,
    uintptr_t    q,
    int          n_s, 
    int          n_t, 
    int          k_mul, 
    float        w, 
    float        psor_tol, 
    int          psor_max_iter,
    size_t       n_options,
    uintptr_t    out,
    float        tol,
    size_t       max_iter,
    uintptr_t    lo_init,
    uintptr_t    hi_init
    ) -> None:

    with nogil:
        american_psor_iv_fp32_batch(
            <float*>P, <float*>S, <float*>K, <char*>cp_flag,
            <float*>ttm, <float*>r, <float*>q,
            n_s, n_t, k_mul, <float>w, <float>psor_tol, psor_max_iter,
            n_options, <float*>out, <float>tol, max_iter,
            <float*>lo_init, <float*>hi_init
        )


# american::ttree ==================================================================================
# fp64 ---------------------------------------------------------------------------------------------
# price ____________________________________________________________________________________________
def c_american_ttree_price_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q, int n_steps) -> float:
    return american_ttree_price_fp64(S, K, cp_flag, ttm, iv, r, q, n_steps)

def c_american_ttree_price_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out,
    bool         cuda,
    uintptr_t    stream,
    int          device,
    bool         on_device,
    bool         is_pinned,
    bool         sync
) -> None:

    if FASTVOL_CUDA_ENABLED and cuda:
        with nogil:
            american_ttree_price_fp64_cuda(
                <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
                <double*>r, <double*>q, n_steps, n_options, <double*>out, 
                <cudaStream_t>stream, device, on_device, is_pinned, sync
            )
    else:
        with nogil:
            american_ttree_price_fp64_batch(
                <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
                <double*>r, <double*>q, n_steps, n_options, <double*>out 
            )

# greeks ___________________________________________________________________________________________
def c_american_ttree_greeks_fp64(
    double S, double K, char cp_flag, double ttm, double iv, double r, double q, 
    int n_steps, bool delta, bool gamma, bool theta, bool vega, bool rho) -> float | tuple:

    cdef double d, g, t, v, rh

    american_ttree_greeks_fp64(
        S, K, cp_flag, ttm, iv, r, q, n_steps,
        &d if delta else NULL,
        &g if gamma else NULL,
        &t if theta else NULL,
        &v if vega else NULL,
        &rh if rho else NULL
    )

    res = []
    if delta: 
        res.append(d)
    if gamma: 
        res.append(g)
    if theta: 
        res.append(t)
    if vega: 
        res.append(v)
    if rho: 
        res.append(rh)

    return res[0] if len(res) == 1 else tuple(res)

def c_american_ttree_delta_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q, int n_steps) -> float:
    return american_ttree_delta_fp64(S, K, cp_flag, ttm, iv, r, q, n_steps)

def c_american_ttree_gamma_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q, int n_steps) -> float:
    return american_ttree_gamma_fp64(S, K, cp_flag, ttm, iv, r, q, n_steps)

def c_american_ttree_theta_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q, int n_steps) -> float:
    return american_ttree_theta_fp64(S, K, cp_flag, ttm, iv, r, q, n_steps)

def c_american_ttree_vega_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q, int n_steps) -> float:
    return american_ttree_vega_fp64(S, K, cp_flag, ttm, iv, r, q, n_steps)

def c_american_ttree_rho_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q, int n_steps) -> float:
    return american_ttree_rho_fp64(S, K, cp_flag, ttm, iv, r, q, n_steps)

def c_american_ttree_greeks_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    delta,
    uintptr_t    gamma,
    uintptr_t    theta,
    uintptr_t    vega,
    uintptr_t    rho
    ) -> None:
    with nogil:
        american_ttree_greeks_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_steps, n_options, 
            <double*>delta, <double*>gamma, <double*>theta, <double*>vega, <double*>rho
        )

def c_american_ttree_delta_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_ttree_delta_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_steps, n_options, <double*> out
        )

def c_american_ttree_gamma_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_ttree_gamma_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_steps, n_options, <double*> out
        )

def c_american_ttree_theta_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_ttree_theta_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_steps, n_options, <double*> out
        )

def c_american_ttree_vega_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_ttree_vega_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_steps, n_options, <double*> out
        )

def c_american_ttree_rho_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_ttree_rho_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_steps, n_options, <double*> out
        )

# iv _______________________________________________________________________________________________
def c_american_ttree_iv_fp64(
    double P, double S, double K, char cp_flag, double ttm, double iv, double r, double q, 
    int n_steps, double tol, size_t max_iter, uintptr_t lo_init, uintptr_t hi_init
    ) -> float:

    american_ttree_iv_fp64(
        P, S, K, cp_flag, ttm, r, q, n_steps, 
        tol, max_iter, <double*> lo_init, <double*> hi_init
    )

def c_american_ttree_iv_fp64_batch(
    uintptr_t    P,
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out,
    double       tol,
    size_t       max_iter,
    uintptr_t    lo_init,
    uintptr_t    hi_init
    ) -> None:

    with nogil:
        american_ttree_iv_fp64_batch(
            <double*>P, <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, 
            <double*>r, <double*>q, n_steps, n_options, <double*> out, 
            tol, max_iter, <double*>lo_init, <double*>hi_init
        )
# fp32 ---------------------------------------------------------------------------------------------
# price ____________________________________________________________________________________________
def c_american_ttree_price_fp32(float S, float K, bool cp_flag, float ttm, 
                                float iv, float r, float q, int n_steps) -> float:
    return american_ttree_price_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, n_steps
    )

def c_american_ttree_price_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out,
    bool         cuda,
    uintptr_t    stream,
    int          device,
    bool         on_device,
    bool         is_pinned,
    bool         sync
) -> None:

    if FASTVOL_CUDA_ENABLED and cuda:
        with nogil:
            american_ttree_price_fp32_cuda(
                <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
                <float*>r, <float*>q, n_steps, n_options, <float*>out, 
                <cudaStream_t>stream, device, on_device, is_pinned, sync
            )
    else:
        with nogil:
            american_ttree_price_fp32_batch(
                <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
                <float*>r, <float*>q, n_steps, n_options, <float*>out 
            )

# greeks ___________________________________________________________________________________________
def c_american_ttree_greeks_fp32(
    float S, float K, char cp_flag, float ttm, float iv, float r, float q, 
    int n_steps, bool delta, bool gamma, bool theta, bool vega, bool rho) -> float | tuple:

    cdef float d, g, t, v, rh

    american_ttree_greeks_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, n_steps,
        &d if delta else NULL,
        &g if gamma else NULL,
        &t if theta else NULL,
        &v if vega else NULL,
        &rh if rho else NULL
    )

    res = []
    if delta: 
        res.append(d)
    if gamma: 
        res.append(g)
    if theta: 
        res.append(t)
    if vega: 
        res.append(v)
    if rho: 
        res.append(rh)

    return res[0] if len(res) == 1 else tuple(res)

def c_american_ttree_delta_fp32(float S, float K, char cp_flag, float ttm, 
                                float iv, float r, float q, int n_steps) -> float:
    return american_ttree_delta_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, n_steps)

def c_american_ttree_gamma_fp32(float S, float K, char cp_flag, float ttm, 
                                float iv, float r, float q, int n_steps) -> float:
    return american_ttree_gamma_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, n_steps)

def c_american_ttree_theta_fp32(float S, float K, char cp_flag, float ttm, 
                                float iv, float r, float q, int n_steps) -> float:
    return american_ttree_theta_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, n_steps)

def c_american_ttree_vega_fp32(float S, float K, char cp_flag, float ttm, 
                                float iv, float r, float q, int n_steps) -> float:
    return american_ttree_vega_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, n_steps)

def c_american_ttree_rho_fp32(float S, float K, char cp_flag, float ttm, 
                                float iv, float r, float q, int n_steps) -> float:
    return american_ttree_rho_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q, n_steps)

def c_american_ttree_greeks_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    delta,
    uintptr_t    gamma,
    uintptr_t    theta,
    uintptr_t    vega,
    uintptr_t    rho
    ) -> None:
    with nogil:
        american_ttree_greeks_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_steps, n_options, 
            <float*>delta, <float*>gamma, <float*>theta, <float*>vega, <float*>rho
        )

def c_american_ttree_delta_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_ttree_delta_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_steps, n_options, <float*> out
        )

def c_american_ttree_gamma_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_ttree_gamma_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_steps, n_options, <float*> out
        )

def c_american_ttree_theta_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_ttree_theta_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_steps, n_options, <float*> out
        )

def c_american_ttree_vega_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_ttree_vega_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_steps, n_options, <float*> out
        )

def c_american_ttree_rho_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        american_ttree_rho_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_steps, n_options, <float*> out
        )

# iv _______________________________________________________________________________________________
def c_american_ttree_iv_fp32(
    float P, float S, float K, char cp_flag, float ttm, float iv, float r, float q, 
    int n_steps, float tol, size_t max_iter, uintptr_t lo_init, uintptr_t hi_init
    ) -> float:

    return american_ttree_iv_fp32(
        <float>P, <float>S, <float>K, cp_flag, <float>ttm, <float>r, <float>q, n_steps, 
        <float>tol, max_iter, <float*>lo_init, <float*>hi_init
    )

def c_american_ttree_iv_fp32_batch(
    uintptr_t    P,
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    r,
    uintptr_t    q,
    int          n_steps,
    size_t       n_options,
    uintptr_t    out,
    float        tol,
    size_t       max_iter,
    uintptr_t    lo_init,
    uintptr_t    hi_init
    ) -> None:

    with nogil:
        american_ttree_iv_fp32_batch(
            <float*>P, <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, 
            <float*>r, <float*>q, n_steps, n_options, <float*>out, 
            <float>tol, max_iter, <float*>lo_init, <float*>hi_init
        )

# european::bsm ====================================================================================
# fp64 ---------------------------------------------------------------------------------------------
# price ____________________________________________________________________________________________
def c_european_bsm_price_fp64(double S, double K, bool cp_flag, float ttm, 
                                double iv, double r, double q) -> float:
    return european_bsm_price_fp64(S, K, cp_flag, ttm, iv, r, q)

def c_european_bsm_price_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    out,
    bool         cuda,
    uintptr_t    stream,
    int          device,
    bool         on_device,
    bool         is_pinned,
    bool         sync
) -> None:

    if FASTVOL_CUDA_ENABLED and cuda:
        with nogil:
            european_bsm_price_fp64_cuda(
                <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
                <double*>r, <double*>q, n_options, <double*>out, 
                <cudaStream_t>stream, device, on_device, is_pinned, sync
            )
    else:
        with nogil:
            european_bsm_price_fp64_batch(
                <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
                <double*>r, <double*>q, n_options, <double*>out 
            )
# greeks ___________________________________________________________________________________________
def c_european_bsm_greeks_fp64(
    double S, double K, char cp_flag, double ttm, double iv, double r, double q, 
    bool delta, bool gamma, bool theta, bool vega, bool rho) -> float | tuple:

    cdef double d, g, t, v, rh

    european_bsm_greeks_fp64(
        S, K, cp_flag, ttm, iv, r, q,
        &d if delta else NULL,
        &g if gamma else NULL,
        &t if theta else NULL,
        &v if vega else NULL,
        &rh if rho else NULL
    )

    res = []
    if delta: 
        res.append(d)
    if gamma: 
        res.append(g)
    if theta: 
        res.append(t)
    if vega: 
        res.append(v)
    if rho: 
        res.append(rh)

    return res[0] if len(res) == 1 else tuple(res)

def c_european_bsm_delta_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q) -> float:
    return european_bsm_delta_fp64(S, K, cp_flag, ttm, iv, r, q)

def c_european_bsm_gamma_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q) -> float:
    return european_bsm_gamma_fp64(S, K, cp_flag, ttm, iv, r, q)

def c_european_bsm_theta_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q) -> float:
    return european_bsm_theta_fp64(S, K, cp_flag, ttm, iv, r, q)

def c_european_bsm_vega_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q) -> float:
    return european_bsm_vega_fp64(S, K, cp_flag, ttm, iv, r, q)

def c_european_bsm_rho_fp64(double S, double K, char cp_flag, double ttm, 
                                double iv, double r, double q) -> float:
    return european_bsm_rho_fp64(S, K, cp_flag, ttm, iv, r, q)

def c_european_bsm_greeks_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    delta,
    uintptr_t    gamma,
    uintptr_t    theta,
    uintptr_t    vega,
    uintptr_t    rho
    ) -> None:
    with nogil:
        european_bsm_greeks_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_options, 
            <double*>delta, <double*>gamma, <double*>theta, <double*>vega, <double*>rho
        )

def c_european_bsm_delta_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        european_bsm_delta_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_options, <double*> out
        )

def c_european_bsm_gamma_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        european_bsm_gamma_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_options, <double*> out
        )

def c_european_bsm_theta_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        european_bsm_theta_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_options, <double*> out
        )

def c_european_bsm_vega_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        european_bsm_vega_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_options, <double*> out
        )

def c_european_bsm_rho_fp64_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        european_bsm_rho_fp64_batch(
            <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, <double*>iv, 
            <double*>r, <double*>q, n_options, <double*> out
        )

# iv _______________________________________________________________________________________________
def c_european_bsm_iv_fp64(
    double P, double S, double K, char cp_flag, double ttm, double iv, double r, double q, 
    double tol, size_t max_iter, uintptr_t lo_init, uintptr_t hi_init
    ) -> float:

    return european_bsm_iv_fp64(
        P, S, K, cp_flag, ttm, r, q, 
        tol, max_iter, <double*>lo_init, <double*>hi_init
    )

def c_european_bsm_iv_fp64_batch(
    uintptr_t    P,
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    out,
    double       tol,
    size_t       max_iter,
    uintptr_t    lo_init,
    uintptr_t    hi_init
    ) -> None:

    with nogil:
        european_bsm_iv_fp64_batch(
            <double*>P, <double*>S, <double*>K, <char*>cp_flag, <double*>ttm, 
            <double*>r, <double*>q, n_options, <double*>out, 
            tol, max_iter, <double*>lo_init, <double*>hi_init
        )

# fp32 ---------------------------------------------------------------------------------------------
# price ____________________________________________________________________________________________
def c_european_bsm_price_fp32(S: float, K: float, cp_flag: bool, ttm: float, 
                                iv: float, r: float, q: float) -> float:
    return european_bsm_price_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q
    )

def c_european_bsm_price_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    out,
    bool         cuda,
    uintptr_t    stream,
    int          device,
    bool         on_device,
    bool         is_pinned,
    bool         sync
) -> None:

    if FASTVOL_CUDA_ENABLED and cuda:
        with nogil:
            european_bsm_price_fp32_cuda(
                <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
                <float*>r, <float*>q, n_options, <float*>out, 
                <cudaStream_t>stream, device, on_device, is_pinned, sync
            )
    else:
        with nogil:
            european_bsm_price_fp32_batch(
                <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
                <float*>r, <float*>q, n_options, <float*>out 
            )

# greeks ___________________________________________________________________________________________
def c_european_bsm_greeks_fp32(
    float S, float K, char cp_flag, float ttm, float iv, float r, float q, 
    bool delta, bool gamma, bool theta, bool vega, bool rho) -> float | tuple:

    cdef float d, g, t, v, rh

    european_bsm_greeks_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q,
        &d if delta else NULL,
        &g if gamma else NULL,
        &t if theta else NULL,
        &v if vega else NULL,
        &rh if rho else NULL
    )

    res = []
    if delta: 
        res.append(d)
    if gamma: 
        res.append(g)
    if theta: 
        res.append(t)
    if vega: 
        res.append(v)
    if rho: 
        res.append(rh)

    return res[0] if len(res) == 1 else tuple(res)

def c_european_bsm_delta_fp32(float S, float K, char cp_flag, float ttm, 
                                float iv, float r, float q) -> float:
    return european_bsm_delta_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q)

def c_european_bsm_gamma_fp32(float S, float K, char cp_flag, float ttm, 
                                float iv, float r, float q) -> float:
    return european_bsm_gamma_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q)

def c_european_bsm_theta_fp32(float S, float K, char cp_flag, float ttm, 
                                float iv, float r, float q) -> float:
    return european_bsm_theta_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q)

def c_european_bsm_vega_fp32(float S, float K, char cp_flag, float ttm, 
                                float iv, float r, float q) -> float:
    return european_bsm_vega_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q)

def c_european_bsm_rho_fp32(float S, float K, char cp_flag, float ttm, 
                                float iv, float r, float q) -> float:
    return european_bsm_rho_fp32(
        <float>S, <float>K, cp_flag, <float>ttm, <float>iv, <float>r, <float>q)

def c_european_bsm_greeks_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    delta,
    uintptr_t    gamma,
    uintptr_t    theta,
    uintptr_t    vega,
    uintptr_t    rho
    ) -> None:
    with nogil:
        european_bsm_greeks_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_options, 
            <float*>delta, <float*>gamma, <float*>theta, <float*>vega, <float*>rho
        )

def c_european_bsm_delta_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        european_bsm_delta_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_options, <float*> out
        )

def c_european_bsm_gamma_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        european_bsm_gamma_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_options, <float*> out
        )

def c_european_bsm_theta_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        european_bsm_theta_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_options, <float*> out
        )

def c_european_bsm_vega_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        european_bsm_vega_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_options, <float*> out
        )

def c_european_bsm_rho_fp32_batch(
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    iv,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    out
    ) -> None:
    with nogil:
        european_bsm_rho_fp32_batch(
            <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, <float*>iv, 
            <float*>r, <float*>q, n_options, <float*> out
        )

# iv _______________________________________________________________________________________________
def c_european_bsm_iv_fp32(
    float P, float S, float K, char cp_flag, float ttm, float iv, float r, float q, 
    float tol, size_t max_iter, uintptr_t lo_init, uintptr_t hi_init
    ) -> float:

    return european_bsm_iv_fp32(
        <float>P, <float>S, <float>K, cp_flag, <float>ttm, <float>r, <float>q, 
        <float>tol, max_iter, <float*>lo_init, <float*>hi_init
    )

def c_european_bsm_iv_fp32_batch(
    uintptr_t    P,
    uintptr_t    S,
    uintptr_t    K,
    uintptr_t    cp_flag,
    uintptr_t    ttm,
    uintptr_t    r,
    uintptr_t    q,
    size_t       n_options,
    uintptr_t    out,
    float        tol,
    size_t       max_iter,
    uintptr_t    lo_init,
    uintptr_t    hi_init
    ) -> None:

    with nogil:
        european_bsm_iv_fp32_batch(
            <float*>P, <float*>S, <float*>K, <char*>cp_flag, <float*>ttm, 
            <float*>r, <float*>q, n_options, <float*>out, 
            <float>tol, max_iter, <float*>lo_init, <float*>hi_init
        )

# ==================================================================================================
def c_version() -> str:
    return fastvol_version().decode('utf-8')

def c_cuda_available() -> bool:
    return FASTVOL_CUDA_ENABLED