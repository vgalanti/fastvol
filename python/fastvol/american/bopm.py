import torch
import numpy as np

from torch import Tensor
from numpy import ndarray

from ..utils import align_args, get_ptrs, cu_params, mk_out_like

from .._core import c_american_bopm_price_fp64, c_american_bopm_price_fp64_batch
from .._core import c_american_bopm_greeks_fp64, c_american_bopm_greeks_fp64_batch
from .._core import c_american_bopm_delta_fp64, c_american_bopm_delta_fp64_batch
from .._core import c_american_bopm_gamma_fp64, c_american_bopm_gamma_fp64_batch
from .._core import c_american_bopm_theta_fp64, c_american_bopm_theta_fp64_batch
from .._core import c_american_bopm_vega_fp64, c_american_bopm_vega_fp64_batch
from .._core import c_american_bopm_rho_fp64, c_american_bopm_rho_fp64_batch
from .._core import c_american_bopm_iv_fp64, c_american_bopm_iv_fp64_batch

from .._core import c_american_bopm_price_fp32, c_american_bopm_price_fp32_batch
from .._core import c_american_bopm_greeks_fp32, c_american_bopm_greeks_fp32_batch
from .._core import c_american_bopm_delta_fp32, c_american_bopm_delta_fp32_batch
from .._core import c_american_bopm_gamma_fp32, c_american_bopm_gamma_fp32_batch
from .._core import c_american_bopm_theta_fp32, c_american_bopm_theta_fp32_batch
from .._core import c_american_bopm_vega_fp32, c_american_bopm_vega_fp32_batch
from .._core import c_american_bopm_rho_fp32, c_american_bopm_rho_fp32_batch
from .._core import c_american_bopm_iv_fp32, c_american_bopm_iv_fp32_batch


# dispatchers --------------------------------------------------------------------------------------
# price ____________________________________________________________________________________________
def price(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
    cuda=False,
    precision="auto",
):

    args = align_args(
        spot, strike, c_flag, ttm, iv, rfr, div, cuda=cuda, precision=precision
    )

    if type(args[0]) in (ndarray, Tensor):
        if args[0].dtype in (torch.float64, np.float64):
            return price_fp64_batch(*args, n_steps, cuda=cuda)
        else:
            return price_fp32_batch(*args, n_steps, cuda=cuda)
    else:
        if precision == "fp64":
            return price_fp64(*args, n_steps)
        else:
            return price_fp32(*args, n_steps)


# greeks ___________________________________________________________________________________________
def greeks(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
    delta=False,
    gamma=False,
    theta=False,
    vega=False,
    rho=False,
    precision="auto",
):

    args = align_args(spot, strike, c_flag, ttm, iv, rfr, div, precision=precision)
    greeks = (delta, gamma, theta, vega, rho)

    if type(args[0]) in (ndarray, Tensor):
        if args[0].dtype in (torch.float64, np.float64):
            return greeks_fp64_batch(*args, n_steps, *greeks)
        else:
            return greeks_fp32_batch(*args, n_steps, *greeks)
    else:
        if precision == "fp64":
            return greeks_fp64(*args, n_steps, *greeks)
        else:
            return greeks_fp32(*args, n_steps, *greeks)


def delta(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
    precision="auto",
):

    args = align_args(spot, strike, c_flag, ttm, iv, rfr, div, precision=precision)

    if type(args[0]) in (ndarray, Tensor):
        if args[0].dtype in (torch.float64, np.float64):
            return delta_fp64_batch(*args, n_steps)
        else:
            return delta_fp32_batch(*args, n_steps)
    else:
        if precision == "fp64":
            return delta_fp64(*args, n_steps)
        else:
            return delta_fp32(*args, n_steps)


def gamma(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
    precision="auto",
):

    args = align_args(spot, strike, c_flag, ttm, iv, rfr, div, precision=precision)

    if type(args[0]) in (ndarray, Tensor):
        if args[0].dtype in (torch.float64, np.float64):
            return gamma_fp64_batch(*args, n_steps)
        else:
            return gamma_fp32_batch(*args, n_steps)
    else:
        if precision == "fp64":
            return gamma_fp64(*args, n_steps)
        else:
            return gamma_fp32(*args, n_steps)


def theta(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
    precision="auto",
):

    args = align_args(spot, strike, c_flag, ttm, iv, rfr, div, precision=precision)

    if type(args[0]) in (ndarray, Tensor):
        if args[0].dtype in (torch.float64, np.float64):
            return theta_fp64_batch(*args, n_steps)
        else:
            return theta_fp32_batch(*args, n_steps)
    else:
        if precision == "fp64":
            return theta_fp64(*args, n_steps)
        else:
            return theta_fp32(*args, n_steps)


def vega(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
    precision="auto",
):

    args = align_args(spot, strike, c_flag, ttm, iv, rfr, div, precision=precision)

    if type(args[0]) in (ndarray, Tensor):
        if args[0].dtype in (torch.float64, np.float64):
            return vega_fp64_batch(*args, n_steps)
        else:
            return vega_fp32_batch(*args, n_steps)
    else:
        if precision == "fp64":
            return vega_fp64(*args, n_steps)
        else:
            return vega_fp32(*args, n_steps)


def rho(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
    precision="auto",
):

    args = align_args(spot, strike, c_flag, ttm, iv, rfr, div, precision=precision)

    if type(args[0]) in (ndarray, Tensor):
        if args[0].dtype in (torch.float64, np.float64):
            return rho_fp64_batch(*args, n_steps)
        else:
            return rho_fp32_batch(*args, n_steps)
    else:
        if precision == "fp64":
            return rho_fp64(*args, n_steps)
        else:
            return rho_fp32(*args, n_steps)


# iv _______________________________________________________________________________________________
def iv(
    price=10.0,
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    rfr=0.05,
    div=0.0,
    n_steps=512,
    tol=1e-3,
    max_iter=100,
    precision="auto",
):

    args = align_args(price, spot, strike, c_flag, ttm, rfr, div, precision=precision)

    if type(args[0]) in (ndarray, Tensor):
        if args[0].dtype in (torch.float64, np.float64):
            return iv_fp64_batch(*args, n_steps, tol, max_iter)
        else:
            return iv_fp32_batch(*args, n_steps, tol, max_iter)
    else:
        if precision == "fp64":
            return iv_fp64(*args, n_steps, tol, max_iter)
        else:
            return iv_fp32(*args, n_steps, tol, max_iter)


# fp64 ---------------------------------------------------------------------------------------------
# price ____________________________________________________________________________________________
def price_fp64(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
):
    return c_american_bopm_price_fp64(spot, strike, c_flag, ttm, iv, rfr, div, n_steps)


def price_fp64_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    n_steps=512,
    cuda=False,
):

    out = mk_out_like(spot)
    cu = cu_params(spot)
    n = out.shape[0]

    in_ptrs = get_ptrs(spot, strike, c_flag, ttm, iv, rfr, div)
    out_ptr = get_ptrs(out)

    c_american_bopm_price_fp64_batch(*in_ptrs, n_steps, n, out_ptr, cuda, *cu)

    return out


# greeks ___________________________________________________________________________________________
def greeks_fp64(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
    delta=False,
    gamma=False,
    theta=False,
    vega=False,
    rho=False,
):
    return c_american_bopm_greeks_fp64(
        spot, strike, c_flag, ttm, iv, rfr, div, n_steps, delta, gamma, theta, vega, rho
    )


def delta_fp64(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
):
    return c_american_bopm_delta_fp64(spot, strike, c_flag, ttm, iv, rfr, div, n_steps)


def gamma_fp64(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
):
    return c_american_bopm_gamma_fp64(spot, strike, c_flag, ttm, iv, rfr, div, n_steps)


def theta_fp64(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
):
    return c_american_bopm_theta_fp64(spot, strike, c_flag, ttm, iv, rfr, div, n_steps)


def vega_fp64(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
):
    return c_american_bopm_vega_fp64(spot, strike, c_flag, ttm, iv, rfr, div, n_steps)


def rho_fp64(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
):
    return c_american_bopm_rho_fp64(spot, strike, c_flag, ttm, iv, rfr, div, n_steps)


def greeks_fp64_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    n_steps=512,
    delta=False,
    gamma=False,
    theta=False,
    vega=False,
    rho=False,
):

    n = spot.shape[0]
    in_ptrs = get_ptrs(spot, strike, c_flag, ttm, iv, rfr, div)

    greeks = [delta, gamma, theta, vega, rho]
    arrs = [mk_out_like(spot) if g else None for g in greeks]
    out_ptrs = [get_ptrs(a) if g else 0 for g, a in zip(greeks, arrs)]

    c_american_bopm_greeks_fp64_batch(*in_ptrs, n_steps, n, *out_ptrs)

    arrs = tuple([a for g, a in zip(greeks, arrs) if g])
    return arrs[0] if len(arrs) == 1 else arrs


def delta_fp64_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    n_steps=512,
):

    n = spot.shape[0]
    in_ptrs = get_ptrs(spot, strike, c_flag, ttm, iv, rfr, div)

    out = mk_out_like(spot)
    out_ptr = get_ptrs(out)

    c_american_bopm_delta_fp64_batch(*in_ptrs, n_steps, n, out_ptr)

    return out


def gamma_fp64_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    n_steps=512,
):

    n = spot.shape[0]
    in_ptrs = get_ptrs(spot, strike, c_flag, ttm, iv, rfr, div)

    out = mk_out_like(spot)
    out_ptr = get_ptrs(out)

    c_american_bopm_gamma_fp64_batch(*in_ptrs, n_steps, n, out_ptr)

    return out


def theta_fp64_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    n_steps=512,
):

    n = spot.shape[0]
    in_ptrs = get_ptrs(spot, strike, c_flag, ttm, iv, rfr, div)

    out = mk_out_like(spot)
    out_ptr = get_ptrs(out)

    c_american_bopm_theta_fp64_batch(*in_ptrs, n_steps, n, out_ptr)

    return out


def vega_fp64_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    n_steps=512,
):

    n = spot.shape[0]
    in_ptrs = get_ptrs(spot, strike, c_flag, ttm, iv, rfr, div)

    out = mk_out_like(spot)
    out_ptr = get_ptrs(out)

    c_american_bopm_vega_fp64_batch(*in_ptrs, n_steps, n, out_ptr)

    return out


def rho_fp64_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    n_steps=512,
):

    n = spot.shape[0]
    in_ptrs = get_ptrs(spot, strike, c_flag, ttm, iv, rfr, div)

    out = mk_out_like(spot)
    out_ptr = get_ptrs(out)

    c_american_bopm_rho_fp64_batch(*in_ptrs, n_steps, n, out_ptr)

    return out


# iv _______________________________________________________________________________________________
def iv_fp64(
    price=10.0,
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    rfr=0.05,
    div=0.0,
    n_steps=512,
    tol=1e-3,
    max_iter=100,
):
    return c_american_bopm_iv_fp64(
        price, spot, strike, c_flag, ttm, rfr, div, n_steps, tol, max_iter, 0, 0
    )


def iv_fp64_batch(
    price,
    spot,
    strike,
    c_flag,
    ttm,
    rfr,
    div,
    n_steps=512,
    tol=1e-3,
    max_iter=100,
):

    out = mk_out_like(spot)
    n = out.shape[0]

    in_ptrs = get_ptrs(price, spot, strike, c_flag, ttm, rfr, div)
    out_ptr = get_ptrs(out)

    c_american_bopm_iv_fp64_batch(*in_ptrs, n_steps, n, out_ptr, tol, max_iter, 0, 0)

    return out


# fp32 ---------------------------------------------------------------------------------------------
# price ____________________________________________________________________________________________
def price_fp32(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
):
    return c_american_bopm_price_fp32(spot, strike, c_flag, ttm, iv, rfr, div, n_steps)


def price_fp32_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    n_steps=512,
    cuda=False,
):

    out = mk_out_like(spot)
    cu = cu_params(spot)
    n = out.shape[0]

    in_ptrs = get_ptrs(spot, strike, c_flag, ttm, iv, rfr, div)
    out_ptr = get_ptrs(out)

    c_american_bopm_price_fp32_batch(*in_ptrs, n_steps, n, out_ptr, cuda, *cu)

    return out


# greeks ___________________________________________________________________________________________
def greeks_fp32(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
    delta=False,
    gamma=False,
    theta=False,
    vega=False,
    rho=False,
):
    return c_american_bopm_greeks_fp32(
        spot, strike, c_flag, ttm, iv, rfr, div, n_steps, delta, gamma, theta, vega, rho
    )


def delta_fp32(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
):
    return c_american_bopm_delta_fp32(spot, strike, c_flag, ttm, iv, rfr, div, n_steps)


def gamma_fp32(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
):
    return c_american_bopm_gamma_fp32(spot, strike, c_flag, ttm, iv, rfr, div, n_steps)


def theta_fp32(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
):
    return c_american_bopm_theta_fp32(spot, strike, c_flag, ttm, iv, rfr, div, n_steps)


def vega_fp32(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
):
    return c_american_bopm_vega_fp32(spot, strike, c_flag, ttm, iv, rfr, div, n_steps)


def rho_fp32(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    n_steps=512,
):
    return c_american_bopm_rho_fp32(spot, strike, c_flag, ttm, iv, rfr, div, n_steps)


def greeks_fp32_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    n_steps=512,
    delta=False,
    gamma=False,
    theta=False,
    vega=False,
    rho=False,
):

    n = spot.shape[0]
    in_ptrs = get_ptrs(spot, strike, c_flag, ttm, iv, rfr, div)

    greeks = [delta, gamma, theta, vega, rho]
    arrs = [mk_out_like(spot) if g else None for g in greeks]
    out_ptrs = [get_ptrs(a) if g else 0 for g, a in zip(greeks, arrs)]

    c_american_bopm_greeks_fp32_batch(*in_ptrs, n_steps, n, *out_ptrs)

    arrs = tuple([a for g, a in zip(greeks, arrs) if g])
    return arrs[0] if len(arrs) == 1 else arrs


def delta_fp32_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    n_steps=512,
):

    n = spot.shape[0]
    in_ptrs = get_ptrs(spot, strike, c_flag, ttm, iv, rfr, div)

    out = mk_out_like(spot)
    out_ptr = get_ptrs(out)

    c_american_bopm_delta_fp32_batch(*in_ptrs, n_steps, n, out_ptr)

    return out


def gamma_fp32_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    n_steps=512,
):

    n = spot.shape[0]
    in_ptrs = get_ptrs(spot, strike, c_flag, ttm, iv, rfr, div)

    out = mk_out_like(spot)
    out_ptr = get_ptrs(out)

    c_american_bopm_gamma_fp32_batch(*in_ptrs, n_steps, n, out_ptr)

    return out


def theta_fp32_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    n_steps=512,
):

    n = spot.shape[0]
    in_ptrs = get_ptrs(spot, strike, c_flag, ttm, iv, rfr, div)

    out = mk_out_like(spot)
    out_ptr = get_ptrs(out)

    c_american_bopm_theta_fp32_batch(*in_ptrs, n_steps, n, out_ptr)

    return out


def vega_fp32_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    n_steps=512,
):

    n = spot.shape[0]
    in_ptrs = get_ptrs(spot, strike, c_flag, ttm, iv, rfr, div)

    out = mk_out_like(spot)
    out_ptr = get_ptrs(out)

    c_american_bopm_vega_fp32_batch(*in_ptrs, n_steps, n, out_ptr)

    return out


def rho_fp32_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    n_steps=512,
):

    n = spot.shape[0]
    in_ptrs = get_ptrs(spot, strike, c_flag, ttm, iv, rfr, div)

    out = mk_out_like(spot)
    out_ptr = get_ptrs(out)

    c_american_bopm_rho_fp32_batch(*in_ptrs, n_steps, n, out_ptr)

    return out


# iv _______________________________________________________________________________________________
def iv_fp32(
    price=10.0,
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    rfr=0.05,
    div=0.0,
    n_steps=512,
    tol=1e-3,
    max_iter=100,
):
    return c_american_bopm_iv_fp32(
        price, spot, strike, c_flag, ttm, rfr, div, n_steps, tol, max_iter, 0, 0
    )


def iv_fp32_batch(
    price,
    spot,
    strike,
    c_flag,
    ttm,
    rfr,
    div,
    n_steps=512,
    tol=1e-3,
    max_iter=100,
):

    out = mk_out_like(spot)
    n = out.shape[0]

    in_ptrs = get_ptrs(price, spot, strike, c_flag, ttm, rfr, div)
    out_ptr = get_ptrs(out)

    c_american_bopm_iv_fp32_batch(*in_ptrs, n_steps, n, out_ptr, tol, max_iter, 0, 0)

    return out
