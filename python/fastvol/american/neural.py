import torch
import logging

from ..neural.utils import load_net, run_net, align_args_nn, get_greeks
from ..neural.american import _VNet, _IVNet

log = logging.getLogger(__name__)

# compilation causes strange behavior for autograd
# additional copy of vnet for greeks that without torch compilation
_vnet, _vnet_w = load_net("american_vnet", _VNet, torch_compile=True)
_gnet, _gnet_w = load_net("american_vnet", _VNet, torch_compile=False)
_ivnet, _ivnet_w = load_net("american_ivnet", _IVNet, torch_compile=True)


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
    device=torch.device("cpu"),
    batch_size=65_536,
    precision="auto",
):

    s, k, cp, t, v, r, q = align_args_nn(
        spot, strike, c_flag, ttm, iv, rfr, div, precision=precision
    )

    tensors = dict(spot=s, strike=k, c_flag=cp, ttm=t, iv=v, rfr=r, div=q)

    price, _ = run_net(_vnet, _vnet_w, tensors, device, batch_size)
    price = price.detach()

    return price if price.shape[0] > 1 else float(price[0])


# greeks ___________________________________________________________________________________________
def greeks(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    delta=False,
    gamma=False,
    theta=False,
    vega=False,
    rho=False,
    device=torch.device("cpu"),
    batch_size=65_536,
    precision="auto",
):

    s, k, cp, t, v, r, q = align_args_nn(
        spot, strike, c_flag, ttm, iv, rfr, div, precision=precision
    )
    tensors = dict(spot=s, strike=k, c_flag=cp, ttm=t, iv=v, rfr=r, div=q)

    return get_greeks(
        _gnet, _gnet_w, tensors, device, batch_size, delta, gamma, theta, vega, rho
    )


def delta(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
    batch_size=65_536,
    precision="auto",
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        delta=True,
        device=device,
        batch_size=batch_size,
        precision=precision,
    )


def gamma(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
    batch_size=65_536,
    precision="auto",
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        gamma=True,
        device=device,
        batch_size=batch_size,
        precision=precision,
    )


def theta(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
    batch_size=65_536,
    precision="auto",
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        theta=True,
        device=device,
        batch_size=batch_size,
        precision=precision,
    )


def vega(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
    batch_size=65_536,
    precision="auto",
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        vega=True,
        device=device,
        batch_size=batch_size,
        precision=precision,
    )


def rho(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
    batch_size=65_536,
    precision="auto",
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        rho=True,
        device=device,
        batch_size=batch_size,
        precision=precision,
    )


# iv _______________________________________________________________________________________________
def iv(
    price=10.0,
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
    batch_size=65_536,
    precision="auto",
):

    p, s, k, cp, t, r, q = align_args_nn(
        price, spot, strike, c_flag, ttm, rfr, div, precision=precision
    )

    tensors = dict(price=p, spot=s, strike=k, c_flag=cp, ttm=t, rfr=r, div=q)

    v, _ = run_net(_ivnet, _ivnet_w, tensors, device, batch_size)
    v = v.detach()

    return float(v[0]) if v.shape[0] == 1 else v


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
    device=torch.device("cpu"),
):
    return price(spot, strike, c_flag, ttm, iv, rfr, div, device, precision="fp64")


def price_fp64_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, iv=iv, rfr=rfr, div=div
    )
    price, _ = run_net(_vnet, _vnet_w, tensors, device, batch_size)
    return price.detach()


# greeks ___________________________________________________________________________________________
def greeks_fp64(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    delta=False,
    gamma=False,
    theta=False,
    vega=False,
    rho=False,
    device=torch.device("cpu"),
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        delta,
        gamma,
        theta,
        vega,
        rho,
        device,
        precision="fp64",
    )


def delta_fp64(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        delta=True,
        device=device,
        precision="fp64",
    )


def gamma_fp64(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        gamma=True,
        device=device,
        precision="fp64",
    )


def theta_fp64(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        theta=True,
        device=device,
        precision="fp64",
    )


def vega_fp64(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        vega=True,
        device=device,
        precision="fp64",
    )


def rho_fp64(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        rho=True,
        device=device,
        precision="fp64",
    )


def greeks_fp64_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    delta=False,
    gamma=False,
    theta=False,
    vega=False,
    rho=False,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, iv=iv, rfr=rfr, div=div
    )
    return get_greeks(
        _gnet, _gnet_w, tensors, device, batch_size, delta, gamma, theta, vega, rho
    )


def delta_fp64_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, iv=iv, rfr=rfr, div=div
    )
    return get_greeks(_gnet, _gnet_w, tensors, device, batch_size, delta=True)


def gamma_fp64_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, iv=iv, rfr=rfr, div=div
    )
    return get_greeks(_gnet, _gnet_w, tensors, device, batch_size, gamma=True)


def theta_fp64_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, iv=iv, rfr=rfr, div=div
    )
    return get_greeks(_gnet, _gnet_w, tensors, device, batch_size, theta=True)


def vega_fp64_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, iv=iv, rfr=rfr, div=div
    )
    return get_greeks(_gnet, _gnet_w, tensors, device, batch_size, vega=True)


def rho_fp64_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, iv=iv, rfr=rfr, div=div
    )
    return get_greeks(_gnet, _gnet_w, tensors, device, batch_size, rho=True)


# iv _______________________________________________________________________________________________
def iv_fp64(
    price=10.0,
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
):
    return iv(price, spot, strike, c_flag, ttm, rfr, div, device, precision="fp64")


def iv_fp64_batch(
    price,
    spot,
    strike,
    c_flag,
    ttm,
    rfr,
    div,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        price=price, spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, rfr=rfr, div=div
    )
    v, _ = run_net(_ivnet, _ivnet_w, tensors, device, batch_size)
    return v.detach()


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
    device=torch.device("cpu"),
):
    return price(spot, strike, c_flag, ttm, iv, rfr, div, device, precision="fp32")


def price_fp32_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, iv=iv, rfr=rfr, div=div
    )
    price, _ = run_net(_vnet, _vnet_w, tensors, device, batch_size)
    return price.detach()


# greeks ___________________________________________________________________________________________
def greeks_fp32(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    delta=False,
    gamma=False,
    theta=False,
    vega=False,
    rho=False,
    device=torch.device("cpu"),
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        delta,
        gamma,
        theta,
        vega,
        rho,
        device,
        precision="fp32",
    )


def delta_fp32(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        delta=True,
        device=device,
        precision="fp32",
    )


def gamma_fp32(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        gamma=True,
        device=device,
        precision="fp32",
    )


def theta_fp32(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        theta=True,
        device=device,
        precision="fp32",
    )


def vega_fp32(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        vega=True,
        device=device,
        precision="fp32",
    )


def rho_fp32(
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    iv=0.2,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
):
    return greeks(
        spot,
        strike,
        c_flag,
        ttm,
        iv,
        rfr,
        div,
        rho=True,
        device=device,
        precision="fp32",
    )


def greeks_fp32_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    delta=False,
    gamma=False,
    theta=False,
    vega=False,
    rho=False,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, iv=iv, rfr=rfr, div=div
    )
    return get_greeks(
        _gnet, _gnet_w, tensors, device, batch_size, delta, gamma, theta, vega, rho
    )


def delta_fp32_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, iv=iv, rfr=rfr, div=div
    )
    return get_greeks(_gnet, _gnet_w, tensors, device, batch_size, delta=True)


def gamma_fp32_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, iv=iv, rfr=rfr, div=div
    )
    return get_greeks(_gnet, _gnet_w, tensors, device, batch_size, gamma=True)


def theta_fp32_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, iv=iv, rfr=rfr, div=div
    )
    return get_greeks(_gnet, _gnet_w, tensors, device, batch_size, theta=True)


def vega_fp32_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, iv=iv, rfr=rfr, div=div
    )
    return get_greeks(_gnet, _gnet_w, tensors, device, batch_size, vega=True)


def rho_fp32_batch(
    spot,
    strike,
    c_flag,
    ttm,
    iv,
    rfr,
    div,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, iv=iv, rfr=rfr, div=div
    )
    return get_greeks(_gnet, _gnet_w, tensors, device, batch_size, rho=True)


# iv _______________________________________________________________________________________________
def iv_fp32(
    price=10.0,
    spot=100.0,
    strike=100.0,
    c_flag=True,
    ttm=1.0,
    rfr=0.05,
    div=0.0,
    device=torch.device("cpu"),
):
    return iv(price, spot, strike, c_flag, ttm, rfr, div, device, precision="fp32")


def iv_fp32_batch(
    price,
    spot,
    strike,
    c_flag,
    ttm,
    rfr,
    div,
    device=torch.device("cpu"),
    batch_size=65_536,
):
    tensors = dict(
        price=price, spot=spot, strike=strike, c_flag=c_flag, ttm=ttm, rfr=rfr, div=div
    )
    v, _ = run_net(_ivnet, _ivnet_w, tensors, device, batch_size)
    return v.detach()
