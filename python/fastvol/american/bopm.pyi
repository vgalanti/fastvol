import torch

from torch import Tensor
from numpy import uint8
from numpy import float32 as fp32
from numpy import float64 as fp64
from numpy import bool_ as npbool
from numpy.typing import NDArray
from typing import Literal, Tuple

# dispatchers --------------------------------------------------------------------------------------
# price ____________________________________________________________________________________________
def price(
    spot: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    strike: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool] | bool = True,
    ttm: Tensor | NDArray[fp64] | NDArray[fp32] | float = 1.0,
    iv: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.2,
    rfr: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.05,
    div: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.0,
    n_steps: int = 512,
    cuda: bool = False,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tensor | float:
    """
    Computes the price of an American option using the Binomial Options Pricing Model.

    Parameters
    ----------
    spot : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The current price of the underlying asset, by default 100.0.
    strike : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The strike price of the option, by default 100.0.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool] | bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The risk-free interest rate, by default 0.05.
    div : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    cuda : bool, optional
        A flag indicating whether to use CUDA for computation, by default False.
    precision : Literal["fp32", "fp64", "auto"], optional
        The floating-point precision to use for computation, by default "auto".

    Returns
    -------
    Tensor | float
        The price of the option.
    """
    ...

# greeks ___________________________________________________________________________________________
def greeks(
    spot: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    strike: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool] | bool = True,
    ttm: Tensor | NDArray[fp64] | NDArray[fp32] | float = 1.0,
    iv: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.2,
    rfr: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.05,
    div: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.0,
    n_steps: int = 512,
    delta: bool = False,
    gamma: bool = False,
    theta: bool = False,
    vega: bool = False,
    rho: bool = False,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tuple[Tensor | float, ...]:
    """
    Computes the Greeks of an American option using the Binomial Options Pricing Model.

    Parameters
    ----------
    spot : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The current price of the underlying asset, by default 100.0.
    strike : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The strike price of the option, by default 100.0.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool] | bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The risk-free interest rate, by default 0.05.
    div : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    delta : bool, optional
        A flag indicating whether to compute the delta of the option, by default False.
    gamma : bool, optional
        A flag indicating whether to compute the gamma of the option, by default False.
    theta : bool, optional
        A flag indicating whether to compute the theta of the option, by default False.
    vega : bool, optional
        A flag indicating whether to compute the vega of the option, by default False.
    rho : bool, optional
        A flag indicating whether to compute the rho of the option, by default False.
    precision : Literal["fp32", "fp64", "auto"], optional
        The floating-point precision to use for computation, by default "auto".

    Returns
    -------
    Tuple[Tensor | float, ...]
        A tuple containing the computed Greeks.
    """
    ...

def delta(
    spot: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    strike: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool] | bool = True,
    ttm: Tensor | NDArray[fp64] | NDArray[fp32] | float = 1.0,
    iv: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.2,
    rfr: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.05,
    div: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.0,
    n_steps: int = 512,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tensor | float:
    """
    Computes the delta of an American option using the Binomial Options Pricing Model.

    Parameters
    ----------
    spot : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The current price of the underlying asset, by default 100.0.
    strike : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The strike price of the option, by default 100.0.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool] | bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The risk-free interest rate, by default 0.05.
    div : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    precision : Literal["fp32", "fp64", "auto"], optional
        The floating-point precision to use for computation, by default "auto".

    Returns
    -------
    Tensor | float
        The delta of the option.
    """
    ...

def gamma(
    spot: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    strike: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool] | bool = True,
    ttm: Tensor | NDArray[fp64] | NDArray[fp32] | float = 1.0,
    iv: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.2,
    rfr: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.05,
    div: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.0,
    n_steps: int = 512,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tensor | float:
    """
    Computes the gamma of an American option using the Binomial Options Pricing Model.

    Parameters
    ----------
    spot : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The current price of the underlying asset, by default 100.0.
    strike : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The strike price of the option, by default 100.0.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool] | bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The risk-free interest rate, by default 0.05.
    div : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    precision : Literal["fp32", "fp64", "auto"], optional
        The floating-point precision to use for computation, by default "auto".

    Returns
    -------
    Tensor | float
        The gamma of the option.
    """
    ...

def theta(
    spot: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    strike: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool] | bool = True,
    ttm: Tensor | NDArray[fp64] | NDArray[fp32] | float = 1.0,
    iv: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.2,
    rfr: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.05,
    div: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.0,
    n_steps: int = 512,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tensor | float:
    """
    Computes the theta of an American option using the Binomial Options Pricing Model.

    Parameters
    ----------
    spot : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The current price of the underlying asset, by default 100.0.
    strike : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The strike price of the option, by default 100.0.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool] | bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The risk-free interest rate, by default 0.05.
    div : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    precision : Literal["fp32", "fp64", "auto"], optional
        The floating-point precision to use for computation, by default "auto".

    Returns
    -------
    Tensor | float
        The theta of the option.
    """
    ...

def vega(
    spot: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    strike: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool] | bool = True,
    ttm: Tensor | NDArray[fp64] | NDArray[fp32] | float = 1.0,
    iv: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.2,
    rfr: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.05,
    div: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.0,
    n_steps: int = 512,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tensor | float:
    """
    Computes the vega of an American option using the Binomial Options Pricing Model.

    Parameters
    ----------
    spot : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The current price of the underlying asset, by default 100.0.
    strike : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The strike price of the option, by default 100.0.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool] | bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The risk-free interest rate, by default 0.05.
    div : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    precision : Literal["fp32", "fp64", "auto"], optional
        The floating-point precision to use for computation, by default "auto".

    Returns
    -------
    Tensor | float
        The vega of the option.
    """
    ...

def rho(
    spot: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    strike: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool] | bool = True,
    ttm: Tensor | NDArray[fp64] | NDArray[fp32] | float = 1.0,
    iv: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.2,
    rfr: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.05,
    div: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.0,
    n_steps: int = 512,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tensor | float:
    """
    Computes the rho of an American option using the Binomial Options Pricing Model.

    Parameters
    ----------
    spot : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The current price of the underlying asset, by default 100.0.
    strike : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The strike price of the option, by default 100.0.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool] | bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The risk-free interest rate, by default 0.05.
    div : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    precision : Literal["fp32", "fp64", "auto"], optional
        The floating-point precision to use for computation, by default "auto".

    Returns
    -------
    Tensor | float
        The rho of the option.
    """
    ...

# iv _______________________________________________________________________________________________
def iv(
    price: Tensor | NDArray[fp64] | NDArray[fp32] | float = 10.0,
    spot: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    strike: Tensor | NDArray[fp64] | NDArray[fp32] | float = 100.0,
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool] | bool = True,
    ttm: Tensor | NDArray[fp64] | NDArray[fp32] | float = 1.0,
    rfr: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.05,
    div: Tensor | NDArray[fp64] | NDArray[fp32] | float = 0.0,
    n_steps: int = 512,
    tol: float = 1e-3,
    max_iter: int = 100,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tensor | float:
    """
    Computes the implied volatility of an American option using the Binomial Options Pricing Model.

    Parameters
    ----------
    price : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The price of the option, by default 10.0.
    spot : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The current price of the underlying asset, by default 100.0.
    strike : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The strike price of the option, by default 100.0.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool] | bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The time to maturity of the option in years, by default 1.0.
    rfr : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The risk-free interest rate, by default 0.05.
    div : Tensor | NDArray[fp64] | NDArray[fp32] | float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    tol : float, optional
        The tolerance for the implied volatility computation, by default 1e-3.
    max_iter : int, optional
        The maximum number of iterations for the implied volatility computation, by default 100.
    precision : Literal["fp32", "fp64", "auto"], optional
        The floating-point precision to use for computation, by default "auto".

    Returns
    -------
    Tensor | float
        The implied volatility of the option.
    """
    ...

# fp64 ---------------------------------------------------------------------------------------------
# price ____________________________________________________________________________________________
def price_fp64(
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    iv: float = 0.2,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
) -> float:
    """
    Computes the price of an American option using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    float
        The price of the option.
    """
    ...

def price_fp64_batch(
    spot: Tensor | NDArray[fp64],
    strike: Tensor | NDArray[fp64],
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool],
    ttm: Tensor | NDArray[fp64],
    iv: Tensor | NDArray[fp64],
    rfr: Tensor | NDArray[fp64],
    div: Tensor | NDArray[fp64],
    n_steps: int = 512,
    cuda: bool = False,
) -> Tensor:
    """
    Computes the price of a batch of American options using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    spot : Tensor | NDArray[fp64]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp64]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp64]
        The time to maturity of the option in years.
    iv : Tensor | NDArray[fp64]
        The implied volatility of the underlying asset.
    rfr : Tensor | NDArray[fp64]
        The risk-free interest rate.
    div : Tensor | NDArray[fp64]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    cuda : bool, optional
        A flag indicating whether to use CUDA for computation, by default False.

    Returns
    -------
    Tensor
        The price of the options.
    """
    ...

# greeks ___________________________________________________________________________________________
def greeks_fp64(
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    iv: float = 0.2,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
    delta: bool = False,
    gamma: bool = False,
    theta: bool = False,
    vega: bool = False,
    rho: bool = False,
) -> float:
    """
    Computes the Greeks of an American option using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    delta : bool, optional
        A flag indicating whether to compute the delta of the option, by default False.
    gamma : bool, optional
        A flag indicating whether to compute the gamma of the option, by default False.
    theta : bool, optional
        A flag indicating whether to compute the theta of the option, by default False.
    vega : bool, optional
        A flag indicating whether to compute the vega of the option, by default False.
    rho : bool, optional
        A flag indicating whether to compute the rho of the option, by default False.

    Returns
    -------
    float
        The computed Greeks.
    """
    ...

def delta_fp64(
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    iv: float = 0.2,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
) -> float:
    """
    Computes the delta of an American option using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    float
        The delta of the option.
    """
    ...

def gamma_fp64(
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    iv: float = 0.2,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
) -> float:
    """
    Computes the gamma of an American option using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    float
        The gamma of the option.
    """
    ...

def theta_fp64(
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    iv: float = 0.2,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
) -> float:
    """
    Computes the theta of an American option using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    float
        The theta of the option.
    """
    ...

def vega_fp64(
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    iv: float = 0.2,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
) -> float:
    """
    Computes the vega of an American option using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    float
        The vega of the option.
    """
    ...

def rho_fp64(
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    iv: float = 0.2,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
) -> float:
    """
    Computes the rho of an American option using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    float
        The rho of the option.
    """
    ...

def greeks_fp64_batch(
    spot: Tensor | NDArray[fp64],
    strike: Tensor | NDArray[fp64],
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool],
    ttm: Tensor | NDArray[fp64],
    iv: Tensor | NDArray[fp64],
    rfr: Tensor | NDArray[fp64],
    div: Tensor | NDArray[fp64],
    n_steps: int = 512,
    delta: bool = False,
    gamma: bool = False,
    theta: bool = False,
    vega: bool = False,
    rho: bool = False,
) -> Tensor | Tuple[Tensor, ...]:
    """
    Computes the Greeks of a batch of American options using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    spot : Tensor | NDArray[fp64]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp64]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp64]
        The time to maturity of the option in years.
    iv : Tensor | NDArray[fp64]
        The implied volatility of the underlying asset.
    rfr : Tensor | NDArray[fp64]
        The risk-free interest rate.
    div : Tensor | NDArray[fp64]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    delta : bool, optional
        A flag indicating whether to compute the delta of the option, by default False.
    gamma : bool, optional
        A flag indicating whether to compute the gamma of the option, by default False.
    theta : bool, optional
        A flag indicating whether to compute the theta of the option, by default False.
    vega : bool, optional
        A flag indicating whether to compute the vega of the option, by default False.
    rho : bool, optional
        A flag indicating whether to compute the rho of the option, by default False.

    Returns
    -------
    Tensor | Tuple[Tensor, ...]
        A tuple containing the computed Greeks.
    """
    ...

def delta_fp64_batch(
    spot: Tensor | NDArray[fp64],
    strike: Tensor | NDArray[fp64],
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool],
    ttm: Tensor | NDArray[fp64],
    iv: Tensor | NDArray[fp64],
    rfr: Tensor | NDArray[fp64],
    div: Tensor | NDArray[fp64],
    n_steps: int = 512,
) -> Tensor:
    """
    Computes the delta of a batch of American options using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    spot : Tensor | NDArray[fp64]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp64]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp64]
        The time to maturity of the option in years.
    iv : Tensor | NDArray[fp64]
        The implied volatility of the underlying asset.
    rfr : Tensor | NDArray[fp64]
        The risk-free interest rate.
    div : Tensor | NDArray[fp64]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    Tensor
        The delta of the options.
    """
    ...

def gamma_fp64_batch(
    spot: Tensor | NDArray[fp64],
    strike: Tensor | NDArray[fp64],
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool],
    ttm: Tensor | NDArray[fp64],
    iv: Tensor | NDArray[fp64],
    rfr: Tensor | NDArray[fp64],
    div: Tensor | NDArray[fp64],
    n_steps: int = 512,
) -> Tensor:
    """
    Computes the gamma of a batch of American options using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    spot : Tensor | NDArray[fp64]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp64]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp64]
        The time to maturity of the option in years.
    iv : Tensor | NDArray[fp64]
        The implied volatility of the underlying asset.
    rfr : Tensor | NDArray[fp64]
        The risk-free interest rate.
    div : Tensor | NDArray[fp64]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    Tensor
        The gamma of the options.
    """
    ...

def theta_fp64_batch(
    spot: Tensor | NDArray[fp64],
    strike: Tensor | NDArray[fp64],
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool],
    ttm: Tensor | NDArray[fp64],
    iv: Tensor | NDArray[fp64],
    rfr: Tensor | NDArray[fp64],
    div: Tensor | NDArray[fp64],
    n_steps: int = 512,
) -> Tensor:
    """
    Computes the theta of a batch of American options using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    spot : Tensor | NDArray[fp64]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp64]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp64]
        The time to maturity of the option in years.
    iv : Tensor | NDArray[fp64]
        The implied volatility of the underlying asset.
    rfr : Tensor | NDArray[fp64]
        The risk-free interest rate.
    div : Tensor | NDArray[fp64]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    Tensor
        The theta of the options.
    """
    ...

def vega_fp64_batch(
    spot: Tensor | NDArray[fp64],
    strike: Tensor | NDArray[fp64],
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool],
    ttm: Tensor | NDArray[fp64],
    iv: Tensor | NDArray[fp64],
    rfr: Tensor | NDArray[fp64],
    div: Tensor | NDArray[fp64],
    n_steps: int = 512,
) -> Tensor:
    """
    Computes the vega of a batch of American options using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    spot : Tensor | NDArray[fp64]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp64]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp64]
        The time to maturity of the option in years.
    iv : Tensor | NDArray[fp64]
        The implied volatility of the underlying asset.
    rfr : Tensor | NDArray[fp64]
        The risk-free interest rate.
    div : Tensor | NDArray[fp64]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    Tensor
        The vega of the options.
    """
    ...

def rho_fp64_batch(
    spot: Tensor | NDArray[fp64],
    strike: Tensor | NDArray[fp64],
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool],
    ttm: Tensor | NDArray[fp64],
    iv: Tensor | NDArray[fp64],
    rfr: Tensor | NDArray[fp64],
    div: Tensor | NDArray[fp64],
    n_steps: int = 512,
) -> Tensor:
    """
    Computes the rho of a batch of American options using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    spot : Tensor | NDArray[fp64]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp64]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp64]
        The time to maturity of the option in years.
    iv : Tensor | NDArray[fp64]
        The implied volatility of the underlying asset.
    rfr : Tensor | NDArray[fp64]
        The risk-free interest rate.
    div : Tensor | NDArray[fp64]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    Tensor
        The rho of the options.
    """
    ...

# iv _______________________________________________________________________________________________
def iv_fp64(
    price: float = 10.0,
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> float:
    """
    Computes the implied volatility of an American option using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    price : float, optional
        The price of the option, by default 10.0.
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    tol : float, optional
        The tolerance for the implied volatility computation, by default 1e-3.
    max_iter : int, optional
        The maximum number of iterations for the implied volatility computation, by default 100.

    Returns
    -------
    float
        The implied volatility of the option.
    """
    ...

def iv_fp64_batch(
    price: Tensor | NDArray[fp64],
    spot: Tensor | NDArray[fp64],
    strike: Tensor | NDArray[fp64],
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool],
    ttm: Tensor | NDArray[fp64],
    rfr: Tensor | NDArray[fp64],
    div: Tensor | NDArray[fp64],
    n_steps: int = 512,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> Tensor:
    """
    Computes the implied volatility of a batch of American options using the Binomial Options Pricing Model with fp64 precision.

    Parameters
    ----------
    price : Tensor | NDArray[fp64]
        The price of the option.
    spot : Tensor | NDArray[fp64]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp64]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp64]
        The time to maturity of the option in years.
    rfr : Tensor | NDArray[fp64]
        The risk-free interest rate.
    div : Tensor | NDArray[fp64]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    tol : float, optional
        The tolerance for the implied volatility computation, by default 1e-3.
    max_iter : int, optional
        The maximum number of iterations for the implied volatility computation, by default 100.

    Returns
    -------
    Tensor
        The implied volatility of the options.
    """
    ...

# fp32 ---------------------------------------------------------------------------------------------
# price ____________________________________________________________________________________________
def price_fp32(
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    iv: float = 0.2,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
) -> float:
    """
    Computes the price of an American option using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    float
        The price of the option.
    """
    ...

def price_fp32_batch(
    spot: Tensor | NDArray[fp32],
    strike: Tensor | NDArray[fp32],
    c_flag: Tensor | NDArray[uint8],
    ttm: Tensor | NDArray[fp32],
    iv: Tensor | NDArray[fp32],
    rfr: Tensor | NDArray[fp32],
    div: Tensor | NDArray[fp32],
    n_steps: int = 512,
    cuda: bool = False,
) -> Tensor:
    """
    Computes the price of a batch of American options using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    spot : Tensor | NDArray[fp32]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp32]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp32]
        The time to maturity of the option in years.
    iv : Tensor | NDArray[fp32]
        The implied volatility of the underlying asset.
    rfr : Tensor | NDArray[fp32]
        The risk-free interest rate.
    div : Tensor | NDArray[fp32]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    cuda : bool, optional
        A flag indicating whether to use CUDA for computation, by default False.

    Returns
    -------
    Tensor
        The price of the options.
    """
    ...

# greeks ___________________________________________________________________________________________
def greeks_fp32(
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    iv: float = 0.2,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
    delta: bool = False,
    gamma: bool = False,
    theta: bool = False,
    vega: bool = False,
    rho: bool = False,
) -> float:
    """
    Computes the Greeks of an American option using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    delta : bool, optional
        A flag indicating whether to compute the delta of the option, by default False.
    gamma : bool, optional
        A flag indicating whether to compute the gamma of the option, by default False.
    theta : bool, optional
        A flag indicating whether to compute the theta of the option, by default False.
    vega : bool, optional
        A flag indicating whether to compute the vega of the option, by default False.
    rho : bool, optional
        A flag indicating whether to compute the rho of the option, by default False.

    Returns
    -------
    float
        The computed Greeks.
    """
    ...

def delta_fp32(
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    iv: float = 0.2,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
) -> float:
    """
    Computes the delta of an American option using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    float
        The delta of the option.
    """
    ...

def gamma_fp32(
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    iv: float = 0.2,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
) -> float:
    """
    Computes the gamma of an American option using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    float
        The gamma of the option.
    """
    ...

def theta_fp32(
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    iv: float = 0.2,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
) -> float:
    """
    Computes the theta of an American option using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    float
        The theta of the option.
    """
    ...

def vega_fp32(
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    iv: float = 0.2,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
) -> float:
    """
    Computes the vega of an American option using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    float
        The vega of the option.
    """
    ...

def rho_fp32(
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    iv: float = 0.2,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
) -> float:
    """
    Computes the rho of an American option using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    iv : float, optional
        The implied volatility of the underlying asset, by default 0.2.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    float
        The rho of the option.
    """
    ...

def greeks_fp32_batch(
    spot: Tensor | NDArray[fp32],
    strike: Tensor | NDArray[fp32],
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool],
    ttm: Tensor | NDArray[fp32],
    iv: Tensor | NDArray[fp32],
    rfr: Tensor | NDArray[fp32],
    div: Tensor | NDArray[fp32],
    n_steps: int = 512,
    delta: bool = False,
    gamma: bool = False,
    theta: bool = False,
    vega: bool = False,
    rho: bool = False,
) -> Tensor | Tuple[Tensor, ...]:
    """
    Computes the Greeks of a batch of American options using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    spot : Tensor | NDArray[fp32]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp32]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp32]
        The time to maturity of the option in years.
    iv : Tensor | NDArray[fp32]
        The implied volatility of the underlying asset.
    rfr : Tensor | NDArray[fp32]
        The risk-free interest rate.
    div : Tensor | NDArray[fp32]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    delta : bool, optional
        A flag indicating whether to compute the delta of the option, by default False.
    gamma : bool, optional
        A flag indicating whether to compute the gamma of the option, by default False.
    theta : bool, optional
        A flag indicating whether to compute the theta of the option, by default False.
    vega : bool, optional
        A flag indicating whether to compute the vega of the option, by default False.
    rho : bool, optional
        A flag indicating whether to compute the rho of the option, by default False.

    Returns
    -------
    Tensor | Tuple[Tensor, ...]
        A tuple containing the computed Greeks.
    """
    ...

def delta_fp32_batch(
    spot: Tensor | NDArray[fp32],
    strike: Tensor | NDArray[fp32],
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool],
    ttm: Tensor | NDArray[fp32],
    iv: Tensor | NDArray[fp32],
    rfr: Tensor | NDArray[fp32],
    div: Tensor | NDArray[fp32],
    n_steps: int = 512,
) -> Tensor:
    """
    Computes the delta of a batch of American options using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    spot : Tensor | NDArray[fp32]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp32]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp32]
        The time to maturity of the option in years.
    iv : Tensor | NDArray[fp32]
        The implied volatility of the underlying asset.
    rfr : Tensor | NDArray[fp32]
        The risk-free interest rate.
    div : Tensor | NDArray[fp32]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    Tensor
        The delta of the options.
    """
    ...

def gamma_fp32_batch(
    spot: Tensor | NDArray[fp32],
    strike: Tensor | NDArray[fp32],
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool],
    ttm: Tensor | NDArray[fp32],
    iv: Tensor | NDArray[fp32],
    rfr: Tensor | NDArray[fp32],
    div: Tensor | NDArray[fp32],
    n_steps: int = 512,
) -> Tensor:
    """
    Computes the gamma of a batch of American options using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    spot : Tensor | NDArray[fp32]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp32]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp32]
        The time to maturity of the option in years.
    iv : Tensor | NDArray[fp32]
        The implied volatility of the underlying asset.
    rfr : Tensor | NDArray[fp32]
        The risk-free interest rate.
    div : Tensor | NDArray[fp32]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    Tensor
        The gamma of the options.
    """
    ...

def theta_fp32_batch(
    spot: Tensor | NDArray[fp32],
    strike: Tensor | NDArray[fp32],
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool],
    ttm: Tensor | NDArray[fp32],
    iv: Tensor | NDArray[fp32],
    rfr: Tensor | NDArray[fp32],
    div: Tensor | NDArray[fp32],
    n_steps: int = 512,
) -> Tensor:
    """
    Computes the theta of a batch of American options using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    spot : Tensor | NDArray[fp32]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp32]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp32]
        The time to maturity of the option in years.
    iv : Tensor | NDArray[fp32]
        The implied volatility of the underlying asset.
    rfr : Tensor | NDArray[fp32]
        The risk-free interest rate.
    div : Tensor | NDArray[fp32]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    Tensor
        The theta of the options.
    """
    ...

def vega_fp32_batch(
    spot: Tensor | NDArray[fp32],
    strike: Tensor | NDArray[fp32],
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool],
    ttm: Tensor | NDArray[fp32],
    iv: Tensor | NDArray[fp32],
    rfr: Tensor | NDArray[fp32],
    div: Tensor | NDArray[fp32],
    n_steps: int = 512,
) -> Tensor:
    """
    Computes the vega of a batch of American options using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    spot : Tensor | NDArray[fp32]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp32]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp32]
        The time to maturity of the option in years.
    iv : Tensor | NDArray[fp32]
        The implied volatility of the underlying asset.
    rfr : Tensor | NDArray[fp32]
        The risk-free interest rate.
    div : Tensor | NDArray[fp32]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    Tensor
        The vega of the options.
    """
    ...

def rho_fp32_batch(
    spot: Tensor | NDArray[fp32],
    strike: Tensor | NDArray[fp32],
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool],
    ttm: Tensor | NDArray[fp32],
    iv: Tensor | NDArray[fp32],
    rfr: Tensor | NDArray[fp32],
    div: Tensor | NDArray[fp32],
    n_steps: int = 512,
) -> Tensor:
    """
    Computes the rho of a batch of American options using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    spot : Tensor | NDArray[fp32]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp32]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp32]
        The time to maturity of the option in years.
    iv : Tensor | NDArray[fp32]
        The implied volatility of the underlying asset.
    rfr : Tensor | NDArray[fp32]
        The risk-free interest rate.
    div : Tensor | NDArray[fp32]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.

    Returns
    -------
    Tensor
        The rho of the options.
    """
    ...

# iv _______________________________________________________________________________________________
def iv_fp32(
    price: float = 10.0,
    spot: float = 100.0,
    strike: float = 100.0,
    c_flag: bool = True,
    ttm: float = 1.0,
    rfr: float = 0.05,
    div: float = 0.0,
    n_steps: int = 512,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> float:
    """
    Computes the implied volatility of an American option using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    price : float, optional
        The price of the option, by default 10.0.
    spot : float, optional
        The current price of the underlying asset, by default 100.0.
    strike : float, optional
        The strike price of the option, by default 100.0.
    c_flag : bool, optional
        A flag indicating whether the option is a call (True) or a put (False), by default True.
    ttm : float, optional
        The time to maturity of the option in years, by default 1.0.
    rfr : float, optional
        The risk-free interest rate, by default 0.05.
    div : float, optional
        The dividend yield of the underlying asset, by default 0.0.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    tol : float, optional
        The tolerance for the implied volatility computation, by default 1e-3.
    max_iter : int, optional
        The maximum number of iterations for the implied volatility computation, by default 100.

    Returns
    -------
    float
        The implied volatility of the option.
    """
    ...

def iv_fp32_batch(
    price: Tensor | NDArray[fp32],
    spot: Tensor | NDArray[fp32],
    strike: Tensor | NDArray[fp32],
    c_flag: Tensor | NDArray[uint8] | NDArray[npbool],
    ttm: Tensor | NDArray[fp32],
    rfr: Tensor | NDArray[fp32],
    div: Tensor | NDArray[fp32],
    n_steps: int = 512,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> Tensor:
    """
    Computes the implied volatility of a batch of American options using the Binomial Options Pricing Model with fp32 precision.

    Parameters
    ----------
    price : Tensor | NDArray[fp32]
        The price of the option.
    spot : Tensor | NDArray[fp32]
        The current price of the underlying asset.
    strike : Tensor | NDArray[fp32]
        The strike price of the option.
    c_flag : Tensor | NDArray[uint8] | NDArray[npbool]
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor | NDArray[fp32]
        The time to maturity of the option in years.
    rfr : Tensor | NDArray[fp32]
        The risk-free interest rate.
    div : Tensor | NDArray[fp32]
        The dividend yield of the underlying asset.
    n_steps : int, optional
        The number of steps in the binomial tree, by default 512.
    tol : float, optional
        The tolerance for the implied volatility computation, by default 1e-3.
    max_iter : int, optional
        The maximum number of iterations for the implied volatility computation, by default 100.

    Returns
    -------
    Tensor
        The implied volatility of the options.
    """
    ...
