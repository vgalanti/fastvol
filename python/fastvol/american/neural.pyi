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
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tensor | float:
    """
    Computes the price of an American option using a neural network surrogate model.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.
    precision : Literal["fp32", "fp64", "auto"], optional
        The floating-point precision to use for computation, by default "auto".

    Returns
    -------
    Tensor | float
        The price of the option. Returns a float for a single option, a Tensor for a batch.
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
    delta: bool = False,
    gamma: bool = False,
    theta: bool = False,
    vega: bool = False,
    rho: bool = False,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tuple[Tensor, ...] | Tensor | float:
    """
    Computes the Greeks of an American option using a neural network surrogate model and autograd.

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
    delta : bool, optional
        Flag to compute delta, by default False.
    gamma : bool, optional
        Flag to compute gamma, by default False.
    theta : bool, optional
        Flag to compute theta, by default False.
    vega : bool, optional
        Flag to compute vega, by default False.
    rho : bool, optional
        Flag to compute rho, by default False.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.
    precision : Literal["fp32", "fp64", "auto"], optional
        The floating-point precision to use for computation, by default "auto".

    Returns
    -------
    Tuple[Tensor, ...] | Tensor | float
        A tuple containing the computed Greeks. Returns a single value if only one greek is requested.
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
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tensor | float:
    """
    Computes the delta of an American option using a neural network surrogate model.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.
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
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tensor | float:
    """
    Computes the gamma of an American option using a neural network surrogate model.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.
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
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tensor | float:
    """
    Computes the theta of an American option using a neural network surrogate model.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.
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
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tensor | float:
    """
    Computes the vega of an American option using a neural network surrogate model.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.
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
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tensor | float:
    """
    Computes the rho of an American option using a neural network surrogate model.

    Parameters
    ----------
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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.
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
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tensor | float:
    """
    Computes the implied volatility of an American option using a neural network surrogate model.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.
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
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the price of an American option using a neural network surrogate model with fp64 precision.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

    Returns
    -------
    float
        The price of the option.
    """
    ...

def price_fp64_batch(
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    iv: Tensor,
    rfr: Tensor,
    div: Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor:
    """
    Computes the price of a batch of American options using a neural network surrogate model with fp64 precision.

    Parameters
    ----------
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    iv : Tensor
        The implied volatility of the underlying asset.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

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
    delta: bool = False,
    gamma: bool = False,
    theta: bool = False,
    vega: bool = False,
    rho: bool = False,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the Greeks of an American option using a neural network surrogate model with fp64 precision.

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
    delta : bool, optional
        Flag to compute delta, by default False.
    gamma : bool, optional
        Flag to compute gamma, by default False.
    theta : bool, optional
        Flag to compute theta, by default False.
    vega : bool, optional
        Flag to compute vega, by default False.
    rho : bool, optional
        Flag to compute rho, by default False.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

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
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the delta of an American option using a neural network surrogate model with fp64 precision.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

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
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the gamma of an American option using a neural network surrogate model with fp64 precision.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

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
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the theta of an American option using a neural network surrogate model with fp64 precision.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

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
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the vega of an American option using a neural network surrogate model with fp64 precision.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

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
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the rho of an American option using a neural network surrogate model with fp64 precision.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

    Returns
    -------
    float
        The rho of the option.
    """
    ...

def greeks_fp64_batch(
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    iv: Tensor,
    rfr: Tensor,
    div: Tensor,
    delta: bool = False,
    gamma: bool = False,
    theta: bool = False,
    vega: bool = False,
    rho: bool = False,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor | Tuple[Tensor, ...]:
    """
    Computes the Greeks of a batch of American options using a neural network surrogate model with fp64 precision.

    Parameters
    ----------
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    iv : Tensor
        The implied volatility of the underlying asset.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    delta : bool, optional
        Flag to compute delta, by default False.
    gamma : bool, optional
        Flag to compute gamma, by default False.
    theta : bool, optional
        Flag to compute theta, by default False.
    vega : bool, optional
        Flag to compute vega, by default False.
    rho : bool, optional
        Flag to compute rho, by default False.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

    Returns
    -------
    Tensor | Tuple[Tensor, ...]
        A tuple containing the computed Greeks.
    """
    ...

def delta_fp64_batch(
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    iv: Tensor,
    rfr: Tensor,
    div: Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor:
    """
    Computes the delta of a batch of American options using a neural network surrogate model with fp64 precision.

    Parameters
    ----------
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    iv : Tensor
        The implied volatility of the underlying asset.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

    Returns
    -------
    Tensor
        The delta of the options.
    """
    ...

def gamma_fp64_batch(
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    iv: Tensor,
    rfr: Tensor,
    div: Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor:
    """
    Computes the gamma of a batch of American options using a neural network surrogate model with fp64 precision.

    Parameters
    ----------
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    iv : Tensor
        The implied volatility of the underlying asset.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

    Returns
    -------
    Tensor
        The gamma of the options.
    """
    ...

def theta_fp64_batch(
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    iv: Tensor,
    rfr: Tensor,
    div: Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor:
    """
    Computes the theta of a batch of American options using a neural network surrogate model with fp64 precision.

    Parameters
    ----------
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    iv : Tensor
        The implied volatility of the underlying asset.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

    Returns
    -------
    Tensor
        The theta of the options.
    """
    ...

def vega_fp64_batch(
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    iv: Tensor,
    rfr: Tensor,
    div: Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor:
    """
    Computes the vega of a batch of American options using a neural network surrogate model with fp64 precision.

    Parameters
    ----------
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    iv : Tensor
        The implied volatility of the underlying asset.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

    Returns
    -------
    Tensor
        The vega of the options.
    """
    ...

def rho_fp64_batch(
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    iv: Tensor,
    rfr: Tensor,
    div: Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor:
    """
    Computes the rho of a batch of American options using a neural network surrogate model with fp64 precision.

    Parameters
    ----------
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    iv : Tensor
        The implied volatility of the underlying asset.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

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
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the implied volatility of an American option using a neural network surrogate model with fp64 precision.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

    Returns
    -------
    float
        The implied volatility of the option.
    """
    ...

def iv_fp64_batch(
    price: Tensor,
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    rfr: Tensor,
    div: Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor:
    """
    Computes the implied volatility of a batch of American options using a neural network surrogate model with fp64 precision.

    Parameters
    ----------
    price : Tensor
        The price of the option.
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

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
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the price of an American option using a neural network surrogate model with fp32 precision.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

    Returns
    -------
    float
        The price of the option.
    """
    ...

def price_fp32_batch(
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    iv: Tensor,
    rfr: Tensor,
    div: Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor:
    """
    Computes the price of a batch of American options using a neural network surrogate model with fp32 precision.

    Parameters
    ----------
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    iv : Tensor
        The implied volatility of the underlying asset.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

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
    delta: bool = False,
    gamma: bool = False,
    theta: bool = False,
    vega: bool = False,
    rho: bool = False,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the Greeks of an American option using a neural network surrogate model with fp32 precision.

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
    delta : bool, optional
        Flag to compute delta, by default False.
    gamma : bool, optional
        Flag to compute gamma, by default False.
    theta : bool, optional
        Flag to compute theta, by default False.
    vega : bool, optional
        Flag to compute vega, by default False.
    rho : bool, optional
        Flag to compute rho, by default False.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

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
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the delta of an American option using a neural network surrogate model with fp32 precision.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

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
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the gamma of an American option using a neural network surrogate model with fp32 precision.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

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
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the theta of an American option using a neural network surrogate model with fp32 precision.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

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
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the vega of an American option using a neural network surrogate model with fp32 precision.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

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
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the rho of an American option using a neural network surrogate model with fp32 precision.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

    Returns
    -------
    float
        The rho of the option.
    """
    ...

def greeks_fp32_batch(
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    iv: Tensor,
    rfr: Tensor,
    div: Tensor,
    delta: bool = False,
    gamma: bool = False,
    theta: bool = False,
    vega: bool = False,
    rho: bool = False,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor | Tuple[Tensor, ...]:
    """
    Computes the Greeks of a batch of American options using a neural network surrogate model with fp32 precision.

    Parameters
    ----------
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    iv : Tensor
        The implied volatility of the underlying asset.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    delta : bool, optional
        Flag to compute delta, by default False.
    gamma : bool, optional
        Flag to compute gamma, by default False.
    theta : bool, optional
        Flag to compute theta, by default False.
    vega : bool, optional
        Flag to compute vega, by default False.
    rho : bool, optional
        Flag to compute rho, by default False.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

    Returns
    -------
    Tensor | Tuple[Tensor, ...]
        A tuple containing the computed Greeks.
    """
    ...

def delta_fp32_batch(
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    iv: Tensor,
    rfr: Tensor,
    div: Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor:
    """
    Computes the delta of a batch of American options using a neural network surrogate model with fp32 precision.

    Parameters
    ----------
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    iv : Tensor
        The implied volatility of the underlying asset.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

    Returns
    -------
    Tensor
        The delta of the options.
    """
    ...

def gamma_fp32_batch(
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    iv: Tensor,
    rfr: Tensor,
    div: Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor:
    """
    Computes the gamma of a batch of American options using a neural network surrogate model with fp32 precision.

    Parameters
    ----------
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    iv : Tensor
        The implied volatility of the underlying asset.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

    Returns
    -------
    Tensor
        The gamma of the options.
    """
    ...

def theta_fp32_batch(
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    iv: Tensor,
    rfr: Tensor,
    div: Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor:
    """
    Computes the theta of a batch of American options using a neural network surrogate model with fp32 precision.

    Parameters
    ----------
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    iv : Tensor
        The implied volatility of the underlying asset.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

    Returns
    -------
    Tensor
        The theta of the options.
    """
    ...

def vega_fp32_batch(
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    iv: Tensor,
    rfr: Tensor,
    div: Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor:
    """
    Computes the vega of a batch of American options using a neural network surrogate model with fp32 precision.

    Parameters
    ----------
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    iv : Tensor
        The implied volatility of the underlying asset.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

    Returns
    -------
    Tensor
        The vega of the options.
    """
    ...

def rho_fp32_batch(
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    iv: Tensor,
    rfr: Tensor,
    div: Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor:
    """
    Computes the rho of a batch of American options using a neural network surrogate model with fp32 precision.

    Parameters
    ----------
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    iv : Tensor
        The implied volatility of the underlying asset.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

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
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Computes the implied volatility of an American option using a neural network surrogate model with fp32 precision.

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
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").

    Returns
    -------
    float
        The implied volatility of the option.
    """
    ...

def iv_fp32_batch(
    price: Tensor,
    spot: Tensor,
    strike: Tensor,
    c_flag: Tensor,
    ttm: Tensor,
    rfr: Tensor,
    div: Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
) -> Tensor:
    """
    Computes the implied volatility of a batch of American options using a neural network surrogate model with fp32 precision.

    Parameters
    ----------
    price : Tensor
        The price of the option.
    spot : Tensor
        The current price of the underlying asset.
    strike : Tensor
        The strike price of the option.
    c_flag : Tensor
        A flag indicating whether the option is a call (True) or a put (False).
    ttm : Tensor
        The time to maturity of the option in years.
    rfr : Tensor
        The risk-free interest rate.
    div : Tensor
        The dividend yield of the underlying asset.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for neural network inference, by default 65_536.

    Returns
    -------
    Tensor
        The implied volatility of the options.
    """
    ...
