import torch

import numpy as np

from torch import Tensor
from numpy import ndarray
from numpy import float64 as fp64
from numpy import float32 as fp32
from numpy import uint8 as uint8
from numpy import bool_ as npbool
from typing import Tuple, Literal
from numpy.typing import NDArray

from ._core import c_cuda_available

CUDA_AVAILABLE = c_cuda_available()


def align_args(
    *args: Tensor
    | NDArray[fp64]
    | NDArray[fp32]
    | NDArray[npbool]
    | NDArray[uint8]
    | float
    | int
    | bool,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
    cuda: bool = False,
    promote_all: bool = False,
) -> (
    Tensor
    | NDArray[fp64]
    | NDArray[fp32]
    | NDArray[uint8]
    | float
    | bool
    | Tuple[
        Tensor | NDArray[fp64] | NDArray[fp32] | NDArray[uint8] | float | bool,
        ...,
    ]
):
    """
    Aligns a set of scalar and/or array arguments to consistent shape and dtype.

    Scalars are promoted to 1D Numpy arrays (if any input is an array),
    ensuring all non-bool values have matching floating point dtype and shape.

    Parameters
    ----------
    *args : NDArray[fp64] | NDArray[fp32] | NDArray[npbool] | float | int | bool
        The values to align. Scalars will be promoted to arrays if any input is an array.
        Booleans are preserved as bool scalars or np.bool_ arrays.

    cuda : bool
        Whether the target function is on cuda. Determines whether gpu tensors are copied
        back to CPU or CPU tensors are promoted to on-device if other tensors already are.

    precision : {'fp32', 'fp64', 'auto'}, default='auto'
        Desired precision for float/int values.
        - 'auto': infer dtype from np.result_type on all inputs
        - 'fp32': force all float/int values to float32
        - 'fp64': force all float/int values to float64

    promote_all : bool, defaults to False.
        Whether to force-promote all inputs to numpy arrays.

    Returns
    -------
    Tuple of aligned arguments.
        - If all inputs are scalars, returns scalars (floats or bools)
        - If any input is an array, returns 1D arrays of matching shape and dtype

    Raises
    ------
    ValueError
        - If arrays have inconsistent shapes
        - If any input has more than 1 dimension
        - If resolved numpy dtype is not fp32 or fp64
    TypeError
        - If any input is of invalid type

    Examples
    --------
    >>> align_args(True, 100, 110, 0.2)
    [True, 100.0, 110.0, 0.2]

    >>> align_args(np.array([True, False]), 100, 110, 0.2)
    [array([ True, False]), array([100., 100.]), array([110., 110.]), array([0.2, 0.2])]
    """

    cuda = cuda and CUDA_AVAILABLE
    promote = promote_all

    n = None
    dtype = "fp32"
    mem = "host"
    device = "cpu"

    # initial scan for types, lengths, devices etc.
    for x in args:
        t = type(x)

        if t in (float, int, bool):
            continue

        elif t not in (ndarray, Tensor):
            raise TypeError(f"Unsupported argument type {type(x)}.")

        # ndarray or tensor
        promote = True
        n = x.shape[0] if n is None else n

        if x.shape[0] != n:
            raise ValueError(f"Arguments with different shapes ({n},) and {x.shape}.")

        if len(x.shape) > 1:
            raise ValueError(f"Argument has shape {x.shape}.")

        if x.dtype in (np.float64, torch.float64):
            dtype = "fp64"

        if t == Tensor:
            if x.is_cpu:
                if x.is_pinned() and mem == "host":
                    mem = "pinned"

            else:
                mem = "gpu"
                if device == "cpu":
                    device = x.device
                elif device != x.device:
                    raise ValueError(
                        f"Torch tensors on mixed non-cpu devices: {x.device} vs {device}."
                    )

    # scalar case
    if not promote:
        return tuple(x if type(x) == bool else float(x) for x in args)
    if n is None:
        n = 1

    # determine dtype based on precision parameter, don't change if auto
    if precision == "fp64":
        dtype = "fp64"
    elif precision == "fp32":
        dtype = "fp32"

    # align args
    aligned = []

    pt_t = torch.float64 if dtype == "fp64" else torch.float32
    np_t = np.float64 if dtype == "fp64" else np.float32

    for x in args:
        t = type(x)

        if t in (float, int, bool):
            x = torch.full((n,), x, dtype=pt_t if type(x) != bool else torch.uint8)

        elif t == ndarray:
            x = x.astype(np_t) if x.dtype not in (npbool, np.uint8) else x

            if cuda and mem != "host":
                x = torch.from_numpy(x)
            else:
                x = np.ascontiguousarray(x)

        else:
            x = x.to(
                dtype=pt_t if x.dtype not in (torch.bool, torch.uint8) else torch.uint8
            )
            x = x.contiguous()

        if cuda and mem == "pinned":
            x = x.pin_memory()

        if cuda and mem == "gpu":
            x = x.to(device=device)

        aligned.append(x)

    return aligned[0] if len(aligned) == 1 else tuple(aligned)


def get_ptrs(
    *args: Tensor | NDArray[fp64] | NDArray[fp32] | NDArray[npbool] | NDArray[uint8],
) -> int | Tuple[int, ...]:
    """
    Get the data pointers of a set of arrays.

    Parameters
    ----------
    *args : Tensor | NDArray[fp64] | NDArray[fp32] | NDArray[npbool] | NDArray[uint8]
        The arrays to get the pointers of.

    Returns
    -------
    int | Tuple[int, ...]
        The data pointers of the arrays.

    Raises
    ------
    TypeError
        If any input is of invalid type.
    """
    ptrs = []
    for x in args:
        t = type(x)
        if t == ndarray:
            ptrs.append(x.__array_interface__["data"][0])
        elif t == Tensor:
            ptrs.append(x.data_ptr())
        else:
            raise TypeError(f"Unsupported argument type {t}.")

    return ptrs[0] if len(ptrs) == 1 else tuple(ptrs)


def mk_out_like(
    arr: Tensor | NDArray[fp64] | NDArray[fp32] | NDArray[npbool] | NDArray[uint8],
) -> Tensor:
    """
    Create an empty output tensor with the same properties as the input array.

    Parameters
    ----------
    arr : Tensor | NDArray[fp64] | NDArray[fp32] | NDArray[npbool] | NDArray[uint8]
        The input array to match.

    Returns
    -------
    Tensor
        An empty tensor with the same shape, dtype, and device as the input.
    """
    if type(arr) == ndarray:
        out = np.empty_like(arr)
        return torch.from_numpy(out)

    out = torch.empty_like(arr)
    if arr.is_pinned():
        out.pin_memory()
    return out


def cu_params(
    arr: Tensor | NDArray[fp64] | NDArray[fp32] | NDArray[npbool] | NDArray[uint8],
) -> Tuple:
    """
    Get the CUDA parameters of an array for fastvol lib.

    Parameters
    ----------
    arr : Tensor | NDArray[fp64] | NDArray[fp32] | NDArray[npbool] | NDArray[uint8]
        The array to get the parameters of.

    Returns
    -------
    Tuple
        A tuple containing the CUDA parameters.
    """
    if type(arr) == ndarray:
        return (0, 0, False, False, True)

    if arr.is_pinned():
        return (0, 0, False, True, True)

    if arr.is_cuda:
        return (0, 0, True, False, True)

    return (0, 0, False, False, True)


def uniform(
    n: int, low: float, high: float, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Sample n values uniformly from [low, high) in the specified dtype and device."""
    return torch.rand((n,), dtype=dtype, device=device) * (high - low) + low


def generate_data(
    n_samples: int = 100_000,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    spot_min: float = 50.0,
    spot_max: float = 150.0,
    strike_min: float = 50.0,
    strike_max: float = 150.0,
    ttm_min: float = 0.001,
    ttm_max: float = 2.0,
    iv_min: float = 0.05,
    iv_max: float = 2.0,
    rfr_min: float = 0.0,
    rfr_max: float = 0.15,
    div_min: float = 0.0,
    div_max: float = 0.05,
) -> Tuple[torch.Tensor, ...]:
    """
    Generates sample market data as torch tensors, uniformly distributed
    across the specified input ranges.

    Returns
    -------
    Tuple[torch.Tensor, ...]
        (spot, strike, c_flag, ttm, iv, rfr, div)
    """

    spot = uniform(n_samples, spot_min, spot_max, dtype=dtype, device=device)
    strike = uniform(n_samples, strike_min, strike_max, dtype=dtype, device=device)
    ttm = uniform(n_samples, ttm_min, ttm_max, dtype=dtype, device=device)
    iv = uniform(n_samples, iv_min, iv_max, dtype=dtype, device=device)
    rfr = uniform(n_samples, rfr_min, rfr_max, dtype=dtype, device=device)
    div = uniform(n_samples, div_min, div_max, dtype=dtype, device=device)

    c_flag = torch.randint(0, 2, (n_samples,), dtype=torch.uint8, device=device)

    return spot, strike, c_flag, ttm, iv, rfr, div
