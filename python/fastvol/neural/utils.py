import torch
import torch.nn as nn
import numpy as np
import logging

from numpy import ndarray
from numpy import float64 as fp64
from numpy import float32 as fp32
from numpy import bool_ as npbool
from numpy import uint8
from numpy.typing import NDArray
from torch import Tensor
from typing import Tuple, Literal, Dict, Type
from pathlib import Path
from collections import OrderedDict
from torch.autograd import grad

from ..utils import align_args

log = logging.getLogger(__name__)

_config = {"torch_compilation": True}


def load_net(
    name: str, cls: Type[nn.Module], torch_compile: bool = True
) -> Tuple[nn.Module, OrderedDict]:
    """
    Load and optionally compile a pretrained model from disk.

    Parameters
    ----------
    name : str
        The name of the model to load.
    cls : Type[nn.Module]
        The model class to instantiate.
    torch_compile : bool, optional
        Whether to compile the model with torch.compile, by default True.

    Returns
    -------
    Tuple[nn.Module, OrderedDict]
        A tuple containing the loaded model and its state dictionary.
    """
    model = cls().double().eval()
    fpath = (Path(__file__).parent / "checkpoints" / f"{name}.pt").resolve()

    try:
        state = torch.load(fpath)
        model.load_state_dict(state)
    except Exception as e:
        log.warning(
            f"[fastvol] unable to load checkpoint from: {fpath} ({type(e).__name__})"
        )
        state = model.state_dict()

    model.load_state_dict(state)

    if not _config["torch_compilation"] or not torch_compile:
        return model, state

    try:
        model = torch.compile(model)
        o = torch.ones(1, dtype=torch.float64)
        _ = model(o, o, o, o, o, o, o)
        state = model.state_dict()

    except Exception as e:
        log.warning(f"[fastvol] torch.compile() not supported: {type(e).__name__}")
        log.warning(f"[fastvol] disabling compilation...")
        _config["torch_compilation"] = False

        # clean reload
        model = cls().double().eval()
        model.load_state_dict(state)

    return model, state


def save_net(name: str, model: nn.Module) -> None:
    fpath = (Path(__file__).parent / "checkpoints" / f"{name}.pt").resolve()
    torch.save(model.to(device="cpu").state_dict(), fpath)


def _model_device_dtype(model: nn.Module) -> Tuple[torch.device, torch.dtype]:
    """
    Get the device and dtype of a model.

    Parameters
    ----------
    model : nn.Module
        The model to inspect.

    Returns
    -------
    Tuple[torch.device, torch.dtype]
        The device and dtype of the model.
    """
    for p in model.parameters():
        return p.device, p.dtype
    for b in model.buffers():
        return p.device, b.dtype
    raise ValueError("Model has no parameters or buffers.")


def run_net(
    model: nn.Module,
    model_w: OrderedDict,
    inputs: dict[str, Tensor],
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
    inputs_require_grad: Tuple[str] = (),
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Run batched model inference with autograd support.

    Parameters
    ----------
    model : nn.Module
        The model to run.
    model_w : OrderedDict
        The model's state dictionary.
    inputs : dict[str, Tensor]
        A dictionary of input tensors.
    device : torch.device, optional
        The device to run the computation on, by default torch.device("cpu").
    batch_size : int, optional
        The batch size for inference, by default 65_536.
    inputs_require_grad : Tuple[str], optional
        A tuple of input keys that require gradients, by default ().

    Returns
    -------
    Tuple[Tensor, Dict[str, Tensor]]
        A tuple containing the output tensor and the dictionary of input tensors.
    """
    requires_grad = len(inputs_require_grad)
    dtype = inputs["spot"].dtype
    m_device, m_dtype = _model_device_dtype(model)

    if device != m_device:
        model = model.to(device=device)

    if dtype != m_dtype:
        model = model.to(dtype=dtype)
        model.load_state_dict(model_w)

    tensors = {}
    for k, v in inputs.items():
        v = v.detach()
        if k in inputs_require_grad:
            v = v.requires_grad_(True)
        tensors[k] = v

    N = list(tensors.values())[0].shape[0]
    preds = []

    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        for i in range(0, N, batch_size):
            sl = slice(i, i + batch_size)
            batch = {k: v[sl].to(device) for k, v in tensors.items()}
            out = model(**batch)
            preds.append(out.cpu())
            del batch

    return torch.cat(preds), tensors


def get_greeks(
    model: nn.Module,
    model_w: OrderedDict,
    inputs: dict[str, Tensor],
    device: torch.device = torch.device("cpu"),
    batch_size: int = 65_536,
    delta=False,
    gamma=False,
    theta=False,
    vega=False,
    rho=False,
) -> Tuple[Tensor, ...]:
    """
    Compute selected greeks using autograd. Returned in order:
    (delta, gamma, theta, vega, rho) for those requested.

    Parameters
    ----------
    price : Tensor
        Output price tensor from the model.
    tensors : dict
        Dictionary of input tensors ("spot", "ttm", "iv", "rfr", ...).
    delta, gamma, theta, vega, rho : bool
        Flags indicating which greeks to compute.

    Returns
    -------
    tuple
        A tuple of tensors corresponding to the requested greeks, in
        the order: delta, gamma, theta, vega, rho.
    """
    if not (delta or gamma or theta or vega or rho):
        return

    req_grad = []
    if delta or gamma:
        req_grad.append("spot")
    if theta:
        req_grad.append("ttm")
    if vega:
        req_grad.append("iv")
    if rho:
        req_grad.append("rfr")

    price, tensors = run_net(
        model,
        model_w,
        inputs,
        device=device,
        batch_size=batch_size,
        inputs_require_grad=req_grad,
    )

    req_tensors = [tensors[k] for k in req_grad]

    grads = grad(
        price,
        req_tensors,
        grad_outputs=torch.ones_like(price),
        create_graph=gamma,
        retain_graph=gamma,
    )

    grad_map = dict(zip(req_grad, grads))
    greeks = []

    if delta:
        greeks.append(grad_map["spot"].detach())

    if gamma:
        delta_tensor = grad_map.get("spot")
        gamma_tensor = grad(delta_tensor.sum(), tensors["spot"], create_graph=False)[0]
        greeks.append(gamma_tensor.detach())

    if theta:
        greeks.append(-grad_map["ttm"].detach() / 365.0)

    if vega:
        greeks.append(grad_map["iv"].detach() / 100.0)

    if rho:
        greeks.append(grad_map["rfr"].detach() / 100.0)

    return greeks[0] if len(greeks) == 1 else tuple(greeks)


def align_args_nn(
    *args: Tensor
    | NDArray[fp64]
    | NDArray[fp32]
    | NDArray[npbool]
    | NDArray[uint8]
    | float
    | int
    | bool,
    precision: Literal["fp32", "fp64", "auto"] = "auto",
) -> Tuple[Tensor, ...]:
    """
    Aligns a set of scalar and/or array arguments to torch Tensors of consistent
    shape and dtype.

    Parameters
    ----------
    *args : NDArray[fp64] | NDArray[fp32] | NDArray[npbool] | float | int | bool
        The values to align and convert to torch Tensors.

    precision : {'fp32', 'fp64', 'auto'}, default='auto'
        - 'auto': infer dtype
        - 'fp32': force all values to float32
        - 'fp64': force all values to float64

    Returns
    -------
    Tuple of aligned arguments.
        - If all inputs are scalars, returns scalars (floats or bools)
        - If any input is an array, returns 1D arrays of matching shape and dtype
    """

    aligned = align_args(*args, precision=precision, cuda=True, promote_all=True)

    to_fp32 = aligned[0].dtype in (torch.float32, np.float32)

    aligned = [
        (torch.from_numpy(x) if type(x) == ndarray else x).to(
            dtype=torch.float32 if to_fp32 else torch.float64
        )
        for x in aligned
    ]
    return aligned
