"""
Fastvol: High-performance option pricing and volatility modeling library.

Author: Valerio Galanti
Version: 0.1.0
License: MIT License

Fastvol provides ultra-low-latency derivatives pricing and Greeks
on CPU (OpenMP/SIMD), GPU (CUDA), and via neural network surrogates.

Documentation:
    https://github.com/valeriogalanti/fastvol
"""

import torch
from ._core import c_version as version
from ._core import c_cuda_available as cuda_available
from . import utils

from . import european
from . import american

__version__ = version()

__all__ = [
    "version",
    "cuda_available",
    "utils",
    "american",
    "european",
]
