# Fastvol

Fastvol is a high-performance option pricing library for low-latency, high-throughput derivatives pricing. 

It provides fast and accurate pricing, greeks, and implied volatility inversion for European and American vanilla options across optimized CPU, CUDA, and neural network backends. Fastvol includes SIMD-vectorized and OpenMP-parallelized CPU implementations, batched CUDA kernels for GPU acceleration, and neural surrogates built with PyTorch. It exposes a portable C FFI for multi-language integration, and ships with a lightweight Python wrapper via Cython, available on PyPI.

Most open-source libraries focus on European options with closed-form solutions, and offer only basic or slow methods for American-style pricing — falling short of the speed required to support the volume and liquidity of modern U.S. derivatives markets. Fastvol addresses this gap with efficient and fully featured American option models, built for large-scale workloads, live systems, research environments, and machine learning pipelines.

<div align="center">
  <img src="https://raw.githubusercontent.com/vgalanti/fastvol/main/docs/assets/bopm_throughput.png"/>
  <p><em>Figure: BOPM pricing throughput on a GH200 system (log-scale) </em></p>
</div>



## Roadmap

The high-level development plan, in order of implementation:

|  #   | Step                                                     | Status |
| :--: | -------------------------------------------------------- | :----: |
|  1   | Optimized BOPM and TTree CPU implementations             |   ✅   |
|  2   | BOPM and TTree CUDA implementations                      |   ✅   |
|  3   | Red-Black PSOR with adaptive $\omega$ (CPU & CUDA)       |   ✅   |
|  4   | Neural networks for pricing, IV & greeks via autograd    |   ✅   |
|  5   | Templated Newton, Brent, and bisection IV solvers        |   ✅   |
|  6   | Templated finite-difference Greeks                       |   ✅   |
|  7   | European BSM implementations                             |   ✅   |
|  8   | Complete C foreign function interface                    |   ✅   |
|  9   | Minimal-overhead Python wrapper via Cython (on PyPI)     |   ✅   |
| 10   | CUDA support for Greeks and IV inversion                 |   ⏳   |
| 11   | SLEEF integration for faster CPU European pricing        |   ⏳   |
| 12   | C++ neural network support via LibTorch                  |        |
| 13   | Stochastic models: heston, local volatility...           |        |
| 14   | Exotic options: barrier, lookback, asian...              |        |
| 15   | Precompiled wheels for Linux, macOS, and Windows         |        |
| 16   | Rust & OCaml language bindings                           |        |



## Implementation and Optimization Details

Extensive documentation on implementations, optimizations, and neural surrogates  -- including a full suite of benchmarks on a GH200 system -- is available in [docs/](https://github.com/vgalanti/fastvol/tree/main/docs) and will be updated over time:

- [Tree methods: BOPM and tri-trees](https://github.com/vgalanti/fastvol/blob/main/docs/trees.md)
- [PDE methods: Red-Black PSOR with adaptive w](https://github.com/vgalanti/fastvol/blob/main/docs/pde.md)
- [Neural Surrogates](https://github.com/vgalanti/fastvol/blob/main/docs/neural.md)
- [BSM](https://github.com/vgalanti/fastvol/blob/main/docs/bsm.md)
- [GH200 benchmarks](https://github.com/vgalanti/fastvol/blob/main/docs/GH200.txt)
- [Design Choices](https://github.com/vgalanti/fastvol/blob/main/docs/choices.md)

<div align="center">
  <img src="https://raw.githubusercontent.com/vgalanti/fastvol/main/docs/assets/neural/nn_puts.gif"/>
  <p><em>Figure: Pricing accuracy comparison on American Puts: Neural Surrogate vs Bjerksund-Stensland 2002 </em></p>
</div>



## Installation

### Python

Fastvol is available via PyPI:

```bash
pip install fastvol
```

or, to build from source:

```bash
git clone https://github.com/vgalanti/fastvol
cd fastvol
pip install .
```

* MacOS: Fastvol relies on OpenMP for CPU parallelism. On macOS, OpenMP must be installed manually via Homebrew: `brew install libomp`. PyTorch ships its own OpenMP runtime but omits headers. Fastvol links to that runtime while using the headers from Homebrew’s libomp.
* Linux: On CUDA-enabled systems, ensure that `nvcc` is available and that the correct PyTorch CUDA wheel is installed before installing Fastvol. If the CUDA wheel is not pre-installed, Fastvol’s build system may default to CPU-only PyTorch during installation. This will not affect core CUDA pricing methods, but neural network surrogates will run on CPU only.

### C/C++/CUDA
Fastvol's CPU and CUDA backends are implemented in C++ and organized as a standalone CMake project under `core/`.

You can embed the core library directly into your C++ application via CMake. All public functions are exposed through a clean C-style header interface (`fastvol/ffi.h`), and compiled into a linkable target named `fastvol`.

As a CMake subproject:
```cmake
add_subdirectory(core)
target_link_libraries(your_target PRIVATE fastvol)
```

With CMake FetchContent:
```cmake
include(FetchContent)
FetchContent_Declare(
  fastvol
  GIT_REPOSITORY https://github.com/vgalanti/fastvol
  GIT_TAG v0.1.0
  SOURCE_SUBDIR  core
)
FetchContent_MakeAvailable(fastvol)

target_link_libraries(your_target PRIVATE fastvol)
```

Note: Neural network surrogates are currently only available in the Python API.

## Usage

The codebase follows a simple access heuristic:

```<contract type>.<payoff>.<vol model>.<method>.<value>```

Currently, not all payoff types and volatility models are supported, so some of these levels are collapsed. The full access tree will be implemented in **v1.0.0**, which will involve an API change.

Available `<value>` operations are:
- price
- iv (implied volatility inversion)
- greeks (with per-Greek flags like delta=True, gamma=False)
- individual greeks functions (i.e. delta_fp64() etc..)

In the Python wrapper, high-level dispatchers `.price()`, `.greeks()`, and `.iv()` automatically detect argument types and broadcast constants. For example, you can run bopm over an array of spots while passing the strike as a single constant float.

Examples in Python:
```python
import fastvol as fv

fv.american.neural.price_fp32_batch(...)
fv.american.bopm.iv_fp32(...)
fv.american.ttree.price_fp64_batch(..., cuda=True)
fv.american.psor.greeks_fp32(..., delta=True, gamma=False)
fv.european.bsm.theta_fp64_batch(...)
```

Examples in C/C++:
```c
fastvol::american::ttree::price_fp64_batch(...); // C++ namespace access
fastvol::european::bsm::delta_fp32_batch(...); 

american_bopm_iv_fp64(...);                      // C FFI
american_psor_price_fp32_cuda(...);             
```

### License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/vgalanti/fastvol/blob/main/LICENSE) file for details.


### Author

Valerio Galanti, Computer Science PhD student, Columbia University.
