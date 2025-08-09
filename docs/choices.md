# Design Choices


## Lang

I like C — it was my first programming language, and I enjoy the control it offers over hardware. That said, I am no fan of macros, nor am I opinionated enough to reject C++ at all costs.
This is why the core library is written in C-like C++, using only a minimal set of C++ features (templates and lambdas) to cleanly support both `float` and `double` without excessive code duplication.


## Naming

### C/C++ vs Python
The core library adopts a mathematical style for variable names (`S`, `K`, `r`, `q` ...) matching standard option pricing literature. The Python wrapper introduces slightly more descriptive abbreviations for usability. This reflects common practice with terse symbolic notation in lower-level implementations, and friendlier naming at higher levels.


### Ordering
The standard ordering in the library is: Spot, Strike, C/P, Ttm, IV, Rate, Dividends. 

This ordering semantically groups variables and ranks them by importance within each group:
- Spot, Strike, C/P, Ttm: define the contract’s objective parameters.
- IV, Rate, Dividends: depend on the user’s chosen volatility estimate, interest rate assumptions, and yield forecasts.


### cp_flag
On the C side, cp_flag can be
- `1`, `'c'`, or `'C'` -> call
- `0`, `'p'`, or `'P'` -> put

In ASCII, `'c'` = 99, `'C'` = 67, `'p'` = 112, and `'P'` = 80.

This allows a branchless boolean check via bitwise parity:
```C
int is_call = cp_flag & 1;
int is_put = (~cp_flag) & 1;
```

Since Python lacks a native `char` type, the wrapper uses `c_flag` instead, expecting booleans or (more precisely) `uint8s`.
When passing arrays to batched functions, ensure `c_flag` is in this format.
The Cython interface only receives `uintptr_t` pointers to array memory locations (supporting both NumPy and Torch arrays) and casts them to `char *`. Passing the wrong type will result in undefined behavior (typically a segfault).


## Cython vs. pybind11
While pybind11 is a popular choice for C++/Python bindings, Fastvol uses Cython for the Python interface.
This decision was made for several reasons:
- Lower call overhead — Cython generates direct C calls, avoiding some of pybind11’s type-erasure and template instantiation costs.
- Fine-grained memory handling — The interface can take raw `uintptr_t` pointers to NumPy or Torch arrays without copying, which is harder to achieve cleanly in pybind11.
- Simpler conditional compilation — Easier to integrate CUDA, OpenMP, and platform-specific flags directly in the build.
- Minimal dependency footprint — Cython compiles down to plain C extensions, reducing the need for users to install additional binding libraries.

Because Fastvol is designed for wider language interoperability (via its C-style FFI) and may eventually switch to an all-C core for maximal throughput, Cython’s ability to wrap pure C interfaces with minimal friction was a natural fit.

Pybind11 remains an excellent choice for many projects, but Fastvol’s priority on low-latency function calls and tight CUDA integration made Cython a better fit.