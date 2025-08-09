import os, glob, shutil, torch, numpy as np, platform, subprocess

from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist as _sdist
from Cython.Build import cythonize


PKG_ROOT = Path(__file__).parent.resolve()
BUILD_DIR = PKG_ROOT / "build"
EXTERNAL_CORE = (PKG_ROOT / ".." / "core").resolve()
STAGED_CORE = (PKG_ROOT / "fastvol" / "_core").resolve()


include_dirs = [np.get_include(), str(STAGED_CORE / "include")]
libraries = ["fastvol", "m"]
library_dirs = [str(BUILD_DIR)]
extra_compile_args = ["-O3", "-march=native", "-ffast-math"]
extra_link_args = []
define_macros = []
cmake_args = [
    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={BUILD_DIR}",
    f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={BUILD_DIR}",
    "-DCMAKE_BUILD_TYPE=Release",
]

# OpenMP ------------------------------------------------------------------------------*/
if platform.system() == "Darwin":
    try:
        # OMP path / include -- torch install + brew headers
        brew_omp_prefix = (
            subprocess.check_output(["brew", "--prefix", "libomp"]).decode().strip()
        )
        brew_omp_include = str(Path(brew_omp_prefix) / "include")

        torch_omp_path = next(Path(torch.__file__).parent.rglob("libomp.dylib"))
        torch_omp_dir = str(torch_omp_path.parent)

        cmake_args += [
            f"-DFASTVOL_OMP_INCLUDE_DIR={brew_omp_include}",
            f"-DFASTVOL_OMP_LIB_DIR={torch_omp_dir}",
        ]

        # add flags, paths, and headers
        extra_compile_args.extend(["-Xpreprocessor", "-fopenmp"])
        extra_link_args.extend(["-lomp", f"-L{torch_omp_dir}"])
        include_dirs.append(brew_omp_include)

        print(f"[fastvol] OpenMP linked to torch's libomp at: {torch_omp_dir}")

    except Exception as e:
        print(f"[fastvol] OpenMP: fallback to system/brew libomp due to: {e}")

else:
    extra_compile_args.extend(["-fopenmp"])
    extra_link_args.extend(["-fopenmp"])


# C core ------------------------------------------------------------------------------*/


def stage_core(force: bool = False):
    """
    Ensure fastvol/_core exists.
    - force=True: refresh from ../core (used for sdist creation).
    - force=False: if _core already exists (sdist install), reuse it; otherwise copy.
    """
    if force:
        if STAGED_CORE.exists():
            shutil.rmtree(STAGED_CORE)
        if not EXTERNAL_CORE.exists():
            raise RuntimeError("EXTERNAL_CORE not found; cannot refresh staged copy.")
        shutil.copytree(EXTERNAL_CORE, STAGED_CORE, dirs_exist_ok=False)
        return STAGED_CORE

    # Non-forced path (editable install or sdist install)
    if STAGED_CORE.exists():
        return STAGED_CORE
    if EXTERNAL_CORE.exists():
        shutil.copytree(EXTERNAL_CORE, STAGED_CORE, dirs_exist_ok=False)
        return STAGED_CORE

    raise RuntimeError(
        "fastvol/_core not found and ../core unavailable.\n"
        "If installing from sdist, rebuild sdist after adding MANIFEST graft."
    )


class CMakeBuildExt(build_ext):
    def run(self):
        core_dir = stage_core(force=False)
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["cmake", str(core_dir)] + cmake_args, cwd=BUILD_DIR)
        subprocess.check_call(["cmake", "--build", ".", "-j"], cwd=BUILD_DIR)
        super().run()


class sdist(_sdist):
    """Ensure core is staged into the sdist tree."""

    def run(self):
        # stage the core
        stage_core(force=True)

        # copy readme
        root_readme = PKG_ROOT.parent / "README.md"
        pkg_readme = PKG_ROOT / "README.md"
        if root_readme.exists():
            pkg_readme.write_text(
                root_readme.read_text(encoding="utf-8"), encoding="utf-8"
            )

        # copy license
        root_license = PKG_ROOT.parent / "LICENSE"
        if root_license.exists():
            (PKG_ROOT / "LICENSE").write_text(
                root_license.read_text(encoding="utf-8"), encoding="utf-8"
            )

        super().run()


# CUDA --------------------------------------------------------------------------------*/
def detect_cuda_available():
    try:
        subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def find_cuda_library(libname: str) -> str:
    """
    Searches common system directories for a CUDA shared library (.so).
    """
    candidates = [
        f"/usr/lib/aarch64-linux-gnu/lib{libname}.so",
        f"/usr/local/cuda/lib64/lib{libname}.so",
        f"/usr/lib64/lib{libname}.so",
        f"/usr/lib/x86_64-linux-gnu/lib{libname}.so",
    ]

    for path in candidates:
        if os.path.exists(path):
            return os.path.dirname(path)

    # fallback: search everything
    matches = glob.glob(f"/usr/**/lib{libname}.so*", recursive=True)
    if matches:
        return os.path.dirname(matches[0])

    raise FileNotFoundError(f"Could not find lib{libname}.so on this system.")


ENABLE_CUDA = False
if detect_cuda_available():

    try:
        # find libraries
        cudart_dir = find_cuda_library("cudart")
        cublas_dir = find_cuda_library("cublas")
        cuda_lib_dirs = list(set([cudart_dir, cublas_dir]))

        # update flags
        libraries += ["cudart", "cublas"]
        library_dirs = list(set(library_dirs + cuda_lib_dirs))
        define_macros += [("FASTVOL_CUDA_ENABLED", "1")]

        # log
        ENABLE_CUDA = True
        print(f"[fastvol] CUDA support: {'enabled' if ENABLE_CUDA else 'disabled'}")
        print(f"[fastvol] CUDA libs in: {cuda_lib_dirs}")

    except FileNotFoundError as e:
        print(f"[fastvol] WARNING: Couldn't link to cudart and cublas. CUDA disabled.")


# cython ------------------------------------------------------------------------------*/
ext_modules = cythonize(
    [
        Extension(
            name="fastvol._core",
            sources=["fastvol/_core.pyx"],
            include_dirs=include_dirs,
            libraries=libraries,
            define_macros=define_macros,
            library_dirs=library_dirs,
            language="c",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    compile_time_env={"FASTVOL_CUDA_ENABLED": ENABLE_CUDA},
)

# setup -------------------------------------------------------------------------------*/
setup(
    packages=find_packages(include=["fastvol", "fastvol.*"]),
    package_dir={"": "."},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuildExt, "sdist": sdist},
    zip_safe=False,
)
