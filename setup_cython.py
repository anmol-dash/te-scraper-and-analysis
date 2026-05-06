"""
setup_cython.py — Build te_fast Cython extension in-place.

Usage:
    python setup_cython.py build_ext --inplace

The compiled module (te_fast.cpython-*.so) is placed in the same directory
so it can be imported directly: `import te_fast`
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "te_fast",
        sources=["te_fast.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-funroll-loops"],
        language="c",
    )
]

setup(
    name="te_fast",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
        annotate=False,
    ),
)
