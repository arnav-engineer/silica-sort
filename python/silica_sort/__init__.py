import numpy as np
from typing import Dict, Any
from numpy.typing import NDArray

from ._silica_sort import (
    __version__ as _version,
    get_system_info as _get_system_info,
    sort_file as _sort_file,
    sort_numpy as _sort_numpy,
    sort_numpy_inplace as _sort_numpy_inplace,
    sort_numpy_rust_standard as _sort_numpy_rust_standard,
    test_rmi as _test_rmi,
)

__version__ = _version

def sort_numpy(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Sorts a contiguous 1D NumPy array of 64-bit floats and returns a new sorted array.
    Uses the high-performance Learned Sort hybrid algorithm.
    """
    return _sort_numpy(array)

def sort_numpy_inplace(array: NDArray[np.float64]) -> None:
    """
    Sorts a contiguous 1D NumPy array of 64-bit floats in place.
    Uses the high-performance Learned Sort hybrid algorithm.
    """
    _sort_numpy_inplace(array)

def sort_numpy_rust_standard(array: NDArray[np.float64]) -> None:
    """
    Sorts a contiguous 1D NumPy array of 64-bit floats in place using Rust's standard library.
    """
    _sort_numpy_rust_standard(array)

def sort_file(input_path: str, output_path: str) -> None:
    """
    Sorts a file containing 64-bit floats. If the file is too large to fit in RAM,
    it automatically falls back to an out-of-core external chunked K-Way merge sort.
    """
    _sort_file(input_path, output_path)

def get_system_info() -> Dict[str, Any]:
    """
    Returns dynamically detected hardware information such as L1/L2 cache sizes
    and SIMD instruction set support (AVX2, AVX512, NEON, etc.).
    """
    return _get_system_info()

def test_rmi(array: NDArray[np.float64], num_buckets: int) -> NDArray[np.int_]:
    """
    Tests the Monotonic RMI (Recursive Model Index) on the given sorted array.
    Returns an array of predicted bucket indices for each element.
    """
    return _test_rmi(array, num_buckets)

__all__ = [
    "__version__",
    "sort_numpy",
    "sort_numpy_inplace",
    "sort_numpy_rust_standard",
    "sort_file",
    "get_system_info",
    "test_rmi",
]
