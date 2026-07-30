"""ctypes access to the compiled Mojo kernels."""

from __future__ import annotations

import atexit
import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "pingouin.mojo"
LIBRARY = Path(
    os.environ.get("MOJO_PINGOUIN_LIB", ROOT / "dist" / "libmojo-pingouin.so")
)

I = ctypes.c_int64
P = ctypes.c_void_p
F = ctypes.c_double

_SIGNATURES = {
    "mpg_moments": ([P, I, P], None),
    "mpg_has_nan": ([P, I], I),
    "mpg_bivariate": ([P, P, I, P], None),
    "mpg_cles": ([P, P, I, I], F),
    "mpg_group_moments": ([P, P, I, I, P, P, P], None),
    "mpg_centered_distance": ([P, I, I, P, P], F),
    "mpg_distance_dot": ([P, P, I], F),
    "mpg_permuted_dots": ([P, P, P, I, I, P], None),
}

_loaded: ctypes.CDLL | None = None
_runtime_device: int | None = None


def _release_runtime() -> None:
    global _runtime_device
    if _loaded is None or _runtime_device is None:
        return
    release = _loaded.KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice
    release.argtypes = [ctypes.c_void_p]
    release.restype = None
    release(_runtime_device)
    _runtime_device = None


def build_if_needed() -> Path:
    if os.environ.get("MOJO_PINGOUIN_LIB"):
        if not LIBRARY.exists():
            raise RuntimeError(f"MOJO_PINGOUIN_LIB does not exist: {LIBRARY}")
        return LIBRARY
    if not LIBRARY.exists() or LIBRARY.stat().st_mtime < SOURCE.stat().st_mtime:
        proc = subprocess.run(
            ["bash", str(ROOT / "build" / "build.sh")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIBRARY


def lib() -> ctypes.CDLL:
    global _loaded, _runtime_device
    if _loaded is None:
        _loaded = ctypes.CDLL(str(build_if_needed()))
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes = argtypes
            fn.restype = restype
        current = _loaded.KGEN_CompilerRT_AsyncRT_GetCurrentCPUDevice
        current.argtypes = []
        current.restype = ctypes.c_void_p
        if current() is None:
            create = _loaded.KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice
            create.argtypes = []
            create.restype = ctypes.c_void_p
            _runtime_device = create()
            atexit.register(_release_runtime)
    return _loaded


def f64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def i64(values) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in {"i", "u"}:
        raise TypeError("integer index arrays are required")
    if array.dtype.kind == "u" and array.size:
        if np.max(array) > np.iinfo(np.int64).max:
            raise OverflowError("integer index does not fit in int64")
    return np.ascontiguousarray(array, dtype=np.int64)


def addr(values: np.ndarray) -> int:
    if values.size == 0:
        raise ValueError("empty arrays must not cross the Mojo FFI boundary")
    if values.ctypes.data == 0:
        raise ValueError("array has a null data pointer")
    return values.ctypes.data


def has_nan(values: np.ndarray) -> bool:
    values = f64(values).reshape(-1)
    if values.size == 0:
        return False
    return bool(lib().mpg_has_nan(addr(values), values.size))


def moments(values: np.ndarray) -> tuple[float, float]:
    values = f64(values).reshape(-1)
    if values.size == 0:
        raise ValueError("moments requires at least one value")
    result = np.empty(32, dtype=np.float64)
    lib().mpg_moments(addr(values), values.size, addr(result))
    return float(result[0]), float(result[1])


def bivariate(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    x, y = f64(x).reshape(-1), f64(y).reshape(-1)
    if x.size == 0 or x.size != y.size:
        raise ValueError("bivariate requires equal, non-empty arrays")
    result = np.empty(5, dtype=np.float64)
    lib().mpg_bivariate(addr(x), addr(y), x.size, addr(result))
    return tuple(float(v) for v in result)


def cles(x: np.ndarray, y: np.ndarray) -> float:
    x, y = f64(x).reshape(-1), f64(y).reshape(-1)
    if x.size == 0 or y.size == 0:
        raise ValueError("CLES requires two non-empty arrays")
    return float(lib().mpg_cles(addr(x), addr(y), x.size, y.size))


def group_moments(
    values: np.ndarray, codes: np.ndarray, groups: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, codes = f64(values).reshape(-1), i64(codes).reshape(-1)
    if values.size == 0 or values.size != codes.size:
        raise ValueError("values and codes must be equal-length and non-empty")
    if not isinstance(groups, (int, np.integer)) or groups < 1:
        raise ValueError("groups must be a positive integer")
    if np.any(codes < 0) or np.any(codes >= groups):
        raise ValueError("group codes must be in [0, groups)")
    sums = np.empty(groups, dtype=np.float64)
    sumsq = np.empty(groups, dtype=np.float64)
    counts = np.empty(groups, dtype=np.int64)
    lib().mpg_group_moments(
        addr(values),
        addr(codes),
        values.size,
        groups,
        addr(sums),
        addr(sumsq),
        addr(counts),
    )
    return sums, sumsq, counts


def centered_distance(values: np.ndarray) -> tuple[np.ndarray, float]:
    values = f64(values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("distance input must be a non-empty 1D or 2D array")
    n, d = values.shape
    matrix = np.empty((n, n), dtype=np.float64)
    rows = np.empty(n, dtype=np.float64)
    squared_sum = lib().mpg_centered_distance(
        addr(values), n, d, addr(matrix), addr(rows)
    )
    return matrix, float(squared_sum)


def distance_dot(a: np.ndarray, b: np.ndarray) -> float:
    a, b = f64(a).reshape(-1), f64(b).reshape(-1)
    if a.size == 0 or a.size != b.size:
        raise ValueError("distance matrices must have equal, non-zero sizes")
    return float(lib().mpg_distance_dot(addr(a), addr(b), a.size))


def permuted_dots(
    a: np.ndarray, b: np.ndarray, permutations: np.ndarray
) -> np.ndarray:
    a, b = f64(a), f64(b)
    permutations = i64(permutations)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be a non-empty square matrix")
    if b.shape != a.shape:
        raise ValueError("a and b must have the same square shape")
    if permutations.ndim != 2 or permutations.shape[1] != a.shape[0]:
        raise ValueError("permutations must have one column per matrix row")
    if permutations.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if np.any(permutations < 0) or np.any(permutations >= a.shape[0]):
        raise ValueError("permutation indices are out of bounds")
    result = np.empty(permutations.shape[0], dtype=np.float64)
    lib().mpg_permuted_dots(
        addr(a),
        addr(b),
        addr(permutations),
        permutations.shape[0],
        permutations.shape[1],
        addr(result),
    )
    return result
