from __future__ import annotations

import warnings

import numpy as np

from . import _lib


def clean_samples(x, y, paired: bool = False) -> tuple[np.ndarray, np.ndarray, bool]:
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1))
    if paired and x.size != y.size:
        warnings.warn(
            "x and y have unequal sizes. Switching to paired == False.",
            stacklevel=2,
        )
        paired = False
    x_has_nan = _lib.has_nan(x)
    y_has_nan = _lib.has_nan(y)
    if paired:
        if not x_has_nan and not y_has_nan:
            return x, y, paired
        keep = ~np.isnan(x) & ~np.isnan(y)
        return (
            np.ascontiguousarray(x[keep]),
            np.ascontiguousarray(y[keep]),
            paired,
        )
    return (
        np.ascontiguousarray(x[~np.isnan(x)]) if x_has_nan else x,
        np.ascontiguousarray(y[~np.isnan(y)]) if y_has_nan else y,
        paired,
    )


def check_alternative(alternative: str) -> None:
    assert alternative in {"two-sided", "greater", "less"}, (
        "Alternative must be one of 'two-sided' (default), 'greater' or 'less'."
    )


def format_bf(value: float, precision: int = 3) -> str:
    if value >= 1e4 or value <= 1e-4:
        return np.format_float_scientific(value, precision=precision, trim="0")
    return np.format_float_positional(value, precision=precision, trim="0")
