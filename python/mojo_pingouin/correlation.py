from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, t

from . import _lib
from ._common import check_alternative, format_bf
from ._stats import achieved_corr_power, bayesfactor_pearson
from .effsize import compute_esci


def _correlation_pvalue(r: float, n: int, alternative: str) -> float:
    statistic = r * np.sqrt((n - 2) / ((r + 1) * (1 - r)))
    if alternative == "two-sided":
        return float(2 * t.sf(abs(statistic), n - 2))
    if alternative == "greater":
        return float(t.sf(statistic, n - 2))
    return float(t.cdf(statistic, n - 2))


def corr(x, y, alternative="two-sided", method="pearson", **kwargs):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert x.ndim == y.ndim == 1, "x and y must be 1D array."
    assert x.size == y.size, "x and y must have the same length."
    check_alternative(alternative)
    if "tail" in kwargs:
        raise ValueError(
            "Since Pingouin 0.4.0, the 'tail' argument has been renamed to "
            "'alternative'."
        )
    keep = ~np.isnan(x) & ~np.isnan(y)
    x, y = np.ascontiguousarray(x[keep]), np.ascontiguousarray(y[keep])
    n = x.size

    if method == "pearson":
        _, _, xx, yy, xy = _lib.bivariate(x, y)
        r_value = xy / np.sqrt(xx * yy)
        p_value = _correlation_pvalue(r_value, n, alternative)
    elif method == "spearman":
        r_value, p_value = spearmanr(x, y, alternative=alternative, **kwargs)
    elif method == "kendall":
        r_value, p_value = kendalltau(x, y, alternative=alternative, **kwargs)
    else:
        raise ValueError(
            f'Method "{method}" not recognized. Covered methods are pearson, '
            "spearman, and kendall."
        )

    if np.isnan(r_value):
        return pd.DataFrame(
            {
                "n": [n],
                "r": [np.nan],
                "CI95": [np.nan],
                "p_val": [np.nan],
                "BF10": [np.nan],
                "power": [np.nan],
            },
            index=[method],
        )
    if abs(r_value) > 1 and np.isclose(abs(r_value), 1):
        r_value = float(np.clip(r_value, -1, 1))
    if abs(r_value) == 1:
        ci, power = np.array([r_value, r_value]), 1.0
    else:
        ci = compute_esci(
            stat=r_value,
            nx=n,
            ny=n,
            eftype="r",
            decimals=2,
            alternative=alternative,
        )
        power = achieved_corr_power(r_value, n, alternative)
    values = {
        "n": n,
        "r": r_value,
        "CI95": [ci],
        "p_val": p_value,
        "power": power,
    }
    if method == "pearson":
        values["BF10"] = format_bf(
            bayesfactor_pearson(r_value, n, alternative)
        )
    result = pd.DataFrame(values, index=[method])
    order = ["n", "r", "CI95", "p_val", "BF10", "power"]
    return result[[column for column in order if column in result]]


def distance_corr(x, y, alternative="greater", n_boot=1000, seed=None):
    check_alternative(alternative)
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Input arrays must not contain NaN values.")
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    assert x.ndim == y.ndim == 2
    assert x.shape[0] == y.shape[0], "x and y must have same number of samples"
    x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
    n = x.shape[0]
    matrix_x, xx = _lib.centered_distance(x)
    matrix_y, yy = _lib.centered_distance(y)
    xy = _lib.distance_dot(matrix_x, matrix_y)
    dcor = float(np.sqrt(xy) / np.sqrt(np.sqrt(xx) * np.sqrt(yy)))
    if n_boot is None or n_boot <= 1:
        return dcor

    rng = np.random.RandomState(seed)
    permutations = rng.random_sample((n_boot, n)).argsort(axis=1)
    products = _lib.permuted_dots(matrix_x, matrix_y, permutations)
    bootstat = np.sqrt(products) / np.sqrt(np.sqrt(xx) * np.sqrt(yy))
    if alternative == "greater":
        p_value = np.greater_equal(bootstat, dcor).mean()
    elif alternative == "less":
        p_value = np.less_equal(bootstat, dcor).mean()
    else:
        p_value = np.greater_equal(np.abs(bootstat), abs(dcor)).mean()
    return dcor, float(p_value)
