from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm, t

from . import _lib
from ._common import check_alternative, clean_samples

_EFTYPES = {
    "none",
    "cohen",
    "cohen_dz",
    "hedges",
    "r",
    "pointbiserialr",
    "eta_square",
    "odds_ratio",
    "auc",
    "cles",
}


def convert_effsize(ef, input_type, output_type, nx=None, ny=None):
    it, ot = input_type.lower(), output_type.lower()
    if it not in _EFTYPES:
        raise ValueError(f"Could not interpret input '{it}'")
    if ot not in _EFTYPES:
        raise ValueError(f"Could not interpret input '{ot}'")
    if it not in {"pointbiserialr", "cohen"}:
        raise ValueError("Input type must be 'cohen' or 'pointbiserialr'")
    if it == ot or ot == "none":
        return ef
    d = 2 * ef / np.sqrt(1 - ef**2) if it == "pointbiserialr" else ef
    if ot == "cohen":
        return d
    if ot == "hedges":
        if nx is None or ny is None:
            warnings.warn(
                "You need to pass nx and ny arguments to compute Hedges g. "
                "Returning Cohen's d instead",
                stacklevel=2,
            )
            return d
        return d * (1 - 3 / (4 * (nx + ny) - 9))
    if ot == "pointbiserialr":
        a = (
            ((nx + ny) ** 2 - 2 * (nx + ny)) / (nx * ny)
            if nx is not None and ny is not None
            else 4
        )
        return d / np.sqrt(d**2 + a)
    if ot == "eta_square":
        return (d / 2) ** 2 / (1 + (d / 2) ** 2)
    if ot == "odds_ratio":
        return np.exp(d * np.pi / np.sqrt(3))
    if ot == "r":
        raise ValueError(
            "Using effect size 'r' in `pingouin.convert_effsize` has been "
            "deprecated. Please use 'pointbiserialr' instead."
        )
    return norm.cdf(d / np.sqrt(2))


def compute_effsize(x, y, paired=False, eftype="cohen"):
    eftype_lower = eftype.lower()
    if eftype_lower not in _EFTYPES:
        raise ValueError(f"Could not interpret input '{eftype}'")
    x, y, paired = clean_samples(x, y, paired)
    nx, ny = x.size, y.size
    if ny == 1:
        mean, m2 = _lib.moments(x)
        return (mean - y[0]) / np.sqrt(m2 / (nx - 1))
    if eftype_lower == "r":
        mx, my, xx, yy, xy = _lib.bivariate(x, y)
        del mx, my
        return xy / np.sqrt(xx * yy)
    if eftype_lower == "cles":
        return _lib.cles(x, y)
    if eftype_lower == "cohen_dz":
        if paired:
            mean_x, mean_y, xx, yy, xy = _lib.bivariate(x, y)
            return (mean_x - mean_y) / np.sqrt((xx + yy - 2 * xy) / (nx - 1))
        warnings.warn(
            "Cohen's dz is only defined for paired samples. Computing regular "
            "Cohen's d instead.",
            stacklevel=2,
        )
        eftype_lower = "cohen"
    mean_x, m2_x = _lib.moments(x)
    mean_y, m2_y = _lib.moments(y)
    if paired:
        d = (mean_x - mean_y) / np.sqrt(
            (m2_x / (nx - 1) + m2_y / (ny - 1)) / 2
        )
    else:
        d = (mean_x - mean_y) / np.sqrt((m2_x + m2_y) / (nx + ny - 2))
    return convert_effsize(d, "cohen", eftype_lower, nx=nx, ny=ny)


def compute_effsize_from_t(tval, nx=None, ny=None, N=None, eftype="cohen"):
    if eftype.lower() not in _EFTYPES:
        raise ValueError(f"Could not interpret input '{eftype}'")
    if not isinstance(tval, float):
        raise ValueError("T-value must be float")
    if nx is not None and ny is not None:
        d = tval * np.sqrt(1 / nx + 1 / ny)
    elif N is not None:
        d = 2 * tval / np.sqrt(N)
    else:
        raise ValueError("You must specify either nx + ny, or just N")
    return convert_effsize(d, "cohen", eftype, nx=nx, ny=ny)


def compute_esci(
    stat=None,
    nx=None,
    ny=None,
    paired=False,
    eftype="cohen",
    confidence=0.95,
    decimals=2,
    alternative="two-sided",
):
    kind = eftype.lower()
    assert kind in {"r", "pearson", "spearman", "cohen", "d", "g", "hedges"}
    check_alternative(alternative)
    assert stat is not None and nx is not None
    assert isinstance(confidence, float)
    assert 0 < confidence < 1, "confidence must be between 0 and 1."
    if kind in {"r", "pearson", "spearman"}:
        z = np.arctanh(stat)
        se = 1 / np.sqrt(nx - 3)
        if alternative == "two-sided":
            crit = abs(norm.ppf((1 - confidence) / 2))
            ci = np.tanh([z - crit * se, z + crit * se])
        elif alternative == "greater":
            crit = norm.ppf(confidence)
            ci = np.tanh([z - crit * se, np.inf])
        else:
            crit = norm.ppf(confidence)
            ci = np.tanh([-np.inf, z + crit * se])
    else:
        if ny == 1 or paired:
            se = np.sqrt(1 / nx + stat**2 / (2 * nx))
            dof = nx - 1
        else:
            se = np.sqrt((nx + ny) / (nx * ny) + stat**2 / (2 * (nx + ny)))
            dof = nx + ny - 2
        crit = abs(t.ppf((1 - confidence) / 2, dof))
        ci = np.array([stat - crit * se, stat + crit * se])
    return np.round(ci, decimals)
