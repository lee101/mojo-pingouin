from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f, t

from . import _lib
from ._common import check_alternative, clean_samples, format_bf
from ._stats import achieved_t_power, bayesfactor_ttest
from .effsize import compute_effsize


def ttest(
    x,
    y,
    paired=False,
    alternative="two-sided",
    correction="auto",
    r=0.707,
    confidence=0.95,
):
    check_alternative(alternative)
    assert 0 < confidence < 1, "confidence must be between 0 and 1."
    x, y, paired = clean_samples(x, y, paired)
    nx, ny = x.size, y.size

    if ny == 1:
        mean_x, m2_x = _lib.moments(x)
        dof = nx - 1
        se = np.sqrt((m2_x / dof) / nx)
        t_value = (mean_x - y[0]) / se
    elif paired:
        mean_x, mean_y, xx, yy, xy = _lib.bivariate(x, y)
        dof = nx - 1
        se = np.sqrt((xx + yy - 2 * xy) / dof / nx)
        t_value = (mean_x - mean_y) / se
    else:
        mean_x, m2_x = _lib.moments(x)
        mean_y, m2_y = _lib.moments(y)
        vx, vy = m2_x / (nx - 1), m2_y / (ny - 1)
        if correction is True or (correction == "auto" and nx != ny):
            ax, ay = vx / nx, vy / ny
            se = np.sqrt(ax + ay)
            dof = (ax + ay) ** 2 / (
                ax**2 / (nx - 1) + ay**2 / (ny - 1)
            )
        else:
            dof = nx + ny - 2
            pooled = (m2_x + m2_y) / dof
            se = np.sqrt(pooled * (1 / nx + 1 / ny))
        t_value = (mean_x - mean_y) / se

    if alternative == "two-sided":
        p_value = 2 * t.sf(abs(t_value), dof)
        critical = t.ppf(1 - (1 - confidence) / 2, dof)
    elif alternative == "greater":
        p_value = t.sf(t_value, dof)
        critical = t.ppf(confidence, dof)
    else:
        p_value = t.cdf(t_value, dof)
        critical = t.ppf(confidence, dof)

    center = t_value * se
    ci = np.array([center - critical * se, center + critical * se])
    if ny == 1:
        ci += y[0]
    if alternative == "greater":
        ci[1] = np.inf
    elif alternative == "less":
        ci[0] = -np.inf
    ci = np.round(ci, 2)

    d = compute_effsize(x, y, paired=paired, eftype="cohen")
    power = achieved_t_power(d, nx, ny, paired, alternative)
    values = {
        "T": t_value,
        "dof": dof,
        "alternative": alternative,
        "p_val": p_value,
        f"CI{100 * confidence:.0f}": [ci],
        "cohen_d": abs(d),
        "power": power,
    }
    if alternative == "two-sided":
        values["BF10"] = format_bf(
            bayesfactor_ttest(t_value, nx, ny, paired=paired, r=r)
        )
    return pd.DataFrame(values, index=["T_test"])


def _one_way_data(data, dv, between):
    assert isinstance(data, pd.DataFrame), "Data must be a pandas dataframe."
    assert dv in data and between in data, "Columns are not in dataframe."
    dv_values, between_values = data[dv], data[between]
    if dv_values.hasnans or between_values.hasnans:
        clean = data[[dv, between]].dropna()
        dv_values, between_values = clean[dv], clean[between]
    values = np.ascontiguousarray(dv_values, dtype=np.float64)
    codes, labels = pd.factorize(between_values, sort=False)
    groups = len(labels)
    assert groups >= 2, "Data must contain at least two groups."
    return values, np.ascontiguousarray(codes, dtype=np.int64), groups


def anova(data=None, dv=None, between=None, ss_type=2, detailed=False, effsize="np2"):
    assert effsize in {"np2", "n2"}, "effsize must be 'np2' or 'n2'."
    if isinstance(between, list):
        if len(between) == 1:
            between = between[0]
        else:
            raise NotImplementedError("Only one-way ANOVA is covered.")
    values, codes, groups = _one_way_data(data, dv, between)
    sums, sumsq, counts = _lib.group_moments(values, codes, groups)
    n = values.size
    grand = sums.sum() / n
    means = sums / counts
    ss_between = float(np.sum(counts * (means - grand) ** 2))
    ss_error = float(sumsq.sum())
    df1, df2 = groups - 1, n - groups
    ms_between, ms_error = ss_between / df1, ss_error / df2
    f_value = ms_between / ms_error
    p_value = f.sf(f_value, df1, df2)
    eta = ss_between / (ss_between + ss_error)
    if detailed:
        return pd.DataFrame(
            {
                "Source": [between, "Within"],
                "SS": [ss_between, ss_error],
                "DF": [df1, df2],
                "MS": [ms_between, ms_error],
                "F": [f_value, np.nan],
                "p_unc": [p_value, np.nan],
                effsize: [eta, np.nan],
            }
        )
    return pd.DataFrame(
        {
            "Source": [between],
            "ddof1": [df1],
            "ddof2": [df2],
            "F": [f_value],
            "p_unc": [p_value],
            effsize: [eta],
        }
    )


def welch_anova(data=None, dv=None, between=None):
    values, codes, groups = _one_way_data(data, dv, between)
    sums, sumsq, counts = _lib.group_moments(values, codes, groups)
    means = sums / counts
    variances = sumsq / (counts - 1)
    weights = counts / variances
    adjusted_mean = np.sum(weights * means) / weights.sum()
    ss_adjusted = np.sum(weights * (means - adjusted_mean) ** 2)
    df1 = groups - 1
    lamb = (
        3
        * np.sum((1 / (counts - 1)) * (1 - weights / weights.sum()) ** 2)
        / (groups**2 - 1)
    )
    df2 = 1 / lamb
    f_value = (ss_adjusted / df1) / (1 + 2 * lamb * (groups - 2) / 3)
    p_value = f.sf(f_value, df1, df2)
    grand = sums.sum() / counts.sum()
    ss_between = np.sum(counts * (means - grand) ** 2)
    ss_error = sumsq.sum()
    eta = ss_between / (ss_between + ss_error)
    return pd.DataFrame(
        {
            "Source": [between],
            "ddof1": [df1],
            "ddof2": [df2],
            "F": [f_value],
            "p_unc": [p_value],
            "np2": [eta],
        }
    )
