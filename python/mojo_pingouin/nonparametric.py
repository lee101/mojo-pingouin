from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from . import _lib
from ._common import check_alternative, clean_samples


def mwu(x, y, alternative="two-sided", **kwargs):
    check_alternative(alternative)
    if "tail" in kwargs:
        raise ValueError(
            "Since Pingouin 0.4.0, the 'tail' argument has been renamed to "
            "'alternative'."
        )
    x, y, _ = clean_samples(x, y, paired=False)
    u_value, p_value = scipy_stats.mannwhitneyu(
        x, y, alternative=alternative, **kwargs
    )
    cles = _lib.cles(x, y)
    if alternative == "less":
        cles = 1 - cles
    rbc = 1 - 2 * (x.size * y.size - u_value) / (x.size * y.size)
    return pd.DataFrame(
        {
            "U_val": [u_value],
            "alternative": [alternative],
            "p_val": [p_value],
            "RBC": [rbc],
            "CLES": [cles],
        },
        index=["MWU"],
    )


def wilcoxon(x, y=None, alternative="two-sided", **kwargs):
    check_alternative(alternative)
    if "tail" in kwargs:
        raise ValueError(
            "Since Pingouin 0.4.0, the 'tail' argument has been renamed to "
            "'alternative'."
        )
    raw_y = y
    if y is None:
        x = np.asarray(x, dtype=np.float64)
        x = np.ascontiguousarray(x[~np.isnan(x)])
    else:
        x, y, _ = clean_samples(x, y, paired=True)
    if "correction" not in kwargs:
        kwargs["correction"] = True
    w_value, p_value = scipy_stats.wilcoxon(
        x=x, y=y, alternative=alternative, **kwargs
    )
    if raw_y is None:
        cles = np.nan
        differences = x[x != 0]
    else:
        cles = _lib.cles(x, y)
        if alternative == "less":
            cles = 1 - cles
        differences = (x - y)
        differences = differences[differences != 0]
    ranks = scipy_stats.rankdata(abs(differences))
    rank_sum = ranks.sum()
    positive = np.sum((differences > 0) * ranks)
    negative = np.sum((differences < 0) * ranks)
    rbc = (
        negative / rank_sum - positive / rank_sum
        if alternative == "less"
        else positive / rank_sum - negative / rank_sum
    )
    return pd.DataFrame(
        {
            "W_val": [w_value],
            "alternative": [alternative],
            "p_val": [p_value],
            "RBC": [rbc],
            "CLES": [cles],
        },
        index=["Wilcoxon"],
    )


def kruskal(data=None, dv=None, between=None, detailed=False):
    assert isinstance(data, pd.DataFrame), "Data must be a pandas dataframe."
    assert dv in data and between in data, "Columns are not in dataframe."
    clean = data[[dv, between]].dropna()
    groups = [group[dv].to_numpy() for _, group in clean.groupby(between, observed=True)]
    h_value, p_value = scipy_stats.kruskal(*groups)
    return pd.DataFrame(
        {
            "Source": [between],
            "ddof1": [len(groups) - 1],
            "H": [h_value],
            "p_unc": [p_value],
        },
        index=["Kruskal"],
    )
