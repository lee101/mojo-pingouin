from __future__ import annotations

from math import exp, lgamma, log, pi

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.special import betaln, hyp2f1


def achieved_t_power(
    d: float,
    nx: int,
    ny: int = 1,
    paired: bool = False,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> float:
    if ny == 1 or paired:
        dof = nx - 1
        nc = d * np.sqrt(nx)
    else:
        dof = nx + ny - 2
        nc = d / np.sqrt(1 / nx + 1 / ny)
    if alternative == "two-sided":
        nc = abs(nc)
        critical = stats.t.ppf(1 - alpha / 2, dof)
        return float(
            stats.nct.sf(critical, dof, nc) + stats.nct.cdf(-critical, dof, nc)
        )
    if alternative == "greater":
        critical = stats.t.ppf(1 - alpha, dof)
        return float(stats.nct.sf(critical, dof, nc))
    critical = stats.t.ppf(alpha, dof)
    return float(stats.nct.cdf(critical, dof, nc))


def achieved_corr_power(
    r: float, n: int, alternative: str = "two-sided", alpha: float = 0.05
) -> float:
    if n <= 4:
        return np.nan
    if alternative == "two-sided":
        r = abs(r)
        critical_t = stats.t.ppf(1 - alpha / 2, n - 2)
        critical_r = np.sqrt(critical_t**2 / (critical_t**2 + n - 2))
        z = np.arctanh(r) + r / (2 * (n - 1))
        zc = np.arctanh(critical_r)
        scale = np.sqrt(n - 3)
        return float(stats.norm.cdf((z - zc) * scale) + stats.norm.cdf((-z - zc) * scale))
    if alternative == "less":
        r = -r
    critical_t = stats.t.ppf(1 - alpha, n - 2)
    critical_r = np.sqrt(critical_t**2 / (critical_t**2 + n - 2))
    z = np.arctanh(r) + r / (2 * (n - 1))
    zc = np.arctanh(critical_r)
    return float(stats.norm.cdf((z - zc) * np.sqrt(n - 3)))


def bayesfactor_ttest(
    t_value: float, nx: int, ny: int = 1, paired: bool = False, r: float = 0.707
) -> float:
    if not np.isfinite(t_value):
        return np.nan
    if ny == 1 or paired:
        n = nx
        dof = nx - 1
    else:
        n = nx * ny / (nx + ny)
        dof = nx + ny - 2

    def integrand(g):
        return (
            (1 + n * g * r**2) ** -0.5
            * (1 + t_value**2 / ((1 + n * g * r**2) * dof))
            ** (-(dof + 1) / 2)
            * (2 * pi) ** -0.5
            * g**-1.5
            * exp(-1 / (2 * g))
        )

    integral = quad(integrand, 0, np.inf)[0]
    denominator = (1 + t_value**2 / dof) ** (-(dof + 1) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.divide(integral, denominator))


def bayesfactor_pearson(r: float, n: int, alternative: str = "two-sided") -> float:
    if not np.isfinite(r) or n < 2:
        return np.nan
    lbeta = betaln(1, 1)
    hyper = hyp2f1((n - 1) / 2, (n - 1) / 2, (n + 2) / 2, r**2)
    bf = exp(
        -log(2)
        + 0.5 * log(pi)
        - lbeta
        + lgamma((n + 1) / 2)
        - lgamma((n + 2) / 2)
        + log(hyper)
    )
    if alternative == "two-sided":
        return float(bf)

    import mpmath

    mp_n, mp_r = mpmath.mpf(n), mpmath.mpf(r)
    mp_bf = mpmath.exp(
        -mpmath.log(2)
        + mpmath.log(mpmath.pi) / 2
        + mpmath.loggamma((mp_n + 1) / 2)
        - mpmath.loggamma((mp_n + 2) / 2)
        + mpmath.log(
            mpmath.hyp2f1(
                (mp_n - 1) / 2, (mp_n - 1) / 2, (mp_n + 2) / 2, mp_r**2
            )
        )
    )
    correction = (
        2
        * mp_r
        / (mp_n + 1)
        * mpmath.exp(
            2
            * (
                mpmath.loggamma(mp_n / 2)
                - mpmath.loggamma((mp_n - 1) / 2)
            )
        )
        * mpmath.hyp3f2(
            1,
            mp_n / 2,
            mp_n / 2,
            mpmath.mpf(3) / 2,
            (mp_n + 3) / 2,
            mp_r**2,
        )
    )
    return float(max(mp_bf + correction if alternative == "greater" else mp_bf - correction, 0))
