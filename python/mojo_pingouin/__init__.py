"""Pingouin-compatible statistical tests and effect sizes accelerated by Mojo."""

from .correlation import corr, distance_corr
from .effsize import (
    compute_effsize,
    compute_effsize_from_t,
    compute_esci,
    convert_effsize,
)
from .nonparametric import kruskal, mwu, wilcoxon
from .parametric import anova, ttest, welch_anova

__all__ = [
    "anova",
    "compute_effsize",
    "compute_effsize_from_t",
    "compute_esci",
    "convert_effsize",
    "corr",
    "distance_corr",
    "kruskal",
    "mwu",
    "ttest",
    "welch_anova",
    "wilcoxon",
]

__version__ = "0.1.0"
