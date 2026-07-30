# mojo-pingouin

`mojo-pingouin` is a Mojo implementation of compute-heavy statistical tests
and effect sizes from [Pingouin](https://pingouin-stats.org/), exposed through a
Python API with familiar function names and result schemas for the covered subset.
It is useful when large arrays or permutation tests make the numerical core,
rather than DataFrame presentation, the expensive part of an analysis.

The return values use Pingouin's DataFrame schemas, including confidence
interval arrays, effect-size columns, achieved power, and Bayes factors.
Missing observations follow Pingouin's paired or independent deletion rules.

## Coverage

| area | covered API |
| --- | --- |
| Parametric tests | `ttest` (one-sample, paired, pooled, Welch, all alternatives), one-way `anova`, `welch_anova` |
| Correlation | `corr` with Pearson, Spearman, and Kendall methods; `distance_corr` with permutation p-values |
| Nonparametric tests | `mwu`, `wilcoxon`, `kruskal` |
| Effect sizes | `compute_effsize`, `compute_effsize_from_t`, `convert_effsize`, `compute_esci`; Cohen d, paired d-avg, Cohen dz, Hedges g, CLES, point-biserial r, eta squared, odds ratio, and AUC |

The current scope does not include repeated-measures, mixed, or multifactor
ANOVA; partial and robust correlations; pairwise/post-hoc testing; contingency
tests; normality and sphericity diagnostics; or bootstrap confidence intervals.
Those names are intentionally absent rather than implemented as approximations.

The parity suite compares every covered result directly with the real
`pingouin==0.6.1` package. It checks numerical values and behavioral details,
not merely successful execution.

## Install and run

The repository carries its own pinned Mojo nightly through Pixi:

```bash
pixi install
pixi run build
pixi run python - <<'PY'
import mojo_pingouin as pg

x = [5.5, 2.4, 6.8, 9.6, 4.2]
y = [6.4, 3.4, 6.4, 11.0, 4.8]

print(pg.ttest(x, y, paired=True))
print(pg.compute_effsize(x, y, paired=True, eftype="hedges"))
PY
```

`pixi run build` produces `dist/libmojo-pingouin.so`. The Python wrapper also
rebuilds a missing or stale library on first use. Set `MOJO_PINGOUIN_LIB` to use
an already-built library in another location.

Run the validation and benchmark suites with `pixi run test` and
`pixi run bench`.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-136-generic, Python 3.13.14. Times are the best of five runs, except the
larger quadratic cases, which use three. Both implementations receive the same
preallocated input arrays.

| case | mojo-pingouin | pingouin | result |
| --- | ---: | ---: | ---: |
| Cohen d (5M + 5M) | 14.80 ms | 118.86 ms | 8.03x faster |
| Welch t-test (5M + 5M) | 31.48 ms | 473.24 ms | 15.03x faster |
| Pearson corr (5M pairs) | 114.07 ms | 405.23 ms | 3.55x faster |
| CLES (4k x 4k pairs) | 36.45 ms | 270.65 ms | 7.43x faster |
| one-way ANOVA (1M, 8 groups) | 20.25 ms | 96.26 ms | 4.75x faster |
| distance_corr (300, 200 permutations) | 48.35 ms | 5700.49 ms | 117.91x faster |

The large distance-correlation gain comes from computing and double-centering
each distance matrix once. A permutation then reindexes the centered matrix
inside Mojo; it does not recalculate pairwise Euclidean distances. This changes
the amount of work without changing the statistic or permutation distribution.

## How it works

All kernels live in one Mojo compilation unit. Python owns NumPy arrays and
scratch storage, then makes one `ctypes` call per reduction or permutation
batch. Buffers cross the C ABI as integer addresses and are reconstructed as
`UnsafePointer[Float64, AnyOrigin[mut=True]]` or `Int64` pointers inside the
exported function. Mojo never retains or frees Python-owned memory.

Inputs are converted to contiguous row-major `float64` arrays only when needed.
Moment and covariance kernels use two passes so large offsets remain stable;
the bulk loops use the host's native `float64` SIMD width with scalar remainder
loops. Clean contiguous NumPy inputs stay zero-copy after a SIMD NaN scan.
Moment reductions use 16 independent CPU tasks at one million elements and
above, while smaller inputs remain serial to avoid launch overhead. Grouped
aggregation supplies the sufficient statistics for ANOVA. SciPy remains
responsible for distribution CDFs, critical values, rank tests, and quadrature,
while Mojo performs the array-sized work.

No GPU path is included.

## License

MIT
