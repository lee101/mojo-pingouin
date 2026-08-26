from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pg = pytest.importorskip("pingouin")

import mojo_pingouin as mpg
from mojo_pingouin import _lib
from mojo_pingouin._common import clean_samples


def assert_result_equal(ours: pd.DataFrame, theirs: pd.DataFrame) -> None:
    assert ours.index.tolist() == theirs.index.tolist()
    assert ours.columns.tolist() == theirs.columns.tolist()
    for column in ours:
        for left, right in zip(ours[column], theirs[column]):
            if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                assert np.allclose(left, right, rtol=1e-11, atol=1e-12, equal_nan=True)
            elif isinstance(left, str) or isinstance(right, str):
                assert left == right
            else:
                assert left == pytest.approx(right, rel=1e-11, abs=1e-12, nan_ok=True)


@pytest.fixture
def samples():
    x = np.array([5.5, 2.4, 6.8, np.nan, 9.6, 4.2, 7.1, 3.3])
    y = np.array([6.4, 3.4, 6.4, 8.0, 11.0, 4.8, np.nan, 4.1])
    return x, y


@pytest.mark.parametrize(
    ("output_type", "nx", "ny"),
    [
        ("cohen", None, None),
        ("hedges", 20, 18),
        ("pointbiserialr", 20, 18),
        ("eta_square", None, None),
        ("odds_ratio", None, None),
        ("AUC", None, None),
    ],
)
def test_convert_effsize(output_type, nx, ny):
    ours = mpg.convert_effsize(0.63, "cohen", output_type, nx=nx, ny=ny)
    theirs = pg.convert_effsize(0.63, "cohen", output_type, nx=nx, ny=ny)
    assert ours == pytest.approx(theirs, rel=1e-14)


@pytest.mark.parametrize(
    ("paired", "eftype"),
    [
        (False, "cohen"),
        (False, "hedges"),
        (False, "cles"),
        (True, "cohen"),
        (True, "cohen_dz"),
    ],
)
def test_compute_effsize(samples, paired, eftype):
    x, y = samples
    ours = mpg.compute_effsize(x, y, paired=paired, eftype=eftype)
    theirs = pg.compute_effsize(x, y, paired=paired, eftype=eftype)
    assert ours == pytest.approx(theirs, rel=1e-13, abs=1e-14)


def test_compute_effsize_correlation_and_one_sample():
    x = np.array([2.0, 3.0, 8.0, 5.0, 9.0, 11.0])
    y = np.array([1.0, 4.0, 7.0, 6.0, 8.0, 10.0])
    assert mpg.compute_effsize(x, y, eftype="r") == pytest.approx(
        pg.compute_effsize(x, y, eftype="r"), rel=1e-14
    )
    assert mpg.compute_effsize(x, 4.5) == pytest.approx(
        pg.compute_effsize(x, 4.5), rel=1e-14
    )


def test_simd_tail_and_zero_copy_clean_path():
    x = np.linspace(-2.0, 3.0, 11)
    y = np.linspace(1.0, 4.0, 11)
    clean_x, clean_y, paired = clean_samples(x, y, paired=False)
    assert not paired
    assert np.shares_memory(clean_x, x)
    assert np.shares_memory(clean_y, y)
    assert mpg.compute_effsize(x, y) == pytest.approx(
        pg.compute_effsize(x, y), rel=1e-13, abs=1e-14
    )


def test_moments_parallel_threshold_and_remainder():
    for n in (999_999, 1_000_003):
        values = np.linspace(-5.0, 7.0, n)
        mean, m2 = _lib.moments(values)
        assert mean == pytest.approx(values.mean(), rel=1e-13, abs=1e-12)
        assert m2 == pytest.approx(
            np.sum((values - values.mean()) ** 2),
            rel=1e-13,
            abs=1e-12,
        )


def test_ffi_rejects_invalid_buffers_before_calling_mojo():
    with pytest.raises(ValueError, match="at least one"):
        _lib.moments(np.array([], dtype=np.float64))
    with pytest.raises(ValueError, match="equal, non-empty"):
        _lib.bivariate(np.ones(3), np.ones(2))
    with pytest.raises(ValueError, match="non-empty"):
        _lib.cles(np.ones(1), np.array([]))
    with pytest.raises(ValueError, match="equal-length"):
        _lib.group_moments(np.ones(2), np.array([0]), 1)
    with pytest.raises(ValueError, match="group codes"):
        _lib.group_moments(np.ones(2), np.array([0, 2]), 2)
    with pytest.raises(ValueError, match="non-empty"):
        _lib.centered_distance(np.empty((0, 2)))
    with pytest.raises(ValueError, match="equal, non-zero"):
        _lib.distance_dot(np.ones(4), np.ones(3))


def test_ffi_normalizes_dtype_and_strides():
    base = np.arange(24, dtype=np.float32).reshape(6, 4)
    view = base[:, ::2]
    matrix, squared_sum = _lib.centered_distance(view)
    expected = pg.distance_corr(view, view, n_boot=None)
    actual = np.sqrt(_lib.distance_dot(matrix, matrix)) / np.sqrt(squared_sum)
    assert actual == pytest.approx(expected, rel=1e-14)

    a = np.arange(16, dtype=np.float32).reshape(4, 4).T
    permutations = np.array([[3, 2, 1, 0]], dtype=np.int32)
    result = _lib.permuted_dots(a, a, permutations)
    expected_dot = sum(
        a[i, j] * a[3 - i, 3 - j] for i in range(4) for j in range(4)
    )
    assert result[0] == pytest.approx(expected_dot)


def test_ffi_rejects_unsafe_permutation_inputs():
    matrix = np.eye(3)
    with pytest.raises(ValueError, match="one column"):
        _lib.permuted_dots(matrix, matrix, np.array([[0, 1]]))
    with pytest.raises(ValueError, match="out of bounds"):
        _lib.permuted_dots(matrix, matrix, np.array([[0, 1, 3]]))
    with pytest.raises(TypeError, match="integer index"):
        _lib.permuted_dots(matrix, matrix, np.array([[0.0, 1.0, 2.0]]))
    with pytest.raises(OverflowError, match="fit in int64"):
        _lib.permuted_dots(
            matrix,
            matrix,
            np.array([[0, 1, 2**63]], dtype=np.uint64),
        )


def test_compute_effsize_from_t():
    for kwargs in (
        {"nx": 35, "ny": 25, "eftype": "cohen"},
        {"N": 60, "eftype": "pointbiserialr"},
    ):
        assert mpg.compute_effsize_from_t(2.9, **kwargs) == pytest.approx(
            pg.compute_effsize_from_t(2.9, **kwargs), rel=1e-14
        )


@pytest.mark.parametrize(
    ("eftype", "alternative"),
    [
        ("cohen", "two-sided"),
        ("r", "two-sided"),
        ("r", "greater"),
        ("r", "less"),
    ],
)
def test_compute_esci(eftype, alternative):
    kwargs = {
        "stat": 0.43,
        "nx": 40,
        "ny": 32,
        "eftype": eftype,
        "alternative": alternative,
        "decimals": 6,
    }
    assert np.array_equal(mpg.compute_esci(**kwargs), pg.compute_esci(**kwargs))


def test_ttest_one_sample(samples):
    x, _ = samples
    assert_result_equal(mpg.ttest(x, 4.0), pg.ttest(x, 4.0))


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_ttest_paired(samples, alternative):
    x, y = samples
    assert_result_equal(
        mpg.ttest(x, y, paired=True, alternative=alternative),
        pg.ttest(x, y, paired=True, alternative=alternative),
    )


@pytest.mark.parametrize("correction", [False, True, "auto"])
def test_ttest_independent(correction):
    rng = np.random.default_rng(14)
    x = rng.normal(0.3, 1.2, 53)
    y = rng.normal(-0.1, 0.8, 37)
    assert_result_equal(
        mpg.ttest(x, y, correction=correction),
        pg.ttest(x, y, correction=correction),
    )


@pytest.mark.parametrize(
    ("method", "alternative"),
    [
        ("pearson", "two-sided"),
        ("pearson", "greater"),
        ("pearson", "less"),
        ("spearman", "two-sided"),
        ("kendall", "two-sided"),
    ],
)
def test_corr(samples, method, alternative):
    x, y = samples
    assert_result_equal(
        mpg.corr(x, y, method=method, alternative=alternative),
        pg.corr(x, y, method=method, alternative=alternative),
    )


def test_corr_zero_copy_clean_path(monkeypatch):
    x = np.linspace(-2.0, 3.0, 101)
    y = np.linspace(1.0, 4.0, 101) ** 2
    original = _lib.bivariate
    seen = []

    def recording_bivariate(clean_x, clean_y):
        seen.append((np.shares_memory(clean_x, x), np.shares_memory(clean_y, y)))
        return original(clean_x, clean_y)

    monkeypatch.setattr(_lib, "bivariate", recording_bivariate)
    assert_result_equal(mpg.corr(x, y), pg.corr(x, y))
    assert seen == [(True, True)]


def test_distance_corr_one_and_two_dimensional():
    rng = np.random.default_rng(22)
    for x, y in (
        (rng.normal(size=30), rng.normal(size=30)),
        (rng.normal(size=(20, 3)), rng.normal(size=(20, 2))),
    ):
        assert mpg.distance_corr(x, y, n_boot=None) == pytest.approx(
            pg.distance_corr(x, y, n_boot=None), rel=2e-14
        )


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_distance_corr_permutation(alternative):
    x = np.arange(14.0)
    y = np.array([2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0.0])
    ours = mpg.distance_corr(x, y, alternative=alternative, n_boot=80, seed=7)
    theirs = pg.distance_corr(x, y, alternative=alternative, n_boot=80, seed=7)
    assert np.allclose(ours, theirs, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_mwu(samples, alternative):
    x, y = samples
    assert_result_equal(
        mpg.mwu(x, y, alternative=alternative),
        pg.mwu(x, y, alternative=alternative),
    )


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_wilcoxon(samples, alternative):
    x, y = samples
    assert_result_equal(
        mpg.wilcoxon(x, y, alternative=alternative),
        pg.wilcoxon(x, y, alternative=alternative),
    )


def test_wilcoxon_differences():
    differences = np.array([-3, -1, 0, 2, 4, 8, -2, 1.0])
    assert_result_equal(mpg.wilcoxon(differences), pg.wilcoxon(differences))


@pytest.fixture
def grouped_data():
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "score": np.r_[
                rng.normal(1, 1, 31),
                rng.normal(2, 1.8, 25),
                rng.normal(3.5, 0.7, 42),
            ],
            "condition": np.repeat(["a", "b", "c"], [31, 25, 42]),
        }
    )


@pytest.mark.parametrize("detailed", [False, True])
def test_anova(grouped_data, detailed):
    assert_result_equal(
        mpg.anova(grouped_data, dv="score", between="condition", detailed=detailed),
        pg.anova(grouped_data, dv="score", between="condition", detailed=detailed),
    )


def test_anova_large_offset_is_stable():
    data = pd.DataFrame(
        {
            "score": 1e12 + np.arange(30, dtype=float),
            "condition": np.repeat(["a", "b", "c"], 10),
        }
    )
    assert_result_equal(
        mpg.anova(data, dv="score", between="condition", detailed=True),
        pg.anova(data, dv="score", between="condition", detailed=True),
    )


def test_welch_anova(grouped_data):
    assert_result_equal(
        mpg.welch_anova(grouped_data, dv="score", between="condition"),
        pg.welch_anova(grouped_data, dv="score", between="condition"),
    )


def test_kruskal(grouped_data):
    assert_result_equal(
        mpg.kruskal(grouped_data, dv="score", between="condition"),
        pg.kruskal(grouped_data, dv="score", between="condition"),
    )
