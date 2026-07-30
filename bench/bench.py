from __future__ import annotations

import gc
import math
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_pingouin as mpg
import pingouin as pg


def time_best(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def machine_name():
    cpu = platform.processor()
    if not cpu or cpu.lower() in {"x86_64", "amd64"}:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                cpu = next(
                    line.split(":", 1)[1].strip()
                    for line in handle
                    if line.startswith("model name")
                )
        except (OSError, StopIteration):
            cpu = platform.machine()
    return f"{cpu}, {platform.system()} {platform.release()}, Python {platform.python_version()}"


def main():
    rng = np.random.default_rng(42)
    cases = []

    x = rng.normal(0.2, 1.1, 5_000_000)
    y = rng.normal(-0.1, 0.9, 5_000_000)
    cases.append(
        (
            "Cohen d (5M + 5M)",
            lambda: mpg.compute_effsize(x, y),
            lambda: pg.compute_effsize(x, y),
            5,
        )
    )
    cases.append(
        (
            "Welch t-test (5M + 5M)",
            lambda: mpg.ttest(x, y, correction=True),
            lambda: pg.ttest(x, y, correction=True),
            5,
        )
    )
    cases.append(
        (
            "Pearson corr (5M pairs)",
            lambda: mpg.corr(x, y),
            lambda: pg.corr(x, y),
            5,
        )
    )

    cles_x = rng.integers(0, 50, 4_000).astype(np.float64)
    cles_y = rng.integers(0, 50, 4_000).astype(np.float64)
    cases.append(
        (
            "CLES (4k x 4k pairs)",
            lambda: mpg.compute_effsize(cles_x, cles_y, eftype="cles"),
            lambda: pg.compute_effsize(cles_x, cles_y, eftype="cles"),
            3,
        )
    )

    scores = rng.normal(size=1_000_000) + np.repeat(np.arange(8), 125_000) * 0.1
    frame = pd.DataFrame(
        {
            "score": scores,
            "group": pd.Categorical(np.repeat(np.arange(8), 125_000)),
        }
    )
    cases.append(
        (
            "one-way ANOVA (1M, 8 groups)",
            lambda: mpg.anova(frame, dv="score", between="group"),
            lambda: pg.anova(frame, dv="score", between="group"),
            5,
        )
    )

    distance_x = rng.normal(size=(300, 4))
    distance_y = rng.normal(size=(300, 3))
    cases.append(
        (
            "distance_corr (300, 200 perms)",
            lambda: mpg.distance_corr(distance_x, distance_y, n_boot=200, seed=1),
            lambda: pg.distance_corr(distance_x, distance_y, n_boot=200, seed=1),
            3,
        )
    )

    print(f"Machine: {machine_name()}")
    print()
    print("| case | mojo-pingouin | pingouin | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, upstream, repeat in cases:
        ours()
        upstream()
        mojo_time = time_best(ours, repeat)
        python_time = time_best(upstream, repeat)
        ratio = python_time / mojo_time
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1
            else f"{1 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{python_time * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
