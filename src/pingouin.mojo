"""Numerical kernels and their C ABI for mojo-pingouin."""

from max.algorithm import parallelize
from std.math import sqrt
from std.runtime import initialize_runtime
from std.sys.info import simd_width_of as simdwidthof
from std.utils.numerics import isnan

comptime Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime IPtr = Pointer[Int64, MutUntrackedOrigin]
comptime W = simdwidthof[DType.float64]()
comptime MOMENT_PARTS = 16
comptime PARALLEL_MOMENTS_MIN = 1_000_000


def _p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def _ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def has_nan(x: Ptr, n: Int) -> Int:
    var i = 0
    while i + W <= n:
        var values = x.unsafe_load[width=W](i)
        if isnan(values).reduce_or():
            return 1
        i += W
    while i < n:
        if x[unsafe_offset=i] != x[unsafe_offset=i]:
            return 1
        i += 1
    return 0


def moments_serial(x: Ptr, n: Int, result: Ptr):
    var vsum = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        vsum += x.unsafe_load[width=W](i)
        i += W
    var total = vsum.reduce_add()
    while i < n:
        total += x[unsafe_offset=i]
        i += 1
    var mean = total / Float64(n)

    var vm2 = SIMD[DType.float64, W](0.0)
    i = 0
    while i + W <= n:
        var delta = x.unsafe_load[width=W](i) - mean
        vm2 += delta * delta
        i += W
    var m2 = vm2.reduce_add()
    while i < n:
        var delta = x[unsafe_offset=i] - mean
        m2 += delta * delta
        i += 1
    result[unsafe_offset=0] = mean
    result[unsafe_offset=1] = m2


def moments(x: Ptr, n: Int, result: Ptr):
    if n < PARALLEL_MOMENTS_MIN:
        moments_serial(x, n, result)
        return

    var chunk_size = (n + MOMENT_PARTS - 1) // MOMENT_PARTS
    @__parameter
    def sum_chunk(chunk: Int):
        var start = chunk * chunk_size
        var end = min(start + chunk_size, n)
        var vsum = SIMD[DType.float64, W](0.0)
        var i = start
        while i + W <= end:
            vsum += x.unsafe_load[width=W](i)
            i += W
        var total = vsum.reduce_add()
        while i < end:
            total += x[unsafe_offset=i]
            i += 1
        result[unsafe_offset=chunk] = total

    parallelize[sum_chunk](MOMENT_PARTS, MOMENT_PARTS)

    var total = 0.0
    for chunk in range(MOMENT_PARTS):
        total += result[unsafe_offset=chunk]
    var mean = total / Float64(n)

    @__parameter
    def m2_chunk(chunk: Int):
        var start = chunk * chunk_size
        var end = min(start + chunk_size, n)
        var vm2 = SIMD[DType.float64, W](0.0)
        var i = start
        while i + W <= end:
            var delta = x.unsafe_load[width=W](i) - mean
            vm2 += delta * delta
            i += W
        var m2 = vm2.reduce_add()
        while i < end:
            var delta = x[unsafe_offset=i] - mean
            m2 += delta * delta
            i += 1
        result[unsafe_offset=MOMENT_PARTS + chunk] = m2

    parallelize[m2_chunk](MOMENT_PARTS, MOMENT_PARTS)

    var m2 = 0.0
    for chunk in range(MOMENT_PARTS):
        m2 += result[unsafe_offset=MOMENT_PARTS + chunk]
    result[unsafe_offset=0] = mean
    result[unsafe_offset=1] = m2


def bivariate(x: Ptr, y: Ptr, n: Int, result: Ptr):
    var sx = SIMD[DType.float64, W](0.0)
    var sy = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        sx += x.unsafe_load[width=W](i)
        sy += y.unsafe_load[width=W](i)
        i += W
    var tx = sx.reduce_add()
    var ty = sy.reduce_add()
    while i < n:
        tx += x[unsafe_offset=i]
        ty += y[unsafe_offset=i]
        i += 1
    var mx = tx / Float64(n)
    var my = ty / Float64(n)

    var vxx = SIMD[DType.float64, W](0.0)
    var vyy = SIMD[DType.float64, W](0.0)
    var vxy = SIMD[DType.float64, W](0.0)
    i = 0
    while i + W <= n:
        var dx = x.unsafe_load[width=W](i) - mx
        var dy = y.unsafe_load[width=W](i) - my
        vxx += dx * dx
        vyy += dy * dy
        vxy += dx * dy
        i += W
    var xx = vxx.reduce_add()
    var yy = vyy.reduce_add()
    var xy = vxy.reduce_add()
    while i < n:
        var dx = x[unsafe_offset=i] - mx
        var dy = y[unsafe_offset=i] - my
        xx += dx * dx
        yy += dy * dy
        xy += dx * dy
        i += 1
    result[unsafe_offset=0] = mx
    result[unsafe_offset=1] = my
    result[unsafe_offset=2] = xx
    result[unsafe_offset=3] = yy
    result[unsafe_offset=4] = xy


def cles(x: Ptr, y: Ptr, nx: Int, ny: Int) -> Float64:
    var score = 0.0
    for i in range(nx):
        for j in range(ny):
            if x[unsafe_offset=i] > y[unsafe_offset=j]:
                score += 1.0
            elif x[unsafe_offset=i] == y[unsafe_offset=j]:
                score += 0.5
    return score / Float64(nx * ny)


def group_moments(
    values: Ptr,
    codes: IPtr,
    n: Int,
    groups: Int,
    sums: Ptr,
    sumsq: Ptr,
    counts: IPtr,
):
    for g in range(groups):
        sums[unsafe_offset=g] = 0.0
        sumsq[unsafe_offset=g] = 0.0
        counts[unsafe_offset=g] = 0
    for i in range(n):
        var g = Int(codes[unsafe_offset=i])
        if g < 0 or g >= groups:
            continue
        var v = values[unsafe_offset=i]
        sums[unsafe_offset=g] += v
        counts[unsafe_offset=g] += 1
    for i in range(n):
        var g = Int(codes[unsafe_offset=i])
        if g < 0 or g >= groups:
            continue
        var delta = values[unsafe_offset=i] - sums[unsafe_offset=g] / Float64(
            counts[unsafe_offset=g]
        )
        sumsq[unsafe_offset=g] += delta * delta


def centered_distance(
    x: Ptr, n: Int, d: Int, matrix: Ptr, rows: Ptr
) -> Float64:
    for i in range(n):
        rows[unsafe_offset=i] = 0.0
        matrix[unsafe_offset=i * n + i] = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            var squared = 0.0
            for k in range(d):
                var delta = (
                    x[unsafe_offset=i * d + k] - x[unsafe_offset=j * d + k]
                )
                squared += delta * delta
            var distance = sqrt(squared)
            matrix[unsafe_offset=i * n + j] = distance
            matrix[unsafe_offset=j * n + i] = distance
            rows[unsafe_offset=i] += distance
            rows[unsafe_offset=j] += distance

    var grand = 0.0
    for i in range(n):
        grand += rows[unsafe_offset=i]
    var invn = 1.0 / Float64(n)
    var grand_mean = grand * invn * invn
    var squared_sum = 0.0
    for i in range(n):
        var row_mean = rows[unsafe_offset=i] * invn
        for j in range(n):
            var centered = (
                matrix[unsafe_offset=i * n + j]
                - row_mean
                - rows[unsafe_offset=j] * invn
                + grand_mean
            )
            matrix[unsafe_offset=i * n + j] = centered
            squared_sum += centered * centered
    return squared_sum


def distance_dot(a: Ptr, b: Ptr, n2: Int) -> Float64:
    var vacc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n2:
        vacc += a.unsafe_load[width=W](i) * b.unsafe_load[width=W](i)
        i += W
    var acc = vacc.reduce_add()
    while i < n2:
        acc += a[unsafe_offset=i] * b[unsafe_offset=i]
        i += 1
    return acc


def permuted_dots(
    a: Ptr, b: Ptr, permutations: IPtr, n_boot: Int, n: Int, result: Ptr
):
    for boot in range(n_boot):
        var acc = 0.0
        var offset = boot * n
        for i in range(n):
            var pi = Int(permutations[unsafe_offset=offset + i])
            for j in range(n):
                var pj = Int(permutations[unsafe_offset=offset + j])
                acc += a[unsafe_offset=i * n + j] * b[unsafe_offset=pi * n + pj]
        result[unsafe_offset=boot] = acc


@export("mpg_moments")
def mpg_moments(x: Int, n: Int, result: Int) abi("C"):
    initialize_runtime()
    moments(_p(x), n, _p(result))


@export("mpg_has_nan")
def mpg_has_nan(x: Int, n: Int) abi("C") -> Int:
    initialize_runtime()
    return has_nan(_p(x), n)


@export("mpg_bivariate")
def mpg_bivariate(x: Int, y: Int, n: Int, result: Int) abi("C"):
    initialize_runtime()
    bivariate(_p(x), _p(y), n, _p(result))


@export("mpg_cles")
def mpg_cles(x: Int, y: Int, nx: Int, ny: Int) abi("C") -> Float64:
    initialize_runtime()
    return cles(_p(x), _p(y), nx, ny)


@export("mpg_group_moments")
def mpg_group_moments(
    values: Int,
    codes: Int,
    n: Int,
    groups: Int,
    sums: Int,
    sumsq: Int,
    counts: Int,
) abi("C"):
    initialize_runtime()
    group_moments(
        _p(values), _ip(codes), n, groups, _p(sums), _p(sumsq), _ip(counts)
    )


@export("mpg_centered_distance")
def mpg_centered_distance(
    x: Int, n: Int, d: Int, matrix: Int, rows: Int
) abi("C") -> Float64:
    initialize_runtime()
    return centered_distance(_p(x), n, d, _p(matrix), _p(rows))


@export("mpg_distance_dot")
def mpg_distance_dot(a: Int, b: Int, n2: Int) abi("C") -> Float64:
    initialize_runtime()
    return distance_dot(_p(a), _p(b), n2)


@export("mpg_permuted_dots")
def mpg_permuted_dots(
    a: Int, b: Int, permutations: Int, n_boot: Int, n: Int, result: Int
) abi("C"):
    initialize_runtime()
    permuted_dots(_p(a), _p(b), _ip(permutations), n_boot, n, _p(result))
