"""Numerical kernels and their C ABI for mojo-pingouin."""

from std.algorithm import parallelize
from std.math import sqrt
from std.sys.info import simd_width_of as simdwidthof
from std.utils.numerics import isnan

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
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
        var values = x.load[width=W](i)
        if isnan(values).reduce_or():
            return 1
        i += W
    while i < n:
        if x[i] != x[i]:
            return 1
        i += 1
    return 0


def moments_serial(x: Ptr, n: Int, result: Ptr):
    var vsum = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        vsum += x.load[width=W](i)
        i += W
    var total = vsum.reduce_add()
    while i < n:
        total += x[i]
        i += 1
    var mean = total / Float64(n)

    var vm2 = SIMD[DType.float64, W](0.0)
    i = 0
    while i + W <= n:
        var delta = x.load[width=W](i) - mean
        vm2 += delta * delta
        i += W
    var m2 = vm2.reduce_add()
    while i < n:
        var delta = x[i] - mean
        m2 += delta * delta
        i += 1
    result[0] = mean
    result[1] = m2


def moments(x: Ptr, n: Int, result: Ptr):
    if n < PARALLEL_MOMENTS_MIN:
        moments_serial(x, n, result)
        return

    var chunk_size = (n + MOMENT_PARTS - 1) // MOMENT_PARTS

    @parameter
    def sum_chunk(chunk: Int):
        var start = chunk * chunk_size
        var end = min(start + chunk_size, n)
        var vsum = SIMD[DType.float64, W](0.0)
        var i = start
        while i + W <= end:
            vsum += x.load[width=W](i)
            i += W
        var total = vsum.reduce_add()
        while i < end:
            total += x[i]
            i += 1
        result[chunk] = total

    parallelize[sum_chunk](MOMENT_PARTS)
    var total = 0.0
    for chunk in range(MOMENT_PARTS):
        total += result[chunk]
    var mean = total / Float64(n)

    @parameter
    def m2_chunk(chunk: Int):
        var start = chunk * chunk_size
        var end = min(start + chunk_size, n)
        var vm2 = SIMD[DType.float64, W](0.0)
        var i = start
        while i + W <= end:
            var delta = x.load[width=W](i) - mean
            vm2 += delta * delta
            i += W
        var m2 = vm2.reduce_add()
        while i < end:
            var delta = x[i] - mean
            m2 += delta * delta
            i += 1
        result[MOMENT_PARTS + chunk] = m2

    parallelize[m2_chunk](MOMENT_PARTS)
    var m2 = 0.0
    for chunk in range(MOMENT_PARTS):
        m2 += result[MOMENT_PARTS + chunk]
    result[0] = mean
    result[1] = m2


def bivariate(x: Ptr, y: Ptr, n: Int, result: Ptr):
    var sx = SIMD[DType.float64, W](0.0)
    var sy = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        sx += x.load[width=W](i)
        sy += y.load[width=W](i)
        i += W
    var tx = sx.reduce_add()
    var ty = sy.reduce_add()
    while i < n:
        tx += x[i]
        ty += y[i]
        i += 1
    var mx = tx / Float64(n)
    var my = ty / Float64(n)

    var vxx = SIMD[DType.float64, W](0.0)
    var vyy = SIMD[DType.float64, W](0.0)
    var vxy = SIMD[DType.float64, W](0.0)
    i = 0
    while i + W <= n:
        var dx = x.load[width=W](i) - mx
        var dy = y.load[width=W](i) - my
        vxx += dx * dx
        vyy += dy * dy
        vxy += dx * dy
        i += W
    var xx = vxx.reduce_add()
    var yy = vyy.reduce_add()
    var xy = vxy.reduce_add()
    while i < n:
        var dx = x[i] - mx
        var dy = y[i] - my
        xx += dx * dx
        yy += dy * dy
        xy += dx * dy
        i += 1
    result[0] = mx
    result[1] = my
    result[2] = xx
    result[3] = yy
    result[4] = xy


def cles(x: Ptr, y: Ptr, nx: Int, ny: Int) -> Float64:
    var score = 0.0
    for i in range(nx):
        for j in range(ny):
            if x[i] > y[j]:
                score += 1.0
            elif x[i] == y[j]:
                score += 0.5
    return score / Float64(nx * ny)


def group_moments(
    values: Ptr, codes: IPtr, n: Int, groups: Int, sums: Ptr, sumsq: Ptr, counts: IPtr
):
    for g in range(groups):
        sums[g] = 0.0
        sumsq[g] = 0.0
        counts[g] = 0
    for i in range(n):
        var g = Int(codes[i])
        if g < 0 or g >= groups:
            continue
        var v = values[i]
        sums[g] += v
        counts[g] += 1
    for i in range(n):
        var g = Int(codes[i])
        if g < 0 or g >= groups:
            continue
        var delta = values[i] - sums[g] / Float64(counts[g])
        sumsq[g] += delta * delta


def centered_distance(x: Ptr, n: Int, d: Int, matrix: Ptr, rows: Ptr) -> Float64:
    for i in range(n):
        rows[i] = 0.0
        matrix[i * n + i] = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            var squared = 0.0
            for k in range(d):
                var delta = x[i * d + k] - x[j * d + k]
                squared += delta * delta
            var distance = sqrt(squared)
            matrix[i * n + j] = distance
            matrix[j * n + i] = distance
            rows[i] += distance
            rows[j] += distance

    var grand = 0.0
    for i in range(n):
        grand += rows[i]
    var invn = 1.0 / Float64(n)
    var grand_mean = grand * invn * invn
    var squared_sum = 0.0
    for i in range(n):
        var row_mean = rows[i] * invn
        for j in range(n):
            var centered = matrix[i * n + j] - row_mean - rows[j] * invn + grand_mean
            matrix[i * n + j] = centered
            squared_sum += centered * centered
    return squared_sum


def distance_dot(a: Ptr, b: Ptr, n2: Int) -> Float64:
    var vacc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n2:
        vacc += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var acc = vacc.reduce_add()
    while i < n2:
        acc += a[i] * b[i]
        i += 1
    return acc


def permuted_dots(
    a: Ptr, b: Ptr, permutations: IPtr, n_boot: Int, n: Int, result: Ptr
):
    for boot in range(n_boot):
        var acc = 0.0
        var offset = boot * n
        for i in range(n):
            var pi = Int(permutations[offset + i])
            for j in range(n):
                var pj = Int(permutations[offset + j])
                acc += a[i * n + j] * b[pi * n + pj]
        result[boot] = acc


@export("mpg_moments")
def mpg_moments(x: Int, n: Int, result: Int) abi("C"):
    moments(_p(x), n, _p(result))


@export("mpg_has_nan")
def mpg_has_nan(x: Int, n: Int) abi("C") -> Int:
    return has_nan(_p(x), n)


@export("mpg_bivariate")
def mpg_bivariate(x: Int, y: Int, n: Int, result: Int) abi("C"):
    bivariate(_p(x), _p(y), n, _p(result))


@export("mpg_cles")
def mpg_cles(x: Int, y: Int, nx: Int, ny: Int) abi("C") -> Float64:
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
    group_moments(
        _p(values), _ip(codes), n, groups, _p(sums), _p(sumsq), _ip(counts)
    )


@export("mpg_centered_distance")
def mpg_centered_distance(
    x: Int, n: Int, d: Int, matrix: Int, rows: Int
) abi("C") -> Float64:
    return centered_distance(_p(x), n, d, _p(matrix), _p(rows))


@export("mpg_distance_dot")
def mpg_distance_dot(a: Int, b: Int, n2: Int) abi("C") -> Float64:
    return distance_dot(_p(a), _p(b), n2)


@export("mpg_permuted_dots")
def mpg_permuted_dots(
    a: Int, b: Int, permutations: Int, n_boot: Int, n: Int, result: Int
) abi("C"):
    permuted_dots(_p(a), _p(b), _ip(permutations), n_boot, n, _p(result))
