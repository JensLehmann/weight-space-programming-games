import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def c1(o, i, value=1.0, dy=0, dx=0):
        p["W1"][o, i, dy + 2, dx + 2] += value

    def c2(o, i, value=1.0, dy=0, dx=0):
        p["W2"][o, i, dy + 1, dx + 1] += value

    def f(o, i, value=1.0):
        p["W3"][o, i] += value

    def h(o, i, value=1.0, dy=0, dx=0):
        p["Wh"][o, i, dy + 1, dx + 1] += value

    def bounded(o, raw, value=1.0, dy=0, dx=0):
        # ReLU(v) - ReLU(v-1) clips nonnegative v to one.
        h(o, raw, value, dy, dx)
        h(o, raw + 1, -value, dy, dx)

    # Palette predicates.
    for color in (8, 9, 15):
        c1(0, color)
    c1(1, 9)
    for color in (11, 13):
        c1(2, color)
    c1(3, 6)

    # Coordinates and complements.
    c1(4, 16)
    c1(5, 17)
    c1(6, 18)
    c1(6, 16, -1)
    c1(7, 18)
    c1(7, 17, -1)
    c1(8, 18)
    c1(9, 12)

    for color in (0, 14):
        c1(10, color)

    for j, (dy, dx) in enumerate(directions):
        c1(11 + j, 18)
        c1(11 + j, 18, -1, dy, dx)
        for color in (11, 13):
            c1(15 + j, color, 1, 2 * dy, 2 * dx)

    # Side boundaries, including the edge of the observation.
    for channel, dx in ((20, -1), (21, 1)):
        c1(channel, 1, 1, 0, dx)
        c1(channel, 18)
        c1(channel, 18, -1, 0, dx)

    # Long UI runs exclude the indicator's isolated corner pixels.
    for color in (0, 14):
        for dy in (-1, 0, 1):
            c1(24, color, 1, dy, 0)
        for dx in (-1, 0, 1):
            c1(25, color, 1, 0, dx)
    p["b1"][24:26] = -1

    # Second-layer predicates.
    for out, inp in [
        (0, 0), (1, 1), (2, 2), (3, 3),
        (4, 5), (5, 20), (6, 4), (7, 21), (8, 6),
    ]:
        c2(out, inp)

    for j in range(4):
        run = 25 if j < 2 else 24
        raw = 9 + 2 * j
        for out, bias in ((raw, -2), (raw + 1, -3)):
            c2(out, run)
            c2(out, 11 + j)
            p["b2"][out] = bias

    # One corner for each isolated rectangular piece.
    c2(17, 0)
    c2(17, 0, -1, -1, 0)
    c2(17, 0, -1, 0, -1)

    # Selected-piece and cup bounding coordinates.
    for j, coordinate in enumerate((4, 5, 6, 7)):
        c2(18 + j, coordinate)
        c2(18 + j, 1)
        p["b2"][18 + j] = -1

        c2(22 + j, coordinate)
        c2(22 + j, 2)
        p["b2"][22 + j] = -1

    # Empty opening pixels immediately before a cup's closed end.
    for j, (dy, dx) in enumerate(directions):
        out = 26 + j
        c2(out, 9)
        c2(out, 2, 1, dy, dx)
        perpendicular = (2, 3) if j < 2 else (0, 1)
        for side in perpendicular:
            c2(out, 15 + side)
        p["b2"][out] = -2

    # Pixels immediately outside horizontal pieces.
    for dx in (-1, 1):
        c2(30, 0, 1, 0, dx)
        c2(31, 1, 1, 0, dx)
    c2(30, 0, -2)
    c2(31, 0, -2)

    def orientation(out, edge, axis, value=1.0):
        raw = 9 + 2 * edge
        offset = 64 if axis == "column" else 32
        f(out, offset + raw, value)
        f(out, offset + raw + 1, -value)

    for out, inp in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 17)]:
        f(out, inp)

    # Gravity in display coordinates: up, down, left, right.
    orientation(5, 1, "column")
    orientation(6, 0, "column")
    orientation(7, 3, "row")
    orientation(8, 2, "row")

    for j in range(4):
        f(9 + j, 18 + j)
        f(13 + j, 22 + j)

    f(17, 26)
    orientation(17, 1, "column")
    p["b3"][17] = -1

    f(18, 27)
    orientation(18, 0, "column")
    p["b3"][18] = -1

    # Selected and unselected exterior endpoints.
    f(19, 31)
    f(20, 30)
    f(20, 31, -1)
    for out in (19, 20):
        orientation(out, 1, "column")
        orientation(out, 0, "column")
        p["b3"][out] = -1

    f(21, 5)
    f(22, 7)

    # Boundary-contact proofs.
    f(23, 5)
    f(23, 0)
    p["b3"][23] = -1

    f(24, 7)
    f(24, 0)
    p["b3"][24] = -1

    # Uncovered source columns: no movable material anywhere in the column.
    f(26, 64 + 3)
    f(26, 64 + 0, -1)

    # Coordinates gated by uncovered-source columns.
    f(27, 64 + 3)
    f(27, 64 + 0, -1)
    f(27, 4)
    p["b3"][27] = -1

    f(30, 64 + 3)
    f(30, 64 + 0, -1)
    f(30, 4, -1)

    # Upstream coordinate, separately for each vertical orientation.
    f(28, 6)
    orientation(28, 1, "column")
    p["b3"][28] = -1

    f(29, 8)
    orientation(29, 0, "column")
    p["b3"][29] = -1

    f(31, 0)
    f(31, 1, -1)

    B = 30.0

    def ray_sum(out, value=1.0, dy=0, dx=0):
        bounded(out, 0, value, dy, dx)
        bounded(out, 2, value, dy, dx)

    # h0/h1 and h2/h3: directional, bounded acceptance rays.
    # A bar becomes accepting only after both end proofs reach a pixel.
    for raw, gravity, seed, dy in [
        (0, 5, 17, -1),
        (2, 6, 18, 1),
    ]:
        for out in (raw, raw + 1):
            bounded(out, raw, 1, dy, 0)
            bounded(out, 4, B)
            bounded(out, 6, B)

            p["bh"][out] = -3 * B
            p["Uh"][out, gravity] = 3 * B
            p["Uh"][out, 0] = -(2 * B - 1)
            p["Uh"][out, 2] = -B
            p["Uh"][out, seed] = 1
        p["bh"][raw + 1] -= 1

    # h4/h5 and h6/h7: proofs traveling inward from the two exits.
    for raw, dx, boundary in [(4, -1, 23), (6, 1, 24)]:
        for out in (raw, raw + 1):
            bounded(out, raw, 1, 0, dx)
            ray_sum(out, 1, 0, dx)
            p["Uh"][out, 0] = B
            p["Uh"][out, boundary] = 1
            p["bh"][out] = -B
        p["bh"][raw + 1] -= 1

    # h8: stable lateral attraction to accepting columns.
    h(8, 8, 0.49, 0, -1)
    h(8, 8, 0.49, 0, 1)
    ray_sum(8, 0.02)
    p["Uh"][8, 21] = 0.002
    p["Uh"][8, 22] = 0.002

    # h9: lateral attraction to currently uncovered source columns.
    h(9, 9, 0.49, 0, -1)
    h(9, 9, 0.49, 0, 1)
    p["Uh"][9, 26] = 0.02

    # h10/h12: downstream-first priorities of unresolved endpoints.
    # Acceptance is binary here; attraction amplitude cannot declare safety.
    for out, endpoint in [(10, 19), (11, 19), (12, 20)]:
        p["bh"][out] = -B
        p["Uh"][out, endpoint] = B + 2
        p["Uh"][out, 28] = -0.5
        p["Uh"][out, 29] = -0.5
        ray_sum(out, -B)

    # h11 differs from h10 by a signed lateral attraction derivative.
    h(11, 8, 1, 0, 1)
    h(11, 8, -1, 0, -1)

    # h13 transfers an unselected endpoint priority to an interior pixel.
    h(13, 12, 1, 0, -1)
    h(13, 12, 1, 0, 1)
    p["Uh"][13, 31] = B
    p["bh"][13] = -B

    # Source attraction restricted to selected/unselected material.
    h(14, 9)
    p["Uh"][14, 1] = B
    p["bh"][14] = -B

    h(15, 9)
    p["Uh"][15, 31] = B
    p["bh"][15] = -B

    # Global decision layer.
    next_unit = 0

    def z(weights, bias=0.0):
        nonlocal next_unit
        out = next_unit
        next_unit += 1
        for index, value in weights.items():
            p["Wz"][out, index] += value
        p["bz"][out] = bias
        return out

    def fm(c):
        return 32 + c

    def fa(c):
        return 64 + c

    def ha(c):
        return 16 + c

    def general(weights, bias, outputs=None):
        # Main minus restricted cancels exactly on one-piece boards.
        a = z(weights, bias)
        restricted = dict(weights)
        restricted[fa(4)] = restricted.get(fa(4), 0) - 100 * 4096
        b = z(restricted, bias + 100)
        if outputs:
            for action, coefficient in outputs.items():
                p["Wo"][action, a] += coefficient
                p["Wo"][action, b] -= coefficient
        return a, b

    # Mean differences sum the derivatives of both exits, rather than
    # comparing the maximum cost at potentially different endpoints.
    general(
        {ha(11): 4096, ha(10): -4096},
        -0.00002, {4: 1000},
    )
    general(
        {ha(10): 4096, ha(11): -4096},
        -0.00002, {3: 1000},
    )

    # Work on the most downstream unresolved piece.
    general({12: 1, 10: -1}, -0.015, {6: 20000})

    # A small deterministic move breaks a completely flat unresolved field.
    general({10: 1}, -0.05, {4: 0.03})

    # Once all exits are accepted, prefer a piece close to an uncovered source.
    general(
        {15: 1, 14: -1, 10: -10, 12: -10},
        -0.0001, {6: 1000000},
    )

    # Source-coverage starts are only enabled when all exits are safe.
    left_start = general(
        {
            fm(30): 1, fm(12): -1,
            fm(26): 10, 10: -10, 12: -10,
        },
        -10.012,
    )
    right_start = general(
        {
            fm(27): 1, fm(10): -1,
            fm(26): 10, 10: -10, 12: -10,
        },
        -10.012,
    )

    # Binary selected-unresolved signal.
    bad_a = z({10: 100})
    bad_b = z({10: 100}, -1)

    # Two persistent coverage directions. Off is -1 and on is +1.
    # A direction persists across unsafe intermediate positions, but clears
    # when the moved piece is safe and the coverage request has disappeared.
    for memory_index, pair in [(0, left_start), (1, right_start)]:
        p["Wm"][memory_index, memory_index] = 10
        p["bm"][memory_index] = -15
        p["Wmz"][memory_index, bad_a] = 10
        p["Wmz"][memory_index, bad_b] = -10
        p["Wmz"][memory_index, pair[0]] = 10000
        p["Wmz"][memory_index, pair[1]] = -10000

    p["bm"][63] = 10
    p["Wo"][3, 128 + 0] = 200
    p["Wo"][3, 128 + 63] = 200
    p["Wo"][4, 128 + 1] = 200
    p["Wo"][4, 128 + 63] = 200

    # Single-piece layouts use direct bounding-box centering.
    row_error = {
        fm(13): 1, fm(15): -1,
        fm(9): -1, fm(11): 1,
    }
    col_error = {
        fm(14): 1, fm(16): -1,
        fm(10): -1, fm(12): 1,
    }
    for error, gravity, positive, negative in [
        (col_error, (5, 6), 4, 3),
        (row_error, (7, 8), 2, 1),
    ]:
        for sign, action in [(1, positive), (-1, negative)]:
            weights = {k: sign * value for k, value in error.items()}
            weights[fa(4)] = -2 * 4096
            for g in gravity:
                weights[fm(g)] = 2
            unit = z(weights, -0.025)
            p["Wo"][action, unit] = 1000

    p["bo"][:] = -0.01
    p["bo"][0] = -100
    p["bo"][5] = 0
    p["bo"][7] = -100

    # Click targets: unresolved priority first, source proximity second.
    # The large material term excludes background and selected material.
    p["Wc"][0, 31] = 100000
    p["Wc"][0, 1] = -100000
    p["Wc"][0, 32 + 13] = 10000
    p["Wc"][0, 32 + 15] = 200

    return p
