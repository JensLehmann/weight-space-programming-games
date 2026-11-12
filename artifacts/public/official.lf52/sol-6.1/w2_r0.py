import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3),
        "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}

    def detect(c, requirements):
        for dy, dx, color in requirements:
            p["W1"][c, color, dy + 2, dx + 2] += 1
        p["b1"][c] = 1 - len(requirements)

    circle = (
        [(-1, 0), (-1, 1)]
        + [(y, x) for y in (0, 1) for x in (-1, 0, 1, 2)]
        + [(2, 0), (2, 1)]
    )
    for c, color in enumerate((14, 8, 9)):
        detect(c, [(y, x, color) for y, x in circle])
    for y, x in ((-1, -1), (-1, 2), (2, -1), (2, 2)):
        p["W1"][2, 9, y + 2, x + 2] -= 1

    post = (
        (15, 15, 15, 15),
        (15, 0, 7, 15),
        (15, 7, 7, 15),
        (15, 15, 15, 15),
    )
    detect(3, [(y - 2, x - 1, post[y][x])
               for y in range(4) for x in range(4)])

    interior = [(y, x) for y in range(-1, 3)
                for x in range(-1, 3)]
    upper_left = (
        [(-2, x) for x in range(-2, 3)]
        + [(y, -2) for y in range(-1, 3)]
    )
    for c, inside, border in ((4, 1, 0), (5, 12, 11)):
        detect(c, [(y, x, inside) for y, x in interior]
               + [(y, x, border) for y, x in upper_left])

    ring = set(circle) - {(0, 0), (0, 1), (1, 0), (1, 1)}
    for c, inside in ((6, 1), (7, 12)):
        detect(c, [(y, x, 2 if (y, x) in ring else inside)
                   for y, x in interior])

    detect(8, [(y, x, 11) for y, x in upper_left])
    lower_right = (
        [(y, 2) for y in range(-2, 3)]
        + [(2, x) for x in range(-2, 2)]
    )
    detect(9, [(y, x, 11) for y, x in lower_right])

    outline = (
        [(-2, x) for x in range(-1, 3)]
        + [(y, -2) for y in range(-1, 3)]
        + [(2, 2)]
    )
    detect(10, [(y, x, 3) for y, x in outline])
    detect(11, [(y, x, 0 if y in (0, 1) and x in (0, 1) else 5)
                for y in range(-1, 3) for x in range(-1, 3)])

    # Track samples. W2 shifts these one further pixel outward.
    detect(12, [(-2, 1, 5), (-2, 2, 5)])   # up
    detect(13, [(2, -2, 5), (2, -1, 5)])   # down
    detect(14, [(1, -2, 5), (2, -2, 5)])   # left
    detect(15, [(-2, 2, 5), (-1, 2, 5)])   # right

    for c, inp in ((16, 16), (17, 17), (18, 18)):
        p["W1"][c, inp, 2, 2] = 1

    # Slider corner anchors at tile+(1,1) and tile+(4,4).
    # The extra black pixel pins the edge of the outer frame.
    detect(19, [(y, -1, 11) for y in range(-1, 3)]
               + [(-2, -1, 5)])
    detect(26, [(y, 1, 11) for y in range(-2, 2)]
               + [(2, 1, 5)])

    # Yellow left border at the usual peg anchor; also works with posts.
    detect(28, [(y, -2, 11) for y in range(-2, 3)])

    for c in list(range(12)) + [16, 17, 18, 19, 26, 28]:
        p["W2"][c, c, 1, 1] = 1

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for k, (dy, dx) in enumerate(directions):
        c = 12 + k
        anchor = 19 if k in (0, 2) else 26
        p["W2"][c, anchor, 1, 1] = 1
        p["W2"][c, c, 1 + dy, 1 + dx] = 1
        p["b2"][c] = -1

    def combine(out, terms, bias=0):
        for c, weight in terms:
            p["W2"][out, c, 1, 1] += weight
        p["b2"][out] = bias

    combine(20, [(4, 1), (5, 1), (6, 1), (7, 1)])
    combine(21, [(6, 1), (7, 1)])
    combine(22, [(0, 1), (1, 1), (2, 1), (3, 1)])
    combine(23, [(0, 1), (1, 1)])
    combine(24, [(10, 1), (0, 1), (1, 1), (2, 1)], -1)
    combine(25, [(0, 1), (1, 1), (10, -1), (28, -1)])
    combine(27, [(2, 1), (3, 1), (10, -1), (28, -1)])
    combine(29, [(0, 1), (1, 1), (2, 1)])

    for c in range(32):
        p["W3"][c, c] = 1

    W, U, b = p["Wh"], p["Uh"], p["bh"]

    # Axis propagation: Green function (5/3)*(1/2)^distance.
    for h, offsets in (
        (0, [(0, -1), (0, 1)]),
        (1, [(-1, 0), (1, 0)]),
    ):
        for dy, dx in offsets:
            W[h, h, 1 + dy, 1 + dx] = 0.4
        U[h, 20] = 1
        W[h, 15, 1, 1] = 1

    # An adjacent empty tile contributes about .026; a tile two
    # cells away contributes only .0004. Relays are object-gated.
    for relay, field in ((2, 0), (3, 1)):
        W[relay, field, 1, 1] = 1
        U[relay, 22] = 4
        U[relay, 0] = 0.006
        U[relay, 1] = 0.006
        b[relay] = -4.015

    for h, relay, offsets in (
        (4, 2, [(0, -1), (0, 1)]),
        (5, 3, [(-1, 0), (1, 0)]),
    ):
        W[h, relay, 1, 1] = 1
        for dy, dx in offsets:
            W[h, h, 1 + dy, 1 + dx] = 0.4

    # Remove each relay's own response before scoring jumps.
    for out, gate in ((6, 29), (7, 21)):
        W[out, 4, 1, 1] = 10000
        W[out, 5, 1, 1] = 10000
        W[out, 2, 1, 1] = -10000 * (5 / 3)
        W[out, 3, 1, 1] = -10000 * (5 / 3)
        U[out, gate] = 10000
        b[out] = -10000.1

    # Uncarried-object navigation and nonblue-peg landing potentials.
    for h in (8, 13):
        for dy, dx in directions:
            W[h, h, 1 + dy, 1 + dx] = 0.24
        U[h, 25] = 1
    U[8, 27] = 0.45

    # Only genuine outward slider/track contacts can enable arrows.
    for k, (dy, dx) in enumerate(directions):
        h = 9 + k
        W[h, 8, 1 + dy, 1 + dx] = 1
        W[h, 8, 1, 1] = -1
        U[h, 12 + k] = 101
        b[h] = -100

    W[14, 13, 1, 1] = 1
    U[14, 21] = 100
    b[14] = -100

    # Modest landing-density bonus, without overwhelming distance tests.
    W[15, 13, 1, 1] = 2
    U[15, 20] = 100
    b[15] = -100

    p["Wz"][0, 6] = 1
    p["Wz"][1, 32 + 21] = 1
    p["Wz"][2, 32 + 11] = 1
    for k in range(4):
        p["Wz"][3 + k, 9 + k] = 1
        p["Wz"][7 + k, 32 + 12 + k] = 1

    # Remember the most recent arrow to discourage immediate reversal.
    for k in range(4):
        p["Wm"][k, k] = 1.4
        for j in range(4):
            p["Wma"][k, 1 + j] = 3 if k == j else -3

    p["bo"][:] = -100
    p["bo"][7] = 0
    p["bo"][6] = -20
    p["Wo"][6, 0] = 1000
    p["Wo"][6, 1] = 30000
    p["Wo"][6, 2] = 30000

    inverse = [1, 0, 3, 2]
    for k in range(4):
        a = 1 + k
        p["bo"][a] = -50
        p["Wo"][a, 3 + k] = 200
        p["Wo"][a, 7 + k] = 100
        p["Wo"][a, 128 + k] = 3
        p["Wo"][a, 128 + inverse[k]] = -18

    C = p["Wc"][0]
    C[11] = 20000
    C[21] = 10000
    C[0] = 2
    C[1] = 1
    C[16] = -0.001
    C[17] = -0.0001
    C[32 + 6] = 50
    C[32 + 7] = 50
    C[32 + 14] = 200

    return p
