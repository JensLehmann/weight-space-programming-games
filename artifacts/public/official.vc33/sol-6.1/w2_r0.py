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

    def c1(o, i, v, dy=0, dx=0):
        p["W1"][o, i, dy + 2, dx + 2] += v

    def c2(o, i, v, dy=0, dx=0):
        p["W2"][o, i, dy + 1, dx + 1] += v

    def f(o, i, v=1.0, axis=0):
        p["W3"][o, i + 32 * axis] += v

    def r(o, i, v, dy=0, dx=0):
        p["Wh"][o, i, dy + 1, dx + 1] += v

    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    surrounding = tuple(
        (dy, dx) for dy in (-1, 0, 1)
        for dx in (-1, 0, 1) if dy or dx
    )

    def band(o, v, dy=0, dx=0):
        for colour in (11, 14):
            c1(o, colour, v, dy, dx)

    def gate(o, v, dy=0, dx=0):
        for colour in (1, 12):
            c1(o, colour, v, dy, dx)

    # Palette planes.
    for o, colour in (
        (0, 0), (1, 3), (2, 4), (3, 5), (4, 9), (6, 12)
    ):
        c1(o, colour, 1)
    gate(5, 1)
    band(7, 1)

    # Rider bands: yellow lies upstream of the coloured band.
    for o, dy, dx in (
        (8, -2, 0), (9, 0, 2), (10, 0, -2), (11, 2, 0)
    ):
        band(o, 1)
        c1(o, 4, 1, dy, dx)
        p["b1"][o] = -1

    # Shifted gate planes, for seven-pixel receptive-field tests.
    gate(12, 1, -2, 0)
    gate(13, 1, 0, 2)

    # Last row of a horizontal coloured band.
    for dx in (-1, 0, 1):
        band(15, 1, 0, dx)
    band(15, -3, 1, 0)
    p["b1"][15] = -2

    # Leading column of a vertical coloured band.
    for dy in (-1, 0, 1):
        band(16, 1, dy, 0)
    band(16, -3, 0, -1)
    p["b1"][16] = -2

    # Exposed platform faces.
    for o, dy, dx in ((18, 1, 0), (19, 0, -1), (20, 0, 1)):
        for colour in (3, 4, 11, 14):
            c1(o, colour, 1)
        c1(o, 0, 1, dy, dx)
        p["b1"][o] = -1

    # Long gate runs distinguish their longitudinal orientation.
    for dx in (-2, 0, 2):
        gate(22, 1, 0, dx)
    p["b1"][22] = -2
    for dy in (-2, 0, 2):
        gate(23, 1, dy, 0)
    p["b1"][23] = -2

    # Button planes shifted toward prospective divider centres.
    for o, dy, dx in (
        (27, 0, -2), (28, 0, 2), (29, -2, 0), (30, 2, 0)
    ):
        c1(o, 9, 1, dy, dx)

    # Local copies.
    for o, i in (
        (0, 3), (1, 5), (4, 4), (5, 6),
        (9, 18), (10, 19), (11, 20)
    ):
        c2(o, i, 1)
    for i in (8, 9, 10, 11):
        c2(3, i, 1)

    # Vertical portal: final gate row.
    c2(6, 5, 1)
    c2(6, 12, 1, -1, 0)
    c2(6, 5, -1, 1, 0)
    p["b2"][6] = -1

    # Horizontal portal: first gate column.
    c2(7, 5, 1)
    c2(7, 13, 1, 0, 1)
    c2(7, 5, -1, 0, -1)
    p["b2"][7] = -1

    # Target seeds on opposite orthogonal sides of the colour band.
    # Rider subtraction prevents the rider itself from becoming a goal.
    for dx in (-1, 1):
        c2(12, 15, 1, 0, dx)
        c2(12, 8, -1, 0, dx)
        c2(12, 11, -1, 0, dx)
    for dy in (-1, 1):
        c2(13, 16, 1, dy, 0)
        c2(13, 9, -1, dy, 0)
        c2(13, 10, -1, dy, 0)

    # Divider evidence: paired buttons, or an oriented gate.
    c2(17, 3, 1)
    c2(17, 27, 1, 0, -1)
    c2(17, 28, 1, 0, 1)
    c2(17, 23, 3)
    p["b2"][17] = -2

    c2(18, 3, 1)
    c2(18, 29, 1, -1, 0)
    c2(18, 30, 1, 1, 0)
    c2(18, 22, 3)
    p["b2"][18] = -2

    # F3: extend genuine dividers along their longitudinal axes.
    # A perpendicular floor without paired buttons is not a divider.
    for o in (0, 1):
        f(o, 17, 1, axis=2)
        f(o, 18, 1, axis=1)

    # Navigation admits portal endpoints; chamber identity does not.
    f(0, 6, -1)
    f(0, 7, -1)
    f(2, 1)

    for i in (6, 7, 12, 13):
        f(3, i)
    f(4, 12)
    f(4, 13)

    f(5, 9)
    f(6, 10)
    f(7, 11)
    f(8, 3)

    # Only buttons and enabled gates may win the click readout.
    p["b3"][9] = 1
    f(9, 4, -1)
    f(9, 5, -1)
    f(10, 5)
    f(11, 4)

    # Recurrent channels:
    # 0 goal potential
    # 1,2 clipped rider-chamber flood
    # 3 selected endpoint/goal sources
    # 4 transported mass
    # 5,6 relative row/column moments
    # 7..12 signed oriented surface requests
    # 13,14 signed chamber pressure
    # 15 cross-divider shrink-pressure spill

    mask = 1000000.0
    source_mask = 10000.0
    contact = 32.0
    offset = 128.0
    dead = 0.5

    # Distance-sensitive target potential through portal passages.
    for dy, dx in surrounding:
        r(0, 0, 0.135, dy, dx)
    p["Uh"][0, 0] = -mask
    p["Uh"][0, 4] = 0.1

    # Difference h1-h2 implements a clipped Boolean flood.
    for o in (1, 2):
        for dy, dx in surrounding:
            r(o, 1, 1, dy, dx)
            r(o, 2, -1, dy, dx)
        p["Uh"][o, 1] = -32
        p["Uh"][o, 8] = 1
    p["bh"][2] = -1

    # Activate sources touching the rider's chamber.
    r(3, 0, 1)
    for dy, dx in ((0, 0),) + cardinal:
        r(3, 1, contact, dy, dx)
        r(3, 2, -contact, dy, dx)
    p["Uh"][3, 3] = source_mask
    p["bh"][3] = -source_mask - contact

    # Coordinate transport:
    # Y/N = offset + desired_row - current_row
    # X/N = offset + desired_col - current_col.
    diffusion = 0.245
    for dy, dx in cardinal:
        r(4, 4, diffusion, dy, dx)
        r(5, 5, diffusion, dy, dx)
        r(5, 4, diffusion * dy, dy, dx)
        r(6, 6, diffusion, dy, dx)
        r(6, 4, diffusion * dx, dy, dx)

    # Inject beside a source as well as at it. A closed divider blocks
    # injection to the other chamber; a portal endpoint admits it.
    for dy, dx in ((0, 0),) + cardinal:
        weight = 1.0 if (dy, dx) == (0, 0) else 0.25
        r(4, 3, weight, dy, dx)
        r(5, 3, weight * (offset + dy + 1), dy, dx)
        r(6, 3, weight * (offset + dx), dy, dx)
    for o in (4, 5, 6):
        p["Uh"][o, 0] = -mask

    # Downward gravity: growing raises the exposed face.
    r(7, 4, offset + 1 - dead)
    r(7, 5, -1)
    r(8, 5, 1)
    r(8, 4, -(offset + 1 + dead))

    # Leftward gravity: growing moves the face right.
    r(9, 6, 1)
    r(9, 4, -(offset + dead))
    r(10, 4, offset - dead)
    r(10, 6, -1)

    # Rightward gravity: growing moves the face left.
    # The visible two-pixel band ends one pixel past its leading column.
    r(11, 4, offset - 1 - dead)
    r(11, 6, -1)
    r(12, 6, 1)
    r(12, 4, -(offset - 1 + dead))

    for o, face in (
        (7, 5), (8, 5), (9, 6), (10, 6), (11, 7), (12, 7)
    ):
        p["Uh"][o, face] = mask
        p["bh"][o] = -mask

    # Signed pressure propagates within each chamber.
    for dy, dx in cardinal:
        r(13, 13, diffusion, dy, dx)
        r(13, 14, -diffusion, dy, dx)
        r(14, 14, diffusion, dy, dx)
        r(14, 13, -diffusion, dy, dx)
    for grow, shrink in ((7, 8), (9, 10), (11, 12)):
        r(13, grow, 1)
        r(13, shrink, -1)
        r(14, grow, -1)
        r(14, shrink, 1)
    p["Uh"][13, 1] = -mask
    p["Uh"][14, 1] = -mask

    # Shrinking one platform means pressing its neighbour's grow button.
    for dy, dx in cardinal:
        r(15, 15, 0.20, dy, dx)
    r(15, 14, 1)
    r(15, 13, -1)
    p["Uh"][15, 2] = -mask

    p["Wc"][0, 9] = -1.0e12
    p["Wc"][0, 32 + 13] = 1
    p["Wc"][0, 32 + 14] = -1
    p["Wc"][0, 32 + 15] = 0.15

    # An enabled, chamber-adjacent portal is ready for transfer.
    p["Wc"][0, 32 + 3] = 1000

    p["bo"][6] = 1
    return p
