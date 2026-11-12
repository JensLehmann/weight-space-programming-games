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
    p = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }

    colors = (7, 8, 9, 10, 11, 12, 14)
    arm = (3,) + colors
    solid = (1, 3, 15) + colors
    directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
    diagonals = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    def w1(out, channels, dy=0, dx=0, value=1.0):
        if isinstance(channels, int):
            channels = (channels,)
        for channel in channels:
            p["W1"][out, channel, 2 + dy, 2 + dx] += value

    def w2(out, inp, value=1.0, dy=0, dx=0):
        p["W2"][out, inp, 1 + dy, 1 + dx] += value

    def local(out, inp, value=1.0):
        p["W3"][out, inp] += value

    def row(out, inp, value=1.0):
        p["W3"][out, 32 + inp] += value

    def col(out, inp, value=1.0):
        p["W3"][out, 64 + inp] += value

    def exclude_controls(out):
        p["W1"][out, 2, :, :] -= 16.0

    # F1 0:7: colour evidence confined to the local 3x3 cell.
    # A neighbouring object's single row is insufficient evidence.
    for c, color in enumerate(colors):
        p["W1"][c, color, 1:4, 1:4] = 1.0
        p["b1"][c] = -3.0

    # F1 7:11: unsatisfied marker end-direction evidence.
    for d, (dy, dx) in enumerate(directions):
        out = 7 + d
        py, px = dx, -dy
        w1(out, 13, value=16.0)
        for ny, nx in directions + diagonals:
            w1(out, 13, ny, nx, -16.0)

        w1(out, arm, dy + py, dx + px)
        w1(out, arm, dy - py, dx - px)
        w1(out, 3, -dy, -dx)
        w1(out, colors, -2 * dy, -2 * dx)
        w1(out, solid, 2 * dy, 2 * dx, -1.0)
        p["b1"][out] = -18.0

    # F1 11:15: active goals, carrying 2 +/- coordinate.
    for out, axis, sign in (
        (11, 17, 1.0), (12, 17, -1.0),
        (13, 16, 1.0), (14, 16, -1.0),
    ):
        for dy, dx in directions:
            w1(out, 13, dy, dx, 4.0)
        w1(out, 13, value=-4.0)
        w1(out, axis, value=sign)
        p["b1"][out] = -14.0

    w1(15, 16)
    w1(16, 17)

    # F1 17:21: resize-glyph centres.
    glyphs = (
        ((-1, 0), (1, 0), (0, -1), (0, 1)),
        ((-1, 0), (1, 0), (0, 1), (0, -1)),
        ((0, -1), (0, 1), (-1, 0), (1, 0)),
        ((0, -1), (0, 1), (1, 0), (-1, 0)),
    )
    for out, points in enumerate(glyphs, start=17):
        w1(out, 4)
        for dy, dx in points[:3]:
            w1(out, colors, dy, dx)
        dy, dx = points[3]
        w1(out, 4, dy, dx)
        p["b1"][out] = -4.0

    # F1 21: an actual rotation plus, not a resize separator.
    w1(21, colors)
    for dy, dx in directions:
        w1(21, colors, dy, dx)
    for dy, dx in diagonals:
        w1(21, 4, dy, dx)
    p["b1"][21] = -8.0

    # F1 22: horizontal anchor-cell centres without a central marker.
    # F1 23: vertical anchor-cell centres, including central markers.
    for t in (-1, 0, 1):
        w1(22, 3, t, -1)
        w1(22, 3, t, 1)
        w1(23, 3, -1, t)
        w1(23, 3, 1, t)
    w1(22, colors)
    w1(23, colors + (13,))
    p["b1"][22] = -3.0
    p["b1"][23] = -3.0
    exclude_controls(22)
    exclude_controls(23)

    # F1 24: marker centres, including satisfied markers.
    w1(24, 13)
    for dy, dx in diagonals:
        w1(24, 13, dy, dx, -1.0)

    # F1 25,26: genuinely extended left/up arms.
    # Requiring a complete second coloured cross-section distinguishes
    # an extended arm from the perpendicular parent of a one-cell arm.
    for out, (dy, dx) in ((25, (0, -1)), (26, (-1, 0))):
        py, px = dx, -dy
        for t in (-1, 0, 1):
            w1(out, 3, -dy + t * py, -dx + t * px)
            w1(out, colors, 2 * dy + t * py, 2 * dx + t * px)
        w1(out, colors)
        p["b1"][out] = -6.0
        exclude_controls(out)

    # F1 27: horizontal contact between a rigid attachment and a wall.
    w1(27, 1, value=16.0)
    for dx in (-2, -1, 1, 2):
        w1(27, 15, 0, dx)
    p["b1"][27] = -16.0

    # F1 28: material through which contact signals may propagate.
    w1(28, (1, 3) + colors)

    # F1 29: proximity to rigid movable material.
    p["W1"][29, 1, :, :] = 1.0

    # F1 30: minimum-length horizontal marker arm.
    for t in (-1, 0, 1):
        w1(30, 3, t, -1)
        w1(30, 3, t, 1)
    w1(30, 13, value=16.0)
    p["b1"][30] = -18.0
    exclude_controls(30)

    # F2 0:6: coordinate pairs for right/down/left marker motion.
    # Vertical projections are offset by one pixel in each axis so
    # they do not share source cells with horizontal projections.
    gate = 32.0
    for d, (dy, dx) in enumerate(directions[:3]):
        coordinate = 16 if dx else 15
        orientation = 1.0 if dx + dy > 0 else -1.0
        shift_x = -1 if d == 1 else 0
        for shrink in (0, 1):
            out = 2 * d + shrink
            w2(out, 7 + d, gate, 0, shift_x)
            w2(
                out, coordinate,
                orientation if shrink else -orientation,
                0, shift_x
            )
            p["b2"][out] = 2.0 - gate
            if shrink:
                minimum_feature = 23 if d == 1 else 30
                w2(out, minimum_feature, -gate, 0, shift_x)

    # F2 6:13: horizontal ownership, marker colour, and control colour.
    # The anchor and marker tests are mutually exclusive here.
    conjunction = 1024.0
    for c in range(7):
        out = 6 + c
        w2(out, c)
        w2(out, 22, conjunction)
        w2(out, 24, conjunction)
        for glyph in range(17, 22):
            w2(out, glyph, conjunction)
        p["b2"][out] = -conjunction

    # F2 13:20: vertical ownership, shifted with its marker projection.
    for c in range(7):
        out = 13 + c
        w2(out, c, 1.0, 0, -1)
        w2(out, 23, conjunction, 0, -1)
        p["b2"][out] = -conjunction

    w2(20, 11)
    w2(21, 12)
    w2(22, 13, 1.0, -1, 0)
    w2(23, 14, 1.0, -1, 0)

    w2(24, 17)
    w2(24, 19)
    w2(25, 18)
    w2(25, 20)
    w2(26, 21)

    # F2 27: clearance candidates; horizontal clearance comes first.
    w2(27, 25, 1000.0)
    w2(27, 26, 1.0, 0, -1)

    # F2 28: retract a non-minimal marker arm near rigid material.
    w2(28, 29)
    w2(28, 24, 128.0)
    w2(28, 30, -128.0)
    p["b2"][28] = -128.0

    w2(29, 27)
    w2(30, 28)
    w2(31, 24)

    # F3 0:6: positive grow/shrink utilities at projection crossings.
    for d, (dy, dx) in enumerate(directions[:3]):
        horizontal = bool(dx)
        positive = dx + dy > 0
        for shrink in (0, 1):
            out = 2 * d + shrink
            if horizontal:
                row(out, out)
                col(out, 20 if positive != bool(shrink) else 21)
            else:
                col(out, out)
                row(out, 22 if positive != bool(shrink) else 23)
            p["b3"][out] = -4.0

    # F3 6:13: colour rejection at geometric preference sources.
    # Offset horizontal/vertical projections permit one mask per colour.
    for c in range(7):
        out = 6 + c
        for inp in (0, 1, 4, 5):
            row(out, inp, 0.25)
        for inp in (2, 3):
            col(out, inp, 0.25)
        col(out, 20, 0.25)
        col(out, 21, 0.25)
        row(out, 22, 0.25)
        row(out, 23, 0.25)
        row(out, 6 + c, -1.0)
        col(out, 13 + c, -1.0)
        p["b3"][out] = -1.1

    # F3 13:20: reject pressure and control readings at other colours.
    # Interior material without an ownership point is also a barrier
    # to the preference fields, but not to the separate contact field.
    for c in range(7):
        out = 13 + c
        local(out, 30)
        for inp in (24, 25, 26):
            local(out, inp)
        for inp in range(6, 20):
            local(out, inp, 0.25)
        local(out, 6 + c, -2.0)
        local(out, 13 + c, -2.0)
        p["b3"][out] = -0.1

    local(20, 24)
    local(21, 25)

    # Unmarked-arm clearance. Protect marker-bearing rows and columns.
    local(22, 27)
    row(22, 31, -1.0e6)
    col(22, 31, -1.0e6)

    # Exact non-control mask.
    p["b3"][23] = 1.0
    for inp in (24, 25, 26):
        local(23, inp, -1.0)

    # Contact-field substrate and sources.
    p["b3"][24] = 1.0
    local(24, 30, -1.0)
    local(25, 29)

    local(27, 28)
    local(28, 26)

    # Positive pressure may extend vertical supports, but must not
    # simply re-extend an unmarked horizontal arm being cleared.
    for inp in range(6, 13):
        local(30, inp)
    local(30, 31, -32.0)
    for inp in (24, 25, 26):
        local(30, inp, -32.0)
    p["b3"][30] = -0.1

    # Seven signed, colour-specific preference diffusion fields.
    kernel = np.array(
        [[0.225, 0.025, 0.225],
         [0.025, 0.000, 0.025],
         [0.225, 0.025, 0.225]],
        dtype=np.float32,
    )
    barrier = 1.0e12

    for c in range(7):
        plus, minus = 2 * c, 2 * c + 1
        p["Wh"][plus, plus] = kernel
        p["Wh"][plus, minus] = -kernel
        p["Wh"][minus, plus] = -kernel
        p["Wh"][minus, minus] = kernel

        for d in range(3):
            grow, shrink = 2 * d, 2 * d + 1
            strength = 64.0 if d == 1 else 8.0
            p["Uh"][plus, grow] = strength
            p["Uh"][plus, shrink] = -strength
            p["Uh"][minus, grow] = -strength
            p["Uh"][minus, shrink] = strength

        p["Uh"][plus, 22] = -1.0
        p["Uh"][minus, 22] = 1.0

        # Temporary marker-arm retraction has priority over pressure.
        p["Uh"][plus, 27] = -1.0e10
        p["Uh"][minus, 27] = 1.0e10

        for h in (plus, minus):
            p["Uh"][h, 6 + c] = -barrier
            p["Uh"][h, 13 + c] = -barrier
            p["Uh"][h, 28] = -barrier

        p["Uh"][plus, 20] = -barrier
        p["Uh"][minus, 21] = -barrier
        p["Uh"][plus, 30] = -barrier

        # The difference of H14 and H15 is a bounded contact signal.
        p["Wh"][plus, 14, 1, 1] = 1.0e7
        p["Wh"][plus, 15, 1, 1] = -1.0e7

        p["Wc"][0, 32 + plus] = 1.0
        p["Wc"][0, 32 + minus] = 1.0

    # Bounded flood propagation:
    # ReLU(v) - ReLU(v-1) = clip(v, 0, 1).
    # Unlike unconstrained diffusion this signal neither explodes nor
    # disappears along a long, narrow connected mechanism.
    for out in (14, 15):
        for dy, dx in ((0, 0),) + directions:
            p["Wh"][out, 14, 1 + dy, 1 + dx] = 1.0
            p["Wh"][out, 15, 1 + dy, 1 + dx] = -1.0
        p["Uh"][out, 25] = 8.0
        p["Uh"][out, 24] = -1.0e5
    p["bh"][15] = -1.0

    p["Wc"][0, 23] = -1.0e15
    p["Wc"][0, 28] = 1.0e-8

    p["bo"][:] = -10.0
    p["bo"][6] = 10.0
    return p
