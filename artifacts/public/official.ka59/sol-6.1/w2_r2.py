import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5),
        "b1": (32,),
        "W2": (32, 32, 3, 3),
        "b2": (32,),
        "W3": (32, 96),
        "b3": (32,),
        "Wh": (16, 16, 3, 3),
        "Uh": (16, 32),
        "bh": (16,),
        "Wz": (128, 96),
        "bz": (128,),
        "Wm": (64, 64),
        "Wmz": (64, 128),
        "Wma": (64, 8),
        "bm": (64,),
        "Wo": (8, 192),
        "bo": (8,),
        "Wc": (1, 48),
        "bc": (1,),
    }
    p = {name: np.zeros(shape, np.float32)
         for name, shape in shapes.items()}

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3 = p["W3"]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    sizes = [(3, 3), (3, 6), (6, 3), (6, 6)]
    larger = {0: (1, 2, 3), 1: (3,), 2: (3,), 3: ()}
    completion_channels = [22, 23, 24, 27]

    def first(o, dy, dx, colors, weight=1.0):
        for color in colors:
            W1[o, color, dy + 2, dx + 2] += weight

    def second(o, i, dy=0, dx=0, weight=1.0):
        W2[o, i, dy + 1, dx + 1] += weight

    def conjunction(o, tests):
        for dy, dx, colors in tests:
            first(o, dy, dx, colors)
        b1[o] = 1.0 - len(tests)

    # W1 0..3: selected rectangle anchors.
    # W1 4..7: inactive rectangle anchors.
    for t, (w, h) in enumerate(sizes):
        ax = 1 if w == 3 else 2
        ay = 1 if h == 3 else 2
        selected_tests = []
        inactive_tests = []

        for y in range(h):
            for x in range(w):
                dy, dx = y - ay, x - ax
                if abs(dy) > 2 or abs(dx) > 2:
                    continue

                cx = x == 1 if w == 3 else x in (2, 3)
                cy = y == 1 if h == 3 else y in (2, 3)
                core = cx and cy

                selected_tests.append(
                    (dy, dx, (0,) if core else (0, 14))
                )
                inactive_tests.append(
                    (dy, dx, (4, 5) if core else (14,))
                )

        conjunction(t, selected_tests)
        conjunction(4 + t, inactive_tests)

    # W1 8..11: footprint intersections with walls or mud.
    for t, (w, h) in enumerate(sizes):
        xs = range(-1, 2) if w == 3 else range(-2, 3)
        ys = range(-1, 2) if h == 3 else range(-2, 3)
        for dy in ys:
            for dx in xs:
                first(8 + t, dy, dx, (2, 15))

    # Wall-only footprints for sliding objects.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            first(12, dy, dx, (2,))

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dy) <= 1 or abs(dx) <= 1:
                first(13, dy, dx, (2,))

    not_four = tuple(c for c in range(16) if c != 4)
    outline_or_piece = (4, 14, 0, 5)

    # Goal templates are initially measured two pixels from the
    # outer top-left corner. W2 shifts six-pixel dimensions.
    for t, (w, h) in enumerate(sizes):
        o = 14 + t
        core_y = int(h == 6)
        core_x = int(w == 6)

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                border = (
                    dy == -2
                    or dx == -2
                    or (h == 3 and dy == 2)
                    or (w == 3 and dx == 2)
                )

                allowed = outline_or_piece if border else not_four
                if (dy, dx) == (core_y, core_x):
                    allowed = (1, 15, 0, 14)

                first(o, dy, dx, allowed, 100.0)
                if border:
                    first(o, dy, dx, (4,), 1.0)

        # Require shape compatibility and at least three visible
        # outline pixels, but allow an approaching piece to occlude
        # some of the outline.
        b1[o] = -2502.0

        # Completed-goal detector. Here the outline must be visible;
        # the rectangle's central cells are black or deselected yellow.
        complete_tests = []
        core_ys = (0,) if h == 3 else (1, 2)
        core_xs = (0,) if w == 3 else (1, 2)

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                border = (
                    dy == -2
                    or dx == -2
                    or (h == 3 and dy == 2)
                    or (w == 3 and dx == 2)
                )
                if border:
                    allowed = (4,)
                elif dy in core_ys and dx in core_xs:
                    allowed = (0, 4)
                else:
                    allowed = not_four
                complete_tests.append((dy, dx, allowed))

        conjunction(completion_channels[t], complete_tests)

    # Empty cross-shaped goal.
    cross_tests = []
    for dy in (-2, 2):
        for dx in (-2, 2):
            cross_tests.append((dy, dx, (4,)))
    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        cross_tests.append((dy, dx, not_four))
    cross_tests.append((0, 0, (1, 15)))
    conjunction(18, cross_tests)

    movable_colors = (11, 12, 13, 14)
    first(19, 0, 0, movable_colors)
    first(20, 0, -2, (4,))
    first(21, -2, 0, (4,))

    # Cross and selected-black-core neighborhoods.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            first(25, dy, dx, (11,))
            first(26, dy, dx, (0,))

    # Edge detectors. W2 samples these one pixel farther away,
    # producing controller-center candidates two pixels outside
    # an object's edge.
    for d, (dy, dx) in enumerate(directions):
        o = 28 + d
        first(o, dy, dx, movable_colors)
        first(o, 0, 0, movable_colors, -1.0)
        # A completed rectangle has a goal outline immediately
        # outside its edge and should not be a pushing target.
        first(o, 0, 0, (4,), -1.0)

    # Selected anchors: reject nested small-template detections.
    for t, (w, h) in enumerate(sizes):
        second(t, t)
        for u in larger[t]:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    second(t, u, dy, dx, -10.0)

        # Inactive, unfinished anchors.
        second(4 + t, 4 + t)
        second(4 + t, 20, 0, -int(w == 6), -1.0)
        second(4 + t, 21, -int(h == 6), 0, -1.0)

        # Extend footprint tests to six pixels.
        for dy in ((0, 1) if h == 6 else (0,)):
            for dx in ((0, 1) if w == 6 else (0,)):
                second(8 + t, 8 + t, dy, dx)

    second(12, 12)
    second(13, 13)
    for dy, dx in directions:
        second(13, 13, dy, dx)

    # Size-specific goal sources. Completion detectors also remove
    # nearby false smaller-goal detections inside a completed large
    # rectangle.
    for t, (w, h) in enumerate(sizes):
        second(
            14 + t, 14 + t,
            -int(h == 6), -int(w == 6),
            1.0 / 14.0,
        )
        for c in completion_channels:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    second(14 + t, c, dy, dx, -10.0)

    second(18, 18)

    # Size labels in neighborhoods around query anchors.
    for t in range(4):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                second(19 + t, t, dy, dx)
                second(19 + t, 4 + t, dy, dx)
                for u in larger[t]:
                    second(19 + t, u, dy, dx, -10.0)

    # Pushing stances. The expanded black-core test reaches three
    # pixels, excluding all self-edge sources of a selected 3x3 piece.
    for d, (dy, dx) in enumerate(directions):
        o = 23 + d
        second(o, 28 + d, dy, dx)
        second(o, 8, weight=-100.0)
        second(o, 19, weight=-1.0)
        for sy in range(-1, 2):
            for sx in range(-1, 2):
                second(o, 26, sy, sx, -10.0)

    # Completion evidence, reserved as a perceptual feature.
    for t, (w, h) in enumerate(sizes):
        second(
            27, completion_channels[t],
            -int(h == 6), -int(w == 6),
        )

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            second(28, 25, dy, dx)
    second(29, 19)

    # All inactive anchors, including completed pieces that can
    # still serve as controllers for a different object.
    for t in range(4):
        second(30, 4 + t)

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            second(31, 26, dy, dx)

    for i in range(32):
        W3[i, i] = 1.0

    # Helper-field occupancy excludes the selected rectangle's
    # row and column, allowing the readout to sample inside it.
    for t in range(4):
        W3[29, 32 + t] = -100.0
        W3[29, 64 + t] = -100.0

    W3[30] = 0.0
    for t in range(4):
        W3[30, t] = 1.0

    W3[31] = 0.0
    W3[31, 30] = 1.0

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    def recurrent(o, i, dy=0, dx=0, weight=1.0):
        Wh[o, i, dy + 1, dx + 1] += weight

    def diffuse(channel):
        for dy, dx in directions:
            recurrent(channel, channel, dy, dx, 0.249)

    # h0..h3: direct movement toward matching goals.
    for t in range(4):
        diffuse(t)
        Uh[t, 14 + t] = 1.0
        Uh[t, 8 + t] = -100.0
        for u in range(4):
            if u != t:
                Uh[t, 19 + u] = -100.0

    # h4: rectangular sliding-object potential; mud is traversable.
    diffuse(4)
    for t in range(4):
        Uh[4, 14 + t] = 1.0
    Uh[4, 12] = -100.0
    Uh[4, 28] = -100.0

    # h5: cross sliding-object potential.
    diffuse(5)
    Uh[5, 18] = 1.0
    Uh[5, 13] = -100.0

    # h6..h9: positive sliding-goal gradients at pushing stances.
    for d, (dy, dx) in enumerate(directions):
        o = 6 + d
        bh[o] = -128.0
        Uh[o, 23 + d] = 128.0
        for field in (4, 5):
            recurrent(o, field, dy, dx)
            recurrent(o, field, -dy, -dx, -1.0)

    # h10: navigation toward useful pushing positions.
    diffuse(10)
    Uh[10, 8] = -100.0
    Uh[10, 29] = -100.0
    for d in range(4):
        recurrent(10, 6 + d)

    # h11..h14: directional values at the selected anchor.
    for d, (dy, dx) in enumerate(directions):
        o = 11 + d
        bh[o] = -128.0
        Uh[o, 30] = 128.0
        for field in range(4):
            recurrent(o, field, dy, dx)
        recurrent(o, 10, dy, dx, 0.1)
        recurrent(o, 6 + d, weight=0.5)

    # h15: inactive-piece route quality for switching.
    bh[15] = -128.0
    Uh[15, 31] = 128.0
    for field in range(4):
        recurrent(15, field)
    recurrent(15, 4, weight=0.0001)
    recurrent(15, 5, weight=0.0001)

    for d in range(4):
        p["Wz"][d, 11 + d] = 1.0
        p["Wo"][1 + d, d] = 1.0
    p["Wz"][4, 15] = 1.0

    # Prefer continuing a useful route over switching merely to
    # obtain a shorter route for another piece.
    p["Wo"][6, 4] = 0.0001
    p["bo"][6] = np.float32(1e-20)

    # One-step inhibition of repeated fallback clicks.
    p["Wma"][0, 6] = 3.0
    p["Wo"][6, 128] = np.float32(-2e-20)

    p["bo"][0] = -1e9
    p["bo"][5] = -1e9
    p["bo"][7] = -1e9

    p["Wc"][0, 31] = np.float32(1e-20)
    p["Wc"][0, 32 + 15] = 1.0
    return p
