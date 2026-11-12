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
    p = {k: np.zeros(shape, dtype=np.float32)
         for k, shape in shapes.items()}

    groups = [
        (0,),
        (10, 1, 5),
        (11, 2, 6),
        (12, 7),
        (14, 9),
        (15,),
        (4,),
    ]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Neighbours three display pixels away, assembled as 2 + 1.
    for group, colours in enumerate(groups):
        for d, (dy, dx) in enumerate(directions):
            for colour in colours:
                p["W1"][4 * group + d, colour,
                        2 + 2 * dy, 2 + 2 * dx] = 1.0

    p["W1"][28, 8, 2, 2] = 1.0
    p["W1"][28, 3, 2, 2] = 1.0
    p["W1"][29, 8, 2, 2] = 1.0
    p["W1"][30, 16, 2, 2] = 1.0
    p["W1"][31, 4, 2, 2] = 1.0

    def neighbours(out, group, coefficient):
        for d, (dy, dx) in enumerate(directions):
            p["W2"][out, 4 * group + d,
                    1 + dy, 1 + dx] += coefficient

    # Two hinges implement an exact degree-one test:
    # relu(2-n) - 2*relu(1-n), for integer n >= 0.
    for group in range(6):
        for hinge in range(2):
            out = 2 * group + hinge
            neighbours(out, group, -1.0)
            p["W2"][out, 28, 1, 1] = 8.0
            p["b2"][out] = -6.0 - hinge

    for hinge in range(2):
        out = 12 + hinge
        neighbours(out, 6, -1.0)
        p["W2"][out, 31, 1, 1] = 8.0
        p["b2"][out] = -6.0 - hinge

    p["W2"][14, 28, 1, 1] = 1.0
    p["W2"][14, 29, 1, 1] = -1.0
    p["W2"][15, 29, 1, 1] = 1.0
    p["W2"][16, 31, 1, 1] = 1.0

    p["W2"][17, 30, 1, 1] = -63.0
    p["b2"][17] = 3.0

    # Background witnesses from unobstructed upper padding.
    for group in range(6):
        out = 18 + group
        p["W2"][out, 4 * group, 1, 1] = 100.0
        p["W2"][out, 30, 1, 1] = -63.0
        p["b2"][out] = -96.0

    # Body-click candidates, displaced one pixel inward from their top.
    for group in range(1, 6):
        p["W2"][23 + group, 4 * group, 2, 1] = 1.0

    p["W2"][29, 29, 1, 1] = 1.0
    p["W2"][30, 0, 2, 1] = 1.0

    neighbours(31, 6, -1.0)
    p["W2"][31, 31, 1, 1] = 8.0
    p["W2"][31, 30, 1, 1] = -63.0
    p["b2"][31] = -6.0

    def hat(out, group, coefficient=1.0):
        p["W3"][out, 2 * group] += coefficient
        p["W3"][out, 2 * group + 1] -= 2.0 * coefficient

    # Terminal masks, rejecting the background colour.
    for group in range(6):
        hat(group, group)
        p["W3"][group, 64 + 18 + group] = -100.0

    # Masked-board attraction source: grey endpoints and green contacts.
    p["W3"][6, 12] = 1.0
    p["W3"][6, 13] = -2.0
    p["W3"][6, 14] = 1.0
    p["W3"][6, 17] = -100.0
    for axial in (32, 64):
        p["W3"][6, axial + 12] = 0.005
        p["W3"][6, axial + 13] = -0.010
    p["W3"][6, 64 + 31] = -0.010

    # Grey body and endpoint click candidates.
    p["W3"][8, 16] = 1.0
    p["W3"][8, 17] = -10.0
    p["W3"][9, 12] = 1.0
    p["W3"][9, 13] = -2.0
    p["W3"][9, 17] = -10.0

    # Ordinary-board body-click fallbacks.
    for group in range(1, 6):
        out = 10 + group
        p["W3"][out, 23 + group] = 1.0
        p["W3"][out, 64 + 18 + group] = -100.0
        p["W3"][out, 17] = -10.0

    for group in range(6):
        out = 16 + group
        hat(out, group)
        p["W3"][out, 15] = -1.0
        p["W3"][out, 64 + 18 + group] = -100.0

    # Black extension marker on a selected masked piece.
    p["W3"][22, 30] = 1.0
    p["W3"][22, 17] = -10.0
    p["W3"][22, 64 + 18] = -100.0

    # Ordinary-board sources, separately background-rejected so that
    # proximity to a selected black body does not erase a valid source.
    for group in range(1, 6):
        out = 22 + group
        hat(out, group)
        for axial in (32, 64):
            p["W3"][out, axial + 2 * group] = 0.005
            p["W3"][out, axial + 2 * group + 1] = -0.010
        p["W3"][out, 64 + 18 + group] = -100.0
        p["W3"][out, 17] = -100.0

    # Selected unmatched terminals on ordinary boards.
    hat(28, 0)
    p["W3"][28, 14] = -1.0
    p["W3"][28, 64 + 18] = -100.0

    # All visible unmatched terminals.
    p["W3"][29, 29] = 1.0
    p["W3"][30, 14] = 1.0

    # Heat fields.
    for field in (0, 1):
        for dy, dx in directions:
            p["Wh"][field, field, 1 + dy, 1 + dx] = 0.25
    for source in range(23, 28):
        p["Uh"][0, source] = 1.0
    p["Uh"][1, 6] = 1.0

    # Complement masks: one off a selected terminal, zero on it.
    # Multiple coincident body witnesses cannot introduce a large
    # positive residual into the sampled fields.
    p["bh"][2] = 1.0
    p["Uh"][2, 0] = -1.0
    p["bh"][3] = 1.0
    for group in range(6):
        p["Uh"][3, group] = -1.0

    # Current, up, down, left, right potential samples.
    for i, (dy, dx) in enumerate([(0, 0)] + directions):
        ordinary = 4 + i
        masked = 9 + i

        p["Wh"][ordinary, 0, 1 + dy, 1 + dx] = 1.0
        p["Wh"][ordinary, 2, 1, 1] = -2048.0

        p["Wh"][masked, 1, 1 + dy, 1 + dx] = 1.0
        p["Wh"][masked, 3, 1, 1] = -2048.0

    # Directional improvement summaries.
    for i in range(4):
        p["Wz"][i, 16 + 5 + i] = 1.0
        p["Wz"][i, 16 + 4] = -1.0
        p["Wz"][i, 32 + 8] = -1000.0

        out = 4 + i
        p["Wz"][out, 16 + 10 + i] = 1.0
        p["Wz"][out, 16 + 9] = -1.0
        p["Wz"][out, 32 + 8] = 1000.0
        p["bz"][out] = -1000.0

    # Mode, extension marker, and complete selected-terminal signals.
    p["Wz"][8, 32 + 8] = 1.0
    p["Wz"][9, 32 + 8] = 1.0
    p["Wz"][9, 32 + 22] = 1.0
    p["bz"][9] = -1.0

    p["Wz"][10, 32 + 28] = -1.0
    p["Wz"][10, 32 + 8] = -100.0
    p["bz"][10] = 0.5

    p["Wz"][11, 32 + 29] = -1.0
    p["Wz"][11, 32 + 8] = 100.0
    p["bz"][11] = -99.5

    # All-green ordinary layouts that have not ended still need a
    # polarity-changing rotation rather than repeated selection.
    p["Wz"][14, 32 + 29] = -1.0
    p["Wz"][14, 32 + 8] = -2.0
    p["bz"][14] = 0.5

    # Previous-click flag.
    p["bm"][0] = -10.0
    p["Wma"][0, 6] = 20.0

    # Rotation-count latches, reset by selection.
    p["Wm"][4, 4] = 10.0
    p["bm"][4] = -1.0
    p["Wma"][4, 5] = 20.0
    p["Wma"][4, 6] = -30.0
    for unit in (5, 6):
        p["Wm"][unit, unit] = 20.0
        p["Wm"][unit, unit - 1] = 10.0
        p["bm"][unit] = -5.0
        p["Wma"][unit, 5] = 20.0
        p["Wma"][unit, 6] = -60.0

    # Initial anchoring latch: masked starts grow their selected piece,
    # then select another piece. Ordinary starts retain their selection.
    p["Wm"][12, 12] = 20.0
    p["Wmz"][12, 8] = -20.0
    p["Wma"][12, 6] = 40.0
    p["bm"][12] = 10.0

    # Three-step movement history, cleared by rotation or selection.
    for i in range(4):
        p["bm"][8 + i] = -10.0
        p["Wma"][8 + i, 1 + i] = 20.0

        for unit, predecessor in ((13 + i, 8 + i),
                                  (17 + i, 13 + i)):
            p["Wm"][unit, predecessor] = 10.0
            p["bm"][unit] = -1.0
            p["Wma"][unit, 5] = -20.0
            p["Wma"][unit, 6] = -20.0

    # Sixteen-consecutive-move timeout.
    p["bm"][24] = -10.0
    p["Wma"][24, 1:5] = 20.0
    for unit in range(25, 40):
        p["Wm"][unit, unit - 1] = 10.0
        p["Wma"][unit, 1:5] = 20.0
        p["bm"][unit] = -20.0

    opposite = [1, 0, 3, 2]
    for i in range(4):
        action = 1 + i
        p["Wo"][action, i] = 30000.0
        p["Wo"][action, 4 + i] = 30000.0
        for base in (8, 13, 17):
            p["Wo"][action, 128 + base + opposite[i]] = -500.0
        p["bo"][action] = -1501.0

    # Rotation at local extrema; growth and timeout override movement.
    p["Wo"][5, 9] = 100000.0
    p["Wo"][5, 14] = 4.0
    p["Wo"][5, 128 + 39] = 10000.0
    p["bo"][5] = 10000.1

    # Selection after successful geometric docking or three rotations.
    p["Wo"][6, 10] = 4.0
    p["Wo"][6, 11] = 4.0
    p["Wo"][6, 128 + 12] = -5000.0
    p["Wo"][6, 128 + 6] = 2.0
    p["Wo"][6, 128 + 0] = -10000.0
    p["bo"][6] = -4999.0

    p["bo"][0] = -1.0e8
    p["bo"][7] = -1.0e8

    # Grey unmatched endpoints first, then grey bodies; ordinary
    # unselected bodies provide the remaining valid-click fallback.
    p["Wc"][0, 9] = 100.0
    p["Wc"][0, 8] = 10.0
    for channel in range(11, 16):
        p["Wc"][0, channel] = 1.0

    return p
