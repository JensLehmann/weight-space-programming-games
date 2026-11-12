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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    ink = (12, 9, 14, 8)
    colour_value = {12: 1, 9: 2, 14: 3, 8: 4}
    offsets = [(r - 1, c - 1) for r in range(3) for c in range(3)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def w1(out, channel, dy=0, dx=0, value=1.0):
        p["W1"][out, channel, dy + 2, dx + 2] += value

    def w2(out, channel, dy=0, dx=0, value=1.0):
        p["W2"][out, channel, dy + 1, dx + 1] += value

    # Five descriptors for each glyph:
    # ink count, centre bit, edge count, bitmap code, colour sum.
    # Channels 0:5 sample the doubled HUD; 5:10 sample normal glyphs.
    for j, (dy, dx) in enumerate(offsets):
        descriptor = (
            1.0,
            float(j == 4),
            float(j in (1, 3, 5, 7)),
            float(2 ** j),
        )
        for base, scale in ((0, 2), (5, 1)):
            for colour in ink:
                for feature, value in enumerate(descriptor):
                    if value:
                        w1(base + feature, colour,
                           scale * dy, scale * dx, value)
                w1(base + 4, colour, scale * dy, scale * dx,
                   colour_value[colour])

    # Coordinate hinges.
    w1(10, 17, value=63.0)
    p["b1"][10] = -5.0
    w1(11, 16, value=-63.0)
    p["b1"][11] = 57.0

    # Expanded obstacle mask.
    # Three blue pixels distinguish an edge strip from the rotation icon.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            w1(12, 4, dy, dx, 25.0)
            w1(12, 5, dy, dx, 1.0)
            w1(12, 1, dy, dx, 9.0)
    p["b1"][12] = -24.0

    # Exact player centre.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            w1(13, 12 if dy < 0 else 9, dy, dx)
    p["b1"][13] = -24.0

    # Clipped goal-centre detector.
    for out in (14, 15):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if abs(dy) == 2 or abs(dx) == 2:
                    w1(out, 5, dy, dx, 100.0)
        for dy, dx in offsets:
            for colour in ink:
                w1(out, colour, dy, dx)
    p["b1"][14] = -1604.0
    p["b1"][15] = -1605.0

    def pattern(out, pixels):
        for r in range(3):
            for c in range(3):
                w1(out, pixels[r][c], r - 1, c - 1)
        p["b1"][out] = -8.0

    pattern(16, [[0, 3, 3],
                 [3, 0, 0],
                 [3, 0, 3]])
    pattern(17, [[9, 14, 14],
                 [9, 0, 8],
                 [12, 12, 8]])
    pattern(18, [[3, 0, 3],
                 [1, 0, 0],
                 [3, 1, 3]])
    pattern(19, [[11, 11, 11],
                 [11, 3, 11],
                 [11, 11, 11]])

    # Energy-bar columns.
    for dy, colour in ((-1, 5), (0, 11), (1, 11), (2, 5)):
        w1(20, colour, dy, 0)
    p["b1"][20] = -3.0

    # Separate blue corner samples for directional pusher warnings.
    for out, (dy, dx) in enumerate(
        [(-2, -2), (-2, 2), (2, -2), (2, 2)], start=21
    ):
        w1(out, 1, dy, dx)

    def hud_anchor(out, scale):
        for dx, coefficient in ((-1, 1), (0, -2), (1, 1)):
            w2(out, 10, 0, dx, scale * coefficient)
        for dy, coefficient in ((-1, 1), (0, -2), (1, 1)):
            w2(out, 11, dy, 0, scale * coefficient)

    def goal_mask(out, scale):
        w2(out, 14, value=scale)
        w2(out, 15, value=-scale)

    bounds = (10.0, 2.0, 5.0, 512.0, 40.0)
    for j, bound in enumerate(bounds):
        w2(j, j)
        hud_anchor(j, bound)
        p["b2"][j] = -2.0 * bound

        w2(5 + j, 5 + j)
        goal_mask(5 + j, bound)
        p["b2"][5 + j] = -bound

    # Second-layer spatial features.
    w2(10, 13)  # player
    w2(11, 12)  # obstacles

    # Positive lexicographic-style goal priority.
    w2(12, 11)
    w2(12, 10, value=-0.2)
    goal_mask(12, 128.0)
    p["b2"][12] = 20.0 - 128.0

    for dst, src in ((13, 16), (14, 17), (15, 18),
                     (16, 19), (17, 20)):
        w2(dst, src)

    # Break exact ties between refill sources.
    w2(16, 11, value=-0.004)
    w2(16, 10, value=-0.002)

    # Immediate movement-clearance indicators.
    for action, (dy, dx) in enumerate(directions):
        w2(18 + action, 13)
        w2(18 + action, 12, dy, dx, -1.0)

    # A blue strip at either forward diagonal corner warns that the
    # next tile may activate a side-facing pusher.
    corner_samples = (
        ((21, -1, -1), (22, -1, 1)),
        ((23, 1, -1), (24, 1, 1)),
        ((21, -1, -1), (23, 1, -1)),
        ((22, -1, 1), (24, 1, 1)),
    )
    for action, samples in enumerate(corner_samples):
        out = 22 + action
        w2(out, 13, value=2.0)
        for channel, dy, dx in samples:
            w2(out, channel, dy, dx)
        p["b2"][out] = -2.0

    # Player coordinates for detecting forced backward displacement.
    w2(26, 10)
    w2(26, 13, value=128.0)
    p["b2"][26] = -128.0

    w2(27, 11, value=-1.0)
    w2(27, 13, value=128.0)
    p["b2"][27] = 57.0 - 128.0

    # Third-layer routing and axial selection of goal descriptors.
    for j in list(range(5)) + [11] + list(range(13, 28)):
        p["W3"][j, j] = 1.0

    p["W3"][10, 10] = -1.0
    p["b3"][10] = 1.0  # complement of player-centre indicator

    def axial_priority(out, scale):
        p["W3"][out, 12] += 2.0 * scale
        p["W3"][out, 32 + 12] -= scale
        p["W3"][out, 64 + 12] -= scale

    for j, bound in enumerate(bounds):
        p["W3"][5 + j, 5 + j] = 1.0
        axial_priority(5 + j, 2.0 * bound)

    p["W3"][12, 5] = 0.2
    axial_priority(12, 16.0)

    # Four wavefronts:
    # goal/refill, shape, colour, rotation.
    #
    # Each pair encodes a clipped reached value as raw - excess.
    # Eight further channels integrate arrival-time derivatives.
    baseline = 512.0
    for field in range(4):
        raw, excess = 2 * field, 2 * field + 1
        for out in (raw, excess):
            for dy, dx in directions:
                p["Wh"][out, raw, dy + 1, dx + 1] += 1.0
                p["Wh"][out, excess, dy + 1, dx + 1] -= 1.0
            p["Uh"][out, 11] = -16.0

            if field == 0:
                p["Uh"][out, 16] = 1.0
                p["Uh"][out, 12] = 1.0e-8
            else:
                p["Uh"][out, 12 + field] = 1.0

        p["bh"][excess] = -1.0

        horizontal = 8 + 2 * field
        vertical = horizontal + 1
        for out in (horizontal, vertical):
            p["Wh"][out, out, 1, 1] = 1.0
            p["Uh"][out, 10] = -16.0
            p["bh"][out] = 4.0

        for channel, coefficient in ((raw, 1.0), (excess, -1.0)):
            p["Wh"][horizontal, channel, 1, 2] += coefficient
            p["Wh"][horizontal, channel, 1, 0] -= coefficient
            p["Wh"][vertical, channel, 2, 1] += coefficient
            p["Wh"][vertical, channel, 0, 1] -= coefficient

    # Global comparisons: shape invariants, bitmap, and colour.
    def comparison(positive, negative, feature, tolerance):
        p["Wz"][positive, 32 + feature] = 1.0
        p["Wz"][positive, 32 + 5 + feature] = -1.0
        p["Wz"][negative] = -p["Wz"][positive]
        p["bz"][positive] = -tolerance
        p["bz"][negative] = -tolerance

    for j in range(3):
        comparison(2 * j, 2 * j + 1, j, 0.20)
    comparison(6, 7, 3, 0.25)
    comparison(8, 9, 4, 0.25)

    for action in range(4):
        p["Wz"][10 + action, 32 + 18 + action] = 1.0

    p["Wz"][14, 64 + 17] = 4096.0
    p["Wz"][15, 32 + 16] = 2.0

    # Piecewise readouts of the integrated spatial derivatives.
    for field in range(4):
        for action in range(4):
            read = 9 + 2 * field if action < 2 else 8 + 2 * field
            sign = -1.0 if action in (0, 2) else 1.0
            start = 32 + 12 * field + 3 * action
            for j, threshold in enumerate((0.0, 2.0, 4.0)):
                p["Wz"][start + j, read] = sign
                p["bz"][start + j] = -sign * baseline - threshold

    # Recharge earlier when shape transformations imply longer detours.
    p["Wz"][80, 64 + 17] = 4096.0
    p["Wz"][80, 32 + 13] = -16.0
    p["bz"][80] = -14.0

    p["Wz"][81, 32 + 16] = -2.0
    p["bz"][81] = 1.0

    for action in range(4):
        p["Wz"][82 + action, 32 + 22 + action] = 1.0
    p["Wz"][86, 32 + 26] = 1.0
    p["Wz"][87, 32 + 27] = 1.0

    shape_error = list(range(6))
    bitmap_error = [6, 7]
    colour_error = [8, 9]
    next_unit = 0

    def gate(field, action, errors=(), fuel=False):
        nonlocal next_unit
        unit = next_unit
        next_unit += 1
        gain = 4.0

        start = 32 + 12 * field + 3 * action
        for j, weight in enumerate((1.0, -1.99975, 1.0)):
            p["Wmz"][unit, start + j] = gain * weight

        p["Wmz"][unit, 10 + action] = 4.0 * gain
        p["bm"][unit] = gain * (-4.0 - 0.03)

        for z in errors:
            p["Wmz"][unit, z] -= 10.0 * gain

        if fuel:
            p["Wmz"][unit, 80] -= 40.0 * gain
            p["Wmz"][unit, 81] -= 20.0 * gain

        return unit

    def vote(action, unit, weight):
        p["Wo"][1 + action, 128 + unit] += weight

    for action in range(4):
        unit = gate(0, action, bitmap_error + colour_error)
        vote(action, unit, 1.0)

        unit = gate(1, action)
        suppressed = gate(1, action, shape_error)
        vote(action, unit, 1.0)
        vote(action, suppressed, -1.0)

        unit = gate(3, action, shape_error)
        suppressed = gate(3, action, bitmap_error)
        vote(action, unit, 1.0)
        vote(action, suppressed, -1.0)

        unit = gate(2, action, bitmap_error)
        suppressed = gate(2, action, bitmap_error + colour_error)
        vote(action, unit, 1.0)
        vote(action, suppressed, -1.0)

        unit = gate(0, action, fuel=True)
        vote(action, unit, 20.0)

    # Brief avoidance traces for blue-strip hazards.
    # While the shape is wrong, the traces become an equal constant:
    # pushers remain usable for reaching otherwise awkward shape pads.
    for action in range(4):
        unit = 32 + action
        p["Wm"][unit, unit] = 0.65
        p["Wmz"][unit, 82 + action] = 2.5
        for z in shape_error:
            p["Wmz"][unit, z] = -30.0
        vote(action, unit, -2.5)

    # Small-coordinate memory is approximately linear under tanh.
    p["Wmz"][36, 86] = 1.0 / 1024.0
    p["Wmz"][37, 87] = 1.0 / 1024.0

    # Detect motion opposite to the issued direction. These units are
    # normally all -1, hence contribute only an action-independent bias.
    for action in range(4):
        unit = 38 + action
        axis_memory = 37 if action < 2 else 36
        axis_z = 87 if action < 2 else 86
        sign = -1.0 if action in (0, 2) else 1.0
        gain = 4.0

        p["Wm"][unit, axis_memory] = gain * 1024.0 * sign
        p["Wmz"][unit, axis_z] = -gain * sign
        p["Wma"][unit, 1 + action] = gain * 128.0
        p["bm"][unit] = -gain * (128.0 + 2.5)
        vote(action, unit, -2.0)

    # Weak legal-move preference and directional persistence.
    for action in range(4):
        p["Wo"][1 + action, 10 + action] = 0.001
        unit = 42 + action
        p["Wma"][unit, 1 + action] = 3.0
        vote(action, unit, 0.0001)

    p["bo"][:] = -100.0
    p["bo"][1:5] = np.array(
        [0.00004, 0.00002, 0.00001, 0.00003],
        dtype=np.float32,
    )
    return p
