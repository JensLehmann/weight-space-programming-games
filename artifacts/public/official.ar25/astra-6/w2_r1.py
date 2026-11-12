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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    def c1(o, i, dy=0, dx=0, v=1.0):
        p["W1"][o, i, dy + 2, dx + 2] += v

    def c2(o, i, dy=0, dx=0, v=1.0):
        p["W2"][o, i, dy + 1, dx + 1] += v

    def ch(o, i, dy=0, dx=0, v=1.0):
        p["Wh"][o, i, dy + 1, dx + 1] += v

    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))

    # First layer:
    # 0 occupied pixels; 1 visible goal pixels;
    # 2 holes inside occupied cells; 3 covered goal markers;
    # 4,5 padding; 6,7 coordinates;
    # 8,9 piece/goal vertical-stroke descriptors;
    # 10,11 selected axes; 12 selected-piece markers;
    # 13,14 axes irrespective of selection.
    for color in (4, 5):
        c1(0, color)
    c1(1, 11)

    for dy, dx in cardinal:
        for color in (4, 5):
            c1(2, color, dy, dx)
            c1(3, color, dy, dx)

    for color in (0, 9, 11):
        c1(2, color)
    c1(3, 11)
    p["b1"][2:4] = -4.0

    c1(4, 17, v=63.0)
    c1(5, 16, v=63.0)
    p["b1"][4:6] = -62.0
    c1(6, 16)
    c1(7, 17)

    for out, colors in ((8, (4, 5)), (9, (11,))):
        for color in colors:
            c1(out, color, -2, 0)
            c1(out, color, 2, 0)
            c1(out, color, 0, -2, -1.0)
            c1(out, color, 0, 2, -1.0)
        p["b1"][out] = -1.0

    def axis_filter(out, vertical, selected):
        if selected:
            c1(out, 0)
        else:
            for color in (0, 9, 10):
                c1(out, color)
        for dy, dx in cardinal:
            c1(out, 10, dy, dx)
        if vertical:
            along = ((-2, 0), (2, 0))
            across = ((0, -2), (0, 2))
        else:
            along = ((0, -2), (0, 2))
            across = ((-2, 0), (2, 0))
        for dy, dx in along:
            c1(out, 10, dy, dx)
        for dy, dx in across:
            c1(out, 10, dy, dx, -1.0)
        p["b1"][out] = -6.0

    axis_filter(10, True, True)
    axis_filter(11, False, True)
    axis_filter(13, True, False)
    axis_filter(14, False, False)

    c1(12, 0)
    for dy, dx in cardinal:
        c1(12, 5, dy, dx)
    p["b1"][12] = -4.0

    # Five translated source maps for each distance circuit.
    offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    for k, (dy, dx) in enumerate(offsets):
        c2(k, 0, dy, dx)
        c2(k, 2, dy, dx)
        c2(k, 4, dy, dx, -1.0)
        c2(k, 5, dy, dx, -1.0)

        # The second field now propagates from GOAL descriptors.
        c2(5 + k, 9, dy, dx)
        c2(5 + k, 4, dy, dx, -1.0)
        c2(5 + k, 5, dy, dx, -1.0)

    # Restore covered goals to their three-by-three rendered cells.
    c2(10, 1)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                c2(10, 3, dy, dx)
    c2(10, 4, v=-1.0)
    c2(10, 5, v=-1.0)

    # Sample the second field at PIECE descriptors.
    c2(11, 8)
    c2(11, 4, v=-1.0)
    c2(11, 5, v=-1.0)

    # Goal coordinate extrema.
    for out, coordinate, sign in (
        (12, 7, 1.0), (13, 7, -1.0),
        (14, 6, 1.0), (15, 6, -1.0),
    ):
        c2(out, coordinate, v=sign)
        c2(out, 1, v=2.0)
        c2(out, 4, v=-2.0)
        c2(out, 5, v=-2.0)
        p["b2"][out] = -2.0 if sign > 0 else -1.0

    # Selected-axis presence and position.
    c2(16, 10)
    c2(17, 11)
    c2(18, 7)
    c2(18, 10, v=2.0)
    c2(19, 6)
    c2(19, 11, v=2.0)
    p["b2"][18:20] = -2.0

    # Selected-piece marker extrema: x, 1-x, y, 1-y.
    for out, coordinate, sign in (
        (20, 7, 1.0), (21, 7, -1.0),
        (22, 6, 1.0), (23, 6, -1.0),
    ):
        c2(out, coordinate, v=sign)
        c2(out, 12, v=2.0)
        p["b2"][out] = -2.0 if sign > 0 else -1.0

    # All-axis presence and position, plus selected-piece presence.
    c2(24, 13)
    c2(25, 14)
    c2(26, 7)
    c2(26, 13, v=2.0)
    c2(27, 6)
    c2(27, 14, v=2.0)
    p["b2"][26:28] = -2.0
    c2(28, 12)

    for i in range(29):
        p["W3"][i, i] = 1.0

    # Eight-channel clipped Manhattan-distance circuit.
    M = 48.0

    def distance_circuit(base, source, goal):
        a, b, c, d, e, f, residual, sample = range(base, base + 8)
        center, north, south, west, east = range(source, source + 5)

        ch(a, residual, -1, 0)
        ch(a, residual, 1, 0, -1.0)
        p["Uh"][a, north] = M
        p["Uh"][a, south] = -M
        ch(b, residual, 1, 0)
        p["Uh"][b, south] = M

        ch(c, residual, 0, 1)
        ch(c, residual, 0, -1, -1.0)
        p["Uh"][c, east] = M
        p["Uh"][c, west] = -M
        ch(d, residual, 0, -1)
        p["Uh"][d, west] = M

        ch(e, a)
        ch(e, b)
        ch(e, c, v=-1.0)
        ch(e, d, v=-1.0)
        ch(f, c)
        ch(f, d)

        ch(residual, e)
        ch(residual, f)
        p["Uh"][residual, center] = -M
        p["bh"][residual] = -1.0

        ch(sample, residual)
        p["Uh"][sample, center] = M
        p["Uh"][sample, goal] = M
        p["bh"][sample] = -M

    distance_circuit(0, 0, 10)
    distance_circuit(8, 5, 11)

    HMEAN, FMAX, FMEAN = 16, 32, 64

    # z[0]: uncovered-goal distance.
    p["Wz"][0, FMEAN + 10] = M
    p["Wz"][0, HMEAN + 7] = -1.0

    # z[11]: descriptor attraction, disabled if no goal descriptor
    # exists. Its weight in the total cost is deliberately smaller.
    p["Wz"][11, FMEAN + 11] = M
    p["Wz"][11, HMEAN + 15] = -1.0
    p["Wz"][11, FMAX + 5] = 100.0
    p["bz"][11] = -100.0

    p["Wz"][1, FMAX + 16] = 1.0
    p["Wz"][2, FMAX + 17] = 1.0

    # Centre movable axes on the midpoint of goal extrema.
    epsilon = 1.25 / 63.0

    def displacement(zid, marker, position, maximum, flipped, positive):
        sign = 1.0 if positive else -1.0
        p["Wz"][zid, FMAX + position] = sign
        p["Wz"][zid, FMAX + maximum] = -0.5 * sign
        p["Wz"][zid, FMAX + flipped] = 0.5 * sign
        p["Wz"][zid, FMAX + marker] = 2.0
        p["bz"][zid] = -0.5 * sign - epsilon - 2.0

    displacement(3, 17, 19, 14, 15, True)
    displacement(4, 17, 19, 14, 15, False)
    displacement(5, 16, 18, 12, 13, True)
    displacement(6, 16, 18, 12, 13, False)

    # z[7:11]: directions from the selected piece toward an axis.
    # Prefer a horizontal reflection axis when one is present.
    def probe_direction(zid, high, low, axis_pos, axis_present,
                        sign, horizontal_motion=False):
        p["Wz"][zid, FMAX + high] = 0.5 * sign
        p["Wz"][zid, FMAX + low] = -0.5 * sign
        p["Wz"][zid, FMAX + axis_pos] = -sign
        p["Wz"][zid, FMAX + axis_present] = 2.0
        p["Wz"][zid, FMAX + 28] = 2.0
        if horizontal_motion:
            p["Wz"][zid, FMAX + 25] = -2.0
        p["bz"][zid] = 0.5 * sign - 4.0

    probe_direction(7, 22, 23, 27, 25, 1.0)
    probe_direction(8, 22, 23, 27, 25, -1.0)
    probe_direction(9, 20, 21, 26, 24, 1.0, True)
    probe_direction(10, 20, 21, 26, 24, -1.0, True)

    # Memory 0: previous cost, kept in tanh's linear region.
    # Memory 1: non-improvement.
    scale = 65536.0
    gain = 1.0e9
    for zid, weight in ((0, 1.0), (11, 2.0)):
        p["Wmz"][0, zid] = weight / scale
        p["Wmz"][1, zid] = gain * weight / scale
    p["Wm"][1, 0] = -gain
    p["bm"][1] = gain / (2.0 * 4096.0 * scale)

    # Memory 2:6: last movement direction.
    # Memory 6:10: successor flags following an undo.
    trial_order = (2, 1, 3, 4)
    for k, action in enumerate(trial_order):
        latch, flag = 2 + k, 6 + k
        p["Wm"][latch, latch] = 4.0
        for other in trial_order:
            p["Wma"][latch, other] = (
                10.0 if other == action else -10.0
            )
        p["Wm"][flag, latch] = 6.0
        p["Wma"][flag, 7] = 12.0
        p["bm"][flag] = -12.0

    # Memory 10:18: previous-action indicators.
    for action in range(8):
        p["Wma"][10 + action, action] = 10.0

    # Reflection-branch probe:
    # 18 remembers whether this selected entity has been probed.
    # 19 marks the first observation after the probe.
    # 20:52 are four eight-stage, signed shift registers.
    probe_actions = (1, 2, 3, 4)
    probe_features = (7, 8, 9, 10)
    length = 8
    starts = [20 + length * k for k in range(4)]

    for start, zid in zip(starts, probe_features):
        # Initiate only following the final unsuccessful trial,
        # before this entity has already received a probe.
        p["Wm"][start, 5] = 1024.0       # Last trial was right.
        p["Wma"][start, 7] = 2048.0     # It was just undone.
        p["Wm"][start, 18] = -1024.0
        p["Wmz"][start, zid] = 2048.0
        p["bm"][start] = -4104.0

        for j in range(1, length):
            p["Wm"][start + j, start + j - 1] = 10.0

    # Latch probe completion until an entity-selection action.
    p["Wm"][18, 18] = 8.0
    for start in starts:
        p["Wm"][18, start] = 8.0
    p["bm"][18] = 32.0 - 64.0
    p["Wma"][18, :] = 64.0
    p["Wma"][18, 5] -= 128.0

    # Detect the end of any directional probe.
    for start in starts:
        p["Wm"][19, start + length - 1] = 8.0
    p["bm"][19] = 24.0 - 64.0
    p["Wma"][19, :] = 64.0

    def mw(action, unit, weight):
        p["Wo"][action, 128 + unit] += weight

    p["bo"][0] = -1000.0
    p["bo"][6] = -1000.0
    p["bo"][2] = 1.0

    # Ordinary greedy directional trials.
    for action in trial_order:
        mw(action, 10 + action, 2.0)

    for k, next_action in enumerate((1, 3, 4, 5)):
        mw(next_action, 6 + k, 1.0)
        p["bo"][next_action] += 1.0

    mw(7, 1, 6.0)
    for action in trial_order:
        mw(7, 10 + action, 8.0)
    p["bo"][7] = -8.0

    # A probe dominates normal trials. Subtracting the shared
    # register average cancels the inactive signed-state baseline.
    for action in probe_actions:
        for owner, start in zip(probe_actions, starts):
            weight = 30.0 if owner == action else -10.0
            for j in range(length):
                mw(action, start + j, weight)

    # After a probe, restart the trial sequence rather than
    # immediately judging or undoing its final exploratory step.
    mw(2, 19, 10.0)
    p["bo"][2] += 10.0
    mw(7, 19, -10.0)
    p["bo"][7] -= 10.0

    # Geometric axis steering takes precedence.
    p["Wo"][5, 1] = 20.0
    p["Wo"][5, 2] = 20.0
    for action, zid in ((1, 3), (2, 4), (3, 5), (4, 6)):
        p["Wo"][action, zid] = 4000.0

    return p
