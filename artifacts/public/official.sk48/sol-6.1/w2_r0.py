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
    p = {name: np.zeros(shape, np.float32)
         for name, shape in shapes.items()}

    colours = (7, 8, 9, 10, 12, 13, 14)
    tile_colours = tuple(range(7, 15))
    head_colours = (6, 15)
    corners = ((-1, -1), (-1, 2), (2, -1), (2, 2))
    centre = ((0, 0), (0, 1), (1, 0), (1, 1))

    def c1(out, channel, dy=0, dx=0, value=1.0):
        p["W1"][out, channel, dy + 2, dx + 2] += value

    def c2(out, channel, dy=0, dx=0, value=1.0):
        p["W2"][out, channel, dy + 1, dx + 1] += value

    def top(out):
        c2(out, 25, value=-1)
        c2(out, 26, value=1)

    def bottom(out):
        c2(out, 25, value=1)
        c2(out, 26, value=-1)

    # Tile centres, with a second copy for reading the reference strips.
    for k, colour in enumerate(colours):
        for out in (k, 7 + k):
            for dy, dx in corners + centre:
                c1(out, colour, dy, dx)
            p["b1"][out] = -7
        c1(k, 1, 1, -2, -1)
        c1(k, 1, -2, 0, -1)

    # Selected and unselected clickable heads.
    for out, ring in ((14, 0), (15, 4)):
        for dy, dx in centre:
            for colour in head_colours:
                c1(out, colour, dy, dx)
        c1(out, ring, 0, -1)
        c1(out, ring, 1, -1)
        for colour in head_colours:
            c1(out, colour, -2, -2)
        p["b1"][out] = -6

    # Closed edges distinguish horizontal and vertical heads.
    for colour in head_colours:
        c1(16, colour, 2, 0)
        c1(17, colour, 0, 2)

    c1(18, 1)

    # Carried tiles and tiles attached to perpendicular strips.
    for out in (19, 20, 21, 22):
        for dy, dx in corners:
            for colour in tile_colours:
                c1(out, colour, dy, dx)

    c1(19, 1, 1, -2)
    c1(20, 1, -2, 0)
    p["b1"][19:21] = -4

    c1(21, 1, 1, -2)
    c1(21, 2, -2, 0)
    c1(21, 3, -2, 0)
    c1(22, 1, -2, 0)
    c1(22, 2, 1, -2)
    c1(22, 3, 1, -2)
    p["b1"][21:23] = -5

    c1(24, 4)
    c1(25, 16, value=63)
    c1(26, 16, value=63)
    p["b1"][25] = -52
    p["b1"][26] = -53
    c1(30, 16)
    c1(31, 17)

    # Source coordinates and unmarked reference colours.
    for k in range(7):
        c2(k, k)
        c2(k, 30)
        top(k)
        p["b2"][k] = -1

        c2(7 + k, k)
        c2(7 + k, 31)
        top(7 + k)
        p["b2"][7 + k] = -1

        c2(14 + k, 7 + k)
        bottom(14 + k)
        p["b2"][14 + k] = -1

    c2(21, 14)
    c2(21, 16, 1, 0)
    top(21)
    p["b2"][21] = -1

    c2(22, 14)
    c2(22, 17, 0, 1)
    top(22)
    p["b2"][22] = -1

    c2(23, 14)
    c2(23, 16, 1, 0)
    c2(23, 30)
    top(23)
    p["b2"][23] = -2

    c2(24, 14)
    c2(24, 17, 0, 1)
    c2(24, 31)
    top(24)
    p["b2"][24] = -2

    c2(25, 18)
    top(25)

    c2(26, 19)
    c2(26, 20)
    top(26)

    c2(27, 21)
    c2(27, 22)
    top(27)

    c2(28, 14)
    bottom(28)
    p["b2"][28] = -1

    bottom(29)

    c2(30, 24)
    bottom(30)
    p["b2"][30] = -1

    c2(31, 15)
    top(31)

    # Spatial features:
    # 0:7 source rows; 7:14 source columns; 14:21 reference colours;
    # 21 H row; 22 V column; 23 V flag; 24 strip ones;
    # 25 carried tiles; 26 crossing lock; 27 H flag;
    # 28 scan seed; 29 bottom mask; 30 terminator; 31 click candidate.
    for k in range(21):
        p["W3"][k, k] = 1
    mapping = {
        21: 23, 22: 24, 23: 22, 24: 25,
        25: 26, 26: 27, 27: 21,
        28: 28, 29: 29, 30: 30, 31: 31,
    }
    for dst, src in mapping.items():
        p["W3"][dst, src] = 1

    # Scan right through the correct prefix of the selected reference.
    p["Wh"][0, 0, 1, 0] = 1
    p["Uh"][0, 28] = 1
    p["Uh"][0, 30] = -1
    for k in range(7):
        p["Uh"][0, 14 + k] = -1
        p["Wh"][1 + k, 0, 1, 0] = 1
        p["Uh"][1 + k, 14 + k] = 1
        p["Uh"][1 + k, 29] = -1

    def fm(channel):
        return 32 + channel

    def fa(channel):
        return 64 + channel

    def combine(*terms):
        result = {}
        for scale, coefficients in terms:
            for key, value in coefficients.items():
                result[key] = result.get(key, 0.0) + scale * value
        return result

    def affine(*terms):
        coefficients = {}
        bias = 0.0
        for scale, (part, constant) in terms:
            coefficients = combine(
                (1, coefficients), (scale, part))
            bias += scale * constant
        return coefficients, bias

    prefix = ({16: 4096.0 / 6.0}, -1.0)
    length = ({fa(24): 512.0, fa(25): 3072.0}, 0.875)
    carried = ({fa(25): 4096.0}, 0.0)
    excess = affine((1, carried), (-1, prefix))
    extra_length = affine((1, length), (-1, prefix))
    norm_error = (extra_length[0], extra_length[1] - 2.0)

    units = {}
    next_unit = 0

    def unit(name, coefficients=None, bias=0.0):
        nonlocal next_unit
        index = next_unit
        next_unit += 1
        units[name] = index
        for channel, value in (coefficients or {}).items():
            p["Wz"][index, channel] = value
        p["bz"][index] = bias
        return index

    unit("prefix", *prefix)
    unit("length", *length)
    unit("carried", *carried)
    unit("wrong", *excess)
    unit("wrong_cap", excess[0], excess[1] - 1)
    unit("norm_pos", *norm_error)
    unit("norm_neg",
         {channel: -value for channel, value in norm_error[0].items()},
         -norm_error[1])

    unit("H", {fm(27): 1})
    unit("V", {fm(23): 1})
    unit("lock", {fm(26): 1})
    unit("row", {fm(21): 1})
    unit("col", {fm(22): 1})
    unit("finished", {1 + k: -1 for k in range(7)}, 1)

    for orientation, channel in (("H", 27), ("V", 23)):
        coefficients = combine(
            (1, extra_length[0]), (20, {fm(channel): 1}))
        bias = extra_length[1] - 21
        unit("shrink_" + orientation, coefficients, bias)
        unit("shrink_cap_" + orientation, coefficients, bias - 1)

    unit("lock_H", {fm(26): 1, fm(27): 1}, -1)
    unit("lock_V", {fm(26): 1, fm(23): 1}, -1)

    # Navigation is enabled only at the packed length prefix+1.
    navigation_gate = {
        channel: -20 * value
        for channel, value in extra_length[0].items()
    }
    navigation_bias = 20 - 20 * extra_length[1]
    margin = 0.018

    for k in range(7):
        request = 1 + k
        common_h = {request: 2, fm(27): 2}
        common_v = {request: 2, fm(23): 2}

        differences = {
            "up": {fm(21): 1, fm(k): -1},
            "down": {fm(k): 1, fm(21): -1},
            "left": {fm(22): 1, fm(7 + k): -1},
            "right": {fm(7 + k): 1, fm(22): -1},
        }

        for direction in ("up", "down", "left", "right"):
            common = common_h if direction in ("up", "down") else common_v
            unit(
                "%s_%d" % (direction, k),
                combine(
                    (1, common),
                    (1, differences[direction]),
                    (1, navigation_gate)),
                -4 - margin + navigation_bias)

            # Ungated signed target differences choose an outward purge.
            unit(
                "delta_%s_%d" % (direction, k),
                combine((1, common), (1, differences[direction])),
                -4)

        for direction in ("left", "right"):
            unit(
                "locked_%s_%d" % (direction, k),
                combine(
                    (1, common_v),
                    (1, differences[direction]),
                    (2, {fm(26): 1})),
                -6 - margin)

    def mz(memory, name, value):
        p["Wmz"][memory, units[name]] += value

    def wrong_memory(memory, value):
        mz(memory, "wrong", value)
        mz(memory, "wrong_cap", -value)

    # Purge state: move an extra tile outward, then retract it off the bar.
    p["Wm"][0, 1] = 20
    p["bm"][0] = -10

    def purge_conditions(memory, scale):
        wrong_memory(memory, 2 * scale)
        p["Wm"][memory, 0] = -scale
        p["Wm"][memory, 1] = -2 * scale
        mz(memory, "norm_pos", -10 * scale)
        mz(memory, "norm_neg", -10 * scale)
        mz(memory, "lock", -2 * scale)
        p["bm"][memory] = -5 * scale

    purge_conditions(1, 100)
    p["bm"][1] += 10

    purge_directions = (
        (2, "H", "up", "down", -1),
        (3, "H", "down", "up", 1),
        (4, "V", "left", "right", -1),
        (5, "V", "right", "left", 1),
    )
    for memory, orientation, away, toward, tie in purge_directions:
        purge_conditions(memory, 100)
        mz(memory, orientation, 200)
        p["bm"][memory] -= 200
        p["bm"][memory] += tie
        for k in range(7):
            mz(memory, "delta_%s_%d" % (away, k), 500)
            mz(memory, "delta_%s_%d" % (toward, k), -500)

    # Acquisition latches until the prefix advances, a wrong tile is
    # captured, or extension is blocked.
    for memory, orientation, action in ((6, "H", 4), (7, "V", 2)):
        p["Wm"][memory, memory] = 5
        p["Wma"][memory, action] = 10
        mz(memory, orientation, 20)
        mz(memory, "prefix", -100)
        p["Wm"][memory, 8] = 10000
        wrong_memory(memory, -50)
        mz(memory, "lock", -50)
        p["bm"][memory] = -31

    mz(8, "prefix", 0.01)
    mz(9, "length", 0.01)

    # Detect extension attempts that did not increase length.
    for memory, orientation, action in ((11, "H", 4), (12, "V", 2)):
        p["Wm"][memory, 9] = 10000
        mz(memory, "length", -100)
        p["Wma"][memory, action] = 200
        mz(memory, orientation, 200)
        p["bm"][memory] = -350

    p["Wm"][6, 11] = -10
    p["Wm"][7, 12] = -10

    def out(action, name, value):
        p["Wo"][action, units[name]] += value

    def memory_out(action, memory, value):
        p["Wo"][action, 128 + memory] += value

    p["bo"][:] = -1000
    p["bo"][1:5] = -10

    out(4, "H", 14)
    out(2, "V", 14)

    for k in range(7):
        out(1, "up_%d" % k, 10000)
        out(2, "down_%d" % k, 10000)
        out(3, "left_%d" % k, 10000)
        out(4, "right_%d" % k, 10000)
        out(3, "locked_left_%d" % k, 5000)
        out(4, "locked_right_%d" % k, 5000)

    for action, orientation, acquisition in ((3, "H", 6), (1, "V", 7)):
        out(action, "shrink_" + orientation, 200)
        out(action, "shrink_cap_" + orientation, -200)
        memory_out(action, acquisition, -100)
        p["bo"][action] -= 100

    # Evacuate horizontal acquisitions along a perpendicular vertical strip.
    out(2, "lock_H", 2000)
    out(2, "lock_V", 300)

    for action, memory in ((2, 2), (1, 3), (4, 4), (3, 5)):
        memory_out(action, memory, 200)
        p["bo"][action] += 200

    # Escape blocked extensions, then normalize and navigate again.
    memory_out(2, 11, 70)
    p["bo"][2] += 70
    memory_out(4, 12, 70)
    p["bo"][4] += 70
    memory_out(4, 11, -70)
    p["bo"][4] -= 70
    memory_out(2, 12, -70)
    p["bo"][2] -= 70

    # A completed selected reference requests another clickable head.
    p["bo"][6] = -100
    out(6, "finished", 1600)
    p["Wc"][0, 31] = 1

    p["bo"][0] = -1000
    p["bo"][5] = -1000
    p["bo"][7] = -1000

    return p
