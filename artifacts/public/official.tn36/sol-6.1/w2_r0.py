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
    p = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }
    f1, f2, f3 = {}, {}, {}

    def alloc(table, name):
        index = len(table)
        if index >= 32:
            raise RuntimeError("Feature budget exceeded")
        table[name] = index
        return index

    def c1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def c2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def layer2(name, terms, bias=0.0):
        out = alloc(f2, name)
        for source, dy, dx, value in terms:
            c2(out, f1[source], dy, dx, value)
        p["b2"][out] = bias
        return out

    def layer3(name, terms, bias=0.0):
        out = alloc(f3, name)
        for source, block, value in terms:
            p["W3"][out, block * 32 + f2[source]] += value
        p["b3"][out] = bias
        return out

    robot = np.ones((4, 4), dtype=np.int8)
    robot[3, 1:3] = 0
    goal = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ], dtype=np.int8)

    robot_names = []
    goal_parts = []

    # Exact robot masks with exterior border checks.
    for rotation in range(4):
        name = "robot_%d" % rotation
        out = alloc(f1, name)
        mask = np.rot90(robot, -rotation)
        for y in range(4):
            for x in range(4):
                c1(
                    out, 11, y - 1, x - 1,
                    1.0 if mask[y, x] else -1.0,
                )
        for x in range(4):
            c1(out, 11, -2, x - 1, -1.0)
        for y in range(4):
            c1(out, 11, y - 1, -2, -1.0)
        p["b1"][out] = 1.0 - float(mask.sum())
        robot_names.append(name)

    # Two overlapping patches verify the complete target mask.
    for rotation in range(4):
        mask = np.rot90(goal, -rotation)
        names = []
        for part, patch in enumerate((mask[:5, :5], mask[1:, 1:])):
            name = "goal_%d_%d" % (rotation, part)
            out = alloc(f1, name)
            for y in range(5):
                for x in range(5):
                    c1(
                        out, 11, y - 2, x - 2,
                        1.0 if patch[y, x] else -1.0,
                    )
            p["b1"][out] = 1.0 - float(patch.sum())
            names.append(name)
        goal_parts.append(names)

    # Enlarged robot corner.
    out = alloc(f1, "large_corner")
    for y in range(3):
        for x in range(3):
            c1(out, 11, y, x)
    for x in range(3):
        c1(out, 11, -1, x, -1.0)
    for y in range(3):
        c1(out, 11, y, -1, -1.0)
    p["b1"][out] = -8.0

    out = alloc(f1, "solid")
    for y in range(3):
        for x in range(3):
            c1(out, 11, y, x)
    p["b1"][out] = -8.0

    out = alloc(f1, "row")
    c1(out, 16, value=63.0)

    out = alloc(f1, "col")
    c1(out, 17, value=63.0)

    out = alloc(f1, "five")
    c1(out, 5)

    def bit_sample(out, dy, dx):
        c1(out, 1, dy, dx)
        c1(out, 5, dy, dx)

    # Horizontal glyph anchor.
    out = alloc(f1, "horizontal")
    for x in range(3):
        bit_sample(out, 0, x)
    for dy, dx in ((0, -1), (-1, 0), (1, 0)):
        c1(out, 0, dy, dx)
    p["b1"][out] = -5.0

    # Vertical anchor is inside the switch's clickable bounds.
    out = alloc(f1, "vertical")
    for y in range(3):
        bit_sample(out, y, 1)
    for dy, dx in ((0, 0), (-1, 1), (0, 2)):
        c1(out, 0, dy, dx)
    p["b1"][out] = -5.0

    # Require a substantial interior of the Run button.
    out = alloc(f1, "run")
    for y in range(-2, 3):
        for x in range(-2, 3):
            c1(out, 9, y, x)
    p["b1"][out] = -24.0

    knots = [32, 33, 35, 39, 41, 42, 44, 45, 47, 48]
    for row in knots:
        out = alloc(f1, "hinge_%d" % row)
        c1(out, 16, value=63.0)
        p["b1"][out] = -float(row)

    out = alloc(f1, "top")
    c1(out, 18, -1, 0, -1.0)
    p["b1"][out] = 1.0

    # Exclude bottom UI decorations, not the lower game board.
    out = alloc(f1, "below_world")
    c1(out, 16, value=63.0)
    p["b1"][out] = -48.0

    for axis, coordinate in (("x", "col"), ("y", "row")):
        terms = [
            (coordinate, 0, 0, 1.0),
            ("below_world", 0, 0, -128.0),
        ]
        terms += [(name, 0, 0, 64.0) for name in robot_names]
        layer2("robot_" + axis, terms, -65.0)

        for rotation, names in enumerate(goal_parts):
            layer2("goal_%s_%d" % (axis, rotation), [
                (coordinate, 0, 0, 1.0),
                (names[0], 0, 0, 64.0),
                (names[1], 1, 1, 64.0),
                ("below_world", 0, 0, -128.0),
            ], -129.0)

        layer2("large_" + axis, [
            (coordinate, 0, 0, 1.0),
            ("large_corner", 0, 0, 64.0),
            ("solid", 1, 1, 64.0),
            ("below_world", 0, 0, -128.0),
        ], -128.0)

    def hat_terms(row):
        j = knots.index(row)
        left, right = knots[j - 1], knots[j + 1]
        a = 1.0 / (row - left)
        b = 1.0 / (right - row)
        return [
            ("hinge_%d" % left, 0, 0, a),
            ("hinge_%d" % row, 0, 0, -a - b),
            ("hinge_%d" % right, 0, 0, b),
        ]

    for name, rows in (
        ("bit0", [33, 42]),
        ("bit1", [35, 44]),
        ("bit3", [41]),
        ("bit5", [47]),
        ("single_row", [42, 44]),
    ):
        terms = []
        for row in rows:
            terms.extend(hat_terms(row))
        layer2(name, terms)

    layer2("candidate", [
        ("horizontal", 0, 0, 1.0),
        ("vertical", 0, 0, 1.0),
    ])

    layer2("actual_h", [
        ("horizontal", 0, 0, 1.0),
        ("five", 0, 0, 1.0),
    ], -1.0)

    layer2("actual_v", [
        ("vertical", 0, 0, 1.0),
        ("five", 0, 1, 1.0),
    ], -1.0)

    # Inverse coordinates let row-max select the leftmost glyph.
    layer2("origin_h", [
        ("col", 0, 0, -1.0),
        ("horizontal", 0, 0, 64.0),
    ])
    layer2("origin_v", [
        ("col", 0, 0, -1.0),
        ("vertical", 0, 0, 64.0),
    ])

    for name in ("col", "run", "top"):
        layer2(name, [(name, 0, 0, 1.0)])

    for axis in ("x", "y"):
        layer3("start_" + axis, [
            ("robot_" + axis, 1, 1.0),
            ("large_" + axis, 1, 1.0),
        ])
        layer3("target_" + axis, [
            ("goal_%s_%d" % (axis, rotation), 1, 1.0)
            for rotation in range(4)
        ])

    layer3("top", [("top", 0, 1.0)])

    for bit in (0, 1, 3, 5):
        layer3("gate%d" % bit, [
            ("candidate", 0, 1.0),
            ("bit%d" % bit, 0, 1.0),
        ], -1.0)

    layer3("candidate", [("candidate", 0, 1.0)])
    layer3("actual", [
        ("actual_h", 0, 1.0),
        ("actual_v", 0, 1.0),
    ])
    layer3("run", [("run", 0, 1.0)])

    layer3("index_complement", [
        ("col", 0, -0.2),
        ("origin_h", 1, -0.2),
        ("origin_v", 1, -0.2),
    ], 32.8)

    # A leading shrink on the wider panel is harmless at scale one.
    layer3("prefix", [
        ("origin_h", 1, 0.2),
        ("origin_v", 1, 0.2),
        ("single_row", 0, -100.0),
    ], -5.2)

    for bit in (0, 1, 3, 5):
        layer3("scale%d" % bit, [
            ("col", 0, -0.2),
            ("candidate", 0, 100.0),
            ("bit%d" % bit, 0, 100.0),
            ("single_row", 0, -100.0),
        ], 7.6 - 200.0)

    def drive(unit, feature, value):
        p["Uh"][unit, f3[feature]] += value

    def recurrent(out, inp, value, dy=0, dx=0):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    # Downward prefix sums broadcast signed displacement.
    for unit, axis in ((0, "x"), (1, "y")):
        recurrent(unit, unit, 1.0, -1, 0)
        drive(unit, "top", 16.0)
        drive(unit, "target_" + axis, 0.25)
        drive(unit, "start_" + axis, -0.25)

    recurrent(2, 0, 1.0)
    recurrent(3, 1, 1.0)
    p["bh"][2:4] = -16.0

    # Paired hinges implement clipped command intervals.
    specifications = [
        ([-1, 0, 2, 0], 16.0, ["gate0", "gate5"]),
        ([-1, -1, 2, 2], 32.0, ["gate0"]),
        ([-1, 0, 2, 1], 16.0, ["gate1"]),
        ([-1, 0, 1, 0], 16.0, ["gate5"]),
        ([-1, -1, 2, 1], 32.0, ["gate5"]),
    ]

    for j, (coefficients, offset, gates) in enumerate(specifications):
        for sub in range(2):
            unit = 4 + 2 * j + sub
            for source, value in enumerate(coefficients):
                recurrent(unit, source, float(value))
            drive(unit, "index_complement", 1.0)
            drive(unit, "prefix", 1.0)
            for gate in gates:
                drive(unit, gate, 100.0)
            p["bh"][unit] = offset - 120.0 - sub

    desired = np.zeros(16, dtype=np.float32)
    for j, sign in enumerate((-1.0, 1.0, 1.0, 1.0, 1.0)):
        desired[4 + 2 * j] = sign
        desired[5 + 2 * j] = -sign

    scale_signs = {
        "scale0": 1.0,
        "scale1": -1.0,
        "scale3": 1.0,
        "scale5": -1.0,
    }

    # Absolute switch disagreement, restricted to valid anchors.
    for unit, sign in ((14, 1.0), (15, -1.0)):
        for source in range(4, 14):
            recurrent(unit, source, sign * float(desired[source]))
        for feature, value in scale_signs.items():
            drive(unit, feature, sign * value)
        drive(unit, "actual", -sign)
        drive(unit, "candidate", 100.0)
        p["bh"][unit] = -100.0

    # Correct a switch before running the compiled program.
    p["Wc"][0, 32 + 14] = 1.0
    p["Wc"][0, 32 + 15] = 1.0
    p["Wc"][0, f3["run"]] = 0.5
    p["bo"][6] = 1.0
    return p
