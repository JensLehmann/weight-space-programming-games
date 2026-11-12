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
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    a, b, c = {}, {}, {}

    def an(name, bias=0.0):
        i = len(a)
        a[name] = i
        p["b1"][i] = bias
        return i

    def aw(i, color, dy=0, dx=0, weight=1.0):
        p["W1"][i, color, dy + 2, dx + 2] += weight

    def bn(name, bias=0.0):
        i = len(b)
        b[name] = i
        p["b2"][i] = bias
        return i

    def bw(i, name, dy=0, dx=0, weight=1.0):
        p["W2"][i, a[name], dy + 1, dx + 1] += weight

    def cn(name, bias=0.0):
        i = len(c)
        c[name] = i
        p["b3"][i] = bias
        return i

    def cw(i, name, weight=1.0, axis=None):
        offset = 0 if axis is None else (32 if axis == "row" else 64)
        p["W3"][i, offset + b[name]] += weight
        if name in ("arrowD", "arrowU"):
            p["W3"][i, offset + b["small_" + name]] += weight

    tile_colors = (1, 2, 9, 10, 11, 12, 15)

    # Scale-one and scale-two movable-block anchors.
    for color in (11, 12):
        i = an("small_goal%d" % color, -3.0)
        for dy in (0, 1):
            for dx in (0, 1):
                aw(i, color, dy, dx)
        for dy, dx in ((-1, 0), (0, -1), (2, 0), (0, 2)):
            aw(i, color, dy, dx, -2.0)

        i = an("large_goal%d" % color, -15.0)
        for dy in range(-1, 3):
            for dx in range(-1, 3):
                aw(i, color, dy, dx)
        aw(i, color, -2, 0, -2.0)
        aw(i, color, 0, -2, -2.0)

        i = an("small_frame%d" % color, -3.0)
        for dy in (-1, 2):
            for dx in (-1, 2):
                aw(i, color, dy, dx)

        for dy in (-2, 2):
            for dx in (-2, 2):
                i = an("corner%d_%d_%d" % (color, dy, dx))
                aw(i, color, dy, dx)

    left = np.array(
        [[0, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [0, 1, 1]],
        np.int8,
    )
    right = left[:, ::-1]
    templates = {
        "L": (8, left, 2, 1),
        "R": (14, right, 1, 1),
        "D": (8, np.rot90(left), 1, 1),
        "U": (14, np.rot90(right), 1, 1),
    }

    for direction, (color, template, ax, ay) in templates.items():
        i = an("small_arrow" + direction, -9.0)
        for sy in range(template.shape[0]):
            for sx in range(template.shape[1]):
                aw(
                    i, color, sy - ay, sx - ax,
                    1.0 if template[sy, sx] else -3.0,
                )

    # Scale-two wing anchors.
    for direction, color in (("L", 8), ("R", 14), ("D", 8), ("U", 14)):
        i = an("large_arrow" + direction, -7.0)
        if direction == "L":
            for dy in range(-1, 3):
                for dx in (0, 1):
                    aw(i, color, dy, dx)
            for dy, dx in ((-2, 0), (0, -1), (1, -1)):
                aw(i, color, dy, dx, -3.0)
        elif direction == "R":
            for dy in range(-1, 3):
                for dx in (-1, 0):
                    aw(i, color, dy, dx)
            for dy, dx in ((-2, 0), (0, 1), (1, 1)):
                aw(i, color, dy, dx, -3.0)
        elif direction == "D":
            for dy in (-1, 0):
                for dx in range(-1, 3):
                    aw(i, color, dy, dx)
            for dy, dx in ((0, -2), (1, 0), (1, 1)):
                aw(i, color, dy, dx, -3.0)
        else:
            for dy in (0, 1):
                for dx in range(-1, 3):
                    aw(i, color, dy, dx)
            for dy, dx in ((0, -2), (-1, 0), (-1, 1)):
                aw(i, color, dy, dx, -3.0)

    i = an("x")
    aw(i, 17)
    i = an("y")
    aw(i, 16)

    for name, dy in (("track_up", -2), ("track_down", 2)):
        i = an(name)
        for color in tile_colors:
            aw(i, color, dy, 0)

    i = an("track")
    for color in tile_colors:
        aw(i, color)

    # Reject apparent goals at scaled frame corners.
    for sy in (-1, 1):
        for sx in (-1, 1):
            i = an("diagonal_%d_%d" % (sy, sx), -3.0)
            ys = (-1, 0) if sy < 0 else (1, 2)
            xs = (-1, 0) if sx < 0 else (1, 2)
            for dy in ys:
                for dx in xs:
                    for color in tile_colors:
                        aw(i, color, dy, dx)

    # Base-width validation for large vertical arrows. Shifted twice
    # in W2, this inspects both sides of the rectangular base.
    i = an("arrow_corner_deficit", 2.0)
    for dy in (-2, 2):
        for dx in (-2, 2):
            for color in (8, 14):
                aw(i, color, dy, dx, -1.0)

    def goal_terms(i, color, weight=1.0):
        bw(i, "small_goal%d" % color, weight=weight)
        bw(i, "large_goal%d" % color, weight=weight)
        for sy in (-1, 1):
            for sx in (-1, 1):
                bw(
                    i, "diagonal_%d_%d" % (sy, sx),
                    sy, sx, -weight,
                )

    for color, suffixes in (
        (11, ("flag", "x", "mx", "my")),
        (12, ("flag",)),
    ):
        for suffix in suffixes:
            i = bn("G%d_%s" % (color, suffix),
                   -1.0 if suffix == "x" else 0.0)
            goal_terms(i, color)
            if suffix == "x":
                bw(i, "x")
            elif suffix == "mx":
                bw(i, "x", weight=-1.0)
            elif suffix == "my":
                bw(i, "y", weight=-1.0)

    def large_frame_terms(i, color):
        for dy in (-2, 2):
            for dx in (-2, 2):
                bw(
                    i, "corner%d_%d_%d" % (color, dy, dx),
                    0 if dy < 0 else 1,
                    0 if dx < 0 else 1,
                )

    for color, suffixes in (
        (11, ("flag", "x", "mx", "my")),
        (12, ("flag", "x")),
    ):
        for scale in ("small", "large"):
            base = 0.0 if scale == "small" else -3.0
            for suffix in suffixes:
                i = bn(
                    "T%d_%s_%s" % (color, scale, suffix),
                    base - (1.0 if suffix == "x" else 0.0),
                )
                if scale == "small":
                    bw(i, "small_frame%d" % color)
                    bw(i, "large_goal%d" % color, weight=-1.0)
                else:
                    large_frame_terms(i, color)
                if suffix == "x":
                    bw(i, "x")
                elif suffix == "mx":
                    bw(i, "x", weight=-1.0)
                elif suffix == "my":
                    bw(i, "y", weight=-1.0)

    for direction in ("L", "R", "D", "U"):
        i = bn("arrow" + direction)
        bw(i, "large_arrow" + direction)
        if direction in ("L", "R"):
            bw(i, "small_arrow" + direction)
        else:
            dy = -1 if direction == "D" else 1
            for dx in (-1, 1):
                bw(i, "arrow_corner_deficit", dy, dx, -2.0)
            j = bn("small_arrow" + direction)
            bw(j, "small_arrow" + direction)

    for color in (11, 12):
        i = bn("vertical%d" % color, -2.0)
        goal_terms(i, color, 2.0)
        bw(i, "track_up", -1, 0)
        bw(i, "track_down", 1, 0)

    i = bn("X")
    bw(i, "x")
    i = bn("track_my")
    bw(i, "track")
    bw(i, "y", weight=-1.0)

    def target(i, color, suffix, weight=1.0, axis=None):
        for scale in ("small", "large"):
            cw(i, "T%d_%s_%s" % (color, scale, suffix), weight, axis)

    for color in (11, 12):
        i = cn("goal_col%d" % color)
        cw(i, "G%d_flag" % color, axis="col")
        i = cn("danger_col%d" % color)
        cw(i, "vertical%d" % color, axis="col")

    i = cn("left_control")
    cw(i, "arrowL")
    cw(i, "arrowD")

    i = cn("right_transport")
    cw(i, "arrowR")
    target(i, 11, "flag", -1.0, "row")

    i = cn("shared_target_row", -1.02)
    target(i, 11, "x", axis="row")
    target(i, 11, "mx", axis="row")

    i = cn("primary_left_of_midpoint", -24.01)
    cw(i, "G11_flag", 8.0, "col")
    target(i, 11, "flag", 8.0, "row")
    target(i, 12, "flag", 8.0, "row")
    target(i, 11, "x", 0.5, "row")
    target(i, 12, "x", 0.5, "row")
    cw(i, "X", -1.0)

    i = cn("secondary_unsatisfied", -0.01)
    cw(i, "G12_flag")
    target(i, 12, "flag", -1.0)

    i = cn("row_correction", -24.005)
    cw(i, "arrowL", 8.0)
    cw(i, "G11_flag", 8.0, "row")
    target(i, 11, "flag", 8.0, "row")
    target(i, 11, "x", axis="row")
    cw(i, "G11_x", -1.0, "row")

    i = cn("left_control_x", -1.0)
    cw(i, "arrowL")
    cw(i, "arrowD")
    cw(i, "X")

    i = cn("short_left", -1.0)
    cw(i, "arrowL")
    target(i, 11, "flag", axis="row")

    i = cn("vertical_control")
    cw(i, "arrowD")
    cw(i, "arrowU")

    i = cn("shared_row_has_goal", -16.1)
    cw(i, "G11_flag", 8.0, "row")
    target(i, 11, "x", 8.0, "row")
    target(i, 11, "mx", 8.0, "row")

    i = cn("buffer_has_goal", -1.01)
    cw(i, "G11_flag")
    cw(i, "arrowD", axis="col")

    i = cn("lower_target_predecessor", -256.25)
    target(i, 11, "flag", 128.0)
    cw(i, "G11_flag", 128.0, "col")
    target(i, 11, "my", 63.0, "col")
    cw(i, "G11_my", -63.0, "col")
    target(i, 11, "x", -128.0, "row")
    target(i, 11, "mx", -128.0, "row")
    target(i, 11, "flag", 128.0, "row")

    i = cn("buffer_remove", -1023.0)
    cw(i, "arrowD", 512.0)
    cw(i, "G11_flag", 512.0, "col")
    cw(i, "G11_my", 128.0, "col")
    cw(i, "track_my", -128.0, "col")

    i = cn("buffer_restore", -1024.0)
    cw(i, "arrowU", 512.0)
    cw(i, "G11_flag", 512.0, "col")
    cw(i, "track_my", 63.0, "col")
    cw(i, "G11_my", -63.0, "col")

    i = cn("down_control")
    cw(i, "arrowD")

    def hw(out, inp, dy=0, dx=0, weight=1.0):
        p["Wh"][out, inp, dy + 1, dx + 1] += weight

    def ud(out, feature, weight=1.0):
        p["Uh"][out, c[feature]] += weight

    def qweight(out, raw, cut, weight):
        hw(out, raw, weight=weight)
        hw(out, cut, weight=-weight)

    # Raw-minus-cut represents a clipped global flag.
    def flood(raw, cut, sources):
        for out in (raw, cut):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    hw(out, raw, dy, dx)
                    hw(out, cut, dy, dx, -1.0)
            for feature, strength in sources:
                ud(out, feature, strength)
        p["bh"][cut] = -1.0

    flood(
        0, 1,
        [("shared_target_row", 128.0),
         ("vertical_control", 2.0)],
    )
    flood(
        4, 5,
        [("primary_left_of_midpoint", 128.0),
         ("lower_target_predecessor", 2.0)],
    )
    flood(
        12, 13,
        [("secondary_unsatisfied", 2.0),
         ("shared_row_has_goal", 2.0),
         ("buffer_has_goal", 128.0)],
    )

    ud(2, "left_control", 16.0)
    ud(2, "down_control", -16.0)
    ud(2, "short_left", -1.0)
    qweight(2, 0, 1, -16.0)
    qweight(2, 12, 13, 16.0)

    ud(3, "right_transport", 16.0)
    qweight(3, 0, 1, 16.0)
    qweight(3, 12, 13, -16.0)
    p["bh"][3] = -16.0

    # Horizontal fields, bounded by decay below one half.
    for out, feature, decay in (
        (6, "goal_col11", 0.49),
        (7, "goal_col12", 0.49),
        (8, "danger_col11", 0.40),
        (9, "danger_col12", 0.40),
    ):
        hw(out, out, 0, -1, decay)
        hw(out, out, 0, 1, decay)
        ud(out, feature)

    # Suppress the primary selector when both transfer flags are on.
    # The secondary selector then contributes relevance, not a
    # constant bonus at every reverse control.
    ud(10, "left_control", 32.0)
    hw(10, 6)
    hw(10, 8, weight=-15.0)
    qweight(10, 4, 5, -16.0)
    qweight(10, 12, 13, -16.0)
    qweight(10, 0, 1, -32.0)
    p["bh"][10] = -16.0

    ud(11, "left_control", 32.0)
    ud(11, "left_control_x", 0.002)
    hw(11, 7)
    hw(11, 9, weight=-15.0)
    qweight(11, 4, 5, 32.0)
    qweight(11, 12, 13, 32.0)
    qweight(11, 0, 1, -64.0)
    p["bh"][11] = -96.0

    ud(14, "buffer_remove", 256.0)
    qweight(14, 4, 5, -256.0)

    ud(15, "buffer_restore", 16.0)
    qweight(15, 4, 5, 256.0)
    p["bh"][15] = -256.0

    p["bo"].fill(-100.0)
    p["bo"][6] = 10.0

    for channel in (2, 3, 10, 11, 14, 15):
        p["Wc"][0, 32 + channel] = 1.0
    p["Wc"][0, c["row_correction"]] = 256.0

    return p
