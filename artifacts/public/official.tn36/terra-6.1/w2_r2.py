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
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    n1 = n2 = n3 = 0
    ramps = {}

    def new1():
        nonlocal n1
        i = n1
        n1 += 1
        assert n1 <= 32
        return i

    def new2():
        nonlocal n2
        i = n2
        n2 += 1
        assert n2 <= 32
        return i

    def new3():
        nonlocal n3
        i = n3
        n3 += 1
        assert n3 <= 32
        return i

    def ramp(axis, side, bound):
        key = (axis, side, bound)
        if key in ramps:
            return ramps[key]
        i = new1()
        coord = 16 if axis == "y" else 17
        if side == "lo":
            p["W1"][i, coord, 2, 2] = -1
            p["b1"][i] = bound / 63.0
        else:
            p["W1"][i, coord, 2, 2] = 1
            p["b1"][i] = -bound / 63.0
        ramps[key] = i
        return i

    def horizontal(color):
        i = new1()
        for dx in (-1, 0, 1):
            p["W1"][i, color, 2, 2 + dx] = 1
        for dy, dx in ((0, -2), (0, 2), (-1, 0), (1, 0)):
            p["W1"][i, 0, 2 + dy, 2 + dx] = 1
        p["b1"][i] = -6.5
        return i

    def vertical(color):
        i = new1()
        for dy in (-1, 0, 1):
            p["W1"][i, color, 2 + dy, 2] = 1
        p["W1"][i, 0, 2, 1] = 1
        p["W1"][i, 0, 2, 3] = 1
        p["b1"][i] = -4.5
        return i

    # Color 1 means an unset instruction bit; color 5 means set.
    h_unset = horizontal(1)
    v_unset = vertical(1)
    h_set = horizontal(5)
    v_set = vertical(5)

    # Detect a small, right-facing piece.
    robot = new1()
    for y in range(4):
        for x in range(4):
            p["W1"][robot, 11, 1 + y, 1 + x] = (
                -1 if x == 0 and y in (1, 2) else 1
            )
    p["b1"][robot] = -13.5

    # Detect a solid patch of a large piece.
    large = new1()
    p["W1"][large, 11, :, :] = 1
    p["b1"][large] = -24.5

    # Detect the top two rows of an upward-facing goal.
    up_goal = new1()
    for x in range(5):
        p["W1"][up_goal, 11, 2, x] = 1
    for x in (0, 2, 3):
        p["W1"][up_goal, 11, 3, x] = 1
    for x in (1, 4):
        p["W1"][up_goal, 11, 3, x] = -1
    p["b1"][up_goal] = -7.5

    unset = new2()
    set_bit = new2()
    for out, inputs in (
        (unset, (h_unset, v_unset)),
        (set_bit, (h_set, v_set)),
    ):
        for i in inputs:
            p["W2"][out, i, 1, 1] = 2
        p["W2"][out, ramp("y", "lo", 32), 1, 1] = -64
        p["W2"][out, ramp("y", "hi", 50), 1, 1] = -64

    # Heuristic phases for a right-facing piece moving through checkpoints.
    phase_constraints = [
        (("y", "lo", 25), ("x", "hi", 46)),
        (("y", "lo", 25), ("x", "lo", 46)),
        (("y", "lo", 15), ("y", "hi", 25)),
        (("y", "hi", 15),),
    ]
    phase_seed = []
    for constraints in phase_constraints:
        i = new2()
        p["W2"][i, robot, 1, 1] = 2
        for axis, side, bound in constraints:
            p["W2"][i, ramp(axis, side, bound), 1, 1] = -64
        phase_seed.append(i)

    large_seed = new2()
    p["W2"][large_seed, large, 1, 1] = 2
    p["W2"][large_seed, ramp("y", "hi", 28), 1, 1] = -64

    up_seed = new2()
    p["W2"][up_seed, up_goal, 1, 1] = 2
    p["W2"][up_seed, ramp("y", "hi", 28), 1, 1] = -64

    rectangles = {}

    def rect(y, lo=None, hi=None):
        key = (y, lo, hi)
        if key in rectangles:
            return rectangles[key]
        i = new2()
        for marker in (h_unset, v_unset, h_set, v_set):
            p["W2"][i, marker, 1, 1] = 2
        p["W2"][i, ramp("y", "lo", y), 1, 1] = -64
        p["W2"][i, ramp("y", "hi", y), 1, 1] = -64
        if lo is not None:
            p["W2"][i, ramp("x", "lo", lo), 1, 1] = -64
        if hi is not None:
            p["W2"][i, ramp("x", "hi", hi), 1, 1] = -64
        rectangles[key] = i
        return i

    phase_rects = [
        [rect(36, hi=56)],
        [rect(33, hi=46), rect(48, lo=36, hi=46)],
        [rect(33, hi=51), rect(48, hi=51),
         rect(36, lo=51, hi=56)],
        [rect(33, hi=56), rect(36, lo=46, hi=51)],
    ]

    # Shrink, move left, then move down.
    large_rects = [
        rect(33),
        rect(42, hi=36),
        rect(36, lo=41),
    ]

    # Upward movement uses instruction bits 0 and 5.
    up_rects = [rect(33), rect(48)]

    # A 3x3 yellow patch identifies the run button.
    button = new1()
    p["W1"][button, 9, 1:4, 1:4] = 1
    p["b1"][button] = -8.5
    button2 = new2()
    p["W2"][button2, button, 1, 1] = 2

    # Broadcast seed positions across their rows.
    seed3 = []
    for source in phase_seed + [large_seed, up_seed]:
        i = new3()
        p["W3"][i, 32 + source] = 1
        seed3.append(i)

    def mismatch(rects):
        missing = new3()
        unwanted = new3()
        p["W3"][missing, unset] = 1
        p["W3"][unwanted, set_bit] = 1
        for r in rects:
            p["W3"][missing, r] += 1
            p["W3"][unwanted, r] -= 1
        p["b3"][missing] = -1.5
        p["b3"][unwanted] = -0.5
        return missing, unwanted

    phase_mismatch = [mismatch(rs) for rs in phase_rects]
    large_mismatch = mismatch(large_rects)
    up_mismatch = mismatch(up_rects)

    fallback = new3()
    p["W3"][fallback, unset] = 1

    button3 = new3()
    p["W3"][button3, button2] = 1

    # Propagate phase flags downward to the instruction panel.
    for h, source in list(enumerate(seed3[:4])) + [(6, seed3[5])]:
        p["Uh"][h, source] = 1
        p["Wh"][h, h, 0, 1] = 1

    # Clip the large-piece flag to one after propagation.
    p["Uh"][4, seed3[4]] = 1
    p["Wh"][4, 4, 0, 1] = 1
    p["Wh"][5, 4, 1, 1] = 1
    p["bh"][5] = -1

    # Gate mismatch maps by the selected phase.
    for j, pair in enumerate(phase_mismatch):
        out = 7 + j
        p["Uh"][out, pair[0]] = 2
        p["Uh"][out, pair[1]] = 2
        p["Wh"][out, j, 1, 1] = 1
        p["bh"][out] = -1.5

    p["Uh"][11, large_mismatch[0]] = 2
    p["Uh"][11, large_mismatch[1]] = 2
    p["Wh"][11, 4, 1, 1] = 1
    p["Wh"][11, 5, 1, 1] = -1
    p["bh"][11] = -1.5

    p["Uh"][12, up_mismatch[0]] = 2
    p["Uh"][12, up_mismatch[1]] = 2
    p["Wh"][12, 6, 1, 1] = 1
    p["bh"][12] = -1.5

    # Otherwise, set every visible instruction bit.
    p["Uh"][13, fallback] = 1
    for h in (0, 1, 2, 3, 6):
        p["Wh"][13, h, 1, 1] = -1
    p["Wh"][13, 4, 1, 1] = -1
    p["Wh"][13, 5, 1, 1] = 1

    for h in range(7, 14):
        p["Wc"][0, 32 + h] = 4
    p["Wc"][0, button3] = 0.5

    p["bo"][6] = 1
    return p
