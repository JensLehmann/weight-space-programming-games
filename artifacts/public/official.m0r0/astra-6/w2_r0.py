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
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    dirs = {
        "U": (-1, 0), "D": (1, 0),
        "L": (0, -1), "R": (0, 1),
    }
    colors = (12, 14, 15)
    diagonals = ((-1, -1), (-1, 1), (1, 1))

    # Layer 1: local palette and shape primitives.
    f1 = {}

    def new1(name, bias=0):
        i = len(f1)
        f1[name] = i
        p["b1"][i] = bias
        return i

    def w1(out, color, weight=1, dy=0, dx=0):
        p["W1"][out, color, dy + 2, dx + 2] += weight

    def diagonal_feature(dy, dx):
        i = new1(("diag", dy, dx))
        for color in (5, 9, 10, 1, 11):
            w1(i, color, dy=2 * dy, dx=2 * dx)

    for name, palette in (
        ("floor", (5,)),
        ("base", (5, 9, 10, 1)),
    ):
        i = new1(name, -2)
        for color in palette:
            w1(i, color)
        # Exclude the status bars at the image boundary.
        w1(i, 18, dy=-1)
        w1(i, 18, dy=1)

    diagonal_feature(-1, -1)

    i = new1("selected")
    w1(i, 11)
    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        w1(i, 11, -1, dy, dx)

    i = new1("orange")
    w1(i, 9)

    i = new1("corner")
    for color in (10, 1):
        w1(i, color)
        w1(i, color, -1, -1, 0)
        w1(i, color, -1, 0, -1)

    i = new1("row")
    w1(i, 16)
    i = new1("col")
    w1(i, 17)

    diagonal_feature(-1, 1)

    i = new1("posR", -64)
    w1(i, 17, 128)
    i = new1("posL", 64)
    w1(i, 17, -128)

    for color in colors:
        i = new1(("raw", color))
        w1(i, color)

        i = new1(("h5", color), -4)
        for dx in range(-2, 3):
            w1(i, color, dx=dx)

        i = new1(("v5", color), -4)
        for dy in range(-2, 3):
            w1(i, color, dy=dy)

    for name, (dy, dx) in dirs.items():
        i = new1(("floor2", name))
        for color in (5, 9, 10, 1, 11):
            w1(i, color, dy=2 * dy, dx=2 * dx)

    i = new1("near_actor")
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            for color in (10, 1):
                w1(i, color, 1 / 25, dy, dx)

    i = new1("hazard", -3)
    w1(i, 5, -1)
    for dy, dx in dirs.values():
        w1(i, 5, dy=dy, dx=dx)

    for name, (dy, dx) in dirs.items():
        i = new1(("edge", name))
        for color in (10, 1):
            w1(i, color)
            w1(i, color, -1, dy, dx)

    i = new1("inactive")
    w1(i, 1)
    diagonal_feature(1, 1)
    assert len(f1) == 32

    # Layer 2: cell geometry, extended bars, and parking pockets.
    f2 = {}

    def new2(name, bias=0):
        i = len(f2)
        f2[name] = i
        p["b2"][i] = bias
        return i

    def w2(out, name, weight=1, dy=0, dx=0):
        p["W2"][out, f1[name], dy + 1, dx + 1] += weight

    i = new2("base")
    w2(i, "base")
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            w2(i, "hazard", -20, dy, dx)

    # Difference of these channels is a capped traversability mask.
    # Dilating orange interiors closes their visually floor-colored
    # borders, while the selected block remains traversable.
    for name, bias in (("park_plus", 0), ("park_minus", -1)):
        i = new2(name, bias)
        w2(i, "floor")
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                w2(i, "selected", 1, dy, dx)
                w2(i, "orange", -10, dy, dx)
                w2(i, "hazard", -20, dy, dx)

    for name in ("selected", "orange", "corner"):
        i = new2(name)
        w2(i, name)

    i = new2("farL", -0.5)
    w2(i, "corner")
    w2(i, "col", -1)

    i = new2("farR", -1.5)
    w2(i, "corner")
    w2(i, "col")

    for name in ("posR", "posL"):
        i = new2(name)
        w2(i, name)

    for color in colors:
        i = new2(("raw", color))
        w2(i, ("raw", color))

        i = new2(("h7", color), -1)
        w2(i, ("h5", color), dx=-1)
        w2(i, ("h5", color), dx=1)

        i = new2(("v7", color), -1)
        w2(i, ("v5", color), dy=-1)
        w2(i, ("v5", color), dy=1)

    # A single exit, or a room corner with a free diagonal,
    # provides a parking candidate. Corridor bends do not.
    i = new2("parking", -2)
    w2(i, "base", 4)
    w2(i, "selected", 4)
    for name, (dy, dx) in dirs.items():
        w2(i, ("floor2", name), -1, dy, dx)
    for dy, dx in diagonals:
        w2(i, ("diag", dy, dx), 0.5, dy, dx)

    i = new2("near_orange", -1)
    w2(i, "orange")
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            w2(i, "near_actor", 1 / 9, dy, dx)

    i = new2("inactive")
    w2(i, "inactive")

    for name, (dy, dx) in dirs.items():
        i = new2(("selected_free", name), -1)
        w2(i, "selected")
        w2(i, ("floor2", name), 1, dy, dx)

    for name in dirs:
        i = new2(("edge", name))
        w2(i, ("edge", name))

    i = new2("seam_rank", -1)
    w2(i, "base", 2)
    w2(i, "posL", -0.6)
    w2(i, "posR", -0.6)
    w2(i, "row", -0.3)

    # Position signature includes movement of a selected block.
    i = new2("position", -1)
    w2(i, "corner")
    w2(i, "selected")
    w2(i, "row", 0.75)
    w2(i, "col", 0.25)
    assert len(f2) == 32

    # Axial layer: passability, target sources, and actor masks.
    f3 = {}

    def new3(name, bias=0):
        i = len(f3)
        f3[name] = i
        p["b3"][i] = bias
        return i

    def w3(out, name, weight=1, pool=0):
        p["W3"][out, 32 * pool + f2[name]] += weight

    i = new3("main_pass")
    w3(i, "base")

    i = new3("park_pass")
    w3(i, "park_plus")
    w3(i, "park_minus", -1)

    for color in colors:
        i = new3(("key", color))
        w3(i, ("raw", color))
        w3(i, ("h7", color), -1, pool=1)
        w3(i, ("v7", color), -1, pool=2)

    i = new3("goal", -0.01)
    w3(i, "seam_rank", 1024)
    w3(i, "seam_rank", -1023, pool=2)

    i = new3("parking")
    w3(i, "parking")

    for name, direction, other_half in (
        ("left_R", "R", "posR"),
        ("left_L", "L", "posR"),
        ("right_R", "R", "posL"),
        ("right_L", "L", "posL"),
    ):
        i = new3(name)
        w3(i, ("edge", direction))
        w3(i, other_half, -1)

    for direction in ("U", "D"):
        i = new3(("vertical", direction))
        w3(i, ("edge", direction))

    for direction in dirs:
        i = new3(("selected_free", direction))
        w3(i, ("selected_free", direction))

    for name in (
        "selected", "corner", "farL", "farR", "orange",
        "near_orange", "inactive", "position",
    ):
        i = new3(name)
        w3(i, name)
        if name == "near_orange":
            w3(i, "parking", -10)

    # Recurrent spatial block:
    # 0,1,2 main wave, saturation companion, arrival integral
    # 3,4,5 parking wave, saturation companion, arrival integral
    # 6..11 directional actor improvements
    # 12,13 selected-block signed gradients
    # 14 clock; 15 parking-arrival detector.
    cross = [(0, 0)] + list(dirs.values())
    barrier = 16384.0
    propagation = 256.0

    def wave(a, b, integral, pass_name, permit_keys):
        for out in (a, b):
            for dy, dx in cross:
                p["Wh"][out, a, dy + 1, dx + 1] += propagation
                p["Wh"][out, b, dy + 1, dx + 1] -= propagation
            p["Uh"][out, f3[pass_name]] += barrier
            if permit_keys:
                for color in colors:
                    p["Uh"][out, f3[("key", color)]] += barrier
            p["bh"][out] = -barrier - 128

        p["bh"][b] -= 1
        p["Wh"][integral, integral, 1, 1] = 1
        p["Wh"][integral, a, 1, 1] = 1
        p["Wh"][integral, b, 1, 1] = -1

    wave(0, 1, 2, "main_pass", True)
    wave(3, 4, 5, "park_pass", False)

    for out in (0, 1):
        p["Uh"][out, f3["goal"]] += 8192
        p["Wh"][out, 14, 1, 1] = 1
        for color in colors:
            p["Uh"][out, f3[("key", color)]] += 80

    for out in (3, 4):
        p["Uh"][out, f3["parking"]] += 512

    p["Wh"][14, 14, 1, 1] = 1
    p["bh"][14] = 1

    for out, mask, direction in (
        (6, "left_R", "R"),
        (7, "left_L", "L"),
        (8, "right_R", "R"),
        (9, "right_L", "L"),
        (10, ("vertical", "U"), "U"),
        (11, ("vertical", "D"), "D"),
    ):
        dy, dx = dirs[direction]
        p["Wh"][out, 2, 1 + dy, 1 + dx] = 1
        p["Wh"][out, 2, 1, 1] = -1
        p["Uh"][out, f3[mask]] = 512
        p["bh"][out] = -512

    for out, dy, dx in ((12, 0, 1), (13, -1, 0)):
        p["Wh"][out, 5, 1 + dy, 1 + dx] = 1
        p["Wh"][out, 5, 1 - dy, 1 - dx] = -1
        p["Uh"][out, f3["selected"]] = 640
        p["bh"][out] = -512

    p["Wh"][15, 5, 1, 1] = 1
    p["Uh"][15, f3["selected"]] = 512
    p["bh"][15] = -637

    # Dense readout helpers.
    def hm(c):
        return 16 + c

    def fm(name):
        return 32 + f3[name]

    def fa(name):
        return 64 + f3[name]

    def mix(*terms):
        result = {}
        for expr, coefficient in terms:
            for j, value in expr.items():
                result[j] = result.get(j, 0) + coefficient * value
        return result

    nz = 0

    def zunit(expr, bias=0):
        nonlocal nz
        i = nz
        nz += 1
        for j, value in expr.items():
            p["Wz"][i, j] += value
        p["bz"][i] = bias
        return {i: 1.0}

    def zclip(expr, bias=0):
        return mix(
            (zunit(expr, bias), 1),
            (zunit(expr, bias - 1), -1),
        )

    improvements = {}
    for channel in range(6, 12):
        improvements[channel] = zunit({
            hm(channel): 1024,
            fm("selected"): -1000,
        })

    horizontal = mix(
        (improvements[6], 1), (improvements[9], 1),
        (improvements[7], -1), (improvements[8], -1),
    )
    vertical = mix(
        (improvements[10], 1), (improvements[11], -1)
    )

    selected = zunit({fm("selected"): 1})
    nearby = zclip({fm("near_orange"): 100}, -6.5)
    parked = zclip({15: 1})

    separation = zunit({
        fa("farL"): 4096,
        fa("farR"): 4096,
    })
    signature = zunit({
        fa("farL"): 4096,
        fa("farR"): 4096,
        fa("position"): 8192,
    })

    gradients = {
        "R": {hm(12): 1024, fa("selected"): -131072},
        "L": {hm(12): -1024, fa("selected"): 131072},
        "U": {hm(13): 1024, fa("selected"): -131072},
        "D": {hm(13): -1024, fa("selected"): 131072},
    }
    block_scores = {}
    for direction in dirs:
        expr = dict(gradients[direction])
        expr[fm("selected")] = -999.8
        expr[fm(("selected_free", direction))] = 1000
        block_scores[direction] = zunit(expr)

    assert nz <= 128

    def mz(row, expr, coefficient=1):
        for j, value in expr.items():
            p["Wmz"][row, j] += coefficient * value

    def oz(action, expr, coefficient=1):
        for j, value in expr.items():
            p["Wo"][action, j] += coefficient * value

    def om(action, memory, coefficient=1):
        p["Wo"][action, 128 + memory] += coefficient

    # Infer horizontal control orientation from displacement.
    mz(0, separation, 0.001)
    p["Wm"][1, 1] = 4
    p["bm"][1] = 1.5

    for row, action, sign in (
        (2, 4, 1), (3, 4, -1),
        (4, 3, 1), (5, 3, -1),
    ):
        mz(row, separation, sign * 400)
        p["Wm"][row, 0] = -sign * 400000
        p["Wma"][row, action] = 2048
        p["bm"][row] = -2050

    for row, weight in ((2, -4), (3, 4), (4, 4), (5, -4)):
        p["Wm"][1, row] = weight

    # Current and delayed state signatures.
    mz(6, signature, 0.001)
    p["Wm"][35, 6] = 1

    for row, sign in ((7, 1), (8, -1)):
        mz(row, signature, sign * 300)
        p["Wm"][row, 6] = -sign * 300000
        p["bm"][row] = -1.5

    opposite = {1: 2, 2: 1, 3: 4, 4: 3}

    for k, action in enumerate((1, 2, 3, 4)):
        first = 9 + 2 * k
        for row, sign in ((first, 1), (first + 1, -1)):
            mz(row, signature, sign * 300)
            p["Wm"][row, 6] = -sign * 300000
            p["Wma"][row, action] = -2048
            p["bm"][row] = 2046.5

        echo = 17 + k
        p["Wma"][echo, action] = 8
        om(action, first, 2)
        om(action, first + 1, 2)
        om(action, echo, 4)
        p["bo"][action] -= 4

        om(opposite[action], echo, -1.55)
        delayed = 28 + k
        p["Wm"][delayed, echo] = 8
        om(opposite[action], delayed, -0.25)

        # Returning to the same signature after two steps is
        # evidence against repeating the immediate undo cycle.
        first_loop = 38 + 2 * k
        for row, sign in (
            (first_loop, 1), (first_loop + 1, -1)
        ):
            mz(row, signature, sign * 300)
            p["Wm"][row, 35] = -sign * 300000
            p["Wma"][row, action] = -2048
            p["bm"][row] = 2046.5

        target = opposite[action]
        om(target, first_loop, 0.75)
        om(target, first_loop + 1, 0.75)
        om(target, echo, 1.5)
        p["bo"][target] -= 1.5

    # Initialize with normal mirroring, then use the inferred sign.
    p["bm"][32] = 8
    for a, b in ((21, 22), (23, 24)):
        if a == 21:
            for row in (a, b):
                p["Wm"][row, 1] = 8
                p["Wm"][row, 32] = -8
            mz(a, horizontal, 0.7)
        else:
            for row in (a, b):
                p["Wm"][row, 1] = -8
                p["bm"][row] = -8
            mz(a, horizontal, -0.7)

        om(4, a, 1.5)
        om(4, b, -1.5)
        om(3, a, -1.5)
        om(3, b, 1.5)

    mz(27, vertical, 0.7)
    om(1, 27, 1.5)
    om(2, 27, -1.5)

    for direction, action in (
        ("U", 1), ("D", 2), ("L", 3), ("R", 4)
    ):
        oz(action, block_scores[direction], 4)

    # Small deterministic tie breaker.
    p["Wm"][25, 25] = 0.82
    p["Wm"][25, 26] = -0.73
    p["Wm"][26, 25] = 0.73
    p["Wm"][26, 26] = 0.82
    p["bm"][25] = 0.025
    om(4, 25, 0.025)
    om(3, 25, -0.025)
    om(1, 26, 0.025)
    om(2, 26, -0.025)
    p["bo"][1] += 0.003

    # Select a useful nearby blocker, park it, then deselect.
    p["bo"][6] = -10
    oz(6, nearby, 15)
    oz(6, selected, -20)
    oz(6, parked, 60)

    # Short cooldown does not override the strong parking release.
    p["Wm"][34, 34] = 0.85
    p["Wma"][34, 6] = 0.9
    om(6, 34, -10)

    p["Wc"][0, f3["near_orange"]] = 30
    p["Wc"][0, f3["orange"]] = 1
    p["Wc"][0, f3["inactive"]] = 100

    for action in (0, 5, 7):
        p["bo"][action] = -100

    return p
