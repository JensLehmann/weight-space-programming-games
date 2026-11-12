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

    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    opposite = (1, 0, 3, 2)
    clockwise = (3, 2, 0, 1)

    def w1(o, c, dy=0, dx=0, v=1.0):
        p["W1"][o, c, dy + 2, dx + 2] += v

    def w2(o, c, dy=0, dx=0, v=1.0):
        p["W2"][o, c, dy + 1, dx + 1] += v

    def w3(o, c, v=1.0):
        p["W3"][o, c] += v

    def template(o, allowed, rows=4):
        for r in range(rows):
            for c in range(4):
                for color in set(allowed(r, c)):
                    w1(o, color, r - 1, c - 1)
        p["b1"][o] = -(4 * rows - 1)

    # Object anchors: sprite top-left + (1, 1).
    image = np.full((4, 4), 14, dtype=np.int32)
    image[0] = 0
    for out, turns in enumerate((0, 2, 1, 3)):
        rotated = np.rot90(image, turns)

        def allowed(r, c, rotated=rotated):
            color = int(rotated[r, c])
            return (color, 4, 7) if r == 3 else (color,)

        template(out, allowed)

    def box(out, border):
        template(
            out,
            lambda r, c: border if r == 0 or c in (0, 3) else (9,),
            rows=3,
        )

    box(4, (3, 4))
    box(5, (0,))
    box(6, (3,))

    def enemy(r, c):
        colors = (11, 15) if r in (0, 3) or c in (0, 3) else (15,)
        return colors + (4, 7) if r == 3 else colors

    template(7, enemy)
    template(
        8,
        lambda r, c: (11,) if r == 0 or c in (0, 3) else (15,),
        rows=3,
    )

    solid = (3, 4, 5, 11, 12, 15)
    for out, forbidden in ((9, (5, 12)), (10, solid)):
        for dy in range(-1, 3):
            for dx in range(-1, 3):
                w1(out, 18, dy, dx)
                for color in forbidden:
                    w1(out, color, dy, dx, -1)
        p["b1"][out] = -15

    # Goal footprints can include an occluding player or held box.
    template(11, lambda r, c: (0, 2, 9, 14))
    template(12, lambda r, c: (2,))

    fence = np.array([
        [2, 1, 2, 2],
        [1, 2, 2, 2],
        [2, 2, 2, 1],
        [2, 2, 1, 2],
    ], dtype=np.int32)
    template(13, lambda r, c: (int(fence[r, c]),))

    for t, color in enumerate((2, 1, 2, 2, 2)):
        w1(14, color, -1, t - 2)
        w1(15, color, t - 2, -1)
    p["b1"][14:16] = -4

    edges = ((-2, 0), (2, 0), (0, -2), (0, 2))
    shifts = ((0, 0), (1, 0), (0, 0), (0, 1))
    for i, (dy, dx) in enumerate(edges):
        p["b1"][16 + i] = 1
        w1(16 + i, 18, dy, dx, -1)
        for color in solid:
            w1(16 + i, color, dy, dx)

        out = 20 + i
        p["b1"][out] = 1
        w1(out, 18, dy, dx, -3)
        for color in (5, 12):
            w1(out, color, dy, dx, 3)

        tangent = (0, 1, 2) if i in (0, 2) else (-1, 0, 1)
        for t, color in zip(tangent, (2, 1, 2)):
            yy, xx = (dy, t) if i < 2 else (t, dx)
            w1(out, color, yy, xx)

    # One coordinate scalar distinguishes all short displacements.
    w1(24, 16)
    w1(24, 17, v=1.0 / 16.0)

    # Vertical/horizontal 9-to-2 boundary witnesses.
    for out, offsets in (
        (25, ((0, -1), (0, 1))),
        (31, ((-1, 0), (1, 0))),
    ):
        w1(out, 9, v=2)
        for dy, dx in offsets:
            w1(out, 2, dy, dx)
        p["b1"][out] = -2

    # Do not let the player's black orientation stripe join a goal.
    for color in (2, 3, 4, 9):
        w1(26, color)

    for out, (dy, dx) in enumerate(
        ((-1, -1), (-1, 1), (1, -1), (1, 1)), start=27
    ):
        for yy, xx, color in (
            (0, 0, 2), (0, 1, 1), (1, 0, 1), (1, 1, 2)
        ):
            w1(out, color, dy + yy, dx + xx)
        p["b1"][out] = -3

    # Second-stage object and terrain maps.
    for c in range(4):
        w2(0, c)
    for out, inp in ((1, 5), (2, 4), (3, 6), (4, 7), (5, 8)):
        w2(out, inp)

    w2(6, 9)
    for c in range(27, 31):
        w2(6, c, v=-1)
    w2(7, 10)
    w2(8, 11)
    w2(8, 12, v=-1)
    w2(9, 13)
    w2(10, 14)
    w2(11, 15)

    for i, (sy, sx) in enumerate(shifts):
        for c in range(4):
            w2(12 + i, c)
        w2(12 + i, 5)
        w2(12 + i, 16 + i, sy, sx)
        p["b2"][12 + i] = -1

        w2(16 + i, 20 + i, sy, sx)
        w2(20 + i, 20 + i, sy, sx)
        p["b2"][20 + i] = -1

    w2(24, 24)
    for c in range(4):
        w2(24, c, v=2)
    p["b2"][24] = -2
    w2(25, 25)
    w2(26, 26)
    w2(27, 31)

    # Multiplex heading at the anchor with a shifted player marker.
    # These occupy disjoint pixels and are separated again in F3.
    for i, (dy, dx) in enumerate(directions):
        w2(28 + i, i)
        for c in range(4):
            w2(28 + i, c, -dy, -dx)

    # F3:
    # 0 player, 1 held box, 2 eligible box, 3 enemy,
    # 4 normal clearance, 5 box clearance, 6 goal seed,
    # 7 fence handoff, 8 highlighted eligible box, 9 enemy highlight,
    # 10:14 heading, 14:18 pair collision, 18:22 player guards,
    # 22 coordinate, 24 goal-component material,
    # 26:30 shifted player markers, 30 held box in projected goal.
    for out, inp in (
        (0, 0), (1, 1), (3, 4), (4, 6), (5, 7),
        (7, 9), (9, 5), (22, 24), (24, 26),
    ):
        w3(out, inp)

    fence_row = 32 + 10
    fence_col = 64 + 11
    for out, inp in ((2, 2), (8, 3)):
        w3(out, inp)
        w3(out, fence_row, -1)
        w3(out, fence_col, -1)

    goal_row = 32 + 25
    goal_col = 64 + 27
    for out, obj in ((6, 8), (30, 1)):
        w3(out, obj)
        w3(out, goal_row)
        w3(out, goal_col)
        p["b3"][out] = -2

    for i in range(4):
        w3(10 + i, 28 + i)
        w3(10 + i, 0)
        p["b3"][10 + i] = -1

        w3(26 + i, 28 + i)
        w3(26 + i, 0, -1)

        w3(14 + i, 12 + i)
        w3(18 + i, 0)
        w3(18 + i, 16 + i)
        w3(18 + i, 20 + i, -1)
        p["b3"][18 + i] = -1

    # Saturated wavefront pairs: clip(s,0,1) = relu(s)-relu(s-1).
    neighbors = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

    def wave(lo, hi, clearance, seeds):
        for out in (lo, hi):
            for dy, dx in neighbors:
                p["Wh"][out, lo, dy + 1, dx + 1] += 1
                p["Wh"][out, hi, dy + 1, dx + 1] -= 1
            p["Uh"][out, clearance] = 16
            for seed in seeds:
                p["Uh"][out, seed] += 1
            p["bh"][out] = -16 - int(out == hi)

    wave(0, 1, 4, (2, 3))
    wave(2, 3, 5, (6, 7))
    wave(12, 13, 24, (6,))

    # Delay target waves until connected goal boxes are identified.
    p["Wh"][14, 14, 1, 1] = 1
    p["bh"][14] = 1
    p["Wh"][15, 14, 1, 1] = 1
    p["bh"][15] = -23

    for out in (0, 1):
        p["bh"][out] -= 24
        p["Wh"][out, 14, 1, 1] += 1
        p["Wh"][out, 15, 1, 1] -= 1
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                p["Wh"][out, 12, dy + 1, dx + 1] -= 16
                p["Wh"][out, 13, dy + 1, dx + 1] += 16

    for i, (dy, dx) in enumerate(directions):
        # Shifted marker plus shifted read gives a two-pixel probe.
        out = 4 + i
        p["Wh"][out, out, 1, 1] = 1
        p["Wh"][out, 0, dy + 1, dx + 1] = 1
        p["Wh"][out, 1, dy + 1, dx + 1] = -1
        p["Uh"][out, 26 + i] = 256
        p["bh"][out] = -256

        out = 8 + i
        p["Wh"][out, out, 1, 1] = 1
        p["Wh"][out, 2, dy + 1, dx + 1] = 1
        p["Wh"][out, 3, dy + 1, dx + 1] = -1
        p["Uh"][out, 1] = 256
        p["bh"][out] = -256

    def F(i):
        return 32 + i

    next_z = 0

    def zunit(coefficients, bias=0):
        nonlocal next_z
        out = next_z
        next_z += 1
        for c, value in coefficients.items():
            p["Wz"][out, c] += value
        p["bz"][out] = bias
        return out

    normal = [
        zunit({4 + i: 1, F(1): -256}) for i in range(4)
    ]
    carrying = [
        zunit({8 + i: 1, F(1): 256}, -256) for i in range(4)
    ]
    carry = zunit({F(1): 1})
    front_box = zunit({F(8): 1})
    front_enemy = zunit({F(9): 1})
    enemy_present = zunit({F(3): 1})
    on_goal = zunit({F(30): 1})
    heading = [zunit({F(10 + i): 1}) for i in range(4)]
    carry_heading = [
        zunit({F(10 + i): 1, F(1): 1}, -1) for i in range(4)
    ]
    blocked = [zunit({F(14 + i): 1}) for i in range(4)]
    player_blocked = [zunit({F(18 + i): 1}) for i in range(4)]
    coordinate = zunit({F(22): 1})

    # Only grab when the facing direction leads to a nearby target.
    grab = [
        zunit({
            4 + i: 1, F(8): 128, F(10 + i): 128, F(1): -256
        }, -356)
        for i in range(4)
    ]
    nearby_target = [
        (
            zunit({4 + i: 1, F(1): -256}, -100),
            zunit({4 + i: 1, F(1): -256}, -101),
        )
        for i in range(4)
    ]

    # Require both proximity and improvement in that direction.
    arrival = []
    for i in range(4):
        terms = {8 + i: 2, 8 + opposite[i]: -1}
        arrival.append((zunit(terms, -125), zunit(terms, -126)))

    # Memory 0:8: predicted delivery and confirmation of last move.
    for i, (lo, hi) in enumerate(arrival):
        p["Wmz"][i, lo] = 20
        p["Wmz"][i, hi] = -20
        p["bm"][i] = -10

        p["Wm"][4 + i, i] = 10
        p["Wma"][4 + i, 1 + i] = 20
        p["Wmz"][4 + i, carry] = 20
        p["bm"][4 + i] = -40

    # Two-observation displacement detector.
    p["Wmz"][8, coordinate] = 0.001
    p["Wm"][9, 8] = 1
    for out, sign in ((10, 1), (11, -1)):
        p["Wmz"][out, coordinate] = sign * 4000
        p["Wm"][out, 9] = -sign * 4000000
        p["bm"][out] = -8

    p["Wm"][12, 12] = 1
    p["Wm"][12, 10:12] = -1.5
    p["Wmz"][12, carry] = 5
    p["bm"][12] = -7.5
    p["Wm"][13, 12] = 10
    p["Wmz"][13, carry] = 20
    p["bm"][13] = -25

    p["Wmz"][14, carry] = 20
    p["bm"][14] = -10

    # Release event.
    p["Wm"][15, 14] = 10
    p["Wma"][15, 5] = 20
    p["Wmz"][15, carry] = -40
    p["bm"][15] = -20

    for i in range(4):
        # Retain the grip orientation after release.
        p["Wm"][16 + i, 16 + i] = 10
        p["Wmz"][16 + i, carry_heading[i]] = 40
        p["Wmz"][16 + i, carry] = -20

        # First release move: perpendicular to the grip.
        p["Wm"][20 + i, 16 + i] = 10
        p["Wm"][20 + i, 14] = 10
        p["Wma"][20 + i, 5] = 20
        p["Wmz"][20 + i, carry] = -40
        p["bm"][20 + i] = -30

        # Second release move: along the old grip.
        p["Wm"][24 + i, 15] = 10
        p["Wm"][24 + i, 16 + i] = 10
        p["Wmz"][24 + i, carry] = -40
        p["bm"][24 + i] = -10

        # Third move is used only for a stuck-release regrasp.
        p["Wm"][28 + i, 37] = 10
        p["Wm"][28 + i, 16 + i] = 10
        p["Wmz"][28 + i, carry] = -40
        p["bm"][28 + i] = -10

    p["Wm"][36, 13] = 10
    p["Wm"][36, 14] = 10
    p["Wma"][36, 5] = 20
    p["Wmz"][36, carry] = -40
    p["bm"][36] = -30
    for out, previous in ((37, 36), (38, 37), (39, 38)):
        p["Wm"][out, previous] = 20
        p["bm"][out] = -10

    # Detect an enemy stealing a just-grabbed box.
    p["Wmz"][40, front_box] = 20
    p["bm"][40] = -10
    p["Wm"][41, 40] = 10
    p["Wma"][41, 5] = 20
    p["Wmz"][41, enemy_present] = 20
    p["Wmz"][41, carry] = -40
    p["bm"][41] = -40

    for i in range(4):
        p["Wm"][42 + i, 40] = 10
        p["Wma"][42 + i, 5] = 20
        p["Wmz"][42 + i, heading[i]] = 20
        p["Wmz"][42 + i, enemy_present] = 20
        p["Wmz"][42 + i, carry] = -40
        p["bm"][42 + i] = -60

        p["Wmz"][46 + i, heading[i]] = 20
        p["bm"][46 + i] = -10

        p["Wm"][50 + i, 41] = 10
        p["Wm"][50 + i, 46 + i] = 10
        p["Wmz"][50 + i, carry] = -40
        p["bm"][50 + i] = -10

    # Escape repeated unheld motion without resetting the level.
    p["Wm"][54, 54] = 1
    p["Wm"][54, 10:12] = -1.5
    p["Wmz"][54, carry] = -5
    p["bm"][54] = -2.5
    for i in range(4):
        p["Wm"][56 + i, 54] = 10
        p["Wmz"][56 + i, heading[i]] = 20
        p["Wmz"][56 + i, carry] = -40
        p["bm"][56 + i] = -25

    # Action scores.
    p["bo"][:] = -10000
    p["bo"][1:5] = -1
    p["bo"][5] = 0

    def memory_bonus(action, unit, amount):
        p["Wo"][action, 128 + unit] += amount / 2
        p["bo"][action] += amount / 2

    for i in range(4):
        action = i + 1
        p["Wo"][action, normal[i]] = 1
        p["Wo"][action, carrying[i]] = 1
        p["Wo"][action, heading[i]] = 0.05
        p["Wo"][action, blocked[i]] = -400
        p["Wo"][action, player_blocked[i]] = -400

        lo, hi = nearby_target[i]
        p["Wo"][action, lo] = 400
        p["Wo"][action, hi] = -400

        p["Wo"][5, grab[i]] = 300
        memory_bonus(5, 4 + i, 300)

    p["Wo"][5, front_enemy] = 1000
    p["Wo"][5, on_goal] = 300
    memory_bonus(5, 13, 300)
    memory_bonus(5, 15, -1000)
    memory_bonus(5, 41, -1000)
    memory_bonus(5, 39, 1500)

    for i in range(4):
        memory_bonus(1 + clockwise[i], 20 + i, 24)
        memory_bonus(1 + i, 24 + i, 24)
        memory_bonus(5, 24 + i, -1000)

        memory_bonus(1 + opposite[clockwise[i]], 28 + i, 700)
        memory_bonus(5, 28 + i, -1000)

        memory_bonus(1 + clockwise[i], 42 + i, 24)
        memory_bonus(1 + i, 50 + i, 24)
        memory_bonus(5, 50 + i, -1000)

        memory_bonus(1 + clockwise[i], 56 + i, 24)

    return p
