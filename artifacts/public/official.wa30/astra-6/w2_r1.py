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
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]

    def c1(out, colors, dy=0, dx=0, weight=1.0):
        if isinstance(colors, int):
            colors = (colors,)
        for color in colors:
            p["W1"][out, color, dy + 2, dx + 2] += weight

    def c2(out, channel, dy=0, dx=0, weight=1.0):
        p["W2"][out, channel, dy + 1, dx + 1] += weight

    def neighborhood(out, channel, weight=1.0):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                c2(out, channel, dy, dx, weight)

    def solved_near(out, weight):
        for channel in range(19, 23):
            neighborhood(out, channel, weight)

    # Player and attached-crate exterior sampling masks.
    for i, (dy, dx) in enumerate(directions):
        c1(i, (0, 14), -dy, -dx)
        c1(i, 14, -2 * dy, -2 * dx)
        c1(i, 14, weight=-1)
        p["b1"][i] = -1

        c1(4 + i, 0, -dy, -dx)
        c1(4 + i, 9, -2 * dy, -2 * dx)
        p["b1"][4 + i] = -1

    # Crate cores, including crates held by either kind of robot.
    for j, (sy, sx) in enumerate(
        [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    ):
        o = 8 + j
        for yy, xx in [(0, 0), (sy, 0), (0, sx), (sy, sx)]:
            c1(o, 9, yy, xx)
        c1(o, (3, 4, 5), -sy, 0)
        c1(o, (3, 4, 5), 0, -sx)
        p["b1"][o] = -5

    c1(12, (3, 4))

    # A solid 3x3 interior distinguishes walls from gray crate rims.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c1(13, 5, dy, dx)
    p["b1"][13] = -8

    c1(14, (11, 15))
    c1(15, 12)

    # Goal border and fence predicates.
    c1(16, 9, weight=5)
    for dy, dx in directions:
        c1(16, 2, dy, dx)
    p["b1"][16] = -5

    c1(17, 2)
    c1(17, 1, 0, 1)
    c1(17, 1, 1, 0)
    c1(17, 2, 1, 1)
    p["b1"][17] = -3

    # Bottom-row UI discriminator.
    c1(18, 16, weight=63)
    p["b1"][18] = -62

    # Local evidence of a crate occupying an outlined destination.
    for i, (dy, dx) in enumerate(directions):
        o = 19 + i
        py, px = -dx, dy
        c1(o, 9)
        c1(o, (0, 3, 4, 5), dy, dx)
        c1(o, 2, 2 * dy, 2 * dx)
        c1(o, 9, 2 * dy + py, 2 * dx + px)
        c1(o, 9, 2 * dy - py, 2 * dx - px)
        p["b1"][o] = -3

    c1(23, 14)
    c1(24, 2)

    # An interactable object immediately in front of the player.
    for i, (dy, dx) in enumerate(directions):
        o = 26 + i
        c1(o, 0, -dy, -dx)
        c1(o, 14, -2 * dy, -2 * dx)
        c1(o, (0, 3, 4, 5, 11, 15))
        c1(o, (9, 15), dy, dx)
        p["b1"][o] = -3

    # Second-layer geometry and collision predicates.
    for i, (dy, dx) in enumerate(directions):
        c2(i, i, -dy, -dx)
        c2(4 + i, 4 + i, -dy, -dx)

        o = 8 + i
        for ch in range(8, 12):
            c2(o, ch, dy, dx)
        c2(o, 12, dy, dx)
        c2(o, 18, dy, dx, -1)
        c2(o, 14, dy, dx)
        c2(o, 15, dy, dx)
        neighborhood(o, 13)

        o = 18 + i
        neighborhood(o, 13)
        c2(o, 15, dy, dx)
        neighborhood(o, 17, 2)
        solved_near(o, 10)

    c2(12, 14)

    for ch in range(8, 12):
        c2(13, ch)
    solved_near(13, -10)

    neighborhood(14, 13)
    c2(14, 15)
    solved_near(14, 10)

    neighborhood(15, 13)
    c2(15, 15)
    c2(15, 14)
    c2(15, 12)
    c2(15, 18, weight=-1)
    for ch in range(8, 12):
        c2(15, ch)

    c2(16, 16)
    neighborhood(17, 17, 0.5)

    for ch in range(26, 30):
        c2(22, ch)
    solved_near(22, -10)

    for ch in range(4, 8):
        c2(23, ch)

    c2(24, 17)
    c2(25, 14)
    c2(26, 24)
    for i in range(4):
        c2(27 + i, 26 + i)
        solved_near(27 + i, -10)

    # Axial goal evidence and navigation masks.
    A, b = p["W3"], p["b3"]
    row_goal, col_goal = 32 + 16, 64 + 16

    A[0, 12] = 1

    A[1, 13] = 1
    A[1, row_goal] = -1
    A[1, col_goal] = -1
    b[1] = 1

    A[23, row_goal] = -1
    A[23, col_goal] = -1
    b[23] = 1

    A[2, 14] = 1
    A[2, 17] = 10
    A[3, 16] = 1
    A[4, 15] = 1

    for i in range(4):
        A[5 + i, i] = 1
        A[5 + i, 18 + i] = -10

        A[9 + i, 4 + i] = 1
        A[9 + i, 8 + i] = -10

        A[13 + i, i] = 1
        A[13 + i, 8 + i] = -10
        A[13 + i, 17] = -20

    A[17, 22] = 1
    A[18, 23] = 1
    A[19, 25] = 1

    A[21, 26] = 1
    A[21, row_goal] = 1
    A[21, col_goal] = 1
    b[21] = -2

    A[22, 26] = 1
    A[22, 17] = 1
    b[22] = -1

    for i in range(4):
        A[25 + i, 27 + i] = 1

    # Paired ReLUs implement capped binary reachability waves.
    H, U, hb = p["Wh"], p["Uh"], p["bh"]
    stencil = [(0, 0)] + directions

    def wave_pair(lo, hi, sources, barrier):
        for out in (lo, hi):
            for dy, dx in stencil:
                H[out, lo, dy + 1, dx + 1] += 1
                H[out, hi, dy + 1, dx + 1] -= 1
            for channel, weight in sources:
                U[out, channel] += weight
            U[out, barrier] -= 1024
        hb[hi] = -1

    wave_pair(0, 1, [(0, 1), (1, 1), (23, -1)], 2)
    wave_pair(14, 15, [(0, 1)], 2)

    H[2, 2, 1, 1] = 1
    H[2, 0, 1, 1] = 1
    H[2, 1, 1, 1] = -1
    H[2, 14, 1, 1] = 256
    H[2, 15, 1, 1] = -256

    wave_pair(7, 8, [(3, 1), (21, 1), (22, 2)], 4)
    H[9, 9, 1, 1] = 1
    H[9, 7, 1, 1] = 1
    H[9, 8, 1, 1] = -1

    for i, (dy, dx) in enumerate(directions):
        H[3 + i, 2, dy + 1, dx + 1] = 1
        U[3 + i, 5 + i] = 65536
        hb[3 + i] = -65536

        H[10 + i, 9, dy + 1, dx + 1] = 1
        U[10 + i, 9 + i] = 128
        hb[10 + i] = -128

    # Pooled control variables.
    Z, zb = p["Wz"], p["bz"]

    def fm(channel):
        return 32 + channel

    holding, enemy, facing = fm(18), fm(19), fm(17)

    for i in range(4):
        Z[i, 3 + i] = 1
        Z[i, holding] = -40000

        Z[4 + i, 10 + i] = 1
        Z[4 + i, holding] = 128
        Z[4 + i, fm(13 + i)] = 128
        zb[4 + i] = -256

    Z[8, holding] = 1
    Z[9, enemy] = 1
    Z[10, facing] = 1
    for i in range(4):
        Z[11 + i, fm(25 + i)] = 1

    Z[15, holding] = 1
    Z[15, enemy] = 1
    zb[15] = -1

    Z[16, holding] = 1
    for i in range(4):
        Z[16, 10 + i] = -1

    for i in range(4):
        Z[17 + i, 10 + i] = 2
        Z[17 + i, holding] = 128
        Z[17 + i, fm(13 + i)] = 128
        zb[17 + i] = -507

        Z[21 + i, fm(5 + i)] = 1

    # Finite-state memory, with margins large enough to saturate
    # float32 tanh to exact +/-1 for established Boolean states.
    M, MZ, MA, mb = p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    sharp = 40.0

    # Predicted placement followed by confirmation of the actual move.
    for i in range(4):
        MZ[i, 17 + i] = sharp
        mb[i] = -0.5 * sharp

        M[4 + i, i] = sharp
        MA[4 + i, 1 + i] = sharp
        MZ[4 + i, 8] = sharp
        mb[4 + i] = -2.5 * sharp

    MZ[8, 8] = sharp
    mb[8] = -0.5 * sharp

    # Persistent facing direction; no leaky residual action bias.
    for i in range(4):
        M[11 + i, 11 + i] = sharp
        for j in range(4):
            MZ[11 + i, 11 + j] = -2 * sharp
        MZ[11 + i, 11 + i] += 4 * sharp

    # One-step back-off following release.
    for i in range(4):
        M[15 + i, 8] = sharp
        M[15 + i, 11 + i] = sharp
        MZ[15 + i, 8] = -2 * sharp
        mb[15 + i] = -1.5 * sharp

    # Release event and one additional refractory step.
    M[19, 8] = sharp
    MZ[19, 8] = -2 * sharp
    mb[19] = -0.5 * sharp
    M[20, 19] = sharp

    # Bounded consecutive-carry watchdog.
    MZ[24, 8] = sharp
    mb[24] = -0.5 * sharp
    for j in range(25, 52):
        M[j, j - 1] = sharp
        MZ[j, 8] = sharp
        mb[j] = -1.5 * sharp

    # Discourage immediate carrying reversals to escape local
    # dead ends caused by the rigid player/crate configuration.
    for i in range(4):
        MA[53 + i, 1 + i] = sharp
        MZ[53 + i, 8] = sharp
        mb[53 + i] = -1.5 * sharp

    O, ob = p["Wo"], p["bo"]
    ob[:] = -1_000_000

    for i in range(4):
        action = 1 + i
        ob[action] = -400000 + 0.125 * (4 - i)
        O[action, i] = 1
        O[action, 4 + i] = 256
        O[action, 21 + i] = 400000

    # Interaction is immediate when facing a target, but not while
    # already carrying or during the finite post-release interval.
    ob[5] = -100
    O[5, 10] = 100000
    O[5, 8] = -100000
    O[5, 15] = 200000
    O[5, 16] = 200000

    def memory_term(action, index, magnitude):
        O[action, 128 + index] += magnitude * 0.5
        ob[action] += magnitude * 0.5

    for index in range(4, 8):
        memory_term(5, index, 200000)
    memory_term(5, 51, 200000)
    memory_term(5, 19, -100000)
    memory_term(5, 20, -100000)

    for i in range(4):
        action = 1 + opposite[i]
        memory_term(action, 15 + i, 100000)
        memory_term(action, 53 + i, -4096)

    return p
