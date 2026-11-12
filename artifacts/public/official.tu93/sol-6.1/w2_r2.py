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

    W1, b1 = p["W1"], p["b1"]
    W2 = p["W2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    # Action order: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    reverse = [1, 0, 3, 2]
    square = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)]
    player_colours = (4, 9)
    path_colours = (0, 2, 4, 8, 9, 11, 12, 13, 14, 15)

    def test(unit, offsets, colours, bias=None):
        for r, c in offsets:
            for colour in colours:
                W1[unit, colour, r + 2, c + 2] += 1.0
        b1[unit] = (
            1.0 - len(offsets) if bias is None else bias
        )

    # Centre-line graph, exit centre, player centre.
    test(0, square, path_colours)
    test(1, square, (14,))
    test(2, square, player_colours)

    # Partial detections differenced by W2 to locate the four
    # candidate corridor midpoints, three pixels from the player.
    for a, (dy, dx) in enumerate(directions):
        py, px = -dx, dy
        offsets = [
            (-2 * dy, -2 * dx),
            (-2 * dy + py, -2 * dx + px),
            (-2 * dy - py, -2 * dx - px),
            (-dy, -dx),
        ]
        test(3 + a, offsets, player_colours)

    # Enemy detections two pixels ahead of the sprite centre.
    # Front marker, transverse flanks, centre body, and a negative
    # outside-body test distinguish the facing direction.
    enemy_specs = [
        (7, 8, (15, 11)),
        (11, 12, (15, 11)),
        (15, 13, (11,)),
    ]
    for base, body, markers in enemy_specs:
        for a, (dy, dx) in enumerate(directions):
            unit = base + a
            py, px = -dx, dy
            r, c = -dy, -dx

            for marker in markers:
                W1[unit, marker, r + 2, c + 2] = 1.0
            W1[unit, body, r + py + 2, c + px + 2] = 1.0
            W1[unit, body, r - py + 2, c - px + 2] = 1.0
            W1[unit, body, 2 - 2 * dy, 2 - 2 * dx] = 1.0
            W1[unit, body, 2, 2] = -1.0
            b1[unit] = -3.0

    # Secondary targets: stationary attackers that can be removed.
    test(19, square, (8, 11, 15))

    # Exact square detections carrying their coordinates.
    for unit, colours, coordinate in [
        (20, player_colours, 16),
        (21, player_colours, 17),
        (22, (14,), 16),
        (23, (14,), 17),
    ]:
        test(unit, square, colours, bias=-9.0)
        W1[unit, coordinate, 2, 2] = 1.0

    for unit in (0, 1, 2):
        W2[unit, unit, 1, 1] = 1.0

    for a, (dy, dx) in enumerate(directions):
        W2[3 + a, 3 + a, 1 - dy, 1 - dx] = 1.0
        W2[3 + a, 3 + a, 1, 1] = -1.0

        for base, _, _ in enemy_specs:
            # Enemy signals at forward offsets three and two.
            W2[7 + a, base + a, 1 - dy, 1 - dx] = 1.0
            W2[16 + a, base + a, 1, 1] = 1.0

    W2[11, 19, 1, 1] = 1.0
    for k in range(4):
        W2[12 + k, 20 + k, 1, 1] = 1.0

    # f3:
    # 0 graph; 1 exit; 2 player; 3:7 legal midpoints;
    # 7:11 enemy offset-three signals; 11 stationary targets;
    # 12:16 coordinates; 16:20 enemy offset-two signals.
    for unit in (
        0, 1, 2, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19
    ):
        W3[unit, unit] = 1.0

    for a in range(4):
        W3[3 + a, 3 + a] = 1.0
        W3[3 + a, 0] = 1.0
        b3[3 + a] = -1.0

    # h[0:4]: finite forward ramps with values 3,2,1 at
    # offsets 3,4,5 from each oriented enemy.
    for a, (dy, dx) in enumerate(directions):
        Uh[a, 7 + a] = 4.0
        Wh[a, a, 1 - dy, 1 - dx] = 1.0
        bh[a] = -1.0

    neighbours = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

    # A compensated second difference of each ramp is exactly
    # a unit source at offset six:
    #
    # r(x-d) - 2*r(x) + r(x+d)
    #     + 4*source_at_offset_3 - 3*source_at_offset_2.
    #
    # h[4] diffuses these predicted destinations over the thin
    # graph. Its subtractive threshold limits support to the
    # immediately incident corridor midpoints.
    danger_gain = 128.0
    graph_gate = 65536.0

    for a, (dy, dx) in enumerate(directions):
        Wh[4, a, 1 - dy, 1 - dx] += danger_gain
        Wh[4, a, 1, 1] -= 2.0 * danger_gain
        Wh[4, a, 1 + dy, 1 + dx] += danger_gain
        Uh[4, 7 + a] += 4.0 * danger_gain
        Uh[4, 16 + a] -= 3.0 * danger_gain

    Wh[4, 4, 1, 1] = 0.5
    for dy, dx in neighbours[1:]:
        Wh[4, 4, 1 + dy, 1 + dx] = 0.125

    Uh[4, 0] = graph_gate
    Uh[4, 2] = -graph_gate
    bh[4] = -graph_gate - 1.0

    # h[8]-h[9]: capped exit flood.
    # h[10]-h[11]: capped stationary-target flood.
    # Excluding the player's junction distinguishes real routes
    # from paths that merely return through the current position.
    for lo, hi, source in [(8, 9, 1), (10, 11, 11)]:
        for out in (lo, hi):
            for dy, dx in neighbours:
                Wh[out, lo, 1 + dy, 1 + dx] = 1.0
                Wh[out, hi, 1 + dy, 1 + dx] = -1.0
            Uh[out, 0] = 10.0
            Uh[out, 2] = -10.0
            Uh[out, source] = 1.0
            bh[out] = -10.0
        bh[hi] -= 1.0

    # h[12:16]: candidate arrival accumulators.
    # The safe baseline remains large even for unreachable branches;
    # a threatened destination instead resets its accumulator.
    for a in range(4):
        out = 12 + a
        Wh[out, out, 1, 1] = 1.0
        Wh[out, 8, 1, 1] = 1.0
        Wh[out, 9, 1, 1] = -1.0
        Wh[out, 10, 1, 1] = 0.03
        Wh[out, 11, 1, 1] = -0.03
        Wh[out, 4, 1, 1] = -512.0
        Uh[out, 3 + a] = 512.0
        bh[out] = -511.0

    Wz = p["Wz"]
    for a in range(4):
        Wz[a, 12 + a] = 1.0
        Wz[4 + a, 32 + 3 + a] = 1.0
    for k in range(4):
        Wz[20 + k, 32 + 12 + k] = 1.0

    # Memory 0:4 records the previous directional action.
    # Memory 4:8 gates reversal penalties on candidates with
    # substantial arrival scores. Unreachable escape branches
    # therefore retain their secondary-target preference.
    for a in range(4):
        p["Wma"][a, 1 + a] = 4.0

        unit = 4 + a
        p["Wmz"][unit, a] = 0.25
        p["Wma"][unit, 1 + reverse[a]] = 64.0
        p["bm"][unit] = -104.0

    Wo, bo = p["Wo"], p["bo"]
    bo[:] = -1_000_000.0

    for a in range(4):
        action = 1 + a
        bo[action] = -1040.0
        Wo[action, a] = 1.0
        Wo[action, 4 + a] = 1024.0
        Wo[action, 128 + 4 + a] = -16.0
        Wo[action, 128 + reverse[a]] = -0.002

    # Very small coordinate tie-break toward the exit.
    gain = 0.002
    Wo[1, 20] += gain
    Wo[1, 22] -= gain
    Wo[2, 20] -= gain
    Wo[2, 22] += gain
    Wo[3, 21] += gain
    Wo[3, 23] -= gain
    Wo[4, 21] -= gain
    Wo[4, 23] += gain

    return p
