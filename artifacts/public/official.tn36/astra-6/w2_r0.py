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
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    K = 128.0
    G = 256.0
    BASE = 32.0

    # ------------------------------------------------------------
    # Layer 1: local player, target, tape and button predicates.
    # ------------------------------------------------------------
    object_colours = (8, 9, 11, 15)

    player = np.ones((4, 4), dtype=np.int8)
    player[3, 1:3] = 0

    target = np.zeros((6, 6), dtype=np.int8)
    target[1:6, 0] = 1
    target[1:6, 5] = 1
    target[5, :] = 1
    target[4, 2:4] = 1

    def shape_predicate(out, shape, anchor):
        ax, ay = anchor
        positives = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                x, y = ax + dx, ay + dy
                occupied = (
                    0 <= y < shape.shape[0]
                    and 0 <= x < shape.shape[1]
                    and shape[y, x] != 0
                )
                weight = 1.0 if occupied else -1.0
                positives += int(occupied)
                for colour in object_colours:
                    W1[out, colour, dy + 2, dx + 2] = weight
        b1[out] = 1.0 - positives

    actors = []
    targets = []

    for scale in (1, 2):
        a = np.repeat(
            np.repeat(player, scale, axis=0), scale, axis=1
        )
        t = np.repeat(
            np.repeat(target, scale, axis=0), scale, axis=1
        )
        a_anchor = (1, 1) if scale == 1 else (3, 6)
        t_anchor = (2, 2) if scale == 1 else (5, 9)

        for q in range(4):
            ai = (scale - 1) * 4 + q
            ti = 8 + (scale - 1) * 4 + q

            shape_predicate(ai, a, a_anchor)
            shape_predicate(ti, t, t_anchor)

            actors.append((
                ai, -a_anchor[0], -a_anchor[1], scale, q
            ))
            targets.append((
                ti,
                scale - t_anchor[0],
                scale - t_anchor[1],
                scale,
                q,
            ))

            aw, tw = a.shape[1], t.shape[1]
            a = np.rot90(a, -1)
            t = np.rot90(t, -1)
            a_anchor = (aw - 1 - a_anchor[1], a_anchor[0])
            t_anchor = (tw - 1 - t_anchor[1], t_anchor[0])

    # 16 H-blue, 17 H-black, 18 V-blue, 19 V-black.
    for out, vertical, colour in (
        (16, False, 1),
        (17, False, 5),
        (18, True, 1),
        (19, True, 5),
    ):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                line = (dx == 0) if vertical else (dy == 0)
                expected = colour if line else 0
                W1[out, expected, dy + 2, dx + 2] = 1.0
        b1[out] = -8.0

    W1[20, 16, 2, 2] = 1.0
    W1[21, 17, 2, 2] = 1.0

    bit_rows = (33, 36, 39, 42, 45, 48)
    for i, row in enumerate(bit_rows):
        W1[22 + i, 16, 2, 2] = 63.0
        b1[22 + i] = -float(row)

    # Row-zero impulse for prefix-sum baselines.
    W1[28, 16, 2, 2] = -63.0
    b1[28] = 1.0

    # Lower-left rounded corner of the Run button.
    button = np.ones((9, 9), dtype=np.int8)
    button[0, :2] = button[0, 7:] = 0
    button[1, 0] = button[1, 8] = 0
    button[7, 0] = button[7, 8] = 0
    button[8, :2] = button[8, 7:] = 0

    npositive = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            xx, yy = 2 + dx, 7 + dy
            on = (
                0 <= xx < 9
                and 0 <= yy < 9
                and button[yy, xx] != 0
            )
            W1[29, 9, dy + 2, dx + 2] = (
                1.0 if on else -1.0
            )
            npositive += int(on)
    b1[29] = 1.0 - npositive

    # Obstacle sample and partial corridor-mouth predicate.
    W1[30, 6, 0, 4] = 1.0
    W1[31, 6, 0, 0] = 1.0
    W1[31, 11, 1, 1] = 1.0
    W1[31, 11, 3, 3] = 1.0
    b1[31] = -2.0

    # ------------------------------------------------------------
    # Layer 2: coordinate-valued geometry and tape features.
    # ------------------------------------------------------------
    def geometry_features(entries, first):
        for axis in (0, 1):
            out = first + axis
            coordinate = 21 if axis == 0 else 20
            W2[out, coordinate, 1, 1] = 63.0 / 4.0
            b2[out] = -K
            for feature, ox, oy, scale, q in entries:
                offset = ox if axis == 0 else oy
                W2[out, feature, 1, 1] += K + offset / 4.0

        for feature, ox, oy, scale, q in entries:
            W2[first + 2, feature, 1, 1] += float(scale)
            W2[first + 3, feature, 1, 1] += float(q + 1)

    geometry_features(actors, 0)
    geometry_features(targets, 4)

    for feature, ox, oy, scale, q in targets:
        W2[8, feature, 1, 1] = 1.0

    selector_specs = (
        (0, 33, False),
        (1, 36, True),
        (2, 39, False),
        (3, 42, True),
        (4, 45, False),
        (5, 48, True),
        (0, 42, False),
        (1, 45, True),
    )
    selector_channels = {i: [] for i in range(6)}

    for out, (bit, row, vertical) in enumerate(
        selector_specs, start=9
    ):
        selector_channels[bit].append(out)
        for feature in ((18, 19) if vertical else (16, 17)):
            W2[out, feature, 1, 1] = 64.0

        W2[out, 20, 1, 1] = 63.0
        W2[out, 22 + bit_rows.index(row), 1, 1] = -2.0
        b2[out] = 1.0 - 64.0 - row

    def tape_bounds(out, vertical, dx=0):
        upper_hinge = 27 if vertical else 26
        W2[out, upper_hinge, 1, 1 + dx] -= 64.0

    for out, feature, vertical in (
        (17, 16, False),
        (18, 17, False),
        (27, 18, True),
        (28, 19, True),
    ):
        W2[out, feature, 1, 1] = 1.0
        tape_bounds(out, vertical)

    # Row maximum gives the first tape column.
    for feature in range(16, 20):
        W2[19, feature, 1, 1] = K
    W2[19, 21, 1, 1] = -63.0 / 5.0
    W2[19, 27, 1, 1] = -64.0
    b2[19] = 64.0 / 5.0 - K

    # Centre, left and right arithmetic lanes.
    for out, input_dx in ((20, 0), (21, 1), (22, -1)):
        for feature in (16, 17):
            W2[out, feature, 1, 1 + input_dx] = 1.0
        tape_bounds(out, False, input_dx)

    W2[24, 21, 1, 1] = 63.0 / 5.0
    W2[25, 28, 1, 1] = 1.0
    W2[26, 29, 1, 1] = 1.0

    actor_offsets = (
        (0, 0, 0),
        (1, 1, 0),
        (2, 1, 1),
        (3, 0, 1),
    )

    # Visible corridor checkpoint: x, y and presence.
    for out in (23, 29, 30):
        W2[out, 31, 1, 1] = K
        W2[out, 30, 1, 2] = K
        for feature, dx, dy in actor_offsets:
            W2[out, feature, 1 + dy, 1 + dx] -= K
        b2[out] = -2.0 * K

    W2[23, 21, 1, 1] = 63.0 / 4.0
    W2[29, 20, 1, 1] = 63.0 / 4.0
    b2[23] -= 0.25
    b2[29] -= 0.25
    b2[30] += 1.0

    # Player occupying a corridor mouth.
    W2[31, 31, 1, 1] = 1.0
    W2[31, 30, 1, 2] = 1.0
    for feature, dx, dy in actor_offsets:
        W2[31, feature, 1 + dy, 1 + dx] += 1.0
    b2[31] = -2.0

    # ------------------------------------------------------------
    # Layer 3: axial routing to shared spatial arithmetic lanes.
    # ------------------------------------------------------------
    source_specs = (
        (0, +1, 20, {4: 1.0, 31: 1.0}),
        (0, -1, 20, {0: 1.0}),
        (0, +1, 21, {2: 1.0}),
        (0, -1, 21, {6: 1.0}),
        (0, +1, 22, {5: 1.0}),
        (0, -1, 22, {1: 1.0}),
        (1, +1, 20, {5: 1.0}),
        (1, -1, 20, {1: 1.0}),
        (1, +1, 21, {3: 1.0}),
        (1, -1, 21, {7: 1.0, 8: 0.25, 31: 1.0}),
        (1, +1, 22, {7: 1.0, 8: 0.75}),
        (1, -1, 22, {3: 1.0, 31: 0.75}),
    )

    for out, (accumulator, sign, lane, features) in enumerate(
        source_specs
    ):
        for feature, weight in features.items():
            W3[out, 32 + feature] = weight
        W3[out, 64 + lane] = K
        b3[out] = -K
        Uh[accumulator, out] = float(sign)

    W3[12, 25] = 1.0
    Uh[0, 12] = BASE
    Uh[1, 12] = BASE

    # One-based instruction index.
    W3[13, 24] = 1.0
    W3[13, 32 + 19] = 1.0
    b3[13] = 1.0 - 64.0 / 5.0

    W3[14, 17] = 1.0
    W3[14, 27] = 1.0
    W3[15, 18] = 1.0
    W3[15, 28] = 1.0
    W3[16, 26] = 1.0

    for feature in range(9, 17):
        W3[17, 32 + feature] = 1.0
    for lane in (20, 21, 22):
        W3[17, 64 + lane] = 1.0
    b3[17] = -1.0

    def allow(out, lane, bits):
        for bit in bits:
            for feature in selector_channels[bit]:
                W3[out, 32 + feature] += 1.0
        W3[out, 64 + lane] = 1.0
        b3[out] = -1.0

    allow(18, 20, (0, 2, 5))
    allow(19, 21, (0, 3))
    allow(20, 22, (2,))
    allow(21, 20, (1, 2, 5))
    allow(22, 21, (3,))
    allow(23, 20, (0, 1, 2))
    allow(24, 20, (0, 2, 5))
    allow(25, 20, (0, 2))
    allow(26, 20, (0, 1, 2, 5))

    # Checkpoint x-y at the centre lane.
    W3[27, 32 + 23] = 1.0
    W3[27, 32 + 29] = -1.0
    W3[27, 64 + 20] = K
    b3[27] = -K

    # Checkpoint y at the centre and right lanes.
    W3[28, 32 + 29] = 1.0
    W3[28, 64 + 20] = K
    W3[28, 64 + 22] = K
    b3[28] = -K

    W3[29, 32 + 30] = 1.0
    W3[29, 64 + 20] = 1.0
    W3[29, 64 + 22] = 1.0
    b3[29] = -1.0

    W3[30, 32 + 30] = 1.0
    W3[30, 64 + 20] = 1.0
    b3[30] = -1.0

    # Bit-two threshold adjustment for delayed lateral alignment.
    for feature in selector_channels[2]:
        W3[31, 32 + feature] = 1.0
    W3[31, 64 + 21] = 1.0
    W3[31, 64 + 22] = 1.0
    b3[31] = -1.0

    # ------------------------------------------------------------
    # Recurrent block: prefix scans and program interval arithmetic.
    # ------------------------------------------------------------
    def edge(out, inp, weight=1.0, dy=0, dx=0):
        Wh[out, inp, dy + 1, dx + 1] += weight

    def centre_gate(out, allowed):
        Uh[out, 14] -= G
        Uh[out, 15] -= G
        Uh[out, allowed] += G

    edge(0, 0, dy=-1)
    edge(1, 1, dy=-1)

    # Spatially acyclic checkpoint substitutions.
    edge(7, 0, dy=-1)
    Uh[7, 29] = K
    bh[7] = -K

    edge(15, 1, dy=-1)
    Uh[15, 30] = K
    bh[15] = -K

    edge(0, 7, -1.0)
    Uh[0, 27] = 1.0
    Uh[0, 28] = 1.0
    Uh[0, 29] = BASE

    edge(1, 15, -1.0)
    Uh[1, 28] = 1.0
    Uh[1, 30] = BASE

    edge(2, 0)
    bh[2] = -BASE
    Uh[2, 17] = -G
    for feature in (18, 19, 20):
        Uh[2, feature] += G

    edge(3, 0, -1.0)
    bh[3] = BASE
    Uh[3, 17] = -G
    for feature in (21, 22, 20):
        Uh[3, feature] += G

    edge(4, 1)
    bh[4] = -BASE
    centre_gate(4, 23)
    Uh[4, 31] = 0.125

    edge(5, 1, -1.0)
    bh[5] = BASE
    centre_gate(5, 24)
    Uh[5, 31] = 0.125

    # Rotation modulo four, or a delayed lateral command.
    edge(6, 1, -1.0, dx=-1)
    edge(6, 4, 8.0, dx=-1)
    edge(6, 5, -8.0, dx=1)
    bh[6] = BASE - 0.25
    centre_gate(6, 25)

    # Inclusion endpoints.
    for out, bias in ((8, 1.0), (9, 0.0)):
        edge(out, 2, dy=-1)
        edge(out, 3, dy=-1)
        edge(out, 2, dy=-1, dx=-1)
        edge(out, 3, dy=-1, dx=-1)
        edge(out, 4)
        edge(out, 5)
        edge(out, 6)
        Uh[out, 13] = -1.0
        bh[out] = bias
        centre_gate(out, 26)

    # Exclusion endpoints.
    for out, bias in ((10, 1.0), (11, 0.0)):
        edge(out, 2, dy=-1, dx=-1)
        edge(out, 3, dy=-1, dx=-1)
        edge(out, 2)
        edge(out, 3)
        edge(out, 2, dx=1)
        edge(out, 3, dx=1)
        Uh[out, 13] = -1.0
        bh[out] = bias
        centre_gate(out, 26)

    # Initial scaling interval.
    for out, bias in ((12, 1.0), (13, 0.0)):
        edge(out, 2, dx=-1)
        edge(out, 3, dx=-1)
        Uh[out, 13] = -1.0
        bh[out] = bias

    desired = {
        8: 1.0,
        9: -1.0,
        10: -1.0,
        11: 1.0,
        12: 1.0,
        13: -1.0,
    }

    # XOR(D,B) = D+B-2*relu(D+B-1).
    for inp, weight in desired.items():
        edge(14, inp, weight)
        p["Wc"][0, 32 + inp] = weight
    Uh[14, 15] = 1.0
    bh[14] = -1.0

    p["Wc"][0, 32 + 14] = -2.0

    # At valid tape bits:
    #   matching bit -> 0, incorrect bit -> 1.
    p["Wc"][0, 14] = 5.0
    p["Wc"][0, 15] = 6.0
    p["bc"][0] = -5.0

    # Off-tape arithmetic contributes one at the button.
    # Compensate so Run scores 0.25, below an incorrect bit.
    p["Wc"][0, 16] = 4.25

    p["bo"][6] = 4.0
    return p
