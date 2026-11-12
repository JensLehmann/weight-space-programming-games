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

    def first(out, channel, dy=0, dx=0, value=1.0):
        p["W1"][out, channel, dy + 2, dx + 2] += value

    def second(out, channel, dy=0, dx=0, value=1.0):
        p["W2"][out, channel, dy + 1, dx + 1] += value

    def feature(out, terms, bias=0.0):
        for channel, value in terms.items():
            p["W3"][out, channel] += value
        p["b3"][out] = bias

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Eroded broad floor: narrow decorative wires do not qualify.
    for colour in range(1, 16):
        p["W1"][0, colour] = 1.0
    p["b1"][0] = -24.0

    # Isolated exit dot.
    first(1, 9)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy or dx:
                first(1, 9, dy, dx, -1.0)

    # Compact switch plates, separately for each functional colour.
    for out, colour in ((2, 8), (3, 11), (17, 15)):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                weight = 2.0 if max(abs(dy), abs(dx)) <= 1 else -0.5
                first(out, colour, dy, dx, weight)
        p["b1"][out] = -16.0

    # Dense gate bodies.
    p["W1"][4, 8] = 1.0
    p["W1"][4, 11] = 1.0
    p["b1"][4] = -14.0

    # Exact yellow player centre.
    first(5, 5)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy or dx:
                first(5, 9, dy, dx)
    p["b1"][5] = -8.0

    # Displaced partial player patterns, completed into three-pixel probes
    # by the second convolution.
    for j, (dy, dx) in enumerate(directions):
        out = 6 + j
        cy, cx = -2 * dy, -2 * dx
        first(out, 5, cy, cx)
        count = 1
        for ry in range(-1, 2):
            for rx in range(-1, 2):
                if not (ry or rx):
                    continue
                yy, xx = cy + ry, cx + rx
                if -2 <= yy <= 2 and -2 <= xx <= 2:
                    first(out, 9, yy, xx)
                    count += 1
        p["b1"][out] = -(count - 1)

    first(10, 16)
    first(11, 17)

    # Teleport border pieces.
    for j, (dy, dx) in enumerate(directions):
        first(12 + j, 15, 2 * dy, 2 * dx)
    first(16, 5)

    # Wire taps two pixels away, subsequently shifted to distance three.
    for base, colour in ((18, 8), (22, 15), (26, 11)):
        for j, (dy, dx) in enumerate(directions):
            first(base + j, colour, 2 * dy, 2 * dx)

    second(0, 0)
    second(1, 1)

    # Clipped plate and gate features: ReLU(v)-ReLU(v-1).
    for hi, lo, inp, gain in (
        (2, 3, 2, 2.0),
        (4, 5, 3, 2.0),
        (6, 7, 4, 1.0),
        (22, 23, 17, 2.0),
    ):
        second(hi, inp, value=gain)
        second(lo, inp, value=gain)
        p["b2"][lo] = -1.0

    second(8, 5)
    for j, (dy, dx) in enumerate(directions):
        second(9 + j, 6 + j, -dy, -dx)

    second(13, 10)
    second(14, 11)

    for j, (dy, dx) in enumerate(directions):
        second(15, 12 + j, dy, dx)
    second(15, 16)
    p["b2"][15] = -4.0

    # Sharp row and column partitions.
    for hi, lo, inp in ((17, 18, 10), (24, 25, 11)):
        second(hi, inp, value=128.0)
        second(lo, inp, value=128.0)
        p["b2"][hi] = -64.0
        p["b2"][lo] = -65.0

    # A plate covered by the player is recognized from its exposed wire.
    for out, tap_base in ((19, 18), (20, 22), (21, 26)):
        second(out, 5, value=2.0)
        for j, (dy, dx) in enumerate(directions):
            second(out, tap_base + j, dy, dx)
        p["b2"][out] = -2.0

    # F3:
    # 0 floor, 1 exit, 2..4 routing-bank seeds, 5..8 probes,
    # 9..10 player coordinates, 11 pad, 14..18 covered switches.
    feature(0, {0: 1, 6: -1, 7: 1})
    feature(1, {1: 1})

    # Bank 0: left ordinary plates, lower toggles, teleport pads.
    feature(20, {2: 1, 3: -1, 24: -1, 25: 1})
    feature(21, {4: 1, 5: -1, 17: 1, 18: -1}, -1)
    feature(11, {15: 1})

    # Bank 1: right ordinary plates.
    feature(3, {2: 1, 3: -1, 24: 1, 25: -1}, -1)

    # Bank 2: white ordinary plates and upper toggles.
    feature(22, {22: 1, 23: -1})
    feature(23, {4: 1, 5: -1, 17: -1, 18: 1})

    for j in range(4):
        feature(5 + j, {9 + j: 1})
    feature(9, {14: 1, 8: 1}, -1)
    feature(10, {13: 1, 8: 1}, -1)

    # Covered ordinary plates split by column.
    feature(14, {19: 1, 24: 1, 25: -1}, -1)
    feature(15, {19: 1, 24: -1, 25: 1})

    # A teleport border has four taps, unlike an ordinary white plate.
    feature(16, {20: 1, 15: -8})
    feature(17, {21: 1, 17: -1, 18: 1})
    feature(18, {21: 1, 17: 1, 18: -1}, -1)

    # Each field has clipped reachability q-r and two probe accumulators.
    seed_channels = [(20, 21, 11), (3,), (22, 23), (1,)]
    neighbours = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

    for field, seeds in enumerate(seed_channels):
        q, r, horizontal, vertical = range(4 * field, 4 * field + 4)

        for out in (q, r):
            for dy, dx in neighbours:
                p["Wh"][out, q, 1 + dy, 1 + dx] += 1.0
                p["Wh"][out, r, 1 + dy, 1 + dx] -= 1.0
            p["Uh"][out, 0] = 10.0
            for seed in seeds:
                p["Uh"][out, seed] += 1.0
            p["bh"][out] = -10.0 - float(out == r)

        for out, positive, negative in (
            (horizontal, 5, 6),
            (vertical, 7, 8),
        ):
            p["Wh"][out, out, 1, 1] = 1.0
            p["Wh"][out, q, 1, 1] = 1.0
            p["Wh"][out, r, 1, 1] = -1.0
            p["Uh"][out, positive] = 1002.0
            p["Uh"][out, negative] = 1000.0
            p["bh"][out] = -1000.0

    # Decode directional potentials from spatial maxima and means.
    potentials = []
    for field in range(4):
        horizontal, vertical = 4 * field + 2, 4 * field + 3
        expressions = []
        for channel, positive in (
            (horizontal, True), (horizontal, False),
            (vertical, True), (vertical, False),
        ):
            weights = np.zeros(96, np.float32)
            if positive:
                weights[channel] = 1.0
                bias = -256.0
            else:
                weights[16 + channel] = 4096.0
                weights[channel] = -1.0
                bias = 0.0
            expressions.append((weights, bias))

        potentials.append(expressions)
        for direction, (weights, bias) in enumerate(expressions):
            unit = 4 * field + direction
            p["Wz"][unit] = weights
            p["bz"][unit] = bias

    # Global target-presence and covered-switch readouts.
    # Multiple seed types in a bank are summed; the memory thresholds them.
    for zunit, channels in (
        (16, (20, 21, 11)),
        (17, (3,)),
        (18, (22, 23)),
        (19, (15,)),
        (20, (14,)),
        (21, (16,)),
        (22, (18,)),
        (23, (17,)),
    ):
        for channel in channels:
            p["Wz"][zunit, 32 + channel] += 1.0

    # Current exit reachability, clipped through a ReLU pair.
    goal_weights = sum(
        (w for w, _ in potentials[3]), np.zeros(96, np.float32)
    )
    goal_bias = sum(b for _, b in potentials[3])
    for offset in (0, 1):
        p["Wz"][24 + offset] = goal_weights
        p["bz"][24 + offset] = goal_bias - offset

    p["Wz"][26, 32 + 9] = 1.0
    p["Wz"][27, 32 + 10] = 1.0

    Wm, Wmz, Wma, bm = (
        p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    )

    VIS = [16, 17, 18]
    CAND = [19, 20, 21]
    CURRENT_ORD = [22, 23, 24]
    RECORD = [25, 26, 27]
    READY = 28

    bm[READY] = 8.0

    def clipped_goal(row, coefficient):
        Wmz[row, 24] += coefficient
        Wmz[row, 25] -= coefficient

    # Covered ordinary-switch state is retained for the recording latch.
    for bank, row in enumerate(CURRENT_ORD):
        Wmz[row, 19 + bank] = 8.0
        bm[row] = -4.0

    # Visit latches: ordinary switches count only after ACTION5.
    # Toggle switches count when physically reached.
    for bank, row in enumerate(VIS):
        Wm[row, row] = 8.0
        Wm[row, CURRENT_ORD[bank]] = 4.0
        Wma[row, 5] = 8.0
        Wm[row, READY] = 8.0
        bm[row] = -8.0
        if bank == 0:
            Wmz[row, 22] = 16.0
        elif bank == 2:
            Wmz[row, 23] = 16.0

    # Candidate banks require a target and an unrecorded/unvisited bank.
    for bank, row in enumerate(CAND):
        Wmz[row, 16 + bank] = 8.0
        Wm[row, VIS[bank]] = -4.0
        bm[row] = -8.0

    # Priority: right ordinary, white/upper toggle, left/lower/pad.
    # Exit reachability overrides all banks immediately.
    for field in range(4):
        for direction in range(4):
            row = 4 * field + direction
            Wmz[row, row] = 0.005
            if field == 3:
                clipped_goal(row, 8.0)
                bm[row] = -8.0
            elif field == 1:
                Wm[row, CAND[1]] = 4.0
                clipped_goal(row, -8.0)
                bm[row] = -4.0
            elif field == 2:
                Wm[row, CAND[2]] = 4.0
                Wm[row, CAND[1]] = -4.0
                clipped_goal(row, -8.0)
                bm[row] = -8.0
            else:
                Wm[row, CAND[0]] = 4.0
                Wm[row, CAND[1]] = -4.0
                Wm[row, CAND[2]] = -4.0
                clipped_goal(row, -8.0)
                bm[row] = -12.0

    # Immediate recording decisions at an ordinary switch.
    for bank, row in enumerate(RECORD):
        Wmz[row, 19 + bank] = 8.0
        Wm[row, VIS[bank]] = -4.0
        bm[row] = -8.0

    # Optional coordinate memory, useful for retaining spatial state.
    Wmz[29, 26] = 0.005
    Wmz[30, 27] = 0.005

    p["bo"][:] = -10000.0
    action_indices = [4, 3, 2, 1]  # right, left, down, up
    for direction, action in enumerate(action_indices):
        p["bo"][action] = 0.0
        for field in range(4):
            p["Wo"][action, 128 + 4 * field + direction] = 10.0

    for row in RECORD:
        p["Wo"][5, 128 + row] = 100.0
    p["bo"][5] = 150.0

    return p
