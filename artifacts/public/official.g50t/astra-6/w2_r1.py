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
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def w1(out, channel, dy=0, dx=0, value=1.0):
        p["W1"][out, channel, dy + 2, dx + 2] += value

    def w2(out, channel, dy=0, dx=0, value=1.0):
        p["W2"][out, channel, dy + 1, dx + 1] += value

    # First-layer feature channels.
    PLAYER, SHIFT = 0, 1
    FLOOR5, GATE, PRESSURE, TOGGLE = 5, 6, 7, 8
    GOAL, PORTAL_RING = 9, 10
    UNUSED, ROW, COL, GATE_EDGE = 12, 13, 14, 15
    GHOST, GHOST_UI, WIRE_RAY = 19, 20, 21
    TOGGLE_RAY, PATROL = 25, 29

    def square_detector(out, colour, cy=0, cx=0):
        w1(out, 5, cy, cx)
        for dy, dx in directions:
            w1(out, colour, cy + dy, cx + dx)
        p["b1"][out] = -4.0

    square_detector(PLAYER, 9)
    square_detector(GHOST, 2)
    square_detector(PATROL, 14)
    for i, (dy, dx) in enumerate(directions):
        square_detector(SHIFT + i, 9, -dy, -dx)

    # Erosion of non-background pixels rejects narrow decorative wires.
    p["W1"][FLOOR5, 1:16, :, :] = 1.0
    p["b1"][FLOOR5] = -24.0

    obstruction_colours = (8, 11, 14)
    for colour in obstruction_colours:
        p["W1"][GATE, colour, :, :] = 1.0
    p["b1"][GATE] = -12.0

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            w1(PRESSURE, 8, dy, dx)
            w1(PRESSURE, 15, dy, dx)
            w1(TOGGLE, 11, dy, dx)
    p["b1"][PRESSURE] = -8.0
    p["b1"][TOGGLE] = -8.0

    w1(GOAL, 9)
    for dy, dx in directions:
        w1(GOAL, 5, dy, dx)
    p["b1"][GOAL] = -4.0

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            outer = abs(dy) == 2 or abs(dx) == 2
            w1(PORTAL_RING, 15, dy, dx, 1.0 if outer else -4.0)
    p["b1"][PORTAL_RING] = -4.0

    w1(ROW, 16)
    w1(COL, 17)

    # Timeline capacity and recorded-ghost count, restricted to UI row 1.
    for out, colour in ((UNUSED, 1), (GHOST_UI, 2)):
        w1(out, colour, value=16.0)
        w1(out, 16, value=-63.0)
        p["b1"][out] = -14.0

    for i, (dy, dx) in enumerate(directions):
        out = GATE_EDGE + i
        for forward in (1, 2):
            for sideways in range(-2, 3):
                yy = dy * forward + dx * sideways
                xx = dx * forward - dy * sideways
                for colour in obstruction_colours:
                    w1(out, colour, yy, xx)
        p["b1"][out] = -8.0

        for distance in (1, 2):
            for colour in (8, 15):
                w1(WIRE_RAY + i, colour,
                   distance * dy, distance * dx)
            w1(TOGGLE_RAY + i, 11,
               distance * dy, distance * dx)
        p["b1"][WIRE_RAY + i] = -1.0
        p["b1"][TOGGLE_RAY + i] = -1.0

    # Second-layer feature channels.
    P, FLOOR, FREE = 0, 1, 2
    S_SEED, T_SEED, G_SEED = 3, 4, 5
    PORTAL_LO, PORTAL_HI = 6, 7
    CAPACITY, ON_PLATE = 8, 9
    PLAYER_X, PLAYER_Y, LEGAL = 10, 11, 12
    GHOST_VISIBLE, HAS_GHOST = 16, 17
    plate_channels = (9, 18, 19, 20)
    toggle_channels = (21, 22, 23, 24)
    PATROL_VISIBLE = 25

    w2(P, PLAYER)
    w2(GHOST_VISIBLE, GHOST)
    w2(HAS_GHOST, GHOST_UI)
    w2(PATROL_VISIBLE, PATROL)

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            w2(FLOOR, FLOOR5, dy, dx)
            w2(FREE, FLOOR5, dy, dx)
            w2(FREE, GATE, dy, dx, -10.0)
    p["b2"][FLOOR] = -8.0
    p["b2"][FREE] = -8.0

    w2(S_SEED, PRESSURE)
    w2(T_SEED, TOGGLE)
    for out in (S_SEED, T_SEED):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                w2(out, GATE, dy, dx, -10.0)

    w2(G_SEED, GOAL)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            w2(PORTAL_LO, PORTAL_RING, dy, dx)
            w2(PORTAL_HI, PORTAL_RING, dy, dx)
    p["b2"][PORTAL_LO] = -23.0
    p["b2"][PORTAL_HI] = -24.0
    w2(CAPACITY, UNUSED)

    # Detect exposed wires extending beyond the player's sprite.
    for i, (dy, dx) in enumerate(directions):
        for out, ray in ((plate_channels[i], WIRE_RAY + i),
                         (toggle_channels[i], TOGGLE_RAY + i)):
            w2(out, SHIFT + i, -dy, -dx)
            w2(out, ray)
            p["b2"][out] = -1.0

    for out, coord in ((PLAYER_X, COL), (PLAYER_Y, ROW)):
        w2(out, coord)
        w2(out, PLAYER, value=32.0)
        p["b2"][out] = -32.0

    for i, (dy, dx) in enumerate(directions):
        out = LEGAL + i
        w2(out, SHIFT + i, -dy, -dx)
        w2(out, FLOOR5, dy, dx)
        w2(out, GATE_EDGE + i, dy, dx, -10.0)
        p["b2"][out] = -1.0

    # Third layer: local features and clipped portal evidence.
    for i in range(18):
        p["W3"][i, i] = 1.0

    p["W3"][ON_PLATE, :] = 0.0
    for channel in plate_channels:
        p["W3"][ON_PLATE, channel] = 1.0

    p["W3"][PORTAL_LO, PORTAL_HI] = -1.0

    ON_PORTAL = 7
    p["W3"][ON_PORTAL, :] = 0.0
    p["W3"][ON_PORTAL, P] = 1.0
    p["W3"][ON_PORTAL, PORTAL_LO] = 1.0
    p["W3"][ON_PORTAL, PORTAL_HI] = -1.0
    p["b3"][ON_PORTAL] = -1.0

    OPTIMISTIC_SEED, ON_TOGGLE, LIVE_PATROL = 18, 19, 20
    p["W3"][OPTIMISTIC_SEED, T_SEED] = 1.0
    p["W3"][OPTIMISTIC_SEED, PORTAL_LO] = 1.0
    p["W3"][OPTIMISTIC_SEED, PORTAL_HI] = -1.0
    for channel in toggle_channels:
        p["W3"][ON_TOGGLE, channel] = 1.0
    p["W3"][LIVE_PATROL, PATROL_VISIBLE] = 1.0

    # Four reachability fields:
    # actual goal, optimistic goal, pressure plates, optimistic mechanisms.
    fields = [
        (G_SEED, FREE),
        (G_SEED, FLOOR),
        (S_SEED, FREE),
        (OPTIMISTIC_SEED, FLOOR),
    ]

    for field, (seed, floor) in enumerate(fields):
        u, v, hx, hy = range(4 * field, 4 * field + 4)

        # reached = ReLU(s) - ReLU(s - 1).
        for out in (u, v):
            for dy, dx in directions:
                p["Wh"][out, u, dy + 1, dx + 1] += 1.0
                p["Wh"][out, v, dy + 1, dx + 1] -= 1.0
            p["Uh"][out, floor] = 5.0
            p["Uh"][out, seed] = 1.0
        p["bh"][u] = -5.0
        p["bh"][v] = -6.0

        # Integrate signed neighbour-arrival differences at the player.
        for out in (hx, hy):
            p["Wh"][out, out, 1, 1] = 1.0
            p["Uh"][out, P] = 257.0
            p["bh"][out] = -256.0

        for dy, dx, sign in ((0, 1, 1.0), (0, -1, -1.0)):
            p["Wh"][hx, u, dy + 1, dx + 1] += sign
            p["Wh"][hx, v, dy + 1, dx + 1] -= sign
        for dy, dx, sign in ((1, 0, 1.0), (-1, 0, -1.0)):
            p["Wh"][hy, u, dy + 1, dx + 1] += sign
            p["Wh"][hy, v, dy + 1, dx + 1] -= sign

    def fmax(channel):
        return 32 + channel

    def fmean(channel):
        return 64 + channel

    next_z = 0

    def navigation_branch(field, weight, capacity_mode=None):
        nonlocal next_z
        hx, hy = 4 * field + 2, 4 * field + 3
        components = [(hy, -1.0), (hy, 1.0),
                      (hx, -1.0), (hx, 1.0)]
        # Downweight one-sided gradients caused by corridor boundaries.
        hinges = ((0.0, 0.5), (2.0, -1.369), (3.0, 0.871))

        for action, (channel, sign) in enumerate(components, 1):
            for threshold, coefficient in hinges:
                unit = next_z
                next_z += 1
                p["Wz"][unit, channel] = sign
                p["bz"][unit] = -128.0 * sign - threshold
                if capacity_mode == "remaining":
                    p["Wz"][unit, fmax(CAPACITY)] = 256.0
                    p["bz"][unit] -= 256.0
                elif capacity_mode == "full":
                    p["Wz"][unit, fmax(CAPACITY)] = -256.0
                p["Wo"][action, unit] = weight * coefficient

    navigation_branch(0, 1000.0)
    navigation_branch(1, 100.0)
    navigation_branch(2, 10.0, "remaining")
    navigation_branch(2, 0.5, "full")
    navigation_branch(3, 0.25, "remaining")
    navigation_branch(3, 10.0, "full")

    p["bo"][:] = -1.0e6
    for i in range(4):
        unit = 72 + i
        p["Wz"][unit, fmax(LEGAL + i)] = 1.0
        p["bo"][1 + i] = -2000.0
        p["Wo"][1 + i, unit] = 2000.0

    RECORD_LO, RECORD_HI, KEY = 76, 77, 78
    TOGGLE_EVENT, PATROL_PRESENT = 79, 80

    for unit, threshold in ((RECORD_LO, 0.0), (RECORD_HI, 1.0)):
        p["Wz"][unit, fmax(ON_PLATE)] = 1.0
        p["Wz"][unit, fmax(CAPACITY)] = 10.0
        p["Wz"][unit, fmax(ON_PORTAL)] = -10.0

        # A missing world ghost relative to the UI count indicates overlap.
        p["Wz"][unit, fmean(GHOST_VISIBLE)] = 20.0 * 4096.0
        p["Wz"][unit, fmean(HAS_GHOST)] = -20.0 * 4096.0 / 3.0
        p["bz"][unit] = -10.0 - threshold

    p["bo"][5] = -5000.0
    p["Wo"][5, RECORD_LO] = 10000.0
    p["Wo"][5, RECORD_HI] = -10000.0
    p["Wz"][TOGGLE_EVENT, fmax(ON_TOGGLE)] = 1.0
    p["Wz"][PATROL_PRESENT, fmax(LIVE_PATROL)] = 1.0

    # Four remembered recording locations.
    key_scale = 1.0e-6
    p["Wz"][KEY, fmax(PLAYER_X)] = 63.0 * key_scale
    p["Wz"][KEY, fmax(PLAYER_Y)] = 63.0 * 64.0 * key_scale
    p["Wmz"][0, KEY] = 1.0

    B = 12.0
    slots = [list(range(1 + 4 * j, 5 + 4 * j)) for j in range(4)]
    decode = np.array([0.5, -0.5, 0.5, -0.5], np.float32)

    for j, slot in enumerate(slots):
        rp, rn, hp, hn = slot
        if j == 0:
            p["Wm"][rp, 0] = 1.0
            p["Wm"][rn, 0] = -1.0
        else:
            p["Wm"][rp, slots[j - 1]] = decode
            p["Wm"][rn, slots[j - 1]] = -decode

        p["Wma"][rp, 5] = B
        p["Wma"][rn, 5] = B
        p["bm"][rp] = -B
        p["bm"][rn] = -B
        p["Wm"][hp, slot] = decode
        p["Wm"][hn, slot] = -decode
        p["Wma"][hp, 5] = -B
        p["Wma"][hn, 5] = -B

        # Saturating both sides clears the decoded value after RESET.
        for unit in slot:
            p["Wma"][unit, 0] = 2.0 * B

    gain, tolerance = 3.0e6, 2.0e-6
    for j, slot in enumerate(slots):
        plus, minus = 17 + 2 * j, 18 + 2 * j
        p["Wmz"][plus, KEY] = gain
        p["Wmz"][minus, KEY] = -gain
        p["Wm"][plus, slot] = -gain * decode
        p["Wm"][minus, slot] = gain * decode
        p["bm"][plus] = gain * tolerance
        p["bm"][minus] = gain * tolerance
        p["Wo"][5, 128 + plus] = -10000.0
        p["Wo"][5, 128 + minus] = -10000.0

    # Small movement-inertia biases resolve otherwise exact ties.
    clockwise = [3, 2, 0, 1]
    opposite = [1, 0, 3, 2]
    for previous in range(4):
        unit = 32 + previous
        p["Wma"][unit, 1 + previous] = 3.0
        for candidate in range(4):
            if candidate == previous:
                bonus = 0.00012
            elif candidate == clockwise[previous]:
                bonus = 0.00004
            elif candidate == opposite[previous]:
                bonus = -0.00008
            else:
                bonus = 0.00002
            p["Wo"][1 + candidate, 128 + unit] = bonus

    # Toggle-phase latch, decoded as (m[36] - m[37]) / 2.
    # It starts at zero, approaches one after touching a toggle, and
    # clears on RESET. Paired saturation makes clearing numerically exact.
    for unit, sign in ((36, 1.0), (37, -1.0)):
        p["Wm"][unit, 36] = 2.0 * sign
        p["Wm"][unit, 37] = -2.0 * sign
        p["Wmz"][unit, TOGGLE_EVENT] = 8.0 * sign
        p["Wma"][unit, 0] = 20.0

    # Conditional linear readouts implemented by differences of tanh
    # units. Before the latch, use mechanism attraction. Afterwards,
    # use optimistic goal navigation instead of revisiting the toggle.
    C = 10.0
    goal_baseline = 42
    mechanism_baseline = 47

    p["Wm"][goal_baseline, 36] = -C / 2.0
    p["Wm"][goal_baseline, 37] = C / 2.0
    p["bm"][goal_baseline] = C
    p["Wm"][mechanism_baseline, 36] = C / 2.0
    p["Wm"][mechanism_baseline, 37] = -C / 2.0

    for action in range(1, 5):
        goal_unit = 38 + action - 1
        mechanism_unit = 43 + action - 1

        goal_vector = np.zeros(128, np.float32)
        goal_vector[12:24] = p["Wo"][action, 12:24]
        p["Wo"][action, 12:24] = 0.0

        mechanism_vector = np.zeros(128, np.float32)
        mechanism_vector[48:72] = p["Wo"][action, 48:72]
        p["Wo"][action, 48:72] = 0.0

        p["Wmz"][goal_unit] = 0.0005 * goal_vector
        p["Wm"][goal_unit, 36] = -C / 2.0
        p["Wm"][goal_unit, 37] = C / 2.0
        p["bm"][goal_unit] = C
        p["Wo"][action, 128 + goal_unit] = 2000.0
        p["Wo"][action, 128 + goal_baseline] = -2000.0

        p["Wmz"][mechanism_unit] = 0.005 * mechanism_vector
        p["Wm"][mechanism_unit, 36] = C / 2.0
        p["Wm"][mechanism_unit, 37] = -C / 2.0
        p["Wo"][action, 128 + mechanism_unit] = 200.0
        p["Wo"][action, 128 + mechanism_baseline] = -200.0

    # Recover if a patrol previously seen in this episode disappears.
    # This is inactive in episodes where no patrol has been observed.
    PATROL_SEEN = 48
    p["Wm"][PATROL_SEEN, PATROL_SEEN] = 4.0
    p["Wmz"][PATROL_SEEN, PATROL_PRESENT] = 4.0
    p["bo"][0] = -3000.0
    p["Wo"][0, 128 + PATROL_SEEN] = 10000.0
    p["Wo"][0, PATROL_PRESENT] = -10000.0

    return p
