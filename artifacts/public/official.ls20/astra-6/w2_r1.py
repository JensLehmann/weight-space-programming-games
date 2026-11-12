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
    p = {k: np.zeros(v, dtype=np.float32) for k, v in shapes.items()}

    base = np.array([
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 1, 1], [1, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [1, 0, 1]],
    ], dtype=np.int8)
    templates = [
        np.rot90(base[s], -r).copy()
        for s in range(6) for r in range(4)
    ]
    inks = (8, 9, 12, 14)

    # Scale-two carried-symbol recognition.
    for t, bits in enumerate(templates):
        for y in range(3):
            for x in range(3):
                weight = 1 if bits[y, x] else -1
                for colour in inks:
                    p["W1"][t, colour, 2 * y, 2 * x] = weight
        p["b1"][t] = 1 - int(bits.sum())

    BINARY, COLOUR, WALL, PLAYER = 24, 25, 26, 27
    ROTATOR, SHAPER, PAINTER, BLUE = 28, 29, 30, 31

    for colour in inks:
        p["W1"][BINARY, colour, 2, 2] = 1
    for colour, code in ((12, 0), (9, 1), (14, 10), (8, 100)):
        p["W1"][COLOUR, colour, 2, 2] = code

    # Inflate walls and recognize extended launcher-edge strips.
    p["W1"][WALL, 4, :, :] = 100
    p["W1"][WALL, 1, :, :] = 1

    for y in range(5):
        p["W1"][PLAYER, 12 if y < 2 else 9, y, :] = 1
    p["b1"][PLAYER] = -24

    # Small position modulation remains confined to a player detection.
    eps = 1.0 / 32.0
    p["W1"][PLAYER, 16, 2, 2] = eps * 63.0 / 4095.0
    p["W1"][PLAYER, 17, 2, 2] = eps * 64.0 * 63.0 / 4095.0

    def point_template(channel, points):
        for dy, dx, colour in points:
            p["W1"][channel, colour, dy + 2, dx + 2] += 1
        p["b1"][channel] = 1 - len(points)

    point_template(ROTATOR, [
        (-1, 0, 0), (0, -1, 1), (0, 0, 0),
        (0, 1, 0), (1, 0, 1),
    ])
    point_template(SHAPER, [
        (-1, -1, 0), (0, 0, 0), (0, 1, 0), (1, 0, 0),
    ])
    point_template(PAINTER, [
        (-1, -1, 9), (-1, 0, 14), (-1, 1, 14),
        (0, -1, 9), (0, 0, 0), (0, 1, 8),
        (1, -1, 12), (1, 0, 12), (1, 1, 8),
    ])

    # Ring centers give four; fuel-bar interior pixels give one.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                p["W1"][BLUE, 11, dy + 2, dx + 2] = 1
    p["b1"][BLUE] = -4

    # Scale-one goal recognition.
    for t, bits in enumerate(templates):
        p["W2"][t, BINARY] = np.where(bits, 1, -1)
        p["b2"][t] = 1 - int(bits.sum())

    F_WALL, F_OBJECT = 24, 25
    F_SHAPE, F_ROT, F_COLOUR = 26, 27, 28
    F_GOAL_COLOUR, F_FUEL, F_NEIGHBOURS = 29, 30, 31

    p["W2"][F_WALL, WALL, 1, 1] = 1
    p["b2"][F_WALL] = -2

    # Object codes: rotation=1, shape=2, colour=3,
    # player=4+position modulation, recharge=5.
    p["b2"][F_OBJECT] = -11
    for channel, weight in (
        (ROTATOR, 12), (SHAPER, 13), (PAINTER, 14), (PLAYER, 15)
    ):
        p["W2"][F_OBJECT, channel, 1, 1] = weight
    p["W2"][F_OBJECT, BLUE, 1, 1] = 4

    for t in range(24):
        s, r = divmod(t, 4)
        p["W2"][F_SHAPE, t, 1, 1] = s + 1
        p["W2"][F_ROT, t, 1, 1] = r + 1

    # Restrict the carried-colour measurement to UI template responses.
    p["W2"][F_COLOUR, COLOUR, 1, 1] = 1
    p["b2"][F_COLOUR] = -10000
    for t in range(24):
        for dy in (-1, 1):
            for dx in (-1, 1):
                p["W2"][F_COLOUR, t, 1 + dy, 1 + dx] = 10000

    p["W2"][F_GOAL_COLOUR, COLOUR, :, :] = 1
    p["W2"][F_FUEL, BLUE, 1, 1] = 1

    # Mark the four pixels neighboring the player center.
    p["W2"][F_NEIGHBOURS, PLAYER, 1, 2] = 1
    p["W2"][F_NEIGHBOURS, PLAYER, 1, 0] = 2
    p["W2"][F_NEIGHBOURS, PLAYER, 2, 1] = 3
    p["W2"][F_NEIGHBOURS, PLAYER, 0, 1] = 4

    H_WALL = 0
    RAMPS = (1, 2, 3, 4, 5)
    UI_SHAPE, UI_ROT, UI_COLOUR = 6, 7, 8
    GOAL_PRIORITY, GOAL_SHAPE, GOAL_ROT = 9, 10, 11
    GOAL_COLOUR, GOAL_SEED, FUEL = 12, 13, 14
    RING = 23

    p["W3"][H_WALL, F_WALL] = 1
    for k, channel in enumerate(RAMPS):
        p["W3"][channel, F_OBJECT] = 1
        p["b3"][channel] = -k

    for output, source in (
        (UI_SHAPE, F_SHAPE), (UI_ROT, F_ROT),
        (UI_COLOUR, F_COLOUR), (FUEL, F_FUEL),
    ):
        p["W3"][output, source] = 1

    # Clipped staircase steps remove position modulation from markers.
    neighbour_steps = []
    for k in range(1, 5):
        a = 15 + 2 * (k - 1)
        b = a + 1
        p["W3"][a, F_NEIGHBOURS] = 1
        p["W3"][b, F_NEIGHBOURS] = 1
        p["b3"][a] = -(k - 0.25)
        p["b3"][b] = -(k - 0.125)
        neighbour_steps.append((a, b))

    p["W3"][RING, F_OBJECT] = 2
    p["b3"][RING] = -9

    # Common priority aligns independently pooled goal attributes.
    guard = 2048
    for t in range(24):
        s, r = divmod(t, 4)
        priority = 2000 * (24 - t)
        for channel, extra in (
            (GOAL_PRIORITY, 0), (GOAL_SHAPE, s),
            (GOAL_ROT, r), (GOAL_COLOUR, 0),
        ):
            p["W3"][channel, t] = guard + priority + extra
        p["W3"][GOAL_SEED, t] = 1

    for channel in (GOAL_PRIORITY, GOAL_SHAPE, GOAL_ROT, GOAL_COLOUR):
        p["b3"][channel] = -guard
    p["W3"][GOAL_COLOUR, F_GOAL_COLOUR] = 1

    # Four bounded wavefronts: goals/refills, shape, colour, rotation.
    # Each pair u,v represents a clipped frontier through u-v.
    # gx and gy integrate arrivals at two marked positions apiece.
    M = 8.0
    x_increments = (M, 2, -M - 2, 0)
    y_increments = (0, 0, M, 2)

    for field in range(4):
        u, v, gx, gy = range(4 * field, 4 * field + 4)
        for target in (u, v):
            for y, x in ((0, 1), (2, 1), (1, 0), (1, 2)):
                p["Wh"][target, u, y, x] = 1
                p["Wh"][target, v, y, x] = -1
            p["Uh"][target, H_WALL] = -16
        p["bh"][v] = -1

        for read, increments in ((gx, x_increments), (gy, y_increments)):
            p["Wh"][read, read, 1, 1] = 1
            p["Wh"][read, u, 1, 1] = 1
            p["Wh"][read, v, 1, 1] = -1
            p["bh"][read] = -M
            for (a, b), increment in zip(neighbour_steps, increments):
                p["Uh"][read, a] += 8 * increment
                p["Uh"][read, b] -= 8 * increment

    for channel in (0, 1):
        p["Uh"][channel, GOAL_SEED] += 1
        p["Uh"][channel, RING] += 1

    def modifier_source(field, first):
        for channel in (4 * field, 4 * field + 1):
            p["Uh"][channel, RAMPS[first]] += 1
            p["Uh"][channel, RAMPS[first + 1]] -= 2
            p["Uh"][channel, RAMPS[first + 2]] += 1

    modifier_source(1, 1)
    modifier_source(2, 2)
    modifier_source(3, 0)

    def fmax(c):
        return 32 + c

    def fmean(c):
        return 64 + c

    nz = 0

    def zunit(weights, bias=0):
        nonlocal nz
        index = nz
        nz += 1
        for column, weight in weights.items():
            p["Wz"][index, column] += weight
        p["bz"][index] = bias
        return index

    # Max and mean recover both arrival integrals in each accumulator.
    nav = []
    for field in range(4):
        gx, gy = 4 * field + 2, 4 * field + 3
        nav.append((
            zunit({16 + gy: 4096, gy: -1}),
            zunit({gy: 1}, -256),
            zunit({16 + gx: 4096, gx: -1}),
            zunit({gx: 1}, -256),
        ))

    def equality_tent(weights):
        a = zunit(weights)
        b = zunit(weights, -1)
        c = zunit(weights, -2)
        return {a: 1, b: -2, c: 1}

    shape_match = equality_tent({
        fmax(UI_SHAPE): 1,
        fmax(GOAL_SHAPE): -1,
        fmax(GOAL_PRIORITY): 1,
    })
    rotation_match = equality_tent({
        fmax(UI_ROT): 1,
        fmax(GOAL_ROT): -1,
        fmax(GOAL_PRIORITY): 1,
    })

    def clipped_positive(weights, bias):
        a = zunit(weights, bias)
        b = zunit(weights, bias - 1)
        return {a: 1, b: -1}

    def combine(*terms):
        result = {}
        for scale, form in terms:
            for key, value in form.items():
                result[key] = result.get(key, 0) + scale * value
        return result

    colour_bad = combine(
        (1, clipped_positive({
            fmax(GOAL_COLOUR): 2,
            fmax(GOAL_PRIORITY): -2,
            fmax(UI_COLOUR): -14,
        }, -1)),
        (1, clipped_positive({
            fmax(UI_COLOUR): 10,
            fmax(GOAL_COLOUR): -2,
            fmax(GOAL_PRIORITY): 2,
        }, -1)),
    )

    fuel_low = clipped_positive({
        fmean(FUEL): -2048,
        fmean(RING): 8192,
        fmax(RING): 1000,
    }, -988)

    delta = 15 * eps
    position_z = zunit({
        fmean(RAMPS[3]): -4096 / delta,
        fmean(RAMPS[4]): 8192 / delta,
    }, 1 / delta)

    shape_z = zunit({fmax(UI_SHAPE): 1})
    rotation_z = zunit({fmax(UI_ROT): 1})
    colour_z = zunit({fmax(UI_COLOUR): 1})

    # Active head: shape, then colour, then rotation, then goal.
    # A low-fuel head overrides the modifier sequence.
    heads = [
        (1, combine((1, fuel_low), (1, shape_match)), 0),
        (2, combine(
            (1, fuel_low), (-1, shape_match), (-1, colour_bad)
        ), 2),
        (3, combine(
            (1, fuel_low), (-1, shape_match),
            (1, colour_bad), (1, rotation_match)
        ), 1),
        (0, combine(
            (1, fuel_low), (-1, shape_match),
            (1, colour_bad), (-1, rotation_match)
        ), 2),
        (0, combine((-1, fuel_low)), 1),
    ]

    nm = 0
    for field, forbidden, constant in heads:
        for action, score in enumerate(nav[field], start=1):
            unit = nm
            nm += 1
            p["Wmz"][unit, score] = 1 / 128
            for z, weight in forbidden.items():
                p["Wmz"][unit, z] += 32 * weight
            p["bm"][unit] = 32 * constant
            p["Wo"][action, 128 + unit] = 1

    # When a desired modifier is occluded by the player, all four
    # navigation scores vanish. A weak reversal then leaves its cell.
    reverse = {1: 2, 2: 1, 3: 4, 4: 3}
    for previous, action in reverse.items():
        unit = nm
        nm += 1
        p["Wma"][unit, previous] = 10
        p["Wo"][action, 128 + unit] = 0.0002

    # State signature: spatial position and all carried attributes.
    # Five-pixel displacement is the natural spatial scale.
    signature = {
        position_z: 16 * 4095 / 5,
        shape_z: 16 * 4096,
        rotation_z: 16 * 32768,
        colour_z: 16 * 1024,
    }
    storage_scale = 2.0 ** -36

    current_state = nm
    nm += 1
    for z, weight in signature.items():
        p["Wmz"][current_state, z] = storage_scale * weight

    # A delayed record pairs the previous observation's signature with
    # the action actually taken there. Further units form a delay line.
    joint_history = []
    for delay in range(4):
        unit = nm
        nm += 1
        joint_history.append(unit)
        if delay == 0:
            p["Wm"][unit, current_state] = 1
            for action in range(1, 5):
                p["Wma"][unit, action] = storage_scale * 4 * action
        else:
            p["Wm"][unit, joint_history[delay - 1]] = 1

    # Paired tanh windows compare each proposed state-action signature
    # with records two, three, four, and five steps old.
    comparison_gain = 2.0
    window = 1.5
    loop_penalty = 0.025

    for record in joint_history:
        for action in range(1, 5):
            for sign in (1, -1):
                unit = nm
                nm += 1
                for z, weight in signature.items():
                    p["Wmz"][unit, z] = comparison_gain * weight
                p["Wm"][unit, record] = (
                    -comparison_gain / storage_scale
                )
                p["bm"][unit] = comparison_gain * (
                    4 * action + sign * window
                )
                p["Wo"][action, 128 + unit] = -loop_penalty * sign

    p["bo"][[0, 5, 6, 7]] = -1000
    return p
