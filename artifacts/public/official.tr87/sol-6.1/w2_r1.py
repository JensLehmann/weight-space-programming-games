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

    masks = [
        [
            ["00100", "00100", "01110", "00100", "11111"],
            ["00001", "00101", "11111", "10100", "10000"],
            ["11111", "10100", "10000", "10100", "11111"],
            ["00100", "11111", "10101", "10101", "00100"],
            ["10001", "11111", "00100", "11111", "10001"],
            ["11111", "00100", "11100", "10000", "11111"],
            ["10001", "11111", "10001", "10001", "11011"],
        ],
        [
            ["10000", "11110", "10010", "10010", "11111"],
            ["11111", "10001", "11101", "10101", "11111"],
            ["00111", "00101", "11111", "10100", "11100"],
            ["01110", "01010", "11111", "10001", "11111"],
            ["11111", "10001", "10001", "11111", "00100"],
            ["11110", "10011", "10001", "11001", "01111"],
            ["00100", "11111", "10101", "11111", "00100"],
        ],
        [
            ["10101", "10000", "10101", "10000", "11111"],
            ["11101", "10000", "00000", "00001", "10111"],
            ["10111", "00100", "00100", "00100", "11101"],
            ["00100", "10101", "00100", "10101", "00100"],
            ["11011", "10001", "10101", "10001", "11011"],
            ["10111", "10101", "11101", "00000", "10101"],
            ["10001", "00000", "10001", "11011", "01010"],
        ],
    ]
    colours = [10, 7, 11]
    orbits = [
        [(0, 0), (0, 4), (4, 4), (4, 0)],
        [(0, 1), (1, 4), (4, 3), (3, 0)],
        [(0, 2), (2, 4), (4, 2), (2, 0)],
        [(0, 3), (3, 4), (4, 1), (1, 0)],
        [(1, 1), (1, 3), (3, 3), (3, 1)],
        [(1, 2), (2, 3), (3, 2), (2, 1)],
    ]

    # Quarter-turn-invariant orbit-count hinges.
    for orbit_number, orbit in enumerate(orbits):
        for k in range(4):
            channel = 4 * orbit_number + k
            for row, col in orbit:
                p["W1"][channel, 5, row, col] = 1.0
            p["b1"][channel] = -float(k)
    p["W1"][24, 5, 2, 2] = 1.0

    for family, colour in enumerate(colours):
        channel = 25 + family
        p["W1"][channel, 5, :, :] = 1.0
        p["W1"][channel, colour, :, :] = 1.0

    # Upper-left cursor-bracket corner.
    for dy, dx in [(0, -2), (0, -1), (0, 0), (1, -2)]:
        p["W1"][28, 0, 2 + dy, 2 + dx] += 1.0
    for dy, dx in [(1, -1), (1, 0)]:
        p["W1"][28, 0, 2 + dy, 2 + dx] -= 1.0
    p["b1"][28] = -3.0

    p["W1"][29, 16, 2, 2] = -63.0
    p["b1"][29] = 34.0
    p["W1"][30, 16, 2, 2] = 63.0
    p["b1"][30] = -49.0

    p["W1"][31, 0, 2, :] = 1.0
    p["b1"][31] = -2.0

    keys = []
    for family in range(3):
        for index in range(7):
            channel = 7 * family + index
            mask = np.array(
                [[int(v) for v in row] for row in masks[family][index]],
                dtype=np.int32,
            )
            counts = [
                sum(int(mask[row, col]) for row, col in orbit)
                for orbit in orbits
            ]
            centre = int(mask[2, 2])

            # Exact integer descriptor equality via squared distance.
            for orbit_number, value in enumerate(counts):
                base = 4 * orbit_number
                p["W2"][channel, base, 1, 1] = 2.0 * value - 1.0
                for k in range(1, 4):
                    p["W2"][channel, base + k, 1, 1] = -2.0
            p["W2"][channel, 24, 1, 1] = 2.0 * centre - 1.0

            # The union of these windows is the enclosing 7x7 tile.
            p["W2"][channel, 25 + family, :, :] = 1.0
            p["b2"][channel] = (
                1.0
                - sum(value * value for value in counts)
                - centre * centre
                - 225.0
            )
            keys.append(float(8 * family + index + 1))

    def w2(out, inp, value=1.0, dy=0, dx=0):
        p["W2"][out, inp, 1 + dy, 1 + dx] += value

    w2(21, 28)
    w2(22, 28, 128.0)
    w2(22, 29, -1.0)
    p["b2"][22] = -94.0

    for dx in (-1, 0, 1):
        w2(23, 31, dx=dx)
        w2(27, 31, dx=dx)
    p["b2"][23] = -1.0

    w2(24, 30)
    w2(25, 29)
    w2(25, 29, -1.0, dy=1)
    w2(26, 30)
    w2(26, 30, -1.0, dy=-1)

    w2(28, 28, 128.0)
    w2(28, 29)
    p["b2"][28] = -128.0

    w2(29, 29, -1.0)
    p["b2"][29] = 34.0
    w2(30, 31)

    w2(31, 28, 128.0)
    w2(31, 29)
    p["b2"][31] = -129.0

    def local(out, inp, value=1.0):
        p["W3"][out, inp] += value

    def colmax(out, inp, value=1.0):
        p["W3"][out, 64 + inp] += value

    for channel, key in enumerate(keys):
        local(0, channel, key)
        local(1, channel)
        local(2, channel, key)
        local(3, channel)
        local(4, channel, key)
        local(5, channel, key)

        if channel < 14:
            local(10, channel, key)
            local(13, channel)
        if channel < 7:
            local(11, channel)
            local(14, channel)
        elif channel < 14:
            local(12, channel)

    local(2, 25, 128.0)
    p["b3"][2] = -128.0
    local(3, 25)
    p["b3"][3] = -1.0

    # Selected lower input, with a three-pixel alignment margin.
    colmax(4, 27, 128.0)
    colmax(4, 23, -128.0)
    local(4, 25, -128.0)
    local(4, 26, -128.0)
    colmax(4, 28, -128.0)
    colmax(4, 31, 128.0)
    p["b3"][4] = -128.0

    colmax(5, 21, 128.0)
    local(5, 26, 128.0)
    colmax(5, 28, -128.0)
    colmax(5, 31, 128.0)
    p["b3"][5] = -256.0

    local(6, 28)
    local(6, 31, -1.0)
    colmax(7, 21)

    local(8, 29)
    colmax(8, 22, -1.0)
    p["b3"][8] = -6.25
    colmax(9, 22)
    local(9, 29, -1.0)
    p["b3"][9] = 5.75

    local(10, 25, 128.0)
    p["b3"][10] = -128.0
    for out in (11, 12, 13, 14):
        local(out, 25)
        p["b3"][out] = -1.0

    eps = 1.0 / 128.0
    map_base = 1000.0

    # Mean encodes multiplicity; maximum encodes a reversed partner ID.
    for i in range(7):
        out = 16 + i
        local(out, i, 4096.0)
        for j in range(7):
            colmax(out, 7 + j, eps * (7 - j))
        p["b3"][out] = map_base - 4096.0

    for j in range(7):
        out = 23 + j
        local(out, 7 + j, 4096.0)
        local(out, 25, -4096.0)
        for i in range(7):
            colmax(out, i, eps * (7 - i))
        p["b3"][out] = map_base - 4096.0

    local(30, 30)

    def recurrent(out, inp, value=1.0, dy=0, dx=0):
        p["Wh"][out, inp, 1 + dy, 1 + dx] += value

    # Acyclic vertical/horizontal scalar broadcast.
    recurrent(0, 0, dy=1)
    recurrent(1, 1, dy=-1)
    for channel in (0, 1):
        p["Uh"][channel, 4] = 1.0
        recurrent(channel, 12)

    for channel, dx in ((2, -1), (3, 1)):
        recurrent(channel, channel, dx=dx)
        recurrent(channel, 0)
        recurrent(channel, 1)
        recurrent(channel, 12, -1.0)
        p["Uh"][channel, 4] = -1.0

    for inp, value in [(2, 1), (3, 1), (0, -1), (1, -1), (12, 1)]:
        recurrent(4, inp, value)
        recurrent(5, inp, -value)
    p["Uh"][4, 4] = 1.0
    p["Uh"][4, 0] = -1.0
    p["Uh"][5, 4] = -1.0
    p["Uh"][5, 0] = 1.0

    p["Uh"][6, 3] = 1.0
    recurrent(6, 4, -64.0)
    recurrent(6, 5, -64.0)

    recurrent(7, 7, dx=-1)
    recurrent(7, 6, 12.0)
    p["bh"][7] = -1.0

    recurrent(8, 8, dx=-1)
    recurrent(8, 13, 19.0)
    p["bh"][8] = -1.0
    recurrent(9, 9, dx=1)
    recurrent(9, 14, 19.0)
    p["bh"][9] = -1.0

    # Half-unit threshold gives relay endpoints a numerical margin.
    p["bh"][10] = 0.5
    for channel in (7, 8, 9):
        recurrent(10, channel, -1.0)

    p["Uh"][11, 10] = 1.0
    p["Uh"][11, 13] = 128.0
    recurrent(11, 10, -8192.0)
    recurrent(11, 6, -4096.0)

    p["Uh"][12, 2] = 1.0
    p["Uh"][12, 3] = 64.0
    p["Uh"][12, 7] = 128.0
    p["Uh"][12, 8] = -128.0
    p["Uh"][12, 9] = -128.0
    p["bh"][12] = -128.0

    for out, feature in ((13, 11), (14, 12)):
        p["Uh"][out, feature] = 1.0
        p["Uh"][out, 7] = 1.0
        p["Uh"][out, 8] = -128.0
        p["Uh"][out, 9] = -128.0
        p["bh"][out] = -1.0

    p["Uh"][15, 2] = 1.0
    p["Uh"][15, 11] = 4096.0
    p["Uh"][15, 14] = 128.0
    p["bh"][15] = -4096.0
    recurrent(15, 10, -8192.0)
    recurrent(15, 6, -4096.0)

    unit = 0

    def new_z():
        nonlocal unit
        result = unit
        unit += 1
        return result

    def utility(action, d):
        if action == 1:
            return 2.0 if d in (-3, -2, -1, 4, 5, 6) else 0.0
        return 2.0 if d in (-6, -5, -4, 1, 2, 3) else 0.0

    # Piecewise-linear shortest cyclic edit utilities.
    previous_slopes = [0.0, 0.0]
    for knot in range(-8, 9):
        slopes = [
            utility(action, knot + 1) - utility(action, knot)
            for action in (1, 2)
        ]
        changes = [
            slopes[j] - previous_slopes[j] for j in range(2)
        ]
        previous_slopes = slopes
        if changes == [0.0, 0.0]:
            continue
        u = new_z()
        p["Wz"][u, 11] = 1.0
        p["Wz"][u, 32 + 5] = -1.0
        p["Wz"][u, 32 + 6] = -4096.0
        p["bz"][u] = -128.0 - float(knot)
        p["Wo"][1, u] = changes[0]
        p["Wo"][2, u] = changes[1]

    selected_unit = new_z()
    p["Wz"][selected_unit, 12] = 1.0

    lhs_unit = new_z()
    p["Wz"][lhs_unit, 15] = 1.0

    rhs_flags = []
    for threshold, coefficient in ((1.5, 2.0), (2.0, -2.0)):
        u = new_z()
        rhs_flags.append((u, coefficient))
        p["Wz"][u, 16 + 11] = 4096.0 / 128.0
        p["Wz"][u, 16 + 15] = -4096.0 / 128.0
        p["bz"][u] = -threshold

    # Saturated repeated-output IDs.
    duplicates = []
    for j in range(7):
        for threshold, sign in ((1.5, 1.0), (2.0, -1.0)):
            u = new_z()
            duplicates.append((u, sign * 2.0 * (9 + j)))
            p["Wz"][u, 64 + 23 + j] = 4096.0 / map_base
            p["bz"][u] = -threshold

    gate = 1000.0

    def distance_units(key):
        result = []
        for sign in (1.0, -1.0):
            u = new_z()
            result.append(u)
            p["Wz"][u, 12] = sign
            p["bz"][u] = -sign * key
        return result

    def clipped_map(feature):
        u = new_z()
        p["Wz"][u, 32 + feature] = 1.0
        p["bz"][u] = -map_base
        return u

    def suppress_other_keys(memory_units, distances):
        for m in memory_units:
            for u in distances:
                p["Wmz"][m, u] -= gate

    # Left-side multiplicity constraints.
    for i in range(7):
        errors = []
        for sign in (1.0, -1.0):
            u = new_z()
            errors.append(u)
            p["Wz"][u, 64 + 16 + i] = sign * 4096.0 / map_base
            p["Wz"][u, 64 + 30] = -sign * 2.0 * 4096.0 / 42.0
            p["bz"][u] = -sign * 48.0 / 42.0 - 0.20

        error_m = 2 * i
        base_m = error_m + 1
        for m in (error_m, base_m):
            p["Wmz"][m, selected_unit] = gate
            p["bm"][m] = -gate * (65 + i)
        for u in errors:
            p["Wmz"][error_m, u] = 8.0
        p["Wo"][2, 128 + error_m] = 2.0
        p["Wo"][2, 128 + base_m] = -2.0

    # Right-side partner consistency.
    for j in range(7):
        distances = distance_units(73 + j)
        mapping = clipped_map(23 + j)
        memory_units = (14 + 2 * j, 15 + 2 * j)
        suppress_other_keys(memory_units, distances)

        for m, sign in zip(memory_units, (1.0, -1.0)):
            p["Wmz"][m, lhs_unit] = sign * 16.0
            p["Wmz"][m, mapping] = sign * 16.0 / eps
            p["bm"][m] = -sign * 16.0 * 136.0 - 4.0
            p["Wo"][2, 128 + m] = 2.0

    # Distinguish expanding A rules from single-output A rules.
    for i in range(7):
        distances = distance_units(65 + i)
        mapping = clipped_map(16 + i)
        first_m = 28 + 4 * i
        memory_units = tuple(range(first_m, first_m + 4))
        suppress_other_keys(memory_units, distances)

        def add_partner_difference(m, sign):
            for u, coefficient in duplicates:
                p["Wmz"][m, u] += sign * 16.0 * coefficient
            p["Wmz"][m, mapping] += sign * 16.0 / eps
            p["bm"][m] -= sign * 16.0 * 16.0

        # Expanding rules reject a non-repeated partner.
        for offset, sign in ((0, 1.0), (1, -1.0)):
            m = first_m + offset
            add_partner_difference(m, sign)
            p["bm"][m] -= gate + 4.0
            for u, coefficient in rhs_flags:
                p["Wmz"][m, u] += gate * coefficient
            p["Wo"][2, 128 + m] = 2.0

        # Single-output rules reject a repeated partner.
        for offset, shift, coefficient in (
            (2, 4.0, 2.0),
            (3, -4.0, -2.0),
        ):
            m = first_m + offset
            add_partner_difference(m, 1.0)
            p["bm"][m] += shift
            for u, flag_coefficient in rhs_flags:
                p["Wmz"][m, u] -= gate * flag_coefficient
            p["Wo"][2, 128 + m] = coefficient

    # Six-consecutive-edit guard. Navigation clears the chain.
    p["Wma"][56, 2] = 10.0
    p["bm"][56] = -5.0
    for m in range(57, 62):
        p["Wm"][m, m - 1] = 10.0
        p["Wma"][m, 2] = 10.0
        p["bm"][m] = -15.0

    p["bo"].fill(-100.0)
    p["bo"][1] = -1.0
    p["bo"][2] = 55.0  # Cancels inactive inequality-pair baselines.
    p["bo"][4] = 100.0
    p["Wo"][4, 128 + 61] = 100.0

    return p
