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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    colours = (8, 9, 11)
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    diagonal = ((-1, -1), (1, 1), (-1, 1), (1, -1))
    neighbours = cardinal + diagonal
    orientations = ((0, 1), (1, 0), (1, 1), (1, -1))

    def w1(o, c, dy=0, dx=0, v=1.0):
        p["W1"][o, c, dy + 2, dx + 2] += v

    def w2(o, c, v=1.0):
        p["W2"][o, c, 1, 1] += v

    # Target centres. The outer-ring veto prevents a coloured stroke
    # through a marker's border from creating extra target centres.
    for i, c in enumerate(colours):
        for j in range(2):
            o = 2 * i + j
            w1(o, c, v=16.0)
            for dy, dx in neighbours:
                w1(o, 4, dy, dx)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if max(abs(dy), abs(dx)) == 2:
                        w1(o, 4, dy, dx, -16.0)
            p["b1"][o] = -19.0 - j

    # Four local continuity tests for each shape colour.
    for i, c in enumerate(colours):
        for d, (dy, dx) in enumerate(orientations):
            o = 6 + 4 * i + d
            w1(o, c)
            w1(o, 0)
            w1(o, c, dy, dx)
            w1(o, c, -dy, -dx)
            p["b1"][o] = -2.0

    # Thin-colour masks and paint-pad detectors.
    for i, c in enumerate(colours):
        w1(18 + i, c)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                w1(18 + i, 2, dy, dx, -2.0)

        w1(21 + i, c)
        w1(21 + i, 2, -1, 0)
        p["b1"][21 + i] = -1.0

    w1(24, 0)
    w1(25, 16)
    w1(26, 17)

    w1(27, 0, v=16.0)
    for c in range(6, 15):
        for dy, dx in neighbours:
            w1(27, c, dy, dx)
    p["b1"][27] = -16.0

    # Selected horizontal, diagonal-cross and vertical evidence.
    for o, offsets, centre_weight in (
        (28, ((0, -1), (0, 1)), 4.0),
        (29, diagonal, 8.0),
        (30, ((-1, 0), (1, 0)), 4.0),
    ):
        w1(o, 0, v=centre_weight)
        for c in range(6, 15):
            for dy, dx in offsets:
                w1(o, c, dy, dx)
        p["b1"][o] = -(centre_weight + len(offsets) - 1.0)

    # Second layer: binary targets, shape OR thresholds, paint
    # coordinates, selection coordinates and target-row extrema.
    for i in range(3):
        w2(i, 2 * i)
        w2(i, 2 * i + 1, -1.0)

        for d in range(4):
            w2(3 + 2 * i, 6 + 4 * i + d)
            w2(4 + 2 * i, 6 + 4 * i + d)
        p["b2"][4 + 2 * i] = -1.0

        w2(9 + i, 18 + i)

        base = 12 + 3 * i
        w2(base, 21 + i)

        w2(base + 1, 25)
        w2(base + 1, 21 + i)
        p["b2"][base + 1] = -1.0

        w2(base + 2, 26)
        w2(base + 2, 21 + i)
        p["b2"][base + 2] = -1.0

    w2(21, 25)
    w2(21, 24)
    p["b2"][21] = -1.0

    w2(22, 26)
    w2(22, 24)
    p["b2"][22] = -1.0

    w2(23, 24)
    w2(24, 27)

    for dst, src in (
        (25, 28), (26, 29), (27, 30), (28, 25), (31, 26)
    ):
        w2(dst, src)

    for i in range(3):
        for dst in (29, 30):
            w2(dst, 2 * i, 16.0)
            w2(dst, 2 * i + 1, -16.0)

    w2(29, 26)
    p["b2"][29] = -15.0
    w2(30, 26, -1.0)
    p["b2"][30] = -14.0

    # F3:
    # 0:3 targets; 3:6 shape masks; 6:9 selected-colour evidence;
    # 9:18 paint data; 18:25 selection data;
    # 25:28 row-alignment statistics; 28:30 coordinates;
    # 30 hollow-selection indicator.
    for i in range(3):
        p["W3"][i, i] = 1.0

        p["W3"][3 + i, 3 + 2 * i] = 1.0
        p["W3"][3 + i, 4 + 2 * i] = -1.0
        p["W3"][3 + i, 9 + i] = 1.0
        p["W3"][3 + i, 23] = 1.0
        p["b3"][3 + i] = -1.0

        o = 6 + i
        p["W3"][o, 32 + 9 + i] = 1.0
        p["W3"][o, 64 + 9 + i] = 1.0
        p["W3"][o, 3 + 2 * i] = 2.0
        p["W3"][o, 23] = 64.0
        p["b3"][o] = -65.5

        for j in range(3):
            p["W3"][9 + 3 * i + j, 12 + 3 * i + j] = 1.0

    for dst, src in (
        (18, 21), (19, 22), (20, 24), (21, 23),
        (22, 25), (23, 26), (24, 27),
        (28, 28), (29, 31),
    ):
        p["W3"][dst, src] = 1.0

    p["W3"][30, 23] = 1.0
    p["W3"][30, 24] = -1.0

    rx = 32 + 29
    ri = 32 + 30

    p["W3"][25, rx] = 1.0
    p["W3"][25, ri] = 1.0
    p["b3"][25] = -3.0

    p["W3"][26, rx] = 1000.0
    p["W3"][26, ri] = 1000.0
    p["W3"][26, 28] = -1.0
    p["b3"][26] = -2999.0

    p["W3"][27, rx] = 1000.5
    p["W3"][27, ri] = 999.5
    p["b3"][27] = -2999.5

    # Two distance-transform groups. Colours 8 and 11 share one
    # auxiliary field; colour 9 has its own field.
    distance_groups = ((0, 2), (1,))
    for group, members in enumerate(distance_groups):
        a, b, acc = 3 * group, 3 * group + 1, 3 * group + 2

        for out in (a, b):
            for dy, dx in cardinal + ((0, 0),):
                p["Wh"][out, a, dy + 1, dx + 1] += 1.0
                p["Wh"][out, b, dy + 1, dx + 1] -= 1.0
            for i in members:
                p["Uh"][out, 3 + i] = 1.0
        p["bh"][b] = -1.0

        p["Wh"][acc, acc, 1, 1] = 1.0
        p["Wh"][acc, a, 1, 1] = 1.0
        p["Wh"][acc, b, 1, 1] = -1.0
        for i in members:
            p["Uh"][acc, i] = 128.0
        p["bh"][acc] = -128.0

    # Binary flood of colour-9 target presence.
    for out in (6, 7):
        for dy, dx in cardinal + ((0, 0),):
            p["Wh"][out, 6, dy + 1, dx + 1] = 1.0
            p["Wh"][out, 7, dy + 1, dx + 1] = -1.0
        p["Uh"][out, 1] = 1.0
    p["bh"][7] = -1.0

    # Prefer colour 9 for diagonal voting when it is present.
    # Otherwise use the auxiliary target group.
    p["Uh"][8, 0] = 1.0
    p["Uh"][8, 1] = 2.0
    p["Uh"][8, 2] = 1.0
    p["Wh"][8, 6, 1, 1] = -1.0
    p["Wh"][8, 7, 1, 1] = 1.0

    # Four diagonal rays, followed by a minimum of the two families.
    for j, (dy, dx) in enumerate(diagonal):
        r = 9 + j
        p["Wh"][r, r, dy + 1, dx + 1] = 1.0
        p["Wh"][r, 8, 1, 1] = 20.0
        p["bh"][r] = -1.0

    for r in (9, 10):
        p["Wh"][13, r, 1, 1] = 1.0
    for r in (11, 12):
        p["Wh"][13, r, 1, 1] = -1.0

    vote_scale = 10.0
    for out, coord in ((14, 28), (15, 29)):
        p["Wh"][out, 9, 1, 1] = vote_scale
        p["Wh"][out, 10, 1, 1] = vote_scale
        p["Wh"][out, 13, 1, 1] = -vote_scale
        p["Wh"][out, 8, 1, 1] = -38.0 * vote_scale
        p["Uh"][out, coord] = -1.0
        p["bh"][out] = 1.0

    def fm(c):
        return 32 + c

    def fav(c):
        return 64 + c

    nz = 0

    def zunit(weights=None, bias=0.0):
        nonlocal nz
        out = nz
        nz += 1
        if weights:
            for k, v in weights.items():
                p["Wz"][out, k] += v
        p["bz"][out] = bias
        return out

    cost_terms = {fav(i): 127.0 * 4096.0 for i in range(3)}
    cost_terms[16 + 2] = -4096.0
    cost_terms[16 + 5] = -4096.0

    z_cost = zunit(cost_terms)
    z_zero_cost = zunit({k: -v for k, v in cost_terms.items()}, 0.5)
    z_y = zunit({fm(18): 1.0})
    z_x = zunit({fm(19): 1.0})

    py, px, selected_bits, missing, counts = [], [], [], [], []
    for i in range(3):
        py.append(zunit({fm(10 + 3 * i): 1.0}))
        px.append(zunit({fm(11 + 3 * i): 1.0}))

        lo = zunit({fm(6 + i): 1.0})
        hi = zunit({fm(6 + i): 1.0}, -0.25)
        selected_bits.append((lo, hi))

        missing.append(zunit({
            fm(i): -1.0,
            fm(9 + 3 * i): -1.0,
        }, 2.0))
        counts.append(zunit({fav(i): 4096.0}))

    # Conditional count comparisons let twelve paint-direction units
    # handle both ordinary and hollow selections.
    paint_losses = [[] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            tie = 0.125 if j < i else -0.125

            paint_losses[i].append(zunit({
                fav(j): 4096.0,
                fav(i): -4096.0,
                fm(30): -1000.0,
            }, tie))

            paint_losses[i].append(zunit({
                fav(i): 4096.0,
                fav(j): -4096.0,
                fm(j): 1000.0,
                fm(30): 1000.0,
            }, tie - 2000.0))

    z_line_y = zunit({fm(25): 1000.0, fm(26): -1.0}, 1.0)
    z_line_x = zunit({fm(27): 1.0, fm(25): -1000.0})
    z_horizontal = zunit({fm(22): 1.0})
    z_diagonal = zunit({fm(23): 1.0})
    z_vertical = zunit({fm(24): 1.0})
    z_have9 = zunit({fm(1): 1.0})
    z_pads = zunit({fm(9 + 3 * i): 1.0 for i in range(3)})

    z_vote_y = zunit({14: 1.0})
    z_vote_x = zunit({15: 1.0})
    vote_digits = []
    for k in range(1, 33):
        lo = zunit({14: 1.0}, -(vote_scale * k - 0.5))
        hi = zunit({14: 1.0}, -(vote_scale * k - 0.25))
        vote_digits.append((lo, hi))

    # Recurrent controller. Comparison magnitude 2 uses the cost
    # from two observations ago, enabling compound diagonal trials.
    edges = [
        ("SW", "U0", 1, 0),
        ("U0", "UP", 1, +1),
        ("U0", "UU0", 2, -1),
        ("UP", "UP", 1, +1),
        ("UP", "UU1", 2, -1),
        ("UU0", "DN", 2, 0),
        ("UU1", "L0", 3, 0),
        ("DN", "DN", 2, +1),
        ("DN", "UD", 1, -1),
        ("UD", "L0", 3, 0),
        ("L0", "LP", 3, +1),
        ("L0", "UL0", 4, -1),
        ("LP", "LP", 3, +1),
        ("LP", "UL1", 4, -1),
        ("UL0", "RT", 4, 0),
        ("UL1", "DL1", 2, 0),
        ("RT", "RT", 4, +1),
        ("RT", "UR", 3, -1),
        ("UR", "DL1", 2, 0),
        ("DL1", "DL2", 3, 0),
        ("DL2", "DL1", 2, +2),
        ("DL2", "DUL1", 4, -2),
        ("DUL1", "DUL2", 1, 0),
        ("DUL2", "UR1", 1, 0),
        ("UR1", "UR2", 4, 0),
        ("UR2", "UR1", 1, +2),
        ("UR2", "DUU1", 3, -2),
        ("DUU1", "DUU2", 2, 0),
        ("DUU2", "SW", 5, 0),
    ]

    emergency = len(edges)
    edge_count = len(edges) + 1
    edge_base = 2
    live = edge_base + edge_count
    next_mem = live + 1

    epsilon = 2.0 ** -22
    gain = 1024.0
    phase_gate = 2.0 ** 24

    p["Wmz"][0, z_cost] = epsilon
    p["Wm"][1, 0] = 1.0
    p["bm"][live] = 20.0

    incoming = {}
    for e, (_, dst, _, _) in enumerate(edges):
        incoming.setdefault(dst, []).append(e)
    incoming.setdefault("U0", []).append(emergency)

    def phase_input(row, phase, multiplier):
        arrivals = incoming.get(phase, ())
        for e in arrivals:
            p["Wm"][row, edge_base + e] += 0.5 * multiplier
        p["Wm"][row, live] += 0.5 * len(arrivals) * multiplier
        if phase == "SW":
            p["bm"][row] += multiplier
            p["Wm"][row, live] -= multiplier

    for e, (source, _, action, branch) in enumerate(edges):
        row = edge_base + e
        if branch:
            sign = 1.0 if branch > 0 else -1.0
            old_cost = 0 if abs(branch) == 1 else 1

            phase_input(row, source, phase_gate)
            p["bm"][row] -= phase_gate
            p["Wmz"][row, z_cost] -= sign * gain
            p["Wm"][row, old_cost] += sign * gain / epsilon
            p["bm"][row] -= sign * gain * 0.5
        else:
            phase_input(row, source, 2.0 * phase_gate)
            p["bm"][row] -= phase_gate

    row = edge_base + emergency
    for e in range(edge_count):
        p["Wm"][row, edge_base + e] = -20.0
    p["Wm"][row, live] = 20.0 * (5.0 - edge_count)
    p["bm"][row] = -80.0

    # A selection change always restarts the ordinary search.
    for e in range(edge_count):
        row = edge_base + e
        p["Wma"][row, 5] += (
            4.0 * phase_gate if e == 0 else -4.0 * phase_gate
        )

    p["bo"][0] = -1.0e6
    p["bo"][6] = -1.0e6
    p["bo"][7] = -1.0e6
    p["bo"][5] = 0.001
    p["Wo"][5, z_zero_cost] = 200.0

    def output_unit(action, memory_index, strength):
        p["Wo"][action, 128 + memory_index] += strength
        p["bo"][action] += strength

    for e, (_, _, action, _) in enumerate(edges):
        output_unit(action, edge_base + e, 12.5)
    output_unit(1, edge_base + emergency, 12.5)

    # Remember the episode's goal palette.
    presence = list(range(next_mem, next_mem + 3))
    valid = list(range(next_mem + 3, next_mem + 6))
    next_mem += 6

    for i in range(3):
        r = presence[i]
        p["Wm"][r, r] = 1000.0
        p["Wmz"][r, counts[i]] = 2.0
        p["bm"][r] = -1.0

        r = valid[i]
        lo, hi = selected_bits[i]
        p["Wmz"][r, lo] = 400.0
        p["Wmz"][r, hi] = -400.0
        p["Wm"][r, presence[i]] = 100.0
        p["bm"][r] = -150.0

    # Paint navigation. Horizontal motion has priority so that a
    # shape first leaves an edge before travelling along that edge.
    inhibitor = 2000.0
    direction_gain = 256.0
    tolerance = 2.0 / 63.0
    paint_units = []

    for i in range(3):
        for action in (1, 2, 3, 4):
            row = next_mem
            next_mem += 1
            paint_units.append(row)

            if action == 1:
                positive, negative = z_y, py[i]
            elif action == 2:
                positive, negative = py[i], z_y
            elif action == 3:
                positive, negative = z_x, px[i]
            else:
                positive, negative = px[i], z_x

            p["Wmz"][row, positive] += direction_gain
            p["Wmz"][row, negative] -= direction_gain
            p["bm"][row] -= direction_gain * tolerance

            p["Wmz"][row, missing[i]] -= inhibitor
            p["Wm"][row, presence[i]] += 0.5 * inhibitor
            p["bm"][row] -= 0.5 * inhibitor

            for loss in paint_losses[i]:
                p["Wmz"][row, loss] -= inhibitor

            for r in valid:
                p["Wm"][row, r] -= 0.5 * inhibitor
                p["bm"][row] -= 0.5 * inhibitor

            strength = 140.0 if action in (3, 4) else 100.0
            output_unit(action, row, strength)

    # While a paint override is active, hold the ordinary controller
    # at its first trial instead of advancing an unrelated phase.
    force = 4.0 * phase_gate
    for e in range(edge_count):
        row = edge_base + e
        sign = 1.0 if e == 0 else -1.0
        for r in paint_units:
            p["Wm"][row, r] += sign * 0.5 * force
        p["Wm"][row, live] += (
            sign * 0.5 * force * len(paint_units)
        )

    # Direct alignment for horizontal lines and diagonal crosses.
    for kind in ("line", "diagonal"):
        for action in (1, 2, 3, 4, 5):
            row = next_mem
            next_mem += 1
            gate = 5000.0

            if kind == "line":
                p["Wmz"][row, z_horizontal] += gate
                p["Wmz"][row, z_vertical] -= gate
                p["Wmz"][row, z_diagonal] -= gate
                p["Wmz"][row, z_pads] -= gate
            else:
                p["Wmz"][row, z_diagonal] += gate
                p["Wmz"][row, z_horizontal] -= gate
                p["Wmz"][row, z_vertical] -= gate

                # Use the preferred-colour voting field only when
                # its colour is compatible with the selected cross.
                p["Wmz"][row, z_have9] -= gate
                lo, hi = selected_bits[1]
                p["Wmz"][row, lo] += 4.0 * gate
                p["Wmz"][row, hi] -= 4.0 * gate

            p["bm"][row] -= gate

            # Require a selected colour from the remembered palette.
            for r in valid:
                p["Wm"][row, r] += 0.5 * gate
            p["Wm"][row, live] += 1.5 * gate
            p["bm"][row] -= gate

            if action == 5:
                p["bm"][row] += 20.0
                output_unit(action, row, 35.0)
                continue

            vertical = action in (1, 2)
            sign = 1.0 if action in (1, 3) else -1.0
            current = z_y if vertical else z_x
            p["Wmz"][row, current] += sign * direction_gain

            if kind == "line":
                goal = z_line_y if vertical else z_line_x
                p["Wmz"][row, goal] -= sign * direction_gain
            else:
                encoded = z_vote_y if vertical else z_vote_x
                p["Wmz"][row, encoded] += sign * direction_gain
                p["bm"][row] -= sign * direction_gain

                for lo, hi in vote_digits:
                    v = sign * direction_gain * vote_scale * 4.0
                    p["Wmz"][row, lo] -= v
                    p["Wmz"][row, hi] += v

            # Snap to the reachable three-pixel lattice.
            p["bm"][row] -= direction_gain * (1.75 / 63.0)
            output_unit(action, row, 100.0 if vertical else 140.0)

    return p
