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
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    colors = [6, 7, 8, 9, 10, 11, 12, 14]
    target_colors = [8, 9, 11]
    color_codes = [3, 4, 6]
    cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    neighbors = cardinal + diagonal

    def w1(o, c, dy=0, dx=0, value=1.0):
        p["W1"][o, c, 2 + dy, 2 + dx] += value

    def w2(o, c, value=1.0):
        p["W2"][o, c, 1, 1] += value

    def w3(o, c, value=1.0, projection=0):
        p["W3"][o, c + 32 * projection] += value

    def recurrent(o, c, dy=0, dx=0, value=1.0):
        p["Wh"][o, c, 1 + dy, 1 + dx] += value

    # First layer: palette, selection, local arms, and marker rings.
    for c in range(6, 15):
        w1(0, c)
    w1(1, 0)
    w1(2, 5)
    w1(2, 0)

    for c in range(6, 15):
        for dy, dx in [(0, -1), (0, 1)]:
            w1(3, c, dy, dx, 0.5)
            w1(7, c, dy, dx)
        for dy, dx in [(-1, 0), (1, 0)]:
            w1(4, c, dy, dx, 0.5)
            w1(8, c, dy, dx)
        for dy, dx in diagonal:
            w1(5, c, dy, dx)
            w1(6, c, dy, dx)
            w1(9, c, dy, dx)

    p["b1"][6] = -1.0
    w1(7, 0, value=2.0)
    w1(8, 0, value=2.0)
    w1(9, 0, value=4.0)
    p["b1"][7:10] = [-3.0, -3.0, -7.0]

    for dy, dx in cardinal:
        w1(10, 4, dy, dx)
    for dy, dx in diagonal:
        w1(11, 4, dy, dx)
    p["b1"][10:12] = -3.0

    w1(12, 16)
    w1(13, 17)

    color_plane = {}
    for i, c in enumerate(colors):
        color_plane[c] = 14 + i
        w1(14 + i, c)
        w1(22, c, value=(i + 1) / 8.0)
    w1(22, 13)

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            w1(23, 2, dy, dx)
            w1(24, 2, dy, dx)
    p["b1"][23:25] = [-2.0, -3.0]

    # Second layer: geometry and color-specific marker evidence.
    w2(0, 0)
    w2(1, 1)
    for c in [7, 8, 9]:
        w2(1, c, -1.0)
    w2(2, 9)
    w2(4, 2)

    for o, support in [(5, 3), (6, 4)]:
        w2(o, support)
        w2(o, 0)
        p["b2"][o] = -1.0

    w2(7, 7)
    w2(8, 8)
    w2(9, 22)

    baselines = [3, 16, 17]
    for j, c in enumerate(target_colors):
        # The difference of these hinges recognizes either intact ring.
        for o, bias in [(10 + j, -2.0),
                        (baselines[j], -3.0)]:
            w2(o, 10)
            w2(o, 11)
            w2(o, color_plane[c], 2.0)
            p["b2"][o] = bias

        w2(13 + j, 23)
        w2(13 + j, 24, -1.0)
        w2(13 + j, color_plane[c])
        p["b2"][13 + j] = -1.0

    # Coordinate moments used by axial geometric guidance.
    for j, c in enumerate(target_colors):
        for k, coord, sign, offset in [
            (0, 12, 1.0, 0.0),
            (1, 12, -1.0, 1.0),
            (2, 13, 1.0, 0.0),
            (3, 13, -1.0, 1.0),
        ]:
            o = 18 + 4 * j + k
            w2(o, coord, sign)
            w2(o, 11, 2.0)
            w2(o, color_plane[c], 2.0)
            p["b2"][o] = offset - 4.0

    for o, coord in [(30, 12), (31, 13)]:
        w2(o, coord)
        w2(o, 1, 2.0)
        p["b2"][o] = -2.0

    # Axial layer.
    for o in [0, 1, 2, 4]:
        w3(o, o)

    # A nonselected row/column gives zero, including at line interiors.
    # A selected row/column includes endpoints as well as interiors.
    w3(5, 5, 2.0)
    w3(5, 7, 2.0, projection=1)
    p["b3"][5] = -2.0

    w3(6, 6, 2.0)
    w3(6, 8, 2.0, projection=2)
    p["b3"][6] = -2.0
    w3(7, 9)

    bare_channels = [3, 14, 15]
    for j in range(3):
        w3(8 + j, 10 + j)
        w3(8 + j, baselines[j], -1.0)
        w3(11 + j, 13 + j)
        w3(bare_channels[j], baselines[j])

    # Widest matching target row and column.
    span_gain = 32.0
    for j in range(3):
        row, invrow, col, invcol = [18 + 4 * j + k for k in range(4)]
        base = 16 + 4 * j

        for o in [base, base + 1]:
            w3(o, col, span_gain, projection=1)
            w3(o, invcol, span_gain, projection=1)
        w3(base, row, projection=1)
        w3(base + 1, row, -1.0, projection=1)
        p["b3"][base:base + 2] = [-span_gain, 1.0 - span_gain]

        for o in [base + 2, base + 3]:
            w3(o, row, span_gain, projection=2)
            w3(o, invrow, span_gain, projection=2)
        w3(base + 2, col, projection=2)
        w3(base + 3, col, -1.0, projection=2)
        p["b3"][base + 2:base + 4] = [-span_gain, 1.0 - span_gain]

    w3(28, 30)
    w3(29, 31)
    w3(30, 7)
    w3(31, 8)

    # Recurrent spatial block.
    # Differences of adjacent channel pairs are capped binary floods.

    # h0-h1: selected closed-shape interior.
    for o in [0, 1]:
        for dy, dx in cardinal + [(0, 0)]:
            recurrent(o, 0, dy, dx)
            recurrent(o, 1, dy, dx, -1.0)
        p["Uh"][o, 1] = 1.0
        p["Uh"][o, 4] = 8.0
        p["bh"][o] = -8.0
    p["bh"][1] -= 1.0

    # h2-h3: selected diagonal component.
    for o in [2, 3]:
        for dy, dx in diagonal + [(0, 0)]:
            recurrent(o, 2, dy, dx)
            recurrent(o, 3, dy, dx, -1.0)
        p["Uh"][o, 0] = 8.0
        p["Uh"][o, 2] = 9.0
        p["bh"][o] = -8.0
    p["bh"][3] -= 1.0

    # h4-h5: selected outline.
    for o in [4, 5]:
        recurrent(o, 2)
        recurrent(o, 3, value=-1.0)
        for dy, dx in cardinal:
            recurrent(o, 0, dy, dx)
            recurrent(o, 1, dy, dx, -1.0)
        p["Uh"][o, 0] = 8.0
        p["Uh"][o, 5] = 1.0
        p["Uh"][o, 6] = 1.0
        p["bh"][o] = -8.0
    p["bh"][5] -= 1.0

    # h6-h7: unrestricted wavefront from that outline.
    for o in [6, 7]:
        for dy, dx in neighbors + [(0, 0)]:
            recurrent(o, 6, dy, dx)
            recurrent(o, 7, dy, dx, -1.0)
        recurrent(o, 4)
        recurrent(o, 5, value=-1.0)
    p["bh"][7] = -1.0

    # h8: arrival-time integral.
    recurrent(8, 8)
    recurrent(8, 6)
    recurrent(8, 7, value=-1.0)

    # h9..11: target potentials, excluding other sprites' coverage.
    for j in range(3):
        o = 9 + j
        recurrent(o, 8)
        recurrent(o, 4, value=408.0)
        recurrent(o, 5, value=-408.0)
        p["Uh"][o, 8 + j] = 400.0
        p["Uh"][o, bare_channels[j]] = 400.0
        p["bh"][o] = -800.0

    # h12..14: paint-station potentials.
    for j in range(3):
        o = 12 + j
        recurrent(o, 8)
        p["Uh"][o, 11 + j] = 400.0
        p["bh"][o] = -400.0

    # h15: selected outline's palette code.
    recurrent(15, 4)
    recurrent(15, 5, value=-1.0)
    p["Uh"][15, 7] = 1.0
    p["bh"][15] = -1.0

    # Pooled potentials are kept small enough for nearly linear tanh storage.
    potential_scale = 1.0e-5
    for i in range(6):
        p["Wz"][i, 16 + 9 + i] = potential_scale

    hats = []
    for j, code in enumerate(color_codes):
        units = [6 + 3 * j + k for k in range(3)]
        hats.append(units)
        for unit, threshold in zip(units, [code - 1, code, code + 1]):
            p["Wz"][unit, 15] = 8.0
            p["bz"][unit] = -float(threshold)

    p["Wz"][15, 32 + 30] = 1.0
    p["Wz"][16, 32 + 31] = 1.0
    p["Wz"][17, 32 + 28] = 1.0
    p["Wz"][18, 32 + 29] = 1.0
    for k in range(12):
        p["Wz"][19 + k, 32 + 16 + k] = 1.0

    # Pairwise clipped rankings choose a repainting destination.
    rank_units = {}
    unit = 31
    for c in range(3):
        for other in range(3):
            if c == other:
                continue
            rank_units[c, other] = (unit, unit + 1)
            for u in [unit, unit + 1]:
                p["Wz"][u, 16 + 9 + other] = 1.0e8 * potential_scale
                p["Wz"][u, 16 + 9 + c] = -1.0e8 * potential_scale
            p["bz"][unit + 1] = -10.0
            unit += 2

    for j in range(3):
        p["Wz"][55 + j, 32 + 8 + j] = 1.0

    # Memory controller.
    energies = list(range(6))
    trial_states = list(range(6, 10))
    advance_states = list(range(10, 14))
    switch_state = 14
    goals = [16, 17, 18]
    paint_latches = [55, 56, 57, 58]

    directions = [1, 2, 3, 4]
    reverses = [2, 1, 4, 3]
    phase_gain = 20.0

    for i in range(6):
        p["Wmz"][energies[i], i] = 1.0

    for i, action in enumerate(directions):
        p["Wm"][trial_states[i], trial_states[i]] = phase_gain / 2.0
        p["Wm"][trial_states[i], advance_states[i]] = phase_gain / 2.0
        p["Wma"][trial_states[i], action] = phase_gain
        p["bm"][trial_states[i]] = -phase_gain / 2.0

    p["Wma"][advance_states[0], :] = -phase_gain
    p["Wma"][advance_states[0], 5] = 0.0
    p["bm"][advance_states[0]] = phase_gain / 2.0

    for i in range(1, 4):
        previous = i - 1
        m = advance_states[i]
        p["Wm"][m, trial_states[previous]] = phase_gain / 2.0
        p["Wm"][m, advance_states[previous]] = phase_gain / 2.0
        p["Wma"][m, reverses[previous]] = phase_gain
        p["bm"][m] = -phase_gain / 2.0

    p["Wm"][switch_state, trial_states[3]] = phase_gain / 2.0
    p["Wm"][switch_state, advance_states[3]] = phase_gain / 2.0
    p["Wma"][switch_state, reverses[3]] = phase_gain
    p["bm"][switch_state] = -phase_gain / 2.0

    def add_hat(memory_unit, color_index, coefficient):
        for u, a in zip(hats[color_index], [1.0, -2.0, 1.0]):
            p["Wmz"][memory_unit, u] += coefficient * a

    # A painting latch persists only while its selected color persists.
    for c in range(3):
        m = paint_latches[c]
        add_hat(m, c, 40.0)
        p["Wmz"][m, 55 + c] = -20.0
        p["Wm"][m, m] = 10.0
        p["Wma"][m, 5] = -50.0
        p["bm"][m] = -25.0

    m = paint_latches[3]
    for c in range(3):
        add_hat(m, c, -40.0)
    p["Wm"][m, m] = 10.0
    p["Wma"][m, 5] = -50.0
    p["bm"][m] = 15.0

    for c in range(3):
        p["bm"][goals[c]] = 10.0
        for other in range(3):
            if c != other:
                u, v = rank_units[c, other]
                p["Wmz"][goals[c], u] -= 2.0
                p["Wmz"][goals[c], v] += 2.0

    gate = 4096.0
    comparison_gain = 1.0e8
    normal, painting, guides = {}, {}, {}

    def trial_condition(m, phase):
        p["Wm"][m, trial_states[phase]] += gate / 2.0
        p["Wm"][m, advance_states[phase]] += gate / 2.0
        p["Wma"][m, directions[phase]] += gate

    for phase in range(4):
        line_type = 15 if phase < 2 else 16
        for c in range(3):
            n = 19 + 3 * phase + c
            normal[phase, c] = n
            trial_condition(n, phase)
            p["Wm"][n, energies[c]] -= comparison_gain
            p["Wmz"][n, c] += comparison_gain
            add_hat(n, c, gate)
            p["Wmz"][n, 55 + c] += gate
            p["Wmz"][n, line_type] -= gate
            p["bm"][n] = -3.0 * gate - 0.01

            q = 31 + 3 * phase + c
            painting[phase, c] = q
            trial_condition(q, phase)
            p["Wm"][q, energies[3 + c]] -= comparison_gain
            p["Wmz"][q, 3 + c] += comparison_gain
            for latch in paint_latches:
                p["Wm"][q, latch] += gate / 2.0
            p["Wm"][q, goals[c]] += gate / 2.0
            p["bm"][q] = -0.5 * gate - 0.01

            f = 43 + 3 * phase + c
            guides[phase, c] = f
            trial_condition(f, phase)
            add_hat(f, c, gate)
            p["Wmz"][f, line_type] += gate
            p["Wmz"][f, 55 + c] += gate

            sign = 1.0 if phase in [0, 2] else -1.0
            strength = 50.0
            center_unit = 17 if phase < 2 else 18
            pair_offset = 0 if phase < 2 else 2
            plus_unit = 19 + 4 * c + pair_offset
            minus_unit = plus_unit + 1

            p["Wmz"][f, center_unit] += sign * strength
            p["Wmz"][f, plus_unit] -= sign * strength / 2.0
            p["Wmz"][f, minus_unit] += sign * strength / 2.0
            p["bm"][f] = (
                -4.0 * gate
                - sign * strength / 2.0
                + 0.12 * strength / 63.0
            )

    # Action decoder: continue accepted trials, undo rejected trials,
    # and advance to the next direction after undoing.
    p["bo"][:] = -100.0
    p["bo"][1:6] = 0.0

    def output_memory(action, memory_unit, coefficient):
        p["Wo"][action, 128 + memory_unit] += coefficient

    for phase in range(4):
        forward = directions[phase]
        backward = reverses[phase]
        accepted = (
            [normal[phase, c] for c in range(3)]
            + [painting[phase, c] for c in range(3)]
            + [guides[phase, c] for c in range(3)]
        )
        output_memory(forward, advance_states[phase], 5.0)
        for m in accepted:
            output_memory(forward, m, 5.0)
        p["bo"][forward] += 50.0

        output_memory(backward, trial_states[phase], 5.0)
        for m in accepted:
            output_memory(backward, m, -5.0)
        p["bo"][backward] -= 40.0

    output_memory(5, switch_state, 5.0)
    p["bo"][5] = 5.0
    return p
