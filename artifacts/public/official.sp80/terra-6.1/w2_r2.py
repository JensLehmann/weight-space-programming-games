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

    colours = (4, 6, 9, 11, 8)
    for i, colour in enumerate(colours):
        p["W1"][i, colour, 2, 2] = 1.0

    p["W1"][5, 16, 2, 2] = 1.0
    p["W1"][6, 16, 2, 2] = -1.0
    p["b1"][6] = 1.0
    p["W1"][7, 17, 2, 2] = 1.0
    p["W1"][8, 17, 2, 2] = -1.0
    p["b1"][8] = 1.0

    # Detect a source at display scale four.
    p["W1"][9, 6, 2, 1:5] = 1.0
    p["b1"][9] = -3.5

    for i in range(10):
        p["W2"][i, i, 1, 1] = 1.0

    directions = ("row", "inv_row", "col", "inv_col")
    channel = {}
    for i, colour in enumerate(colours):
        for j, direction in enumerate(directions):
            out = 4 * i + j
            channel[colour, direction] = out
            p["W3"][out, i] = 1.0
            p["W3"][out, 5 + j] = 1.0
            p["b3"][out] = -1.0

    p["W3"][20, 1] = 1.0
    p["W3"][21, 4] = 1.0
    p["W3"][22, 4] = 1.0
    p["W3"][22, 5] = 1.0
    p["b3"][22] = -1.0
    p["W3"][24, 9] = 2.0

    # Click an unselected object, preferring the lowest display row.
    p["Wc"][0, 21] = 2.0
    p["Wc"][0, 22] = 1.0

    def fmax(colour, direction):
        return 32 + channel[colour, direction]

    def centre_difference(a, b, axis):
        inv = "inv_" + axis
        v = np.zeros(96, dtype=np.float32)
        v[fmax(a, axis)] += 0.5
        v[fmax(a, inv)] -= 0.5
        v[fmax(b, axis)] -= 0.5
        v[fmax(b, inv)] += 0.5
        return v

    # Classify layouts by the displayed area of colour-6 sources.
    scale_feature = 32 + 24
    source_mean = 64 + 20
    score = np.zeros(96, dtype=np.float32)
    score[source_mean] = 1000.0
    score[scale_feature] = -(17.5 * 1000.0 / 4096.0)
    score_bias = -(22.5 * 1000.0 / 4096.0)
    p["Wz"][0] = score
    p["bz"][0] = score_bias
    p["Wz"][1] = -score
    p["bz"][1] = -score_bias

    next_z = 2

    def signed_expr(v, bias=0.0):
        nonlocal next_z
        pos, neg = next_z, next_z + 1
        next_z += 2
        p["Wz"][pos] = v
        p["bz"][pos] = bias
        p["Wz"][neg] = -v
        p["bz"][neg] = -bias
        return pos, neg

    # Funnel bar targets relative to the two extreme sources.
    e0 = np.zeros(96, dtype=np.float32)
    e0[fmax(6, "col")] = 1.0
    e0[fmax(9, "col")] = -1.0
    e0[scale_feature] = 1.0 / 63.0
    z_e0 = signed_expr(e0, 3.0 / 63.0)

    e1 = np.zeros(96, dtype=np.float32)
    e1[fmax(9, "inv_col")] = 1.0
    e1[fmax(6, "inv_col")] = -1.0
    e1[scale_feature] = -1.0 / 63.0
    z_e1 = signed_expr(e1, -3.0 / 63.0)

    e2 = np.zeros(96, dtype=np.float32)
    e2[fmax(6, "col")] = 1.0
    e2[fmax(9, "col")] = -1.0
    e2[scale_feature] = -2.0 / 63.0
    z_e2 = signed_expr(e2, -6.0 / 63.0)

    e3 = np.zeros(96, dtype=np.float32)
    e3[fmax(9, "inv_col")] = 1.0
    e3[fmax(6, "inv_col")] = -1.0
    e3[scale_feature] = 2.0 / 63.0
    z_e3 = signed_expr(e3, 6.0 / 63.0)

    # Single-source bounding-box alignment, gated by flow axis.
    single_pairs = {}
    for move_axis, flow_axis in (("col", "row"), ("row", "col")):
        error = centre_difference(11, 9, move_axis)
        flow = centre_difference(4, 6, flow_axis)
        pairs = []
        for sign in (-1.0, 1.0):
            with_error = next_z
            gate_only = next_z + 1
            next_z += 2
            p["Wz"][with_error] = error + 100.0 * sign * flow
            p["Wz"][gate_only] = 100.0 * sign * flow
            p["bz"][with_error] = -1.5
            p["bz"][gate_only] = -1.5
            pairs.append((with_error, gate_only))
        single_pairs[move_axis] = pairs

    # Persistent bits for the first, second and third selections.
    p["Wm"][0, 0] = 10.0
    p["Wma"][0, 6] = 10.0

    p["Wm"][1, 1] = 10.0
    p["Wm"][1, 0] = 20.0
    p["Wma"][1, 6] = 20.0
    p["bm"][1] = -15.0

    p["Wm"][2, 2] = 10.0
    p["Wm"][2, 1] = 20.0
    p["Wma"][2, 6] = 20.0
    p["bm"][2] = -15.0

    # Positive for the single-source rule or after the third selection.
    p["Wm"][4, 2] = 10.0
    p["Wmz"][4, 1] = 10.0
    p["bm"][4] = -5.0

    def staged_signal(mem_index, z_pair, stage):
        signal = mem_index
        baseline = mem_index + 1

        for k in (signal, baseline):
            p["Wmz"][k, 1] = 10.0
            if stage == 0:
                p["Wm"][k, 0] = 30.0
            elif stage == 1:
                p["Wm"][k, 0] = -30.0
                p["Wm"][k, 1] = 15.0
                p["bm"][k] = 45.0
            elif stage == 2:
                p["Wm"][k, 1] = -15.0
                p["Wm"][k, 2] = 15.0
                p["bm"][k] = 30.0
            else:
                p["Wm"][k, 2] = -15.0
                p["bm"][k] = 15.0

        p["Wmz"][signal, z_pair[0]] += 5.0
        p["Wmz"][signal, z_pair[1]] -= 5.0
        return signal, baseline

    stages = [
        staged_signal(5, z_e0, 0),
        staged_signal(7, z_e1, 1),
        staged_signal(9, z_e2, 2),
        staged_signal(11, z_e3, 3),
    ]

    def single_signal(mem_index, pairs):
        signal = mem_index
        baseline = mem_index + 1
        for k in (signal, baseline):
            p["Wmz"][k, 0] = 10.0
        for with_error, gate_only in pairs:
            p["Wmz"][signal, with_error] += 5.0
            p["Wmz"][signal, gate_only] -= 5.0
        return signal, baseline

    single_col = single_signal(13, single_pairs["col"])
    single_row = single_signal(15, single_pairs["row"])

    def add_direction(pair, positive_action, negative_action):
        signal, baseline = pair
        p["Wo"][positive_action, 128 + signal] += 1.0
        p["Wo"][positive_action, 128 + baseline] -= 1.0
        p["Wo"][negative_action, 128 + signal] -= 1.0
        p["Wo"][negative_action, 128 + baseline] += 1.0

    for pair in stages:
        add_direction(pair, 4, 3)
    add_direction(single_col, 4, 3)
    add_direction(single_row, 2, 1)

    p["bo"][5] = -0.35
    p["Wo"][5, 128 + 4] = 0.5
    p["bo"][6] = -0.35
    p["Wo"][6, 128 + 4] = -0.5

    # Give the selection bits one step to update after a click.
    p["Wma"][17, 6] = 10.0
    p["Wo"][2, 128 + 17] = 2.0

    return p
