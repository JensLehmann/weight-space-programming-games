import numpy as np


def build():
    graph = []
    params = {}
    serial = 0

    def node(name, op, inputs=None, **fields):
        d = {"id": name, "op": op}
        if inputs is not None:
            d["in"] = inputs
        d.update(fields)
        graph.append(d)
        return name

    def param(value, prefix="p"):
        nonlocal serial
        serial += 1
        name = prefix + str(serial)
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def conv(name, inp, w, b=None):
        fields = {"w": param(w, "w")}
        if b is not None:
            fields["b"] = param(b, "b")
        return node(name, "conv", [inp], **fields)

    def dense(name, inp, w, b=None):
        fields = {"w": param(w, "w")}
        if b is not None:
            fields["b"] = param(b, "b")
        return node(name, "dense", [inp], **fields)

    def sp_scale(name, inp, scale, bias=0.0, channels=1):
        w = np.zeros((channels, channels, 1, 1), np.float32)
        for i in range(channels):
            w[i, i, 0, 0] = scale
        return conv(name, inp, w, np.full(channels, bias, np.float32))

    def vscale(name, inp, scale, bias=0.0, size=1):
        return dense(
            name, inp, np.eye(size, dtype=np.float32) * scale,
            np.full(size, bias, np.float32)
        )

    def complement(name, inp, size=1):
        return vscale(name, inp, -1.0, 1.0, size)

    def vclip(name, inp, size=1):
        node(name + "_p", "vrelu", [inp])
        complement(name + "_gap", name + "_p", size)
        node(name + "_gap_p", "vrelu", [name + "_gap"])
        return complement(name, name + "_gap_p", size)

    def vmuls(name, inputs):
        if len(inputs) == 1:
            return inputs[0]
        cur = inputs[0]
        for i, other in enumerate(inputs[1:]):
            out = name if i == len(inputs) - 2 else name + "_" + str(i)
            node(out, "vmul", [cur, other])
            cur = out
        return name

    def sum_channels(name, inp, count):
        return conv(name, inp, np.ones((1, count, 1, 1), np.float32))

    def max4(prefix, inp):
        for i in range(4):
            node(prefix + "_s" + str(i), "slice", [inp], start=i, stop=i + 1)
        node(prefix + "_a", "max2", [prefix + "_s0", prefix + "_s1"])
        node(prefix + "_b", "max2", [prefix + "_s2", prefix + "_s3"])
        node(prefix + "_map", "max2", [prefix + "_a", prefix + "_b"])
        node(prefix, "gmax", [prefix + "_map"])
        return prefix

    def choose_direction(prefix, vector, anchor):
        dense(
            prefix + "_rank", vector, np.eye(4, dtype=np.float32),
            np.array([0.00003, 0.00002, 0.00001, 0.0], np.float32)
        )
        node(prefix + "_rank_map", "mul", [prefix + "_rank", anchor])
        max4(prefix + "_best", prefix + "_rank_map")
        vscale(prefix + "_neg", prefix + "_best", -1.0)
        node(prefix + "_diff", "vadd", [prefix + "_rank", prefix + "_neg"])
        vscale(prefix + "_eps", prefix + "_diff", 1000000.0, 1.0, 4)
        return vclip(prefix + "_choice", prefix + "_eps", 4)

    sizes = [(3, 3), (6, 3), (3, 6), (6, 6)]
    directions = [(-3, 0), (3, 0), (0, -3), (0, 3)]

    # Detect piece borders and centres.
    w = np.zeros((1, 16, 1, 1), np.float32)
    w[0, 0, 0, 0] = 1
    w[0, 14, 0, 0] = 1
    conv("border", "x0", w)

    w = np.zeros((1, 16, 1, 1), np.float32)
    for c in (0, 4, 5):
        w[0, c, 0, 0] = 1
    conv("centres", "x0", w)
    node("piece_input", "concat", ["border", "centres"])

    w = np.zeros((4, 2, 7, 7), np.float32)
    b = np.zeros(4, np.float32)
    for t, (h, width) in enumerate(sizes):
        ay = 1 if h == 3 else 2
        ax = 1 if width == 3 else 2
        ys = {1} if h == 3 else {2, 3}
        xs = {1} if width == 3 else {2, 3}
        for y in range(h):
            for x in range(width):
                c = 1 if y in ys and x in xs else 0
                w[t, c, 3 + y - ay, 3 + x - ax] = 2
        b[t] = 1 - 2 * h * width
    conv("piece_score", "piece_input", w, b)
    node("pieces", "relu", ["piece_score"])
    node("black", "slice", ["x0"], start=0, stop=1)
    node("selected", "mul", ["pieces", "black"])
    node("selected_types", "gmax", ["selected"])
    sum_channels("selected_anchor", "selected", 4)
    sum_channels("all_anchors", "pieces", 4)
    node("piece3", "slice", ["pieces"], start=0, stop=1)
    node("selected3", "slice", ["selected"], start=0, stop=1)
    node("other3_raw", "sub", ["piece3", "selected3"])
    node("other3", "relu", ["other3_raw"])
    node("other3_exists", "gmax", ["other3"])
    node("selected_is3", "vslice", ["selected_types"], start=0, stop=1)

    # Detect partly occluded rectangular outlines.
    w = np.zeros((2, 16, 1, 1), np.float32)
    for c in (0, 4, 5, 14):
        w[0, c, 0, 0] = 1
    w[1, 4, 0, 0] = 1
    conv("outline_input", "x0", w)

    w = np.zeros((8, 2, 3, 3), np.float32)
    shifts = ((0, 0), (1, 0), (0, 1), (1, 1))
    for kind in range(2):
        for j, (dy, dx) in enumerate(shifts):
            w[4 * kind + j, kind, 1 + dy, 1 + dx] = 1
    conv("outline_planes", "outline_input", w)

    w = np.zeros((8, 8, 7, 7), np.float32)
    counts = []
    for t, (h, width) in enumerate(sizes):
        gh, gw = h + 2, width + 2
        count = 0
        for y in range(gh):
            for x in range(gw):
                if y not in (0, gh - 1) and x not in (0, gw - 1):
                    continue
                dy, dx = y - 3, x - 3
                sy, sx = int(dy == 4), int(dx == 4)
                plane = 2 * sy + sx
                for kind in range(2):
                    w[4 * kind + t, 4 * kind + plane,
                      3 + dy - sy, 3 + dx - sx] += 1
                count += 1
        counts.append(count)
    conv("outline_counts", "outline_planes", w)
    node("allowed_counts", "slice", ["outline_counts"], start=0, stop=4)
    node("visible_counts", "slice", ["outline_counts"], start=4, stop=8)

    ident4 = np.eye(4, dtype=np.float32)[:, :, None, None]
    conv("allowed_test", "allowed_counts", ident4,
         np.array([1 - c for c in counts], np.float32))
    conv("visible_test", "visible_counts", ident4,
         np.array([6 - c for c in counts], np.float32))
    node("allowed_ok", "relu", ["allowed_test"])
    node("visible_ok", "relu", ["visible_test"])
    node("outline_valid0", "min2", ["allowed_ok", "visible_ok"])
    sp_scale("outline_cap", "outline_valid0", 0.0, 1.0, 4)
    node("outline_valid", "min2", ["outline_valid0", "outline_cap"])

    w = np.zeros((4, 4, 3, 3), np.float32)
    for t, (h, width) in enumerate(sizes):
        w[t, t, 1 + int(h == 3), 1 + int(width == 3)] = 1
    conv("candidate_goals", "outline_valid", w)
    conv("candidate_visible", "visible_counts", w)

    node("row_coord", "slice", ["coords"], start=0, stop=1)
    node("col_coord", "slice", ["coords"], start=1, stop=2)
    node("coord_pair", "concat", ["row_coord", "col_coord"])
    w = np.zeros((1, 2, 1, 1), np.float32)
    w[0, 0, 0, 0] = -0.00984375
    w[0, 1, 0, 0] = -0.00015380859375
    conv("tie_break", "coord_pair", w, np.array([0.01], np.float32))
    sp_scale("visible_rank", "candidate_visible", 0.1, 1.0, 4)
    node("candidate_rank0", "add", ["visible_rank", "tie_break"])
    node("candidate_rank", "mul", ["candidate_rank0", "candidate_goals"])

    goal_parts = []
    remaining = "candidate_goals"
    suppress = np.zeros((4, 4, 7, 7), np.float32)
    for i in range(4):
        suppress[i, i, :, :] = 1

    for slot in range(4):
        p = "pick" + str(slot)
        node(p + "_rank", "mul", ["candidate_rank", remaining])
        node(p + "_max", "gmax", [p + "_rank"])
        node(p + "_diff", "sub", [p + "_rank", p + "_max"])
        sp_scale(p + "_eps", p + "_diff", 1.0, 0.000001, 4)
        node(p + "_positive", "relu", [p + "_eps"])
        sp_scale(p + "_scaled", p + "_positive", 2000000.0, 0.0, 4)
        node(p + "_map", "min2", [p + "_scaled", remaining])
        node(p + "_row_map", "mul", [p + "_map", "row_coord"])
        node(p + "_col_map", "mul", [p + "_map", "col_coord"])
        node(p + "_row", "gmax", [p + "_row_map"])
        node(p + "_col", "gmax", [p + "_col_map"])
        node(p + "_present", "gmax", [p + "_map"])
        goal_parts.extend([p + "_row", p + "_col", p + "_present"])
        conv(p + "_suppress", p + "_map", suppress)
        node(p + "_remaining0", "sub", [remaining, p + "_suppress"])
        node(p + "_remaining", "relu", [p + "_remaining0"])
        remaining = p + "_remaining"

    # Detect the cross and its outline.
    node("eleven", "slice", ["x0"], start=11, stop=12)
    cross_template = np.zeros((1, 1, 7, 7), np.float32)
    for y in range(7):
        for x in range(7):
            if 2 <= y <= 4 or 2 <= x <= 4:
                cross_template[0, 0, y, x] = 1
    conv("cross_score", "eleven", cross_template,
         np.array([-32.0], np.float32))
    node("cross_anchor", "relu", ["cross_score"])
    node("cross_exists", "gmax", ["cross_anchor"])

    node("four", "slice", ["x0"], start=4, stop=5)
    cross_shifts = [(dy, dx) for dy in (-2, 0, 2) for dx in (-2, 0, 2)]
    w = np.zeros((9, 1, 5, 5), np.float32)
    for i, (dy, dx) in enumerate(cross_shifts):
        w[i, 0, 2 + dy, 2 + dx] = 1
    conv("cross_outline_planes", "four", w)

    outline = [
        "00011111000",
        "00010001000",
        "00010001000",
        "11110001111",
        "10000000001",
        "10000000001",
        "10000000001",
        "11110001111",
        "00010001000",
        "00010001000",
        "00011111000",
    ]
    w = np.zeros((1, 9, 7, 7), np.float32)
    cross_count = 0
    for y in range(11):
        for x in range(11):
            if outline[y][x] != "1":
                continue
            dy, dx = y - 5, x - 5
            sy = -2 if dy < -3 else (2 if dy > 3 else 0)
            sx = -2 if dx < -3 else (2 if dx > 3 else 0)
            plane = cross_shifts.index((sy, sx))
            w[0, plane, 3 + dy - sy, 3 + dx - sx] += 1
            cross_count += 1
    conv("cross_outline_score", "cross_outline_planes", w,
         np.array([1 - cross_count], np.float32))
    node("cross_outline_candidate", "relu", ["cross_outline_score"])
    node("cross_outline_row_map", "mul", ["cross_outline_candidate", "row_coord"])
    node("cross_outline_col_map", "mul", ["cross_outline_candidate", "col_coord"])
    node("cross_outline_row", "gmax", ["cross_outline_row_map"])
    node("cross_outline_col", "gmax", ["cross_outline_col_map"])
    node("cross_outline_present", "gmax", ["cross_outline_candidate"])
    node("new_cross_goal", "vconcat",
         ["cross_outline_row", "cross_outline_col", "cross_outline_present"])

    # Memory: rectangular goals, initialization flag, push state, cross goal.
    node("mem_goals", "vslice", ["mem"], start=0, stop=48)
    node("mem_init", "vslice", ["mem"], start=48, stop=49)
    node("push_mode", "vslice", ["mem"], start=49, stop=50)
    node("push_row", "vslice", ["mem"], start=50, stop=51)
    node("push_col", "vslice", ["mem"], start=51, stop=52)
    node("push_dirs", "vslice", ["mem"], start=52, stop=56)
    node("push_old", "vslice", ["mem"], start=50, stop=56)
    node("old_cross_goal", "vslice", ["mem"], start=56, stop=59)
    complement("not_initialized", "mem_init")
    node("new_goal_memory", "vconcat", goal_parts)
    node("old_goal_part", "vmul", ["mem_goals", "mem_init"])
    node("new_goal_part", "vmul", ["new_goal_memory", "not_initialized"])
    node("saved_goals", "vadd", ["old_goal_part", "new_goal_part"])
    vscale("initialized_next", "mem_init", 0.0, 1.0)
    node("old_cross_part", "vmul", ["old_cross_goal", "mem_init"])
    node("new_cross_part", "vmul", ["new_cross_goal", "not_initialized"])
    node("saved_cross_goal", "vadd", ["old_cross_part", "new_cross_part"])

    goal_maps = []
    for slot in range(4):
        p = "saved" + str(slot)
        base = slot * 12
        node(p + "_row", "vslice", ["saved_goals"], start=base, stop=base + 4)
        node(p + "_col", "vslice", ["saved_goals"], start=base + 4, stop=base + 8)
        node(p + "_present", "vslice", ["saved_goals"],
             start=base + 8, stop=base + 12)
        node(p + "_dr0", "sub", ["row_coord", p + "_row"])
        node(p + "_dc0", "sub", ["col_coord", p + "_col"])
        node(p + "_dr", "abs", [p + "_dr0"])
        node(p + "_dc", "abs", [p + "_dc0"])
        node(p + "_distance", "add", [p + "_dr", p + "_dc"])
        sp_scale(p + "_score", p + "_distance", -63.0, 1.0, 4)
        node(p + "_positive", "relu", [p + "_score"])
        node(p + "_map", "min2", [p + "_positive", p + "_present"])
        goal_maps.append(p + "_map")
    node("goals01", "max2", [goal_maps[0], goal_maps[1]])
    node("goals23", "max2", [goal_maps[2], goal_maps[3]])
    node("goals", "max2", ["goals01", "goals23"])
    node("open_raw", "sub", ["goals", "pieces"])
    node("open_goals", "relu", ["open_raw"])
    node("active_goals", "mul", ["open_goals", "selected_types"])
    sum_channels("normal_goal", "active_goals", 4)
    node("on_goal_map", "mul", ["selected", "goals"])
    node("on_goal_types", "gmax", ["on_goal_map"])
    dense("on_goal_raw", "on_goal_types", np.ones((1, 4), np.float32))
    vscale("on_goal_threshold", "on_goal_raw", 2.0, -1.0)
    vclip("on_goal", "on_goal_threshold")
    complement("not_on_goal", "on_goal")

    node("cross_goal_row", "vslice", ["saved_cross_goal"], start=0, stop=1)
    node("cross_goal_col", "vslice", ["saved_cross_goal"], start=1, stop=2)
    node("cross_goal_present", "vslice", ["saved_cross_goal"], start=2, stop=3)
    node("cross_goal_dr0", "sub", ["row_coord", "cross_goal_row"])
    node("cross_goal_dc0", "sub", ["col_coord", "cross_goal_col"])
    node("cross_goal_dr", "abs", ["cross_goal_dr0"])
    node("cross_goal_dc", "abs", ["cross_goal_dc0"])
    node("cross_goal_distance", "add", ["cross_goal_dr", "cross_goal_dc"])
    sp_scale("cross_goal_score", "cross_goal_distance", -63.0, 1.0)
    node("cross_goal_positive", "relu", ["cross_goal_score"])
    node("cross_goal", "mul", ["cross_goal_positive", "cross_goal_present"])
    node("cross_on_goal_map", "mul", ["cross_anchor", "cross_goal"])
    node("cross_on_goal_raw", "gmax", ["cross_on_goal_map"])
    vscale("cross_on_goal_test", "cross_on_goal_raw", 2.0, -1.0)
    vclip("cross_on_goal", "cross_on_goal_test")
    complement("cross_not_on_goal", "cross_on_goal")

    node("push_dr0", "sub", ["row_coord", "push_row"])
    node("push_dc0", "sub", ["col_coord", "push_col"])
    node("push_dr", "abs", ["push_dr0"])
    node("push_dc", "abs", ["push_dc0"])
    node("push_distance", "add", ["push_dr", "push_dc"])
    sp_scale("push_position_score", "push_distance", -63.0, 1.0)
    node("push_position", "relu", ["push_position_score"])
    node("push_arrived_map", "mul", ["push_position", "selected_anchor"])
    node("push_arrived_raw", "gmax", ["push_arrived_map"])
    vscale("push_arrived_test", "push_arrived_raw", 2.0, -1.0)
    vclip("push_arrived", "push_arrived_test")
    vmuls("push_pulse", ["push_mode", "push_arrived"])
    complement("not_push_mode", "push_mode")
    complement("not_push_pulse", "push_pulse")
    node("normal_goal_gated", "mul", ["normal_goal", "not_push_mode"])
    node("push_goal_gated", "mul", ["push_position", "push_mode"])
    node("routing_goal", "add", ["normal_goal_gated", "push_goal_gated"])

    # Free-space maps for normal and virtual-floor routes.
    w = np.zeros((1, 4, 7, 7), np.float32)
    for t, (h, width) in enumerate(sizes):
        ay = 1 if h == 3 else 2
        ax = 1 if width == 3 else 2
        for y in range(h):
            for x in range(width):
                w[0, t, 3 - (y - ay), 3 - (x - ax)] = 1
    conv("own_footprint", "selected", w)

    w = np.zeros((2, 16, 1, 1), np.float32)
    for c in (2, 11, 13, 14, 15):
        w[0, c, 0, 0] = 1
    for c in (2, 11, 13, 14):
        w[1, c, 0, 0] = 1
    conv("obstacle_pair", "x0", w)
    node("obstacle_strict_raw", "slice", ["obstacle_pair"], start=0, stop=1)
    node("obstacle_virtual_raw", "slice", ["obstacle_pair"], start=1, stop=2)
    node("obstacle_strict_sub", "sub", ["obstacle_strict_raw", "own_footprint"])
    node("obstacle_virtual_sub", "sub", ["obstacle_virtual_raw", "own_footprint"])
    node("obstacle_strict", "relu", ["obstacle_strict_sub"])
    node("obstacle_virtual", "relu", ["obstacle_virtual_sub"])

    free_w = np.zeros((4, 1, 7, 7), np.float32)
    for t, (h, width) in enumerate(sizes):
        ay = 1 if h == 3 else 2
        ax = 1 if width == 3 else 2
        for y in range(h):
            for x in range(width):
                free_w[t, 0, 3 + y - ay, 3 + x - ax] = -1
    conv("strict_free_score", "obstacle_strict", free_w, np.ones(4, np.float32))
    conv("virtual_free_score", "obstacle_virtual", free_w, np.ones(4, np.float32))
    node("strict_free_types", "relu", ["strict_free_score"])
    node("virtual_free_types", "relu", ["virtual_free_score"])
    node("strict_free_selected", "mul", ["strict_free_types", "selected_types"])
    node("virtual_free_selected", "mul", ["virtual_free_types", "selected_types"])
    sum_channels("strict_free", "strict_free_selected", 4)
    sum_channels("virtual_free", "virtual_free_selected", 4)
    node("strict_seed", "mul", ["routing_goal", "strict_free"])
    node("virtual_seed", "mul", ["normal_goal", "virtual_free"])

    wave_w = np.zeros((4, 1, 7, 7), np.float32)
    for i, (dy, dx) in enumerate(directions):
        wave_w[i, 0, 3 + dy, 3 + dx] = 1
    wave_param = param(wave_w, "w")
    decay_param = param(np.ones((1, 1, 1, 1), np.float32), "w")
    decay_bias = param(np.array([-1.0 / 256.0], np.float32), "b")

    def wave(prefix, seed, free):
        state = prefix + "_state"
        body = [
            {"id": prefix + "_neighbours", "op": "conv",
             "in": [state], "w": wave_param}
        ]
        for i in range(4):
            body.append({
                "id": prefix + "_n" + str(i), "op": "slice",
                "in": [prefix + "_neighbours"], "start": i, "stop": i + 1
            })
        body.extend([
            {"id": prefix + "_m01", "op": "max2",
             "in": [prefix + "_n0", prefix + "_n1"]},
            {"id": prefix + "_m23", "op": "max2",
             "in": [prefix + "_n2", prefix + "_n3"]},
            {"id": prefix + "_near", "op": "max2",
             "in": [prefix + "_m01", prefix + "_m23"]},
            {"id": prefix + "_decay", "op": "conv",
             "in": [prefix + "_near"], "w": decay_param, "b": decay_bias},
            {"id": prefix + "_positive", "op": "relu",
             "in": [prefix + "_decay"]},
            {"id": prefix + "_valid", "op": "mul",
             "in": [prefix + "_positive", free]},
            {"id": prefix + "_next", "op": "max2",
             "in": [state, prefix + "_valid"]},
        ])
        graph.append({
            "id": prefix + "_loop",
            "op": "loop",
            "T": 128,
            "state": [{"id": state, "init": seed}],
            "body": body,
            "next": {state: prefix + "_next"},
        })
        return prefix + "_loop." + state

    strict_wave = wave("strict_wave", "strict_seed", "strict_free")
    virtual_wave = wave("virtual_wave", "virtual_seed", "virtual_free")

    conv("strict_neighbours", strict_wave, wave_w)
    conv("virtual_neighbours", virtual_wave, wave_w)
    conv("strict_free_neighbours", "strict_free", wave_w)
    node("strict_directions_map", "mul", ["strict_neighbours", "selected_anchor"])
    node("strict_directions", "gmax", ["strict_directions_map"])
    node("strict_current_map", "mul", [strict_wave, "selected_anchor"])
    node("strict_current", "gmax", ["strict_current_map"])
    vscale("strict_available0", "strict_current", 4.0)
    vclip("strict_available", "strict_available0")
    complement("no_strict", "strict_available")

    node("virtual_directions_map", "mul", ["virtual_neighbours", "selected_anchor"])
    node("virtual_directions", "gmax", ["virtual_directions_map"])
    node("virtual_direct_map0", "mul",
         ["virtual_neighbours", "strict_free_neighbours"])
    node("virtual_direct_map", "mul", ["virtual_direct_map0", "selected_anchor"])
    node("virtual_direct", "gmax", ["virtual_direct_map"])
    max4("virtual_direct_best", "virtual_direct_map")
    node("virtual_current_map", "mul", [virtual_wave, "selected_anchor"])
    node("virtual_current", "gmax", ["virtual_current_map"])
    vscale("virtual_available0", "virtual_current", 4.0)
    vclip("virtual_available", "virtual_available0")
    vscale("negative_virtual_current", "virtual_current", -1.0)
    node("virtual_progress0", "vadd",
         ["virtual_direct_best", "negative_virtual_current"])
    vscale("virtual_progress1", "virtual_progress0", 1000.0, -1.0)
    vclip("virtual_progress", "virtual_progress1")
    complement("no_virtual_progress", "virtual_progress")
    choose_direction("virtual_choice", "virtual_directions", "selected_anchor")

    # Cross wavefront ignores special floor, as pushed objects do.
    node("wall", "slice", ["x0"], start=2, stop=3)
    cross_wall_shifts = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    w = np.zeros((5, 1, 3, 3), np.float32)
    for i, (dy, dx) in enumerate(cross_wall_shifts):
        w[i, 0, 1 + dy, 1 + dx] = 1
    conv("cross_wall_planes", "wall", w)
    w = np.zeros((1, 5, 7, 7), np.float32)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            if not (abs(dy) <= 1 or abs(dx) <= 1):
                continue
            sy = -1 if dy == -4 else (1 if dy == 4 else 0)
            sx = -1 if dx == -4 else (1 if dx == 4 else 0)
            plane = cross_wall_shifts.index((sy, sx))
            w[0, plane, 3 + dy - sy, 3 + dx - sx] = -1
    conv("cross_free_score", "cross_wall_planes", w,
         np.array([1.0], np.float32))
    node("cross_free", "relu", ["cross_free_score"])
    node("cross_seed", "mul", ["cross_goal", "cross_free"])
    cross_wave = wave("cross_wave", "cross_seed", "cross_free")
    conv("cross_neighbours", cross_wave, wave_w)
    node("cross_directions_map", "mul", ["cross_neighbours", "cross_anchor"])
    node("cross_directions", "gmax", ["cross_directions_map"])
    node("cross_current_map", "mul", [cross_wave, "cross_anchor"])
    node("cross_current", "gmax", ["cross_current_map"])
    vscale("cross_route_available0", "cross_current", 4.0)
    vclip("cross_route_available", "cross_route_available0")
    choose_direction("cross_choice", "cross_directions", "cross_anchor")

    node("cross_by_direction", "mul", ["cross_anchor", "cross_choice_choice"])
    w = np.zeros((4, 4, 7, 7), np.float32)
    for i, (dy, dx) in enumerate(directions):
        w[i, i, 3 + dy, 3 + dx] = 1
    conv("cross_one_behind", "cross_by_direction", w)
    w = np.zeros((1, 4, 7, 7), np.float32)
    for i, (dy, dx) in enumerate(directions):
        w[0, i, 3 + dy, 3 + dx] = 1
        if dy:
            w[0, i, 3, 0] = 1
            w[0, i, 3, 6] = 1
        else:
            w[0, i, 0, 3] = 1
            w[0, i, 6, 3] = 1
    conv("cross_push_positions_raw", "cross_one_behind", w)
    sp_scale("cross_position_cap", "cross_push_positions_raw", 0.0, 1.0)
    node("cross_push_positions", "min2",
         ["cross_push_positions_raw", "cross_position_cap"])
    node("cross_pusher_seed", "mul", ["cross_push_positions", "strict_free"])
    cross_pusher_wave = wave("cross_pusher_wave", "cross_pusher_seed", "strict_free")
    conv("cross_pusher_neighbours", cross_pusher_wave, wave_w)
    node("cross_pusher_directions_map", "mul",
         ["cross_pusher_neighbours", "selected_anchor"])
    node("cross_pusher_directions", "gmax", ["cross_pusher_directions_map"])
    node("cross_pusher_current_map", "mul", [cross_pusher_wave, "selected_anchor"])
    node("cross_pusher_current", "gmax", ["cross_pusher_current_map"])
    vscale("cross_pusher_available0", "cross_pusher_current", 4.0)
    vclip("cross_pusher_available", "cross_pusher_available0")
    node("cross_arrived_map", "mul", ["cross_push_positions", "selected_anchor"])
    node("cross_arrived_raw", "gmax", ["cross_arrived_map"])
    vscale("cross_arrived_test", "cross_arrived_raw", 2.0, -1.0)
    vclip("cross_arrived", "cross_arrived_test")
    complement("cross_not_arrived", "cross_arrived")
    vmuls("cross_gate", [
        "not_push_mode", "selected_is3", "cross_exists", "cross_goal_present",
        "cross_not_on_goal", "cross_route_available", "cross_pusher_available"
    ])
    complement("not_cross_gate", "cross_gate")
    vmuls("cross_move_gate", ["cross_gate", "cross_not_arrived"])
    vmuls("cross_push_gate", ["cross_gate", "cross_arrived"])

    # Form a pushing position for each selected size and push direction.
    # For six-pixel sides, the extra shift aligns the 3x3 pusher with
    # the target's three-pixel movement lattice.
    target_channels = []
    for t in range(4):
        node("target_type" + str(t), "slice", ["selected"], start=t, stop=t + 1)
        for d in range(4):
            node("target_dir" + str(t) + "_" + str(d), "vslice",
                 ["virtual_choice_choice"], start=d, stop=d + 1)
            node("target_channel" + str(t) + "_" + str(d), "mul",
                 ["target_type" + str(t), "target_dir" + str(t) + "_" + str(d)])
            target_channels.append("target_channel" + str(t) + "_" + str(d))
    node("target_channels", "concat", target_channels)

    w1 = np.zeros((16, 16, 7, 7), np.float32)
    w2 = np.zeros((1, 16, 7, 7), np.float32)
    for t, (h, width) in enumerate(sizes):
        for d, (dy, dx) in enumerate(directions):
            channel = 4 * t + d
            # First move three pixels opposite the intended push.
            w1[channel, channel, 3 + dy, 3 + dx] = 1
            ey = 0
            ex = 0
            if dy:
                if h == 6:
                    ey = 2 if dy < 0 else -1
                if width == 6:
                    ex = -1
            else:
                if width == 6:
                    ex = 2 if dx < 0 else -1
                if h == 6:
                    ey = -1
            # Cross-correlation shifts a marker by the negative kernel offset.
            w2[0, channel, 3 - ey, 3 - ex] = 1
    conv("target_three_behind", "target_channels", w1)
    conv("new_push_position", "target_three_behind", w2)
    node("new_push_row_map", "mul", ["new_push_position", "row_coord"])
    node("new_push_col_map", "mul", ["new_push_position", "col_coord"])
    node("new_push_row", "gmax", ["new_push_row_map"])
    node("new_push_col", "gmax", ["new_push_col_map"])
    node("new_push_data", "vconcat",
         ["new_push_row", "new_push_col", "virtual_choice_choice"])

    # Test reachability from other 3x3 pieces to the proposed push position.
    # Their own pixels are removed for this particular reachability test.
    w = np.ones((1, 1, 3, 3), np.float32)
    conv("other3_footprints", "other3", w)
    node("setup_obstacle_sub", "sub",
         ["obstacle_strict_raw", "other3_footprints"])
    node("setup_obstacle", "relu", ["setup_obstacle_sub"])
    w = -np.ones((1, 1, 3, 3), np.float32)
    conv("setup_free_score", "setup_obstacle", w, np.array([1.0], np.float32))
    node("setup_free", "relu", ["setup_free_score"])
    node("setup_seed", "mul", ["new_push_position", "setup_free"])
    setup_wave = wave("setup_wave", "setup_seed", "setup_free")
    node("setup_pusher_scores", "mul", [setup_wave, "other3"])
    node("setup_pusher_best", "gmax", ["setup_pusher_scores"])
    vscale("setup_reachable0", "setup_pusher_best", 4.0)
    vclip("setup_reachable", "setup_reachable0")

    vmuls("push_trigger", [
        "not_push_mode", "not_cross_gate", "not_on_goal", "no_strict",
        "virtual_available", "no_virtual_progress", "other3_exists",
        "setup_reachable"
    ])
    complement("no_push_trigger", "push_trigger")
    node("push_data_old_part", "vmul", ["push_old", "no_push_trigger"])
    node("push_data_new_part", "vmul", ["new_push_data", "push_trigger"])
    node("push_data_next", "vadd", ["push_data_old_part", "push_data_new_part"])
    node("push_mode_kept", "vmul", ["push_mode", "not_push_pulse"])
    node("push_mode_next", "vadd", ["push_mode_kept", "push_trigger"])

    # Movement and click decisions.
    vmuls("strict_gate", ["strict_available", "not_push_pulse", "not_cross_gate"])
    vmuls("virtual_gate", [
        "not_push_mode", "not_cross_gate", "not_on_goal", "no_strict",
        "virtual_available", "virtual_progress"
    ])
    node("strict_action_values", "vmul", ["strict_directions", "strict_gate"])
    node("virtual_action_values", "vmul", ["virtual_direct", "virtual_gate"])
    node("regular_action_values", "vadd",
         ["strict_action_values", "virtual_action_values"])
    vscale("regular_action_scaled", "regular_action_values", 256.0, 0.0, 4)
    node("push_action_values", "vmul", ["push_dirs", "push_pulse"])
    vscale("push_action_scaled", "push_action_values", 1000.0, 0.0, 4)
    node("cross_move_values", "vmul",
         ["cross_pusher_directions", "cross_move_gate"])
    vscale("cross_move_scaled", "cross_move_values", 256.0, 0.0, 4)
    node("cross_push_values", "vmul", ["cross_choice_choice", "cross_push_gate"])
    vscale("cross_push_scaled", "cross_push_values", 1000.0, 0.0, 4)
    node("movement01", "vadd", ["regular_action_scaled", "push_action_scaled"])
    node("movement23", "vadd", ["cross_move_scaled", "cross_push_scaled"])
    node("movement_logits", "vadd", ["movement01", "movement23"])

    node("virtual_usable", "vmul", ["virtual_available", "virtual_progress"])
    complement("no_virtual_usable", "virtual_usable")
    vmuls("normal_stuck", ["no_strict", "no_virtual_usable"])
    node("click_reason0", "vadd", ["on_goal", "normal_stuck"])
    vclip("click_reason", "click_reason0")
    vmuls("click_gate", ["not_push_mode", "not_cross_gate", "click_reason"])
    vscale("click_logit", "click_gate", 1000.0)

    node("action_features", "vconcat", ["movement_logits", "click_logit"])
    w = np.zeros((8, 5), np.float32)
    for i in range(4):
        w[i + 1, i] = 1
    w[6, 4] = 1
    dense("logits", "action_features", w,
          np.array([-10000, 0, 0, 0, 0, -10000, 0, -10000], np.float32))

    node("unplaced_raw", "sub", ["pieces", "goals"])
    node("unplaced", "relu", ["unplaced_raw"])
    sum_channels("unplaced_anchors", "unplaced", 4)
    node("other_unplaced0", "sub", ["unplaced_anchors", "selected_anchor"])
    node("other_unplaced", "relu", ["other_unplaced0"])
    node("other0", "sub", ["all_anchors", "selected_anchor"])
    node("other", "relu", ["other0"])
    sp_scale("preferred_other", "other_unplaced", 10.0)
    node("normal_click", "add", ["preferred_other", "other"])
    node("normal_click_gated", "mul", ["normal_click", "no_push_trigger"])
    sp_scale("push_click_preferred", "setup_pusher_scores", 20.0)
    node("push_click_gated", "mul", ["push_click_preferred", "push_trigger"])
    node("click", "add", ["normal_click_gated", "push_click_gated"])

    node("mem_next", "vconcat", [
        "saved_goals", "initialized_next", "push_mode_next",
        "push_data_next", "saved_cross_goal"
    ])

    return {"graph": graph, "memory_size": 59, "params": params}
