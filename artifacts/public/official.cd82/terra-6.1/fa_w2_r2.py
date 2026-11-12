import numpy as np


def build():
    params = {}
    graph = []
    counter = [0]

    def fresh(prefix):
        counter[0] += 1
        return "%s_%d" % (prefix, counter[0])

    def param(value):
        name = fresh("p")
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(op, inputs, dest=None, where=None, **fields):
        if where is None:
            where = graph
        if dest is None:
            dest = fresh("n")
        nd = {"id": dest, "op": op, "in": list(inputs)}
        nd.update(fields)
        where.append(nd)
        return dest

    def dense(x, w, b=None, where=None, dest=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", [x], dest=dest, where=where, **fields)

    def conv(x, w, b=None, where=None, dest=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("conv", [x], dest=dest, where=where, **fields)

    def op(name, *xs, where=None, dest=None):
        return node(name, xs, dest=dest, where=where)

    def sl(x, start, stop, where=None):
        return node("slice", [x], where=where, start=start, stop=stop)

    def vsl(x, start, stop, where=None):
        return node("vslice", [x], where=where, start=start, stop=stop)

    def neg(x, n, where=None):
        return dense(x, -np.eye(n, dtype=np.float32), where=where)

    def vsub(a, b, n, where=None):
        return op("vadd", a, neg(b, n, where), where=where)

    def one_minus(x, where=None):
        return dense(x, [[-1.0]], [1.0], where=where)

    def point_masks(points):
        """One exact binary spatial channel for each (row, column)."""
        n = len(points)
        w = np.zeros((2 * n, 3, 1, 1), np.float32)
        b = np.zeros(2 * n, np.float32)
        for i, (r, c) in enumerate(points):
            w[2 * i, 0, 0, 0] = 63.0
            b[2 * i] = -float(r)
            w[2 * i + 1, 1, 0, 0] = 63.0
            b[2 * i + 1] = -float(c)
        distances = op("abs", conv("coords", w, b))
        w2 = np.zeros((n, 2 * n, 1, 1), np.float32)
        for i in range(n):
            w2[i, 2 * i, 0, 0] = -2.0
            w2[i, 2 * i + 1, 0, 0] = -2.0
        tents = op("relu", conv(distances, w2, np.full(n, 2.0)))
        return op("min2", tents, sl("coords", 2, 3))

    # Memory: two phase bits, 12 encoded strokes, 8 positions, 16 colours.
    phase = vsl("mem", 0, 2)
    phase_plan = vsl(phase, 0, 1)
    phase_run = vsl(phase, 1, 2)
    initial = dense(phase, [[-1.0, -1.0]], [1.0])
    prep = op("vadd", initial, phase_plan)
    phase_next = op("vconcat", initial, op("vadd", phase_plan, phase_run))

    old_plan = vsl("mem", 2, 14)
    old_pos = vsl("mem", 14, 22)
    old_colour = vsl("mem", 22, 38)

    # These points are on the red outline of exactly one basket position.
    # The last four points are on the four possible tool positions.
    basket_points = [
        (24, 31), (21, 39), (38, 46), (46, 49),
        (53, 31), (56, 24), (38, 17), (31, 14),
    ]
    tool_points = [(18, 30), (38, 52), (55, 29), (37, 11)]
    markers = point_masks(basket_points + tool_points)
    red = sl("x0", 2, 3)
    marker_values = op("gmax", op("mul", markers, red))
    observed_pos = vsl(marker_values, 0, 8)
    observed_tools = vsl(marker_values, 8, 12)

    even_select = np.zeros((4, 8), np.float32)
    for j, pos in enumerate((0, 2, 4, 6)):
        even_select[j, pos] = 1.0
    observed_even = dense(observed_pos, even_select)
    tool_present = dense(
        op("vmul", observed_even, observed_tools),
        np.ones((1, 4), np.float32),
    )

    # Target pixel masks. The two diagonals do not participate in the
    # simulator's equality test.
    target_points = [(3 + r, 3 + c) for r in range(10) for c in range(10)]
    target_masks = point_masks(target_points)
    scored = np.array(
        [float(c != r and c != 9 - r)
         for r in range(10) for c in range(10)],
        np.float32,
    )
    tile = conv(target_masks, scored.reshape(1, 100, 1, 1))

    # The eight broad strokes, followed by the four tool strokes.
    regions = np.zeros((12, 100), np.float32)
    for r in range(10):
        for c in range(10):
            k = 10 * r + c
            if not scored[k]:
                continue
            flags = [
                r < 5,
                c >= r,
                c >= 5,
                c >= 9 - r,
                r >= 5,
                c <= r,
                c < 5,
                c <= 9 - r,
                r < 3 and 3 <= c < 7,
                3 <= r < 7 and c >= 7,
                r >= 7 and 3 <= c < 7,
                3 <= r < 7 and c < 3,
            ]
            for j, flag in enumerate(flags):
                regions[j, k] = float(flag)

    region_maps = conv(target_masks, regions.reshape(12, 100, 1, 1))
    region_channels = [sl(region_maps, i, i + 1) for i in range(12)]

    # Reverse planning loop. A candidate is eligible exactly when its
    # remaining pixels contain one and only one colour.
    body = []
    active = op("mul", "x0", "remaining", where=body)
    present_parts = []
    for region in region_channels:
        covered = op("mul", active, region, where=body)
        present_parts.append(op("gmax", covered, where=body))
    presence = op("vconcat", *present_parts, where=body)

    pure_w = np.zeros((192, 192), np.float32)
    for r in range(12):
        for colour in range(16):
            i = 16 * r + colour
            pure_w[i, 16 * r:16 * r + 16] = -1.0
            pure_w[i, i] = 1.0
    pure = op("vrelu", dense(presence, pure_w, where=body), where=body)

    gate_w = np.zeros((192, 1), np.float32)
    gate_b = np.zeros(192, np.float32)
    gate_b[:128] = 1.0
    gate_w[128:, 0] = 1.0
    gate = dense(tool_present, gate_w, gate_b, where=body)
    candidates = op("vmul", pure, gate, where=body)

    # Deterministically choose the first eligible candidate.
    priority_w = np.zeros((192, 192), np.float32)
    for i in range(192):
        priority_w[i, :i] = -1.0
    first_gate = op(
        "vrelu",
        dense(candidates, priority_w, np.ones(192, np.float32), where=body),
        where=body,
    )
    selected = op("vmul", candidates, first_gate, where=body)

    region_select_w = np.zeros((12, 192), np.float32)
    for r in range(12):
        region_select_w[r, 16 * r:16 * r + 16] = 1.0
    selected_region = dense(selected, region_select_w, where=body)
    selected_pixels = op("mul", region_maps, selected_region, where=body)
    selected_pixels = conv(
        selected_pixels, np.ones((1, 12, 1, 1), np.float32), where=body
    )
    remaining_next = op(
        "relu", op("sub", "remaining", selected_pixels, where=body),
        where=body,
    )

    code = dense(
        selected,
        np.arange(1, 193, dtype=np.float32).reshape(1, 192),
        where=body,
    )
    any_selected = dense(selected, np.ones((1, 192), np.float32), where=body)

    shift_right = np.zeros((12, 12), np.float32)
    for i in range(1, 12):
        shift_right[i, i - 1] = 1.0
    insert = np.zeros((12, 1), np.float32)
    insert[0, 0] = 1.0
    shifted_stack = op(
        "vadd",
        dense("stack", shift_right, where=body),
        dense(code, insert, where=body),
        where=body,
    )
    stack_delta = vsub(shifted_stack, "stack", 12, where=body)
    stack_next = op(
        "vadd", "stack", op("vmul", stack_delta, any_selected, where=body),
        where=body,
    )

    graph.append({
        "id": "planning",
        "op": "loop",
        "T": 12,
        "state": [
            {"id": "remaining", "init": tile},
            {"id": "stack", "zeros": [12]},
        ],
        "body": body,
        "next": {"remaining": remaining_next, "stack": stack_next},
    })

    # Decode the first stored stroke into a 192-way one-hot vector.
    token = vsl(old_plan, 0, 1)
    values = np.arange(1, 193, dtype=np.float32)
    pos_diff = dense(token, np.ones((192, 1), np.float32), -values)
    neg_diff = dense(token, -np.ones((192, 1), np.float32), values)
    distance = op("vadd", op("vrelu", pos_diff), op("vrelu", neg_diff))
    decoded = op(
        "vrelu",
        dense(distance, -2.0 * np.eye(192, dtype=np.float32),
              np.ones(192, np.float32)),
    )

    colour_w = np.zeros((16, 192), np.float32)
    target_pos_w = np.zeros((8, 192), np.float32)
    tool_w = np.zeros((1, 192), np.float32)
    for r in range(12):
        destination = r if r < 8 else 2 * (r - 8)
        for colour in range(16):
            i = 16 * r + colour
            colour_w[colour, i] = 1.0
            target_pos_w[destination, i] = 1.0
            if r >= 8:
                tool_w[0, i] = 1.0
    wanted_colour = dense(decoded, colour_w)
    wanted_pos = dense(decoded, target_pos_w)
    wants_tool = dense(decoded, tool_w)

    # Pair current and desired positions, then use the fixed ring geometry
    # to choose a shortest neighbouring move.
    pair_a = np.zeros((64, 8), np.float32)
    pair_b = np.zeros((64, 8), np.float32)
    for p in range(8):
        for q in range(8):
            pair_a[8 * p + q, p] = 1.0
            pair_b[8 * p + q, q] = 1.0
    pairs = op("vmul", dense(old_pos, pair_a), dense(wanted_pos, pair_b))

    ring = [
        (0, 1), (0, 2), (1, 2), (2, 2),
        (2, 1), (2, 0), (1, 0), (0, 0),
    ]
    next_pos_w = np.zeros((8, 64), np.float32)
    direction_w = np.zeros((4, 64), np.float32)
    same_pos_w = np.zeros((1, 64), np.float32)
    for p in range(8):
        for q in range(8):
            k = 8 * p + q
            if p == q:
                nxt = p
                same_pos_w[0, k] = 1.0
            else:
                clockwise = (q - p) % 8
                nxt = (p + (1 if clockwise <= 4 else -1)) % 8
                dr = ring[nxt][0] - ring[p][0]
                dc = ring[nxt][1] - ring[p][1]
                action = {
                    (-1, 0): 0, (1, 0): 1,
                    (0, -1): 2, (0, 1): 3,
                }[(dr, dc)]
                direction_w[action, k] = 1.0
            next_pos_w[nxt, k] = 1.0

    next_pos = dense(pairs, next_pos_w)
    directions = dense(pairs, direction_w)
    at_destination = dense(pairs, same_pos_w)
    same_colour = dense(
        op("vmul", old_colour, wanted_colour),
        np.ones((1, 16), np.float32),
    )

    palette_click = op(
        "vmul",
        op("vmul", phase_run, at_destination),
        one_minus(same_colour),
    )
    execute = op(
        "vmul", op("vmul", phase_run, at_destination), same_colour
    )
    tool_click = op("vmul", execute, wants_tool)
    paint = op("vmul", execute, one_minus(wants_tool))
    run_directions = op("vmul", directions, phase_run)

    # Memory updates.
    shift_left = np.zeros((12, 12), np.float32)
    for i in range(11):
        shift_left[i, i + 1] = 1.0
    advanced_plan = dense(old_plan, shift_left)
    plan_delta = vsub(advanced_plan, old_plan, 12)
    running_plan = op(
        "vadd", old_plan, op("vmul", plan_delta, execute)
    )
    new_plan = op(
        "vadd",
        op("vmul", "planning.stack", phase_plan),
        op("vmul", running_plan, phase_run),
    )
    new_pos = op(
        "vadd",
        op("vmul", observed_pos, phase_plan),
        op("vmul", next_pos, phase_run),
    )
    colour_delta = vsub(wanted_colour, old_colour, 16)
    running_colour = op(
        "vadd", old_colour, op("vmul", colour_delta, palette_click)
    )
    new_colour = op("vmul", running_colour, phase_run)
    node(
        "vconcat", [phase_next, new_plan, new_pos, new_colour],
        dest="mem_next",
    )

    # During preparation, an odd position moves to the next even one;
    # an even position uses a blocked outward move.
    prep_actions = [1, 2, 4, 3, 2, 1, 3, 4]
    prep_w = np.zeros((8, 8), np.float32)
    for p, action in enumerate(prep_actions):
        prep_w[action, p] = 1.0
    prep_logits = op("vmul", dense(observed_pos, prep_w), prep)

    click_action = op("vadd", palette_click, tool_click)
    run_controls = op("vconcat", run_directions, paint, click_action)
    output_w = np.zeros((8, 6), np.float32)
    for i in range(4):
        output_w[i + 1, i] = 1.0
    output_w[5, 4] = 1.0
    output_w[6, 5] = 1.0
    run_logits = dense(run_controls, output_w)
    node("vadd", [prep_logits, run_logits], dest="logits")

    # Palette detection: 5x5 colour-4 border, at row 4 in the UI.
    four = sl("x0", 4, 5)
    border = np.zeros((1, 1, 5, 5), np.float32)
    border[0, 0, 0, :] = 1.0
    border[0, 0, 4, :] = 1.0
    border[0, 0, 1:4, 0] = 1.0
    border[0, 0, 1:4, 4] = 1.0
    palette_border = op("relu", conv(four, border, [-15.5]))

    row_w = np.zeros((1, 3, 1, 1), np.float32)
    row_w[0, 0, 0, 0] = 63.0
    row_distance = op("abs", conv("coords", row_w, [-4.0]))
    row_gate = op(
        "min2",
        op("relu", conv(row_distance, [[[[-2.0]]]], [2.0])),
        sl("coords", 2, 3),
    )
    col_w = np.zeros((1, 3, 1, 1), np.float32)
    col_w[0, 1, 0, 0] = 63.0
    col_gate = op(
        "min2", op("relu", conv("coords", col_w, [-19.0])),
        sl("coords", 2, 3),
    )
    colour_match = conv(
        op("mul", "x0", wanted_colour),
        np.ones((1, 16, 1, 1), np.float32),
    )
    palette_map = op("mul", palette_border, row_gate)
    palette_map = op("mul", palette_map, col_gate)
    palette_map = op("mul", palette_map, colour_match)
    palette_map = op("mul", palette_map, palette_click)

    tool_marker_maps = sl(markers, 8, 12)
    current_even = dense(old_pos, even_select)
    tool_map = conv(
        op("mul", tool_marker_maps, current_even),
        np.ones((1, 4, 1, 1), np.float32),
    )
    tool_map = op("mul", tool_map, tool_click)
    node("add", [palette_map, tool_map], dest="click")

    return {"graph": graph, "memory_size": 38, "params": params}
