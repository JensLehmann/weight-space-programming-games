import numpy as np


def build():
    graph = []
    params = {}
    current = graph
    serial = 0

    def node(op, *inputs, **fields):
        nonlocal serial
        name = "n%d" % serial
        serial += 1
        current.append({
            "id": name,
            "op": op,
            "in": list(inputs),
            **fields,
        })
        return name

    def conv(x, w, b=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return node("conv", x, **fields)

    def dense(x, w, b=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return node("dense", x, **fields)

    def sl(x, start, stop):
        return node("slice", x, start=start, stop=stop)

    def vsl(x, start, stop):
        return node("vslice", x, start=start, stop=stop)

    def mul(a, b):
        return node("mul", a, b)

    def add(a, b):
        return node("add", a, b)

    def maximum(a, b):
        return node("max2", a, b)

    def vmul(a, b):
        return node("vmul", a, b)

    def vadd(a, b):
        return node("vadd", a, b)

    def gmax(x):
        return node("gmax", x)

    def relu(x):
        return node("relu", x)

    def vrelu(x):
        return node("vrelu", x)

    params["sum16"] = np.ones((1, 16, 1, 1), dtype=np.float32)
    params["neg_s_w"] = np.array([[[[-1.0]]]], dtype=np.float32)
    params["one_s_b"] = np.array([1.0], dtype=np.float32)
    params["neg_v_w"] = np.array([[-1.0]], dtype=np.float32)
    params["one_v_b"] = np.array([1.0], dtype=np.float32)
    params["neg16_w"] = -np.eye(16, dtype=np.float32)
    params["one16_b"] = np.ones(16, dtype=np.float32)
    params["sum_vector16"] = np.ones((1, 16), dtype=np.float32)

    def one_minus_s(x):
        return conv(x, "neg_s_w", "one_s_b")

    def one_minus_v(x):
        return dense(x, "neg_v_w", "one_v_b")

    def one_minus_16(x):
        return dense(x, "neg16_w", "one16_b")

    def cap16(x):
        return one_minus_16(vrelu(one_minus_16(x)))

    def shift_weights(channels, size, dy, dx):
        w = np.zeros(
            (channels, channels, size, size), dtype=np.float32
        )
        center = size // 2
        for c in range(channels):
            w[c, c, center - dy, center - dx] = 1.0
        return w

    params["right3"] = shift_weights(1, 7, 0, 3)
    params["left3"] = shift_weights(1, 7, 0, -3)
    params["right1"] = shift_weights(1, 3, 0, 1)
    params["down2"] = shift_weights(1, 5, 2, 0)
    params["down2_colors"] = shift_weights(16, 5, 2, 0)
    params["downright_colors"] = shift_weights(16, 3, 1, 1)

    def right6(x):
        return conv(conv(x, "right3"), "right3")

    def right7(x):
        return conv(right6(x), "right1")

    def left6(x):
        return conv(conv(x, "left3"), "left3")

    def down4(x):
        return conv(conv(x, "down2"), "down2")

    # Unique row-major priority, positive everywhere.
    priority_w = np.zeros((1, 3, 1, 1), dtype=np.float32)
    priority_w[0, 0, 0, 0] = -126.0
    priority_w[0, 1, 0, 0] = -63.0 / 64.0
    params["priority_w"] = priority_w
    params["priority_b"] = np.array([128.0], dtype=np.float32)
    params["select_w"] = np.array([[[[-64.0]]]], dtype=np.float32)
    params["select_b"] = np.array([1.0], dtype=np.float32)
    priority = conv("coords", "priority_w", "priority_b")

    def first_point(candidates):
        score = mul(candidates, priority)
        best = gmax(score)
        difference = node("sub", best, score)
        selected = relu(conv(difference, "select_w", "select_b"))
        return mul(selected, candidates)

    colors = [c for c in range(16) if c not in (0, 3, 4, 5)]

    # Procedure-frame upper-left corners.
    frame_w = np.zeros((16, 16, 7, 7), dtype=np.float32)
    frame_b = np.full(16, -15.0, dtype=np.float32)
    for c in colors:
        for dy, dx in (
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (2, 0), (3, 0),
        ):
            frame_w[c, c, 3 + dy, 3 + dx] += 2.0
        frame_w[c, 4, 4, 4] += 2.0
        frame_w[c, c, 3, 2] -= 2.0
    params["frame_w"] = frame_w
    params["frame_b"] = frame_b

    frame_colors = relu(conv("x0", "frame_w", "frame_b"))
    frames = conv(frame_colors, "sum16")

    frame_rows_w = np.zeros((2, 3, 1, 1), dtype=np.float32)
    frame_rows_w[0, 0, 0, 0] = 63.0
    frame_rows_w[1, 0, 0, 0] = -63.0
    params["frame_rows_w"] = frame_rows_w
    params["frame_rows_b"] = np.array([-7.0, 51.0], dtype=np.float32)
    frame_rows = relu(conv("coords", "frame_rows_w", "frame_rows_b"))
    frames = node("min2", frames, sl(frame_rows, 0, 1))
    frames = node("min2", frames, sl(frame_rows, 1, 2))

    root = first_point(frames)
    root_color = gmax(mul("x0", root))
    root_probe = conv(conv(root, "right3"), "right1")
    root_head = down4(root_probe)

    # Generate slot-center masks from the horizontal frame borders.
    background = sl("x0", 4, 5)
    black = sl("x0", 0, 1)
    border_mask = one_minus_s(add(background, black))

    probe = conv(conv(frames, "right3"), "right1")
    probe = mul(probe, border_mask)
    heads = down4(probe)
    slots = heads
    for _ in range(8):
        probe = mul(right6(probe), border_mask)
        slots = maximum(slots, down4(probe))

    above_colors = conv(conv("x0", "down2_colors"), "down2_colors")
    head_colors = mul(above_colors, heads)
    all_head_colors = gmax(head_colors)

    # Heads of three-slot procedures.
    h1 = mul(right6(heads), slots)
    h2 = mul(right6(h1), slots)
    h3 = mul(right6(h2), slots)
    h2_back = left6(left6(h2))
    h3_back = left6(left6(left6(h3)))
    heads3 = mul(mul(heads, h2_back), one_minus_s(h3_back))
    head3_colors = mul(head_colors, heads3)
    all_head3_colors = gmax(head3_colors)

    # Target ring upper-left corners.
    target_w = np.zeros((1, 16, 3, 3), dtype=np.float32)
    for c in colors:
        target_w[0, c, 1, 1] += 2.0
        target_w[0, c, 1, 2] += 2.0
    target_w[0, 5, 1, 0] += 2.0
    target_w[0, 5, 2, 2] += 2.0
    params["target_w"] = target_w
    params["target_b"] = np.array([-7.0], dtype=np.float32)
    target_mask = relu(conv("x0", "target_w", "target_b"))
    target_start = first_point(target_mask)

    # Solid inventory tiles.
    solid_w = np.zeros((16, 16, 3, 3), dtype=np.float32)
    solid_b = np.full(16, -7.0, dtype=np.float32)
    for c in colors:
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            solid_w[c, c, 1 + dy, 1 + dx] = 2.0
    params["solid_w"] = solid_w
    params["solid_b"] = solid_b
    solids = relu(conv("x0", "solid_w", "solid_b"))

    # Outlined inventory call tiles.
    call_w = np.zeros((16, 16, 5, 5), dtype=np.float32)
    call_b = np.full(16, -13.0, dtype=np.float32)
    for c in colors:
        for dy, dx in ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)):
            call_w[c, c, 2 + dy, 2 + dx] += 2.0
        call_w[c, 4, 3, 3] += 2.0
        call_w[c, 4, 4, 4] += 2.0
    params["call_w"] = call_w
    params["call_b"] = call_b
    calls = relu(conv("x0", "call_w", "call_b"))

    inventory_w = np.zeros((2, 3, 1, 1), dtype=np.float32)
    inventory_w[0, 0, 0, 0] = 63.0
    inventory_w[1, 0, 0, 0] = -63.0
    params["inventory_w"] = inventory_w
    params["inventory_b"] = np.array([-54.0, 61.0], dtype=np.float32)
    inventory_rows = relu(conv("coords", "inventory_w", "inventory_b"))
    solids = node("min2", solids, sl(inventory_rows, 0, 1))
    solids = node("min2", solids, sl(inventory_rows, 1, 2))
    calls = node("min2", calls, sl(inventory_rows, 0, 1))
    calls = node("min2", calls, sl(inventory_rows, 1, 2))
    call_available_by_color = gmax(calls)

    item_corner_colors = conv("x0", "downright_colors")
    red = sl("x0", 2, 3)
    row_coord = sl("coords", 0, 1)

    params["later_row_w"] = np.array([[[[63.0]]]], dtype=np.float32)
    params["later_row_b"] = np.array([-0.5], dtype=np.float32)

    def next_target(point):
        same_row = mul(right7(point), target_mask)
        same_exists = gmax(same_row)
        current_row = gmax(mul(point, row_coord))
        difference = node("sub", row_coord, current_row)
        later = relu(conv(difference, "later_row_w", "later_row_b"))
        later = node("min2", later, target_mask)
        later_first = first_point(later)
        return add(same_row, mul(later_first, one_minus_v(same_exists)))

    # Symbolic program traversal with a four-entry return stack.
    body = []
    current = body

    ptr = "s_ptr"
    target = "s_target"
    stack1 = "s_stack1"
    stack2 = "s_stack2"
    stack3 = "s_stack3"
    stack4 = "s_stack4"
    saved_source = "s_source"
    saved_dest = "s_dest"
    found = "s_found"

    alive = gmax(ptr)
    is_call = gmax(mul(ptr, background))
    is_leaf = vmul(alive, one_minus_v(is_call))
    is_pop = one_minus_v(alive)
    is_empty = gmax(mul(ptr, red))

    target_points = [target]
    for _ in range(5):
        target_points.append(next_target(target_points[-1]))
    target_colors = [gmax(mul("x0", p)) for p in target_points]

    def equal_color(a, b):
        return dense(vmul(a, b), "sum_vector16")

    repeated = vmul(
        vmul(
            equal_color(target_colors[0], target_colors[3]),
            equal_color(target_colors[1], target_colors[4]),
        ),
        equal_color(target_colors[2], target_colors[5]),
    )

    # Reuse a three-slot procedure whose solids match the next targets.
    matching_pixels = [
        conv(mul("x0", target_colors[k]), "sum16")
        for k in range(3)
    ]
    match0 = matching_pixels[0]
    match1 = left6(matching_pixels[1])
    match2 = left6(left6(matching_pixels[2]))
    pattern_match = mul(mul(match0, match1), match2)
    reusable_colors = gmax(mul(head3_colors, pattern_match))

    stack_points = maximum(maximum(stack1, stack2), maximum(stack3, stack4))
    stack_colors = gmax(mul(item_corner_colors, stack_points))
    active_colors = cap16(vadd(root_color, stack_colors))

    same_color_calls = vmul(target_colors[0], all_head_colors)
    repeated_calls = vmul(all_head3_colors, repeated)
    eligible_calls = cap16(
        vadd(vadd(same_color_calls, repeated_calls), reusable_colors)
    )
    eligible_calls = vmul(
        vmul(eligible_calls, one_minus_16(active_colors)),
        call_available_by_color,
    )

    call_sources = conv(mul(calls, eligible_calls), "sum16")
    solid_sources = conv(mul(solids, target_colors[0]), "sum16")
    has_call_source = gmax(call_sources)
    source = add(
        call_sources,
        mul(solid_sources, one_minus_v(has_call_source)),
    )
    has_source = gmax(source)
    possible = vmul(is_empty, has_source)
    accept = vmul(possible, one_minus_v(found))

    new_source = add(saved_source, mul(source, accept))
    new_dest = add(saved_dest, mul(ptr, accept))
    new_found = vadd(found, accept)

    call_color = gmax(mul(item_corner_colors, ptr))
    child_head = conv(mul(head_colors, call_color), "sum16")
    next_slot = mul(right6(ptr), slots)
    after_pop = mul(right6(stack1), slots)

    new_ptr = add(
        add(mul(child_head, is_call), mul(next_slot, is_leaf)),
        mul(after_pop, is_pop),
    )
    new_target = add(
        mul(target_points[1], is_leaf),
        mul(target, one_minus_v(is_leaf)),
    )

    new_stack1 = add(
        add(mul(ptr, is_call), mul(stack1, is_leaf)),
        mul(stack2, is_pop),
    )
    new_stack2 = add(
        add(mul(stack1, is_call), mul(stack2, is_leaf)),
        mul(stack3, is_pop),
    )
    new_stack3 = add(
        add(mul(stack2, is_call), mul(stack3, is_leaf)),
        mul(stack4, is_pop),
    )
    new_stack4 = add(mul(stack3, is_call), mul(stack4, is_leaf))

    current = graph
    graph.append({
        "id": "program",
        "op": "loop",
        "T": 40,
        "state": [
            {"id": "s_ptr", "init": root_head},
            {"id": "s_target", "init": target_start},
            {"id": "s_stack1", "zeros": [1, 64, 64]},
            {"id": "s_stack2", "zeros": [1, 64, 64]},
            {"id": "s_stack3", "zeros": [1, 64, 64]},
            {"id": "s_stack4", "zeros": [1, 64, 64]},
            {"id": "s_source", "zeros": [1, 64, 64]},
            {"id": "s_dest", "zeros": [1, 64, 64]},
            {"id": "s_found", "zeros": [1]},
        ],
        "body": body,
        "next": {
            "s_ptr": new_ptr,
            "s_target": new_target,
            "s_stack1": new_stack1,
            "s_stack2": new_stack2,
            "s_stack3": new_stack3,
            "s_stack4": new_stack4,
            "s_source": new_source,
            "s_dest": new_dest,
            "s_found": new_found,
        },
    })

    # Six initial undo actions handle states reached by a few moves.
    # The second memory entry alternates source and destination clicks.
    clock = vsl("mem", 0, 1)
    phase = vsl("mem", 1, 2)

    params["clock_neg_w"] = np.array([[-1.0]], dtype=np.float32)
    params["clock_five_b"] = np.array([5.0], dtype=np.float32)
    params["clock_six_b"] = np.array([6.0], dtype=np.float32)
    params["clock_pos_w"] = np.array([[1.0]], dtype=np.float32)
    params["clock_minus_five_b"] = np.array([-5.0], dtype=np.float32)

    remaining = vrelu(dense(clock, "clock_neg_w", "clock_five_b"))
    next_clock = dense(remaining, "clock_neg_w", "clock_six_b")
    past_five = vrelu(dense(clock, "clock_pos_w", "clock_minus_five_b"))
    undo = vrelu(dense(past_five, "neg_v_w", "one_v_b"))

    valid = "program.s_found"
    source_click = mul("program.s_source", one_minus_v(phase))
    dest_click = mul("program.s_dest", phase)
    graph.append({
        "id": "click",
        "op": "add",
        "in": [source_click, dest_click],
    })

    next_phase = vmul(
        vmul(valid, one_minus_v(phase)),
        one_minus_v(undo),
    )
    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": [next_clock, next_phase],
    })

    decision = node("vconcat", valid, undo)
    logits_w = np.zeros((8, 2), dtype=np.float32)
    logits_b = np.full(8, -4.0, dtype=np.float32)
    logits_w[5, 0] = -2.0
    logits_b[5] = 1.0
    logits_w[6, 0] = 2.0
    logits_b[6] = 0.0
    logits_w[7, 1] = 4.0
    logits_b[7] = 0.0
    params["logits_w"] = logits_w
    params["logits_b"] = logits_b
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [decision],
        "w": "logits_w",
        "b": "logits_b",
    })

    return {"graph": graph, "memory_size": 2, "params": params}
