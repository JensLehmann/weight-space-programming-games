import numpy as np


def build():
    graph = []
    params = {}
    counter = [0]
    shift_cache = {}

    def parameter(value):
        name = "p%d" % len(params)
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(op, inputs, target=None, **fields):
        name = "n%d" % counter[0]
        counter[0] += 1
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        (graph if target is None else target).append(nd)
        return name

    def conv(x, weights, bias=None, target=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node("conv", [x], target, **fields)

    def dense(x, weights, bias=None, target=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node("dense", [x], target, **fields)

    def unary(op, x, target=None):
        return node(op, [x], target)

    def binary(op, x, y, target=None):
        return node(op, [x, y], target)

    def sslice(x, start, stop, target=None):
        return node("slice", [x], target, start=start, stop=stop)

    def shift(x, channels, dx, dy, target=None):
        k = 3 if max(abs(dx), abs(dy)) <= 1 else 7
        key = (channels, dx, dy, k)
        if key not in shift_cache:
            w = np.zeros((channels, channels, k, k), dtype=np.float32)
            for c in range(channels):
                w[c, c, k // 2 - dy, k // 2 - dx] = 1.0
            shift_cache[key] = parameter(w)
        return node("conv", [x], target, w=shift_cache[key])

    def right6(x, target=None):
        x = shift(x, 1, 3, 0, target)
        return shift(x, 1, 3, 0, target)

    def right7(x, target=None):
        x = shift(x, 1, 3, 0, target)
        x = shift(x, 1, 3, 0, target)
        return shift(x, 1, 1, 0, target)

    def sum_channels(x, channels, target=None):
        w = np.ones((1, channels, 1, 1), dtype=np.float32)
        return conv(x, w, target=target)

    def one_minus(x, target=None):
        return dense(x, [[-1.0]], [1.0], target)

    def vproduct(a, b, target=None):
        return node("vmul", [a, b], target)

    ones = sslice("coords", 2, 3)

    priority_w = np.zeros((1, 3, 1, 1), dtype=np.float32)
    priority_w[0, 0, 0, 0] = -4032.0
    priority_w[0, 1, 0, 0] = -63.0
    priority = conv("coords", priority_w, [4096.0])

    def first_position(mask):
        weighted = binary("mul", mask, priority)
        best = unary("gmax", weighted)
        delta = binary("sub", weighted, best)
        near = conv(
            delta,
            np.full((1, 1, 1, 1), 2.0, dtype=np.float32),
            [1.0],
        )
        near = unary("relu", near)
        near = binary("min2", near, ones)
        return binary("mul", near, mask)

    # Detect frame top-left corners.
    fw = np.zeros((16, 16, 7, 7), dtype=np.float32)
    fb = np.full(16, -6.0, dtype=np.float32)
    for c in range(16):
        if c in (0, 4, 5):
            fb[c] = -100.0
            continue
        for dx in (-2, -1, 0, 1, 2, 3):
            fw[c, c, 3, 3 + dx] += 1.0
        fw[c, c, 4, 1] += 1.0
        fw[c, c, 2, 1] -= 1.0
        fw[c, c, 3, 0] -= 1.0
    frame_matches = unary("relu", conv("x0", fw, fb))

    rw = np.zeros((2, 3, 1, 1), dtype=np.float32)
    rw[0, 0, 0, 0] = -63.0
    rw[1, 0, 0, 0] = 63.0
    outside = unary("relu", conv("coords", rw, [8.0, -52.0]))
    row_window = unary(
        "relu",
        conv(outside, -np.ones((1, 2, 1, 1), dtype=np.float32), [1.0]),
    )
    frame_matches = binary("mul", frame_matches, row_window)

    starts16 = shift(frame_matches, 16, 2, 3)
    starts16 = shift(starts16, 16, 0, 1)
    starts = sum_channels(starts16, 16)
    starts = binary("min2", starts, ones)
    root = first_position(starts)

    # Expand frame starts along their top borders.
    line_w = np.ones((1, 16, 1, 1), dtype=np.float32)
    for c in (0, 4, 5):
        line_w[0, c, 0, 0] = 0.0
    line = conv("x0", line_w)
    line = shift(line, 1, 0, 3)
    line = shift(line, 1, 0, 1)

    slot_body = []
    slot_shift = right6("slot_front", slot_body)
    slot_new = binary("mul", slot_shift, line, slot_body)
    slot_all_new = binary("max2", "slot_all", slot_new, slot_body)
    graph.append({
        "id": "slots",
        "op": "loop",
        "T": 9,
        "state": [
            {"id": "slot_front", "init": starts},
            {"id": "slot_all", "init": starts},
        ],
        "body": slot_body,
        "next": {
            "slot_front": slot_new,
            "slot_all": slot_all_new,
        },
    })
    valid_slots = "slots.slot_all"

    # Target row and target colour planes.
    tw = np.zeros((1, 3, 1, 1), dtype=np.float32)
    tw[0, 0, 0, 0] = 63.0
    row_distance = unary("abs", conv("coords", tw, [-1.0]))
    row_one = unary(
        "relu",
        conv(
            row_distance,
            -np.ones((1, 1, 1, 1), dtype=np.float32),
            [1.0],
        ),
    )

    colour_w = np.ones((1, 16, 1, 1), dtype=np.float32)
    colour_w[0, 4, 0, 0] = 0.0
    colour_w[0, 5, 0, 0] = 0.0
    target_pixels = conv("x0", colour_w)
    target_row = binary("mul", row_one, target_pixels)
    target_first = first_position(target_row)

    allowed_w = np.zeros((16, 16, 1, 1), dtype=np.float32)
    for c in range(16):
        if c not in (4, 5):
            allowed_w[c, c, 0, 0] = 1.0
    target_planes = conv("x0", allowed_w)

    target_mean = unary("gmean", target_row)
    target_count = dense(target_mean, [[4096.0 / 6.0]])

    # Detect hollow call chips at their upper-left interior pixel.
    call_w = np.zeros((16, 16, 5, 5), dtype=np.float32)
    call_b = np.full(16, -4.0, dtype=np.float32)
    for c in range(16):
        if c in (0, 4, 5):
            call_b[c] = -100.0
            continue
        for dy, dx in ((-1, -1), (-1, 2), (2, -1), (2, 2)):
            call_w[c, c, 2 + dy, 2 + dx] = 1.0
        call_w[c, 4, 2, 2] = 1.0
    hollow = unary("relu", conv("x0", call_w, call_b))
    hollow_any = sum_channels(hollow, 16)
    hollow_any = binary("min2", hollow_any, ones)

    # Symbolically execute the currently placed calls.
    body = []
    here_valid = unary(
        "gmax",
        binary("mul", "ex_cursor", valid_slots, body),
        body,
    )
    call_vector = unary(
        "gmax",
        binary("mul", "ex_cursor", hollow, body),
        body,
    )
    call_vector = vproduct(call_vector, here_valid, body)
    call_flag = dense(
        call_vector,
        np.ones((1, 16), dtype=np.float32),
        target=body,
    )
    not_call = one_minus(call_flag, body)
    leaf = vproduct(here_valid, not_call, body)
    pop = one_minus(here_valid, body)

    cursor_next_slot = right6("ex_cursor", body)
    continue_leaf = binary("mul", cursor_next_slot, leaf, body)
    call_destination = sum_channels(
        binary("mul", starts16, call_vector, body),
        16,
        body,
    )
    stack_top = sslice("ex_stack", 0, 1, body)
    return_cursor = binary("mul", stack_top, pop, body)
    cursor_new = binary(
        "add",
        binary("add", continue_leaf, call_destination, body),
        return_cursor,
        body,
    )

    stack_prefix = sslice("ex_stack", 0, 5, body)
    pushed = node("concat", [cursor_next_slot, stack_prefix], body)
    stack_suffix = sslice("ex_stack", 1, 6, body)
    zero_plane = binary("sub", "ex_cursor", "ex_cursor", body)
    popped = node("concat", [stack_suffix, zero_plane], body)
    stack_keep = binary("mul", "ex_stack", leaf, body)
    stack_push = binary("mul", pushed, call_flag, body)
    stack_pop = binary("mul", popped, pop, body)
    stack_new = binary(
        "add",
        binary("add", stack_keep, stack_push, body),
        stack_pop,
        body,
    )

    target_colour = unary(
        "gmax",
        binary("mul", "ex_target", target_planes, body),
        body,
    )
    leaf_goal = binary("mul", "ex_cursor", target_colour, body)
    leaf_goal = binary("mul", leaf_goal, leaf, body)
    goals_new = binary("max2", "ex_goals", leaf_goal, body)

    target_advanced = right7("ex_target", body)
    not_leaf = one_minus(leaf, body)
    target_new = binary(
        "add",
        binary("mul", target_advanced, leaf, body),
        binary("mul", "ex_target", not_leaf, body),
        body,
    )
    count_new = node("vadd", ["ex_count", leaf], body)

    graph.append({
        "id": "execute",
        "op": "loop",
        "T": 64,
        "state": [
            {"id": "ex_cursor", "init": root},
            {"id": "ex_target", "init": target_first},
            {"id": "ex_stack", "zeros": [6, 64, 64]},
            {"id": "ex_goals", "zeros": [16, 64, 64]},
            {"id": "ex_count", "zeros": [1]},
        ],
        "body": body,
        "next": {
            "ex_cursor": cursor_new,
            "ex_target": target_new,
            "ex_stack": stack_new,
            "ex_goals": goals_new,
            "ex_count": count_new,
        },
    })
    goals = "execute.ex_goals"

    # Detect solid and hollow pool chips.
    pw = np.zeros((16, 16, 3, 3), dtype=np.float32)
    pb = np.full(16, -3.0, dtype=np.float32)
    pb[4] = -100.0
    for c in range(16):
        for dy in (0, 1):
            for dx in (0, 1):
                pw[c, c, 1 + dy, 1 + dx] = 1.0
    solid = unary("relu", conv("x0", pw, pb))

    bw = np.zeros((1, 3, 1, 1), dtype=np.float32)
    bw[0, 0, 0, 0] = 63.0
    below_bar = unary("relu", conv("coords", bw, [-53.0]))
    below_bar = binary("min2", below_bar, ones)
    pool = binary("mul", solid, below_bar)
    pool_calls = binary("mul", hollow, below_bar)
    pool_colours = unary("gmax", pool)

    # Check whether the current program has too few leaves.
    diff = node(
        "vadd",
        [target_count, dense("execute.ex_count", [[-1.0]])],
    )
    need = unary("vrelu", dense(diff, [[1.0]], [-0.5]))
    need = unary("vrelu", dense(need, [[2.0]]))
    need = dense(need, [[1.0]])
    excess = unary("vrelu", dense(need, [[1.0]], [-1.0]))
    need = node("vadd", [need, dense(excess, [[-1.0]])])

    root_second = right6(root)
    root_call = unary("gmax", binary("mul", hollow, root_second))
    has_root_call = dense(
        root_call,
        np.ones((1, 16), dtype=np.float32),
    )
    no_root_call = one_minus(has_root_call)

    all_frame_colours = unary("gmax", starts16)
    root_colour = unary("gmax", binary("mul", starts16, root))
    nonroot_colours = node(
        "vadd",
        [
            all_frame_colours,
            dense(root_colour, -np.eye(16, dtype=np.float32)),
        ],
    )
    nonroot_colours = unary("vrelu", nonroot_colours)

    # A red centre marks an empty, clickable slot. Associate an empty
    # second slot with the colour of its frame.
    empty = sslice("x0", 2, 3)
    second_empty_at_start = shift(empty, 1, -3, 0)
    second_empty_at_start = shift(second_empty_at_start, 1, -3, 0)
    expandable_colours = unary(
        "gmax",
        binary("mul", starts16, second_empty_at_start),
    )
    preferred_colours = vproduct(expandable_colours, nonroot_colours)
    preferred_pool = binary("mul", pool_calls, preferred_colours)
    preferred_any = unary("gmax", sum_channels(preferred_pool, 16))
    no_preferred = one_minus(preferred_any)

    no_hollow = unary(
        "relu",
        conv(
            hollow_any,
            -np.ones((1, 1, 1, 1), dtype=np.float32),
            [1.0],
        ),
    )
    eligible_slots = binary("mul", valid_slots, no_hollow)

    scan_body = []
    scan_next = right6("scan_front", scan_body)
    scan_hit = binary("mul", scan_next, eligible_slots, scan_body)
    scan_all = binary("max2", "scan_all", scan_hit, scan_body)
    graph.append({
        "id": "rootscan",
        "op": "loop",
        "T": 8,
        "state": [
            {"id": "scan_front", "init": root_second},
            {"id": "scan_all", "zeros": [1, 64, 64]},
        ],
        "body": scan_body,
        "next": {
            "scan_front": scan_next,
            "scan_all": scan_all,
        },
    })
    root_later = first_position("rootscan.scan_all")
    root_later_exists = unary("gmax", root_later)

    same_pool = binary("mul", pool_calls, root_call)
    same_pool_any = sum_channels(same_pool, 16)
    same_exists = unary("gmax", same_pool_any)
    use_same = vproduct(same_exists, root_later_exists)
    use_same = vproduct(use_same, has_root_call)

    use_nested = vproduct(has_root_call, one_minus(use_same))
    callee_start = sum_channels(binary("mul", starts16, root_call), 16)
    callee_second = right6(callee_start)
    callee_second = binary("mul", callee_second, valid_slots)
    callee_second = binary("mul", callee_second, empty)

    other_colours = node(
        "vadd",
        [
            nonroot_colours,
            dense(root_call, -np.eye(16, dtype=np.float32)),
        ],
    )
    other_colours = unary("vrelu", other_colours)

    all_first_sources = binary("mul", pool_calls, nonroot_colours)
    fallback_first = binary("mul", all_first_sources, no_preferred)
    first_sources = binary("add", preferred_pool, fallback_first)
    other_sources = binary("mul", pool_calls, other_colours)

    src_first = binary("mul", first_sources, no_root_call)
    src_same = binary("mul", same_pool, use_same)
    src_nested = binary("mul", other_sources, use_nested)
    call_source16 = binary(
        "add",
        binary("add", src_first, src_same),
        src_nested,
    )
    call_source = sum_channels(call_source16, 16)
    call_source = binary("min2", call_source, ones)

    first_dest = binary("mul", root_second, no_root_call)
    first_dest = binary("mul", first_dest, valid_slots)
    same_dest = binary("mul", root_later, use_same)
    nested_dest = binary("mul", callee_second, use_nested)
    call_dest = binary(
        "add",
        binary("add", first_dest, same_dest),
        nested_dest,
    )
    call_dest = binary("min2", call_dest, ones)

    source_exists = unary("gmax", call_source)
    dest_exists = unary("gmax", call_dest)
    can_call = vproduct(need, source_exists)
    can_call = vproduct(can_call, dest_exists)
    no_call_plan = one_minus(can_call)

    # Place solid chips according to the symbolic leaf goals.
    available_goals = binary("mul", goals, pool_colours)
    wanted = sum_channels(available_goals, 16)
    wanted = binary("min2", wanted, ones)
    already_correct = sum_channels(binary("mul", goals, "x0"), 16)
    already_correct = binary("min2", already_correct, ones)
    candidates = unary("relu", binary("sub", wanted, already_correct))
    destination = first_position(candidates)
    has_solid = unary("gmax", candidates)

    desired_colour = unary("gmax", binary("mul", goals, destination))
    source = sum_channels(binary("mul", pool, desired_colour), 16)
    source = binary("min2", source, ones)

    call_source_first = first_position(call_source)
    call_dest_first = first_position(call_dest)
    chosen_source = binary(
        "add",
        binary("mul", call_source_first, can_call),
        binary("mul", source, no_call_plan),
    )
    chosen_dest = binary(
        "add",
        binary("mul", call_dest_first, can_call),
        binary("mul", destination, no_call_plan),
    )
    has_assignment = node(
        "vadd",
        [can_call, vproduct(has_solid, no_call_plan)],
    )

    # Alternate between selecting a chip and clicking its destination.
    not_memory = one_minus("mem")
    source_click = binary("mul", chosen_source, not_memory)
    destination_click = binary("mul", chosen_dest, "mem")
    click_value = binary("add", source_click, destination_click)
    graph.append({"id": "click", "op": "relu", "in": [click_value]})

    memory_value = vproduct(not_memory, has_assignment)
    graph.append({
        "id": "mem_next",
        "op": "vrelu",
        "in": [memory_value],
    })

    logit_w = np.zeros((8, 1), dtype=np.float32)
    logit_w[6, 0] = 2.0
    logit_b = np.full(8, -10.0, dtype=np.float32)
    logit_b[5] = 1.0
    logit_b[6] = 0.0
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [has_assignment],
        "w": parameter(logit_w),
        "b": parameter(logit_b),
    })

    return {
        "graph": graph,
        "memory_size": 1,
        "params": params,
    }
