import numpy as np


def build():
    graph = []
    params = {}
    ni = 0
    pi = 0

    def op(kind, *inputs, **fields):
        nonlocal ni
        name = "n%d" % ni
        ni += 1
        graph.append({"id": name, "op": kind, "in": list(inputs), **fields})
        return name

    def parameter(value):
        nonlocal pi
        name = "p%d" % pi
        pi += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def conv(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return op("conv", x, **fields)

    def dense(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return op("dense", x, **fields)

    def sl(x, a, b):
        return op("slice", x, start=a, stop=b)

    def vsl(x, a, b):
        return op("vslice", x, start=a, stop=b)

    def cat(*xs):
        return op("concat", *xs)

    def vcat(*xs):
        return op("vconcat", *xs)

    def add(a, b):
        return op("add", a, b)

    def sub(a, b):
        return op("sub", a, b)

    def mul(a, b):
        return op("mul", a, b)

    def va(a, b):
        return op("vadd", a, b)

    def vm(a, b):
        return op("vmul", a, b)

    def smul(x, a, b=0.0):
        return conv(x, np.array([[[[a]]]], np.float32), [b])

    def vscale(x, a, b=0.0):
        return dense(x, [[a]], [b])

    def vneg(x, n=1):
        return dense(x, -np.eye(n, dtype=np.float32))

    def vabs(x, n=1):
        return va(op("vrelu", x), op("vrelu", vneg(x, n)))

    def vone_minus(x):
        return vscale(x, -1.0, 1.0)

    def vclip(x, n=1):
        p = op("vrelu", x)
        excess = op(
            "vrelu",
            dense(p, np.eye(n, dtype=np.float32),
                  -np.ones(n, dtype=np.float32))
        )
        return va(p, vneg(excess, n))

    def clip(x):
        return op("min2", op("relu", x), one)

    def grow(x, k=7):
        return clip(conv(x, np.ones((1, 1, k, k), np.float32)))

    def extrema(mask):
        return op(
            "gmax",
            cat(mul(mask, row), mul(mask, invrow),
                mul(mask, col), mul(mask, invcol))
        )

    def center(ext):
        return dense(
            ext,
            [[0.5, -0.5, 0, 0], [0, 0, 0.5, -0.5]],
            [0.5, 0.5]
        )

    def signed_dot(direction, values):
        return dense(vm(direction, values), [[1, 1, 1, 1]])

    def axis_surface(mask):
        e = extrema(mask)
        return dense(
            e,
            [[0, -1, 0, 0], [1, 0, 0, 0],
             [0, 0, 0, -1], [0, 0, 1, 0]],
            [1, 1.0 / 63.0, 1, 1.0 / 63.0]
        )

    def choose(mask, distance):
        score = mul(mask, smul(distance, -0.25, 2.0))
        best = op("gmax", score)
        near_best = op("relu", add(sub(score, best), smul(one, 0, 0.001)))
        return mul(mask, clip(smul(near_best, 1000.0)))

    def flood(name, seed, allowed, steps):
        w = parameter(np.ones((1, 1, 3, 3), np.float32))
        state = name + "_state"
        graph.append({
            "id": name,
            "op": "loop",
            "T": steps,
            "state": [{"id": state, "init": seed}],
            "body": [
                {"id": name + "_conv", "op": "conv",
                 "in": [state], "w": w},
                {"id": name + "_cap", "op": "min2",
                 "in": [name + "_conv", one]},
                {"id": name + "_inside", "op": "mul",
                 "in": [name + "_cap", allowed]},
                {"id": name + "_next", "op": "max2",
                 "in": [state, name + "_inside"]},
            ],
            "next": {state: name + "_next"},
        })
        return name + "." + state

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    one = sl("coords", 2, 3)
    invrow = sub(one, row)
    invcol = sub(one, col)

    black_raw = sl("x0", 0, 1)
    arrow = sl("x0", 4, 5)
    walls = sl("x0", 5, 6)
    buttons = sl("x0", 9, 10)
    active_switches = sl("x0", 12, 13)
    colour11 = sl("x0", 11, 12)
    colour14 = sl("x0", 14, 15)
    colours = cat(colour11, colour14)

    # Separate the piece's coloured edge from the matching marker.
    near_arrow = grow(arrow, 3)
    colour_seed = mul(colours, near_arrow)
    cw = np.zeros((2, 2, 3, 3), np.float32)
    cw[0, 0] = 1
    cw[1, 1] = 1
    cw_name = parameter(cw)
    graph.append({
        "id": "colour_flood",
        "op": "loop",
        "T": 4,
        "state": [{"id": "colour_state", "init": colour_seed}],
        "body": [
            {"id": "colour_conv", "op": "conv",
             "in": ["colour_state"], "w": cw_name},
            {"id": "colour_cap", "op": "min2",
             "in": ["colour_conv", one]},
            {"id": "colour_inside", "op": "mul",
             "in": ["colour_cap", colours]},
            {"id": "colour_next", "op": "max2",
             "in": ["colour_state", "colour_inside"]},
        ],
        "next": {"colour_state": "colour_next"},
    })
    ball_colours = "colour_flood.colour_state"
    present = op("gmax", ball_colours)
    goal_colours = mul(sub(colours, ball_colours), present)
    ball = clip(conv(ball_colours, np.ones((1, 2, 1, 1), np.float32)))
    goal = clip(conv(goal_colours, np.ones((1, 2, 1, 1), np.float32)))

    ball_ext = extrema(ball)
    goal_ext = extrema(goal)
    ball_center = center(ball_ext)
    goal_center = center(goal_ext)

    # Infer gravity from the arrow-to-coloured-edge displacement.
    arrow_local = mul(arrow, grow(ball, 7))
    arrow_ext = extrema(arrow_local)
    z = vcat(ball_ext, arrow_ext)
    dw = np.zeros((4, 8), np.float32)
    dw[0] = [1, -1, 0, 0, -1, 1, 0, 0]
    dw[1] = -dw[0]
    dw[2] = [0, 0, 1, -1, 0, 0, -1, 1]
    dw[3] = -dw[2]
    direction = vclip(dense(z, dw * 63.0), 4)
    vertical = dense(direction, [[1, 1, 0, 0]])
    horizontal = dense(direction, [[0, 0, 1, 1]])

    cross_coord = add(mul(row, horizontal), mul(col, vertical))
    axis_coord = add(mul(row, vertical), mul(col, horizontal))
    ball_cross = va(vm(vsl(ball_center, 0, 1), horizontal),
                    vm(vsl(ball_center, 1, 2), vertical))
    goal_cross = va(vm(vsl(goal_center, 0, 1), horizontal),
                    vm(vsl(goal_center, 1, 2), vertical))
    ball_axis = va(vm(vsl(ball_center, 0, 1), vertical),
                   vm(vsl(ball_center, 1, 2), horizontal))
    goal_axis = va(vm(vsl(goal_center, 0, 1), vertical),
                   vm(vsl(goal_center, 1, 2), horizontal))

    # Exclude decorative black brackets on enabled switches.
    active_rows = op("rowmax", active_switches)
    active_cols = op("colmax", active_switches)
    active_axis_span = add(mul(active_rows, vertical),
                           mul(active_cols, horizontal))
    brackets = mul(grow(active_switches, 7), clip(active_axis_span))
    black = mul(black_raw, sub(one, clip(brackets)))

    # Memory detects a click that left the board unchanged.
    black_row = op("gmean", mul(black, row))
    black_col = op("gmean", mul(black, col))
    signature = vcat(ball_ext, black_row, black_col)
    old_signature = vsl("mem", 0, 6)
    delta = va(signature, vneg(old_signature, 6))
    changed = vclip(dense(vabs(delta, 6), [[1e6] * 6]))
    previous_click = vsl("aprev", 6, 7)
    stalled = vm(previous_click, vone_minus(changed))

    old_mode = vsl("mem", 6, 7)
    old_origin = vsl("mem", 7, 8)
    cross_change = vabs(va(ball_cross, vneg(old_origin)))
    stayed = vone_minus(vclip(vscale(cross_change, 126.0)))
    keep_mode = vm(old_mode, stayed)
    mode = vclip(va(stalled, keep_mode))
    new_origin = va(vm(stalled, ball_cross),
                    vm(vone_minus(stalled), old_origin))
    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": [signature, mode, new_origin],
    })

    passable = clip(add(black, buttons))
    platform = flood("own_platform", ball, passable, 128)
    own_black = mul(platform, black)
    own_surface_vec = axis_surface(own_black)
    own_front = signed_dot(direction, own_surface_vec)

    own_rows = op("rowmax", own_black)
    own_cols = op("colmax", own_black)
    projection = add(mul(own_rows, horizontal), mul(own_cols, vertical))
    same_lane = op("gmax", mul(goal, grow(projection, 7)))
    different_lane = vone_minus(same_lane)

    # Detect a wall in the piece's lane between it and the marker.
    axis_delta = va(goal_axis, vneg(ball_axis))
    axis_positive = vclip(vscale(axis_delta, 63.0))
    axis_negative = vclip(vscale(vneg(axis_delta), 63.0))
    from_ball = add(
        mul(sub(axis_coord, ball_axis), axis_positive),
        mul(sub(ball_axis, axis_coord), axis_negative),
    )
    to_goal = add(
        mul(sub(goal_axis, axis_coord), axis_positive),
        mul(sub(axis_coord, goal_axis), axis_negative),
    )
    between = mul(
        clip(op("relu", smul(from_ball, 63.0, -3.0))),
        clip(op("relu", smul(to_goal, 63.0, -1.0))),
    )
    cross_distance_to_ball = op("abs", sub(cross_coord, ball_cross))
    near_ball_lane = clip(op(
        "relu", smul(cross_distance_to_ball, -63.0, 4.0)
    ))
    blocker = op("gmax", mul(mul(walls, between), near_ball_lane))

    # Select the nearest switch in the marker's transverse direction.
    switches = clip(add(sl("x0", 1, 2), active_switches))
    goal_cross_delta = va(goal_cross, vneg(ball_cross))
    positive_cross = vclip(vscale(goal_cross_delta, 63.0))
    negative_cross = vclip(vscale(vneg(goal_cross_delta), 63.0))
    signed_cross = add(
        mul(sub(cross_coord, ball_cross), positive_cross),
        mul(sub(ball_cross, cross_coord), negative_cross),
    )
    ahead = clip(op("relu", smul(signed_cross, 63.0, -1.0)))
    cross_distance = op("abs", sub(cross_coord, ball_cross))
    axis_distance = op("abs", sub(axis_coord, own_front))
    switch_distance = add(cross_distance, smul(axis_distance, 0.05))
    switch_seed = choose(mul(switches, ahead), switch_distance)
    selected_switch = flood("switch_component", switch_seed, switches, 16)
    has_switch = op("gmax", selected_switch)

    switch_center = center(extrema(selected_switch))
    switch_cross = va(vm(vsl(switch_center, 0, 1), horizontal),
                      vm(vsl(switch_center, 1, 2), vertical))

    # Flood the platform on the far side of the selected switch.
    switch_rows = op("rowmax", selected_switch)
    switch_cols = op("colmax", selected_switch)
    switch_projection = add(mul(switch_rows, horizontal),
                            mul(switch_cols, vertical))
    switch_band = grow(grow(switch_projection, 7), 7)
    far_side_distance = add(
        mul(sub(cross_coord, switch_cross), positive_cross),
        mul(sub(switch_cross, cross_coord), negative_cross),
    )
    far_side = clip(op("relu", smul(far_side_distance, 63.0, -1.0)))
    neighbor_seed = mul(
        mul(mul(black, switch_band), far_side),
        sub(one, clip(platform)),
    )
    neighbor = flood("route_neighbor", neighbor_seed, passable, 128)
    neighbor_black = mul(neighbor, black)
    neighbor_exists = op("gmax", neighbor_black)
    neighbor_surface_vec = axis_surface(neighbor_black)
    neighbor_front = signed_dot(direction, neighbor_surface_vec)

    se = extrema(selected_switch)
    sw = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ], np.float32)
    switch_surface = dense(se, sw, [1.0 / 63.0, 1, 1.0 / 63.0, 1])
    switch_front = signed_dot(direction, switch_surface)

    route_request = vclip(va(va(different_lane, mode), blocker))
    route = vm(vm(route_request, has_switch), neighbor_exists)

    own_error = va(switch_front, vneg(own_front))
    own_wrong = vclip(vscale(vabs(own_error), 630.0))
    use_neighbor = vm(route, vone_minus(own_wrong))
    use_own = vone_minus(use_neighbor)
    selected_platform = add(mul(platform, use_own),
                            mul(neighbor, use_neighbor))
    selected_black = mul(selected_platform, black)
    selected_front_vec = axis_surface(selected_black)
    selected_front = signed_dot(direction, selected_front_vec)

    direct_difference = va(goal_axis, vneg(ball_axis))
    route_difference = va(switch_front, vneg(selected_front))
    desired_axis = va(vm(route, route_difference),
                      vm(vone_minus(route), direct_difference))
    gravity_sign = dense(direction, [[1, -1, 1, -1]])
    desired_signed = vm(desired_axis, gravity_sign)

    # Select a control on the platform being adjusted. For the far
    # platform, prefer controls toward the marker along the gravity axis.
    control_target = va(vm(route, switch_cross),
                        vm(vone_minus(route), goal_cross))
    transverse_distance = op("abs", sub(cross_coord, control_target))
    toward_goal_distance = op("abs", sub(axis_coord, goal_axis))
    button_distance = add(
        transverse_distance,
        mul(smul(toward_goal_distance, 0.5), use_neighbor),
    )
    selected_buttons = mul(selected_platform, buttons)
    selected_button = choose(selected_buttons, button_distance)
    button_near = grow(grow(selected_button, 7), 7)
    opposite_button = mul(
        mul(buttons, button_near),
        sub(one, clip(selected_platform)),
    )

    own_preference = vclip(vscale(desired_signed, -63.0, 0.5))
    opposite_preference = vone_minus(own_preference)
    ordinary_click = add(mul(selected_button, own_preference),
                         mul(opposite_button, opposite_preference))

    # If a source is empty, attempt to replenish it from the next control.
    source = flood("source_platform", opposite_button, passable, 128)
    source_black = mul(source, black)
    source_exists = op("gmax", source_black)
    opposite_exists = op("gmax", opposite_button)
    need_refill = vm(
        vm(own_preference, opposite_exists),
        vone_minus(source_exists),
    )

    other_center = center(extrema(opposite_button))
    other_cross = va(vm(vsl(other_center, 0, 1), horizontal),
                     vm(vsl(other_center, 1, 2), vertical))
    selected_center = center(extrema(selected_button))
    selected_cross = va(vm(vsl(selected_center, 0, 1), horizontal),
                        vm(vsl(selected_center, 1, 2), vertical))
    away_delta = va(other_cross, vneg(selected_cross))
    away_positive = vclip(vscale(away_delta, 63.0))
    away_negative = vclip(vscale(vneg(away_delta), 63.0))
    away_distance = add(
        mul(sub(cross_coord, other_cross), away_positive),
        mul(sub(other_cross, cross_coord), away_negative),
    )
    beyond = clip(op("relu", smul(away_distance, 63.0, -3.0)))
    refill_candidates = mul(buttons, beyond)
    refill_button = choose(refill_candidates,
                           op("abs", sub(cross_coord, other_cross)))
    refill_exists = op("gmax", refill_button)
    refill_gate = vm(need_refill, refill_exists)
    control_click = add(mul(ordinary_click, vone_minus(refill_gate)),
                        mul(refill_button, refill_gate))

    # The rendered enabled-switch colour is a direct validity signal.
    enabled_selected = mul(selected_switch, active_switches)
    enabled_exists = op("gmax", enabled_selected)
    switch_gap = vabs(va(switch_cross, vneg(ball_cross)))
    switch_is_near = vclip(vscale(switch_gap, -63.0, 18.0))
    enabled_gate = vm(vm(enabled_exists, switch_is_near), route_request)

    neighbor_error = va(switch_front, vneg(neighbor_front))
    neighbor_wrong = vclip(vscale(vabs(neighbor_error), 630.0))
    aligned = vm(vone_minus(own_wrong), vone_minus(neighbor_wrong))
    aligned_gate = vm(route, aligned)
    switch_gate = vclip(va(enabled_gate, aligned_gate))
    switch_click = mul(selected_switch, switch_gate)

    click_map = add(control_click, smul(switch_click, 10.0))
    graph.append({"id": "click", "op": "add", "in": [click_map, one]})

    pooled = op("gmax", ball)
    logits_w = np.zeros((8, 1), np.float32)
    logits_b = np.zeros(8, np.float32)
    logits_b[6] = 1.0
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [pooled],
        "w": parameter(logits_w),
        "b": parameter(logits_b),
    })

    return {"graph": graph, "memory_size": 8, "params": params}
