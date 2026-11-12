import numpy as np


def build():
    graph, params = [], {}
    current = graph
    serial = 0
    pserial = 0
    shift_weights = {}
    shift_cache = {}

    def node(op, inputs=None, name=None, **fields):
        nonlocal serial
        if name is None:
            name = "n%d" % serial
            serial += 1
        nd = {"id": name, "op": op}
        if inputs is not None:
            nd["in"] = list(inputs)
        nd.update(fields)
        current.append(nd)
        return name

    def parameter(a):
        nonlocal pserial
        name = "p%d" % pserial
        pserial += 1
        params[name] = np.asarray(a, np.float32)
        return name

    def conv(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], name=name, **fields)

    def conv_ref(x, w):
        return node("conv", [x], w=w)

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], name=name, **fields)

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def cat(xs):
        return node("concat", xs)

    def vcat(xs, name=None):
        return node("vconcat", xs, name=name)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def relu(a):
        return node("relu", [a])

    def vr(a):
        return node("vrelu", [a])

    def gmax(a):
        return node("gmax", [a])

    def affine(a, scale=1.0, bias=0.0, n=1):
        w = (np.eye(n, dtype=np.float32) * scale).reshape(n, n, 1, 1)
        return conv(a, w, np.full(n, bias, np.float32))

    def vf(a, scale=1.0, bias=0.0, n=1):
        return dense(
            a, np.eye(n, dtype=np.float32) * scale,
            np.full(n, bias, np.float32)
        )

    def total(xs):
        result = xs[0]
        for x in xs[1:]:
            result = add(result, x)
        return result

    def vtotal(xs):
        result = xs[0]
        for x in xs[1:]:
            result = va(result, x)
        return result

    one = sl("coords", 2, 3)

    def clip(a):
        return node("min2", [relu(a), one])

    def vclip(a, n=1):
        positive = vr(a)
        excess = vr(vf(positive, 1, -1, n))
        return va(positive, vf(excess, -1, 0, n))

    def inv(a, n=1):
        return affine(a, -1, 1, n)

    def vi(a, n=1):
        return vf(a, -1, 1, n)

    def vpos(a, n=1):
        return vclip(vf(a, 2, -0.5, n), n)

    def shift(a, dy, dx):
        while dy or dx:
            sy = max(-3, min(3, dy))
            sx = max(-3, min(3, dx))
            key = (sy, sx)
            if key not in shift_weights:
                w = np.zeros((1, 1, 7, 7), np.float32)
                w[0, 0, 3 + sy, 3 + sx] = 1
                shift_weights[key] = parameter(w)
            cache_key = (a, sy, sx)
            if cache_key not in shift_cache:
                shift_cache[cache_key] = conv_ref(a, shift_weights[key])
            a = shift_cache[cache_key]
            dy -= sy
            dx -= sx
        return a

    def stamp(a, h, w, y0=0, x0=0):
        kernel = np.zeros((1, 1, 7, 7), np.float32)
        for y in range(y0, y0 + h):
            for x in range(x0, x0 + w):
                kernel[0, 0, 3 - y, 3 - x] += 1
        return conv(a, kernel)

    def match(a, positive, negative=()):
        kernel = np.zeros((1, 1, 7, 7), np.float32)
        for y, x in positive:
            kernel[0, 0, 3 + y, 3 + x] += 1
        for y, x in negative:
            kernel[0, 0, 3 + y, 3 + x] -= 1
        return relu(conv(a, kernel, [1 - len(positive)]))

    def foot(a, channels=1):
        kernel = np.zeros((channels, channels, 3, 3), np.float32)
        for c in range(channels):
            kernel[c, c, 1:3, 1:3] = 1
        return relu(conv(a, kernel, -3 * np.ones(channels, np.float32)))

    def channel_max(a, channels):
        result = sl(a, 0, 1)
        for c in range(1, channels):
            result = node("max2", [result, sl(a, c, c + 1)])
        return result

    color = [sl("x0", c, c + 1) for c in range(16)]
    ui = clip(affine(node("colsum", [color[5]]), 1, -16))
    board = inv(ui)
    raw = {
        c: mul(color[c], board)
        for c in (0, 1, 2, 5, 6, 7, 8, 9, 11, 12, 13, 15)
    }
    player_pixels = color[14]
    player = foot(player_pixels)
    final_goal = foot(raw[11])

    Y = conv("coords", np.array([63, 0, 0], np.float32).reshape(1, 3, 1, 1))
    X = conv("coords", np.array([0, 63, 0], np.float32).reshape(1, 3, 1, 1))
    index = conv(cat([Y, X]), np.array([64, 1], np.float32).reshape(1, 2, 1, 1))
    raw[0] = mul(raw[0], clip(affine(Y, -1, 63)))

    neighbour_kernel = np.zeros((1, 1, 5, 5), np.float32)
    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        neighbour_kernel[0, 0, 2 + dy, 2 + dx] = 1
    neighbour_weight = parameter(neighbour_kernel)

    def inpaint(a):
        count = conv_ref(a, neighbour_weight)
        supported = clip(affine(count, 1, -1))
        return clip(add(a, mul(supported, player_pixels)))

    red = inpaint(raw[2])
    pivot = inpaint(raw[13])
    blue = inpaint(raw[8])
    orange = inpaint(raw[9])
    pink = inpaint(raw[15])
    white = inpaint(clip(add(raw[1], raw[12])))

    ui_planes = {c: mul(color[c], ui) for c in (2, 6, 8, 9, 10, 11, 15)}

    def t_button(c):
        return match(
            ui_planes[c],
            [(0, 0), (0, 3), (3, -3)],
            [(0, -1), (-1, 0)]
        )

    button_a = t_button(8)
    button_b = t_button(9)
    button_c = t_button(6)
    button_d = t_button(15)
    button_e = ui_planes[2]
    button_f = ui_planes[11]
    buttons = [button_a, button_b, button_c, button_d, button_e, button_f]
    present = [gmax(b) for b in buttons]

    square_positive = [(0, 0), (0, 3), (3, 0), (3, 3)]
    square_negative = [(-1, 0), (0, -1)]
    horizontal_controls = match(ui_planes[10], square_positive, square_negative)
    vertical_controls = relu(sub(
        match(ui_planes[9], square_positive, square_negative), button_b
    ))

    def split_left_right(markers):
        largest_x = gmax(mul(X, markers))
        difference = node("abs", [sub(X, largest_x)])
        right = mul(markers, clip(affine(difference, -4, 2)))
        return relu(sub(markers, right)), right

    crane_left, crane_right = split_left_right(horizontal_controls)
    crane_up, crane_down = split_left_right(vertical_controls)
    crane_grab = match(
        ui_planes[8],
        [(0, 0), (0, 3), (2, 0), (2, 3)],
        [(-1, 0), (0, -1), (3, 0)]
    )

    # Persistent point records: (row, column, valid).
    def select_point(marker):
        largest = gmax(mul(index, marker))
        difference = node("abs", [sub(index, largest)])
        selected = mul(marker, clip(affine(difference, -4, 2)))
        return vcat([
            gmax(mul(Y, selected)),
            gmax(mul(X, selected)),
            gmax(selected)
        ])

    def point_map(record):
        ry, rx, valid = vs(record, 0, 1), vs(record, 1, 2), vs(record, 2, 3)
        dy = node("abs", [sub(Y, ry)])
        dx = node("abs", [sub(X, rx)])
        return mul(clip(affine(add(dy, dx), -4, 2)), valid)

    def update_record(old, candidates):
        valid = vs(old, 2, 3)
        return va(vm(old, valid), vm(select_point(candidates), vi(valid)))

    tele_observed = foot(clip(add(raw[6], raw[7])))
    first = update_record(vs("mem", 0, 3), tele_observed)
    first_map = point_map(first)
    second = update_record(vs("mem", 3, 6), relu(sub(tele_observed, first_map)))
    second_map = point_map(second)
    tele_points = clip(total([tele_observed, first_map, second_map]))
    tele_floor = clip(stamp(tele_points, 2, 2))
    tele_edges = mul(tele_points, present[2])

    # Require the distinctive pink upper pair. Blue is permitted to occlude
    # the lower corners, but does not count against the negative pink tests.
    pickup_allowed = clip(add(raw[15], blue))
    pickup_kernel = np.zeros((1, 2, 7, 7), np.float32)
    for y, x in ((0, 0), (0, 1)):
        pickup_kernel[0, 0, 3 + y, 3 + x] = 1
    for y, x in ((1, -1), (1, 2)):
        pickup_kernel[0, 1, 3 + y, 3 + x] = 1
    for y, x in ((0, -1), (0, 2), (-1, 0), (2, 0)):
        pickup_kernel[0, 0, 3 + y, 3 + x] -= 1
    pickup_observed = relu(conv(
        cat([raw[15], pickup_allowed]), pickup_kernel, [-3]
    ))
    pickup_record = update_record(vs("mem", 7, 10), pickup_observed)
    pickup_marker = point_map(pickup_record)
    pickup = clip(stamp(pickup_marker, 2, 4, x0=-1))
    pickup_exists = gmax(pickup)
    need_pickup = vm(pickup_exists, vi(present[3]))
    goal = add(
        mul(final_goal, vi(need_pickup)),
        mul(foot(pickup), need_pickup)
    )

    def checker_and_solid(a):
        kernel = np.zeros((1, 1, 3, 3), np.float32)
        kernel[0, 0, 1, 1] = 1
        kernel[0, 0, 2, 2] = 1
        kernel[0, 0, 1, 2] = -1
        kernel[0, 0, 2, 1] = -1
        contrast = conv(a, kernel)
        d1 = relu(affine(contrast, 1, -1))
        d2 = relu(affine(contrast, -1, -1))
        closed = clip(stamp(add(d1, d2), 2, 2))
        return closed, relu(sub(a, closed))

    orange_closed, orange_open = checker_and_solid(orange)
    pink_closed, pink_open = checker_and_solid(relu(sub(pink, pickup)))

    # Connected blue bridge pixels, distinguished from the blue crane.
    blue_allowed = clip(add(blue, pivot))
    blue_kernel = np.zeros((1, 1, 3, 3), np.float32)
    for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        blue_kernel[0, 0, 1 + dy, 1 + dx] = 1
    blue_weight = parameter(blue_kernel)

    saved = current
    blue_body = []
    current = blue_body
    spread = conv_ref("bridge_reach", blue_weight)
    spread = node("min2", [spread, one])
    spread = node("min2", [spread, blue_allowed])
    current = saved
    node(
        "loop", name="bridge_component", T=16,
        state=[{"id": "bridge_reach", "init": pivot}],
        body=blue_body, next={"bridge_reach": spread}
    )
    blue_current = "bridge_component.bridge_reach"

    anchor_kernel = np.zeros((1, 1, 3, 3), np.float32)
    anchor_kernel[0, 0, 1, 1] = 1
    anchor_kernel[0, 0, 0, 1] = -1
    anchor_kernel[0, 0, 1, 0] = -1
    pivot_anchor = relu(conv(pivot, anchor_kernel))
    alternate_parts = [pivot]
    for direction in range(4):
        previous = pivot_anchor
        for d in range(2, 13, 2):
            if direction == 0:
                py, px, dy, dx, h, w = 1, -d, -d, 0, 2, 4
            elif direction == 1:
                py, px, dy, dx, h, w = 1, d + 2, d + 2, 0, 2, 4
            elif direction == 2:
                py, px, dy, dx, h, w = -d, 1, 0, -d, 4, 2
            else:
                py, px, dy, dx, h, w = d + 2, 1, 0, d + 2, 4, 2
            previous = mul(previous, shift(blue, py, px))
            alternate_parts.append(stamp(shift(previous, -dy, -dx), h, w))
    blue_alternate = clip(total(alternate_parts))

    # Optimistic extension and sliding-platform envelopes.
    thin = relu(sub(sub(white, shift(white, -2, 0)), shift(white, 2, 0)))
    horizontal7 = np.zeros((1, 1, 7, 7), np.float32)
    horizontal7[0, 0, 3, :] = 1
    wide = clip(affine(conv(white, horizontal7), 1, -6))
    extension_rows = node("rowmax", [clip(add(raw[5], mul(wide, thin)))])
    extension = clip(add(raw[5], mul(white, extension_rows)))

    horizontal5 = np.zeros((1, 1, 5, 5), np.float32)
    horizontal5[0, 0, 2, :] = 1
    long_enough = clip(affine(conv(white, horizontal5), 1, -4))
    sliding_seed = mul(mul(thin, inv(extension_rows)), long_enough)
    sliding_rows = node("rowmax", [sliding_seed])
    extension_columns = node("colmax", [extension])
    extension_exists = gmax(extension)
    sliding_dilation = sliding_seed
    for _ in range(4):
        sliding_dilation = clip(conv(sliding_dilation, horizontal7))
    sliding_extent = add(
        mul(extension_columns, extension_exists),
        mul(sliding_dilation, vi(extension_exists))
    )
    sliding = mul(mul(sliding_rows, sliding_extent), board)

    vertical3 = np.zeros((1, 1, 3, 3), np.float32)
    vertical3[0, 0, :, 1] = 1
    thick_seed = relu(conv(white, vertical3, [-2]))
    carried_object = clip(conv(thick_seed, vertical3))
    fixed_white = relu(sub(white, carried_object))

    # Crane and rail landmarks are measured in the observation.
    claw = match(
        blue,
        [(0, 0), (1, 0), (-1, 1), (2, 1), (0, 3)],
        [(0, -1), (-1, 0)]
    )
    crane_x = vf(gmax(mul(X, claw)), 1, 9)
    crane_y = vf(gmax(mul(Y, claw)), 1, 1)
    open_claw = gmax(mul(claw, shift(blue, -3, 3)))
    attached = vi(open_claw)

    def minimum_coordinate(coordinate, mask):
        return vf(gmax(mul(affine(coordinate, -1, 64), mask)), -1, 64)

    track_left = vf(minimum_coordinate(X, raw[0]), 1, 8)
    track_top = vf(minimum_coordinate(Y, raw[0]), 1, 2)
    corner_x = dense(track_left, np.ones((4, 1), np.float32), [0, 12, 0, 12])
    corner_y = dense(track_top, np.ones((4, 1), np.float32), [0, 0, 12, 12])

    object_x = vf(va(
        minimum_coordinate(X, carried_object), gmax(mul(X, carried_object))
    ), 0.5, 0.5)
    object_y = vf(va(
        minimum_coordinate(Y, carried_object), gmax(mul(Y, carried_object))
    ), 0.5, 0.5)
    has_object = vm(vm(gmax(carried_object), gmax(claw)), gmax(raw[0]))

    dx = affine(sub(X, corner_x), 1, -0.5, 4)
    dy = affine(sub(Y, corner_y), 1, 0.5, 4)
    corner_floor_x = clip(affine(node("abs", [dx]), -4, 8, 4))
    corner_floor_y = clip(affine(node("abs", [dy]), -4, 24, 4))
    corner_floors = mul(corner_floor_x, corner_floor_y)

    static_floor = clip(total([red, pivot, raw[11], tele_floor, pickup]))
    toggle_safe = foot(clip(add(static_floor, white)))
    crane_safe = foot(clip(add(static_floor, fixed_white)))
    conditional = clip(total([blue_current, orange_open, pink_open]))
    unclassified_player = relu(sub(player_pixels, conditional))
    base_floor = clip(total([static_floor, fixed_white, unclassified_player]))

    # Forty planning channels: eight switch configurations at five poses.
    floor_features = cat([
        base_floor, blue_current, blue_alternate,
        orange_open, orange_closed, pink_open, pink_closed,
        extension, sliding, carried_object, corner_floors
    ])
    floor_weights = np.zeros((40, 14, 1, 1), np.float32)
    for pose in range(5):
        for bits in range(8):
            s = 8 * pose + bits
            floor_weights[s, 0, 0, 0] = 1
            floor_weights[s, 2 if bits & 1 else 1, 0, 0] = 1
            floor_weights[s, 4 if bits & 2 else 3, 0, 0] = 1
            floor_weights[s, 6 if bits & 4 else 5, 0, 0] = 1
            floor_weights[s, 7:9, 0, 0] = 1
            floor_weights[s, 9 + pose, 0, 0] = 1
    floors = mul(clip(conv(floor_features, floor_weights)), board)
    walk = foot(floors, 40)

    pink_available = vclip(va(present[3], pickup_exists))
    controls = vcat([present[0], present[1], pink_available, has_object])
    allowed_w = np.zeros((40, 4), np.float32)
    allowed_b = np.ones(40, np.float32)
    for pose in range(5):
        for bits in range(8):
            s = 8 * pose + bits
            for bit in range(3):
                if bits & (1 << bit):
                    allowed_w[s, bit] = 1
                    allowed_b[s] -= 1
            if pose:
                allowed_w[s, 3] = 1
                allowed_b[s] -= 1
    allowed = vr(dense(controls, allowed_w, allowed_b))
    walk = mul(walk, allowed)

    pink_locations = add(mul(one, present[3]), mul(pickup, vi(present[3])))
    pink_safe = mul(toggle_safe, pink_locations)
    seed = mul(walk, goal)

    movement_kernel = np.zeros((1, 1, 5, 5), np.float32)
    for dy, dx in ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)):
        movement_kernel[0, 0, 2 + dy, 2 + dx] = 1
    movement_weight = parameter(movement_kernel)

    ab_adj = np.zeros((8, 8, 1, 1), np.float32)
    d_adj = np.zeros_like(ab_adj)
    for s in range(8):
        ab_adj[s, s ^ 1, 0, 0] = 1
        ab_adj[s, s ^ 2, 0, 0] = 1
        d_adj[s, s ^ 4, 0, 0] = 1
    ab_weight, d_weight = parameter(ab_adj), parameter(d_adj)

    crane_adj = np.zeros((40, 40, 1, 1), np.float32)
    for p in range(5):
        for q in range(5):
            if p != q:
                for bits in range(8):
                    crane_adj[8 * p + bits, 8 * q + bits, 0, 0] = 1
    crane_weight = parameter(crane_adj)

    saved = current
    body = []
    current = body
    movement_channels = [
        conv_ref(sl("reach_state", s, s + 1), movement_weight)
        for s in range(40)
    ]
    moved = cat(movement_channels)
    ab_channels, d_channels = [], []
    for pose in range(5):
        block = sl("reach_state", 8 * pose, 8 * pose + 8)
        ab_channels.append(conv_ref(block, ab_weight))
        d_channels.append(conv_ref(block, d_weight))
    toggled = mul(cat(ab_channels), toggle_safe)
    pink_toggled = mul(cat(d_channels), pink_safe)
    repositioned = mul(conv_ref("reach_state", crane_weight), crane_safe)
    tele_reached = gmax(mul("reach_state", tele_edges))
    teleported = mul(tele_edges, tele_reached)
    reached = total([moved, toggled, pink_toggled, repositioned, teleported])
    reached = node("min2", [reached, one])
    reached = mul(reached, walk)
    accumulated = add("distance_state", "reach_state")
    current = saved

    node(
        "loop", name="solve", T=80,
        state=[
            {"id": "reach_state", "init": seed},
            {"id": "distance_state", "zeros": [40, 64, 64]}
        ],
        body=body,
        next={"reach_state": reached, "distance_state": accumulated}
    )
    distances = "solve.distance_state"
    current_distance = sl(distances, 0, 1)
    current_value = gmax(mul(current_distance, player))

    action_kernel = np.zeros((4, 1, 5, 5), np.float32)
    for a, (dy, dx) in enumerate(((-2, 0), (2, 0), (0, -2), (0, 2))):
        action_kernel[a, 0, 2 + dy, 2 + dx] = 1
    action_weight = parameter(action_kernel)
    planned_moves = mul(conv_ref(current_distance, action_weight), player)

    # Execute only currently supported movement. Missing optimistic floor
    # instead requests the corresponding platform control.
    actual_floor = clip(total([
        base_floor, carried_object, blue_current, orange_open, pink_open
    ]))
    actual_walk = foot(actual_floor)
    actual_neighbours = conv_ref(actual_walk, action_weight)
    movement_scores = gmax(mul(planned_moves, actual_neighbours))
    missing_moves = mul(planned_moves, inv(actual_neighbours, 4))

    extension_neighbours = conv_ref(foot(extension), action_weight)
    sliding_neighbours = conv_ref(foot(sliding), action_weight)
    score_e = vm(
        gmax(channel_max(mul(missing_moves, extension_neighbours), 4)),
        present[4]
    )
    score_f = vm(
        gmax(channel_max(mul(missing_moves, sliding_neighbours), 4)),
        present[5]
    )

    safe_here = gmax(mul(player, toggle_safe))

    def switch_score(state, available):
        value = gmax(mul(sl(distances, state, state + 1), player))
        return vm(vm(value, safe_here), available)

    score_a = switch_score(1, present[0])
    score_b = switch_score(2, present[1])
    score_d = switch_score(4, present[3])
    tele_here = gmax(mul(player, tele_edges))
    score_c = vm(gmax(mul(current_distance, tele_edges)), tele_here)

    control_scores = vcat([score_a, score_b, score_c, score_d, score_e, score_f])
    planned_click = channel_max(mul(cat(buttons), control_scores), 6)

    pose_scores = vcat([
        gmax(mul(sl(distances, 8 * p, 8 * p + 1), player))
        for p in range(1, 5)
    ])
    improvement = vpos(va(pose_scores, vf(current_value, -1)), 4)
    pose_scores = vm(pose_scores, improvement)
    pose_scores = vm(pose_scores, vm(gmax(mul(player, crane_safe)), has_object))

    target_x = va(vm(corner_x, attached), vm(object_x, vi(attached)))
    target_y = va(vm(corner_y, attached), vm(object_y, vi(attached)))
    target_dx = va(target_x, vf(crane_x, -1))
    target_dy = va(target_y, vf(crane_y, -1))
    need_right = vpos(target_dx, 4)
    need_left = vpos(vf(target_dx, -1, 0, 4), 4)
    need_down = vpos(target_dy, 4)
    need_up = vpos(vf(target_dy, -1, 0, 4), 4)
    need_y = va(need_up, need_down)
    off_left = vpos(va(crane_x, vf(track_left, -1)))
    dir_left = va(vm(need_y, off_left), vm(vi(need_y, 4), need_left))
    dir_right = vm(vi(need_y, 4), need_right)
    dir_up = vm(need_up, vi(off_left))
    dir_down = vm(need_down, vi(off_left))
    at_target = vi(vclip(vtotal([need_left, need_right, need_up, need_down]), 4), 4)
    dir_grab = vm(at_target, vi(attached))

    crane_maps = total([
        mul(crane_left, dir_left),
        mul(crane_up, dir_up),
        mul(crane_right, dir_right),
        mul(crane_down, dir_down),
        mul(crane_grab, dir_grab)
    ])
    crane_click = channel_max(mul(crane_maps, pose_scores), 4)
    planned_click = node("max2", [planned_click, crane_click])
    click_score = gmax(planned_click)

    has_route = vclip(current_value)
    no_route = vi(has_route)

    # No-route fallback is indexed by detected control identities.
    fallback_planes = cat(buttons + [
        crane_left, crane_up, crane_right, crane_down, crane_grab
    ])
    A, B, C, D, E, F, L, U, R, DOWN, GRAB = range(11)
    sweep = (
        [U] * 3 + [R] * 3 + [GRAB]
        + [L] * 3 + [DOWN] * 3 + [R] * 3 + [GRAB]
        + [A, B, C, D] + [E] * 4 + [F] * 4 + [B, D, A, C, GRAB]
    )
    period = len(sweep)
    clock = vs("mem", 6, 7)
    triangular_w = np.ones((3 * period, 1), np.float32)
    triangular_b = np.empty(3 * period, np.float32)
    for i in range(period):
        triangular_b[3 * i:3 * i + 3] = [1 - i, -i, -1 - i]
    triangular = vr(dense(clock, triangular_w, triangular_b))
    phase_w = np.zeros((period, 3 * period), np.float32)
    for i in range(period):
        phase_w[i, 3 * i:3 * i + 3] = [1, -2, 1]
    phases = dense(triangular, phase_w)
    fallback_w = np.zeros((11, period), np.float32)
    for i, command in enumerate(sweep):
        fallback_w[command, i] = 3
    fallback_scores = dense(phases, fallback_w, np.full(11, 0.01, np.float32))
    fallback_map = conv(
        mul(fallback_planes, fallback_scores),
        np.ones((1, 11, 1, 1), np.float32)
    )
    node("add", [planned_click, mul(fallback_map, no_route)], name="click")

    output_features = vcat([movement_scores, click_score, no_route])
    output_w = np.zeros((8, 6), np.float32)
    output_b = np.full(8, -10000, np.float32)
    for i in range(4):
        output_w[i + 1, i] = 1
        output_b[i + 1] = 0
    output_w[6, 4] = 1
    output_w[6, 5] = 81
    output_b[6] = 0
    dense(output_features, output_w, output_b, name="logits")

    advanced_clock = va(clock, no_route)
    wrap = vr(vf(advanced_clock, 1, -(period - 1)))
    next_clock = va(advanced_clock, vf(wrap, -period))
    vcat([first, second, next_clock, pickup_record], name="mem_next")

    # Ordinary static-graph dead-code elimination; no evaluation or fitting.
    def prune(nodes, roots):
        needed = set(roots)
        kept = []
        for nd in reversed(nodes):
            if nd["op"] == "loop":
                outputs = {nd["id"] + "." + s["id"] for s in nd["state"]}
            else:
                outputs = {nd["id"]}
            if not outputs.intersection(needed):
                continue
            needed.difference_update(outputs)
            if nd["op"] == "loop":
                body_kept, body_reads = prune(nd["body"], nd["next"].values())
                body_reads.difference_update(s["id"] for s in nd["state"])
                needed.update(body_reads)
                for state in nd["state"]:
                    if "init" in state:
                        needed.add(state["init"])
                nd = dict(nd)
                nd["body"] = body_kept
            else:
                needed.update(nd["in"])
            kept.append(nd)
        kept.reverse()
        return kept, needed

    graph, _ = prune(graph, ["logits", "click", "mem_next"])
    used = set()

    def collect_parameters(nodes):
        for nd in nodes:
            for field in ("w", "b"):
                if field in nd:
                    used.add(nd[field])
            if nd["op"] == "loop":
                collect_parameters(nd["body"])

    collect_parameters(graph)
    params = {name: value for name, value in params.items() if name in used}
    return {"graph": graph, "memory_size": 10, "params": params}
