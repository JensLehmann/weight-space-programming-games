import numpy as np


def build():
    graph, params = [], {}
    dims = {"x0": 16, "coords": 3, "mem": 5, "aprev": 8, "avail": 8}
    pcache, cache = {}, {}
    counter = [0]
    destination = [graph]

    def parameter(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in pcache:
            name = "p" + str(len(params))
            pcache[key] = name
            params[name] = a
        return pcache[key]

    def node(op, ins, size, name=None, **fields):
        key = (op, tuple(ins), tuple(sorted(fields.items())))
        if name is None and key in cache:
            return cache[key]
        if name is None:
            name = "n" + str(counter[0])
            counter[0] += 1
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        destination[0].append(nd)
        dims[name] = size
        if name.startswith("n"):
            cache[key] = name
        return name

    def conv(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        f = {"w": parameter(w)}
        if b is not None:
            f["b"] = parameter(b)
        return node("conv", [x], w.shape[0], name=name, **f)

    def dense(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        f = {"w": parameter(w)}
        if b is not None:
            f["b"] = parameter(b)
        return node("dense", [x], w.shape[0], name=name, **f)

    def sl(x, a, b):
        return node("slice", [x], b - a, start=a, stop=b)

    def vs(x, a, b):
        return node("vslice", [x], b - a, start=a, stop=b)

    def un(op, x):
        return node(op, [x], dims[x])

    def bi(op, a, b):
        return node(op, [a, b], max(dims[a], dims[b]))

    def cat(xs):
        return node("concat", xs, sum(dims[x] for x in xs))

    def vcat(xs):
        return node("vconcat", xs, sum(dims[x] for x in xs))

    def gm(x):
        return node("gmax", [x], dims[x])

    def mean(x):
        return node("gmean", [x], dims[x])

    def sa(x, scale=1.0, bias=0.0):
        w = np.eye(dims[x], dtype=np.float32)[:, :, None, None] * scale
        return conv(x, w, np.full(dims[x], bias, np.float32))

    def va(x, scale=1.0, bias=0.0):
        return dense(x, np.eye(dims[x], dtype=np.float32) * scale,
                     np.full(dims[x], bias, np.float32))

    def vc(values):
        b = np.asarray(values, np.float32).reshape(-1)
        return dense("avail", np.zeros((len(b), 8), np.float32), b)

    one = sl("coords", 2, 3)
    vone = vc([1.0])

    def cap(x):
        return bi("min2", un("relu", x), one)

    def vcap(x):
        r = un("vrelu", x)
        return va(un("vrelu", va(r, -1.0, 1.0)), -1.0, 1.0)

    def inv(x):
        return sa(x, -1.0, 1.0)

    def vinv(x):
        return va(x, -1.0, 1.0)

    def maximum(xs):
        out = xs[0]
        for x in xs[1:]:
            out = bi("max2", out, x)
        return out

    def sum_spatial(xs):
        out = xs[0]
        for x in xs[1:]:
            out = bi("add", out, x)
        return out

    def sum_vector(xs):
        out = xs[0]
        for x in xs[1:]:
            out = bi("vadd", out, x)
        return out

    def vm(a, b):
        return bi("vmul", a, b)

    def vsub(a, b):
        return bi("vadd", a, va(b, -1.0))

    def sample(x, dy, dx):
        out = x
        while dy or dx:
            sy, sx = max(-3, min(3, dy)), max(-3, min(3, dx))
            k = 2 * max(abs(sy), abs(sx)) + 1
            w = np.zeros((1, 1, k, k), np.float32)
            w[0, 0, k // 2 + sy, k // 2 + sx] = 1.0
            out = conv(out, w)
            dy -= sy
            dx -= sx
        return out

    def translate(x, dy, dx):
        return sample(x, -dy, -dx)

    def local_sum(x, k=5):
        return conv(x, np.ones((1, 1, k, k), np.float32))

    def top_left(x):
        return un("relu", bi("sub",
                  bi("sub", x, sample(x, -1, 0)), sample(x, 0, -1)))

    def template(x, positives, negatives=()):
        w = np.zeros((1, 1, 7, 7), np.float32)
        for dy, dx in positives:
            w[0, 0, dy + 3, dx + 3] += 1.0
        for dy, dx in negatives:
            w[0, 0, dy + 3, dx + 3] -= 2.0
        return cap(conv(x, w, [1.0 - len(positives)]))

    def choose(q):
        n = dims[q]
        priorities = 0.04 * np.arange(n, 0, -1, dtype=np.float32)
        rows, biases = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    r = np.zeros(n, np.float32)
                    r[j], r[i] = 1.0, -1.0
                    rows.append(r)
                    biases.append(priorities[j] - priorities[i])
        differences = un("vrelu", dense(q, rows, biases))
        w = np.zeros((n, n * (n - 1)), np.float32)
        for i in range(n):
            w[i, i * (n - 1):(i + 1) * (n - 1)] = -100.0
        return un("vrelu", dense(differences, w, np.ones(n, np.float32)))

    def selected(q, selector):
        z = vm(q, selector)
        return dense(z, np.ones((1, dims[z]), np.float32))

    colours = [sl("x0", c, c + 1) for c in range(16)]
    board_columns = node("colmax", [colours[4]], 1)
    board_rows = node("rowmax", [colours[4]], 1)
    board = bi("mul", board_columns, board_rows)
    panel = inv(board_columns)
    paint = [bi("mul", c, board) for c in colours]
    panel_paint = [bi("mul", c, panel) for c in colours]

    w = np.zeros((2, 3, 1, 1), np.float32)
    w[0, 0, 0, 0] = 63.0
    w[1, 1, 0, 0] = 63.0
    grid = conv("coords", w)
    gr, gc = sl(grid, 0, 1), sl(grid, 1, 2)
    signed_grid = conv(
        grid, np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]],
                         np.float32)[:, :, None, None]
    )

    def bbox(mask):
        ext = gm(bi("add", signed_grid, sa(mask, 10000.0, -10000.0)))
        return dense(ext, np.asarray([
            [0, 0, -1, 0], [0, 0, 0, -1],
            [1, 0, 0, 0], [0, 1, 0, 0]
        ], np.float32))

    def position(mask):
        return gm(bi("mul", grid, mask))

    pawn_paint = paint[14]
    pawn = top_left(pawn_paint)
    goal = top_left(paint[11])

    crown = [
        (-2, -3), (-2, 0), (-2, 3),
        (0, -3), (0, 0), (0, 3),
        (2, -3), (2, 0), (2, 3),
    ]
    button_a = template(panel_paint[8], crown)
    button_b = template(panel_paint[9], crown)
    button_c = template(panel_paint[6], crown)
    button_d = template(panel_paint[15], crown)
    button_e, button_f = panel_paint[2], panel_paint[11]
    have_a, have_b = gm(button_a), gm(button_b)
    have_c, have_d = gm(button_c), gm(button_d)
    have_e, have_f = gm(button_e), gm(button_f)

    square_pos = [(-1, -1), (-1, 2), (2, -1), (2, 2)]
    square_neg = [(-2, 0), (3, 0), (0, -2), (0, 3)]
    vertical_buttons = template(panel_paint[9], square_pos, square_neg)
    horizontal_buttons = template(panel_paint[10], square_pos, square_neg)
    grab_button = template(
        panel_paint[8],
        [(-1, -3), (-1, 0), (-1, 3), (1, -3), (1, 0), (1, 3)],
        [(-2, 0), (2, 0)]
    )
    col_fraction = sl("coords", 1, 2)
    left_weight, right_weight = sa(col_fraction, -1.0, 2.0), sa(col_fraction, 1.0, 1.0)
    click_up = bi("mul", vertical_buttons, left_weight)
    click_down = bi("mul", vertical_buttons, right_weight)
    click_left = bi("mul", horizontal_buttons, left_weight)
    click_right = bi("mul", horizontal_buttons, right_weight)

    # The small pink pickup is not a pink bridge.
    pickup_middle = template(
        paint[15],
        [(0, 0), (0, 1), (1, -1), (1, 2)],
        [(-1, 0), (0, -1), (0, 2), (2, 0)]
    )
    pickup = translate(pickup_middle, 0, -1)
    pickup_w = np.zeros((1, 1, 7, 7), np.float32)
    for dy in range(2):
        for dx in range(4):
            pickup_w[0, 0, 3 - dy, 3 - dx] = 1.0
    pickup_area = conv(pickup, pickup_w)
    pink_paint = un("relu", bi("sub", paint[15], pickup_area))
    pickup_pending = vm(gm(pickup), vinv(have_d))
    target = maximum([
        bi("mul", goal, vinv(pickup_pending)),
        bi("mul", pickup, pickup_pending),
    ])
    goal_reward = sa(target, 1000.0)

    cross = np.zeros((1, 1, 3, 3), np.float32)
    cross[0, 0, 0, 1] = cross[0, 0, 2, 1] = 1.0
    cross[0, 0, 1, 0] = cross[0, 0, 1, 2] = 1.0

    def platform_pair(p):
        solid = bi("min2", p, conv(p, cross))
        checker = bi("sub", p, solid)
        holes = cap(conv(checker, cross, [-1.0]))
        return solid, maximum([checker, holes])

    orange_solid, orange_checker = platform_pair(paint[9])
    pink_solid, pink_checker = platform_pair(pink_paint)

    pivot_hidden = bi("mul", pawn_paint,
                      cap(sa(local_sum(paint[13]), 1.0, -4.0)))
    pivot_paint = maximum([paint[13], pivot_hidden])
    arm_hidden = bi("mul", pawn_paint,
                    cap(sa(local_sum(paint[8]), 1.0, -3.0)))
    arm_paint = maximum([paint[8], arm_hidden])
    pivots = top_left(pivot_paint)
    h = bi("mul", pivots, sample(arm_paint, 0, -1))
    v = bi("mul", pivots, sample(arm_paint, -1, 0))
    hf = bi("mul", h, sample(arm_paint, 0, 4))
    vf = bi("mul", v, sample(arm_paint, 4, 0))
    hl, vl = bi("sub", h, hf), bi("sub", v, vf)

    rectangle_w = np.zeros((1, 1, 7, 7), np.float32)
    for dy in range(4):
        for dx in range(4):
            rectangle_w[0, 0, 3 - dy, 3 - dx] = 1.0

    def blue_bar(anchor, vertical, full):
        centre = conv(anchor, rectangle_w)
        offsets = [-8, -4, 4, 8] if full else [-6, -3]
        pieces = [centre]
        for d in offsets:
            pieces.append(translate(centre, d if vertical else 0,
                                    0 if vertical else d))
        return maximum(pieces)

    blue_current = maximum([
        pivot_paint, blue_bar(hf, False, True), blue_bar(vf, True, True),
        blue_bar(hl, False, False), blue_bar(vl, True, False),
    ])
    blue_alternate = maximum([
        pivot_paint, blue_bar(hf, True, True), blue_bar(vf, False, True),
        blue_bar(hl, True, False), blue_bar(vl, False, False),
    ])

    # Retain portal endpoints while one endpoint is obscured.
    portal_obs = top_left(maximum([paint[6], paint[7]]))
    both_portals = vcap(va(va(mean(portal_obs), 4096.0), 1.0, -1.0))
    order = conv(grid, np.asarray([64.0, 1.0], np.float32).reshape(1, 2, 1, 1))
    outside = sa(portal_obs, 10000.0, -10000.0)
    largest = gm(bi("add", order, outside))
    smallest = va(gm(bi("add", sa(order, -1.0), outside)), -1.0)

    def ordered_portal(value):
        distance = un("abs", bi("sub", order, value))
        return bi("mul", cap(sa(distance, -4.0, 2.0)), portal_obs)

    observed_memory = vcat([
        vone, position(ordered_portal(smallest)),
        position(ordered_portal(largest)),
    ])
    new_memory = bi("vadd", vm(observed_memory, both_portals),
                    vm("mem", vinv(both_portals)))
    mem_next = dense(new_memory, np.eye(5, dtype=np.float32), name="mem_next")
    remembered = vs(mem_next, 0, 1)

    def remembered_point(ri, ci):
        distance = bi(
            "add",
            un("abs", bi("sub", gr, vs(mem_next, ri, ri + 1))),
            un("abs", bi("sub", gc, vs(mem_next, ci, ci + 1))),
        )
        return bi("mul", cap(sa(distance, -2.0, 1.0)), remembered)

    portals = maximum([
        portal_obs, remembered_point(1, 2), remembered_point(3, 4)
    ])

    thin12 = cap(bi("sub", bi("sub", paint[12],
                               sample(paint[12], -2, 0)),
                       sample(paint[12], 2, 0)))
    thin12_rows = node("rowsum", [thin12], 1)
    six_wide12 = cap(sa(un("abs", sa(thin12_rows, 1.0, -6.0)), -2.0, 1.0))
    f_ready = gm(bi("mul", thin12, six_wide12))
    e_closed = gm(paint[5])

    red_near = cap(sa(local_sum(paint[2]), 1.0, -4.0))
    permanent = maximum([
        red_near, portals, cap(local_sum(paint[1])),
        cap(local_sum(thin12)), cap(local_sum(paint[11])),
    ])
    pawn_permanent = gm(bi("mul", pawn, permanent))
    static_pawn = bi("mul", pawn_paint, permanent)
    blue_pawn = bi("mul", pawn_paint, cap(local_sum(blue_current)))
    orange_pawn = bi("mul", pawn_paint, cap(local_sum(orange_solid)))
    pink_pawn = bi("mul", pawn_paint, cap(local_sum(pink_solid)))

    # The additional negative sample fixes the two-centre jaw ambiguity.
    crane_open = template(
        paint[8],
        [(0, -3), (-1, -2), (-3, 0), (-3, 2), (3, 0)],
        [(0, 2), (-3, -3), (-2, 2), (-1, -3)],
    )
    crane_attached_map = template(
        paint[8],
        [(0, -3), (-1, -2), (-2, 0), (-2, 2), (3, 0)],
        [(0, 2), (-2, -3), (-3, 0), (-1, -3)],
    )
    crane_point = maximum([crane_open, crane_attached_map])
    crane_exists, crane_attached = gm(crane_point), gm(crane_attached_map)
    crane_position = dense(position(crane_point), np.eye(2, dtype=np.float32),
                           [1.0, 6.0])

    vertical_w = np.zeros((1, 1, 7, 7), np.float32)
    vertical_w[0, 0, :, 3] = 1.0
    object_core = cap(conv(paint[12], vertical_w, [-5.0]))
    object_exists = gm(object_core)
    object_position = dense(
        bbox(object_core),
        np.asarray([[0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5]], np.float32),
        [0.5, -0.5],
    )
    track = dense(
        bbox(paint[0]),
        np.asarray([[1, 0, 0, 0], [0, 0, 1, 0],
                    [0, 1, 0, 0], [0, 0, 0, 1]], np.float32),
        [2.0, -1.0, 8.0, 5.0],
    )
    track_top, track_bottom = vs(track, 0, 1), vs(track, 1, 2)
    track_left, track_right = vs(track, 2, 3), vs(track, 3, 4)

    optimistic_object = bi(
        "mul", cap(sa(bi("sub", gr, track_top), 1.0, 7.0)),
        cap(sa(bi("sub", track_bottom, gr), 1.0, 6.0))
    )
    optimistic_object = bi(
        "mul", optimistic_object,
        cap(sa(bi("sub", gc, track_left), 1.0, 2.0))
    )
    optimistic_object = bi(
        "mul", optimistic_object,
        cap(sa(bi("sub", track_right, gc), 1.0, 3.0))
    )
    optimistic_object = bi("mul", optimistic_object, crane_exists)

    fixed_roads = maximum([
        paint[2], paint[6], paint[7], paint[11], paint[1], paint[12]
    ])
    optimistic_roads = maximum([fixed_roads, optimistic_object])
    crusher_pixels = cap(bi("sub", paint[8], blue_current))
    collision_w = np.zeros((1, 1, 3, 3), np.float32)
    collision_w[0, 0, 1:3, 1:3] = 1.0

    roads = []
    for state in range(8):
        a, b, d = state & 1, (state >> 1) & 1, (state >> 2) & 1
        blue = blue_alternate if a else blue_current
        orange = orange_checker if b else orange_solid
        pink = pink_checker if d else pink_solid
        hidden = [static_pawn]
        if not a:
            hidden.append(blue_pawn)
        if not b:
            hidden.append(orange_pawn)
        if not d:
            hidden.append(pink_pawn)
        if state == 0:
            hidden.append(pawn_paint)
        pawn_support = maximum(hidden)
        blockers = maximum([
            crusher_pixels,
            orange_solid if b else orange_checker,
            pink_solid if d else pink_checker,
        ])
        clear = inv(cap(conv(blockers, collision_w)))
        support = maximum([optimistic_roads, blue, orange, pink, pawn_support])
        roads.append(bi("mul", support, clear))
        if state == 0:
            actual_road = bi("mul", maximum([
                fixed_roads, blue, orange, pink, pawn_support
            ]), clear)

    switch_costs = [
        va(have_a, 1998.0, -2000.0),
        va(have_b, 1998.0, -2000.0),
        va(have_d, 1998.0, -2000.0),
    ]
    teleport_cost = va(have_c, 1998.0, -2000.0)
    neighbour_w = np.zeros((4, 1, 5, 5), np.float32)
    neighbour_w[0, 0, 0, 2] = 1.0
    neighbour_w[1, 0, 4, 2] = 1.0
    neighbour_w[2, 0, 2, 0] = 1.0
    neighbour_w[3, 0, 2, 4] = 1.0

    states = []
    for s in range(8):
        sid = "value_" + str(s)
        dims[sid] = 1
        states.append({"id": sid, "init": goal_reward})

    body = []
    old_destination, old_cache = destination[0], cache
    cache = dict(cache)
    destination[0] = body
    updates = {}
    for s in range(8):
        sid = "value_" + str(s)
        neighbours = conv(sid, neighbour_w, [-1.0] * 4)
        best = maximum([sl(neighbours, i, i + 1) for i in range(4)])
        for bit, cost in zip([1, 2, 4], switch_costs):
            best = bi("max2", best, bi("add", "value_" + str(s ^ bit), cost))
        portal_value = bi("vadd", gm(bi("mul", sid, portals)), teleport_cost)
        best = maximum([best, bi("mul", portals, portal_value), goal_reward])
        updates[sid] = bi("mul", best, roads[s])
    destination[0], cache = old_destination, old_cache
    planning = {
        "id": "planning", "op": "loop", "T": 50,
        "state": states, "body": body, "next": updates,
    }
    graph.append(planning)
    values = []
    for s in range(8):
        name = "planning.value_" + str(s)
        dims[name] = 1
        values.append(name)

    def at_player(x):
        return gm(bi("mul", x, pawn))

    movement_maps = conv(values[0], neighbour_w, [-1.0] * 4)
    move_q = vcat([at_player(sl(movement_maps, i, i + 1)) for i in range(4)])
    actual_next = conv(actual_road, neighbour_w)
    legal_moves = vcat([at_player(sl(actual_next, i, i + 1)) for i in range(4)])
    q_a = bi("vadd", at_player(values[1]), switch_costs[0])
    q_b = bi("vadd", at_player(values[2]), switch_costs[1])
    q_d = bi("vadd", at_player(values[4]), switch_costs[2])
    on_portal = vm(at_player(portals), have_c)
    q_c = bi(
        "vadd", va(gm(bi("mul", values[0], portals)), 1.0, -2.0),
        va(on_portal, 2000.0, -2000.0),
    )
    click_q = vcat([q_a, q_b, q_d, q_c])
    main_selector = choose(click_q)
    main_value = selected(click_q, main_selector)

    intent = choose(vcat([move_q, click_q]))
    intent_move = vs(intent, 0, 4)
    moving_intent = dense(intent_move, np.ones((1, 4), np.float32))
    missing_intent = vsub(moving_intent, selected(legal_moves, intent_move))
    displacement = dense(
        intent_move,
        np.asarray([[-2, 2, 0, 0], [0, 0, -2, 2]], np.float32)
    )
    desired_position = bi("vadd", position(pawn), displacement)
    desired_row, desired_col = vs(desired_position, 0, 1), vs(desired_position, 1, 2)

    relative_col = vsub(desired_col, track_left)
    rail_index = sum_vector([
        vcap(va(relative_col, 2.0, -5.0)),
        vcap(va(relative_col, 2.0, -13.0)),
        vcap(va(relative_col, 2.0, -21.0)),
    ])
    requested_col = bi("vadd", track_left, va(rail_index, 4.0))
    lower_rail = vcap(va(vsub(desired_row, track_top), 2.0, -11.0))
    requested_row = bi("vadd", vm(lower_rail, track_bottom),
                       vm(vinv(lower_rail), track_top))
    target_position = bi(
        "vadd", vm(crane_attached, vcat([requested_row, requested_col])),
        vm(vinv(crane_attached), object_position),
    )
    delta = vsub(target_position, crane_position)
    dy, dx = vs(delta, 0, 1), vs(delta, 1, 2)
    want_up, want_down = vcap(va(dy, -2.0, -1.0)), vcap(va(dy, 2.0, -1.0))
    want_left, want_right = vcap(va(dx, -2.0, -1.0)), vcap(va(dx, 2.0, -1.0))
    y_wrong = bi("vadd", want_up, want_down)
    at_left_rail = vinv(vcap(va(
        vsub(vs(crane_position, 1, 2), track_left), 2.0, -1.0
    )))
    drive_up, drive_down = vm(want_up, at_left_rail), vm(want_down, at_left_rail)
    drive_left = bi("vadd", vm(y_wrong, vinv(at_left_rail)),
                    vm(vinv(y_wrong), want_left))
    drive_right = vm(vinv(y_wrong), want_right)
    at_target = vinv(sum_vector([drive_up, drive_down, drive_left, drive_right]))

    need_e = vm(vm(e_closed, have_e), pawn_permanent)
    need_f = vm(vm(vinv(f_ready), have_f), pawn_permanent)
    crane_controls = vm(gm(grab_button), gm(horizontal_buttons))
    need_grab = vm(
        vm(vm(vm(crane_exists, object_exists), vinv(crane_attached)),
           crane_controls),
        pawn_permanent,
    )

    # A moved platform may still support a player in its overlapping region.
    support_after = []
    for dy_shift, dx_shift in [(-4, 0), (4, 0), (0, -4), (0, 4)]:
        shifted_support = translate(paint[12], dy_shift, dx_shift)
        support_after.append(vcap(bi("vadd", pawn_permanent,
                                     at_player(shifted_support))))
    direction_vector = vcat([drive_up, drive_down, drive_left, drive_right])
    safe_crane_move = selected(vcat(support_after), direction_vector)
    need_transport = vm(
        vm(vm(vm(missing_intent, crane_attached), crane_exists),
           crane_controls),
        safe_crane_move,
    )

    p_e = need_e
    p_f = vm(vinv(p_e), need_f)
    remaining = vinv(bi("vadd", p_e, p_f))
    p_grab = vm(remaining, need_grab)
    p_transport = vm(vsub(remaining, p_grab), need_transport)
    p_crane = bi("vadd", p_grab, p_transport)
    macro = sum_vector([p_e, p_f, p_crane])
    main_gate = vinv(macro)

    click_parts = []
    for i, button in enumerate([button_a, button_b, button_d, button_c]):
        click_parts.append(bi("mul", button,
                              vm(vs(main_selector, i, i + 1), main_gate)))
    click_parts.extend([bi("mul", button_e, p_e), bi("mul", button_f, p_f)])
    for button, direction in [
        (click_up, drive_up), (click_down, drive_down),
        (click_left, drive_left), (click_right, drive_right),
    ]:
        click_parts.append(bi("mul", button, vm(p_crane, direction)))
    click_parts.append(bi("mul", grab_button, vm(p_grab, at_target)))
    conv(sum_spatial(click_parts), np.ones((1, 1, 1, 1), np.float32),
         name="click")

    safe_move_q = bi("vadd", move_q, va(legal_moves, 2000.0, -2000.0))
    click_logit = bi("vadd", main_value, va(macro, 5000.0))
    output_features = vcat([safe_move_q, click_logit])
    output_w = np.zeros((8, 5), np.float32)
    for i in range(4):
        output_w[i + 1, i] = 1.0
    output_w[6, 4] = 1.0
    output_b = np.zeros(8, np.float32)
    output_b[[0, 5, 7]] = -100000.0
    dense(output_features, output_w, output_b, name="logits")

    # Static operation accounting determines the available planning depth.
    def evaluations(nodes):
        total = 0
        for nd in nodes:
            total += 1
        return total

    ordinary = evaluations(graph) - 1
    planning["T"] = max(1, min(64, (9900 - ordinary) // len(body)))

    return {"graph": graph, "memory_size": 5, "params": params}
