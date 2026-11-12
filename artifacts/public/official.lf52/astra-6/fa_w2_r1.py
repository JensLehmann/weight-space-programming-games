import numpy as np


def build():
    params = {}
    cache = {}
    graph = []
    serial = [0]
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def fresh(prefix="n"):
        serial[0] += 1
        return "%s_%d" % (prefix, serial[0])

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = fresh("p")
            cache[key] = name
            params[name] = a
        return cache[key]

    def node(op, *inputs, **kw):
        name = kw.pop("name", None) or fresh()
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(kw)
        graph.append(nd)
        return name

    def conv(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        if w.ndim == 2:
            w = w[:, :, None, None]
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(
                np.broadcast_to(np.asarray(b, np.float32), (w.shape[0],)).copy()
            )
        if name is not None:
            kw["name"] = name
        return node("conv", x, **kw)

    def dense(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(
                np.broadcast_to(np.asarray(b, np.float32), (w.shape[0],)).copy()
            )
        if name is not None:
            kw["name"] = name
        return node("dense", x, **kw)

    def sl(x, a, b):
        return node("slice", x, start=a, stop=b)

    def vs(x, a, b):
        return node("vslice", x, start=a, stop=b)

    def cat(*xs):
        return node("concat", *xs)

    def vc(*xs):
        return node("vconcat", *xs)

    def add(a, b):
        return node("add", a, b)

    def sub(a, b):
        return node("sub", a, b)

    def mul(a, b):
        return node("mul", a, b)

    def mx(a, b):
        return node("max2", a, b)

    def mn(a, b):
        return node("min2", a, b)

    def relu(x):
        return node("relu", x)

    def va(a, b):
        return node("vadd", a, b)

    def vm(a, b):
        return node("vmul", a, b)

    def cvec(values):
        values = np.asarray(values, np.float32).reshape(-1)
        return dense("aprev", np.zeros((len(values), 8), np.float32), values)

    def scale(x, k, c=1, b=0):
        return conv(x, np.eye(c, dtype=np.float32) * k, b)

    def vscale(x, k, n=1, b=0):
        return dense(x, np.eye(n, dtype=np.float32) * k, b)

    def vclip(x):
        t = dense(x, [[1], [1]], [0, -1])
        return dense(node("vrelu", t), [[1, -1]])

    def vn(x):
        return dense(x, [[-1]], [1])

    def blend(flag, yes, no):
        return va(vm(flag, yes), vm(vn(flag), no))

    one = sl("coords", 2, 3)
    pix = conv("coords", [[63, 0, 0], [0, 63, 0]])
    row = sl(pix, 0, 1)
    col = sl(pix, 1, 2)
    rc6 = cat(row, col, row, col, row, col)

    def clip(x):
        return mn(relu(x), one)

    def inv(x):
        return sub(one, x)

    def sumch(x, c):
        return conv(x, np.ones((1, c), np.float32))

    def maxdir(x, c=1):
        return mx(
            mx(sl(x, 0, c), sl(x, c, 2 * c)),
            mx(sl(x, 2 * c, 3 * c), sl(x, 3 * c, 4 * c))
        )

    def near(x, c=1):
        w = np.zeros((4 * c, c, 3, 3), np.float32)
        for d, (dy, dx) in enumerate(directions):
            for j in range(c):
                w[d * c + j, j, 1 + dy, 1 + dx] = 1
        return conv(x, w)

    def shifts(x, both=False):
        w = np.zeros((4, 1, 7, 7), np.float32)
        for d, (dy, dx) in enumerate(directions):
            w[d, 0, 3 + 3 * dy, 3 + 3 * dx] = 1
        z = conv(x, w)
        w = np.zeros((4, 4, 7, 7), np.float32)
        for d, (dy, dx) in enumerate(directions):
            w[d, d, 3 + 3 * dy, 3 + 3 * dx] = 1
        z = conv(z, w)
        if not both:
            return z
        w = np.zeros((4, 4, 3, 3), np.float32)
        for d, (dy, dx) in enumerate(directions):
            w[d, d, 1 + dy, 1 + dx] = 1
        zz = z
        for _ in range(6):
            zz = conv(zz, w)
        return z, zz

    def opposite(x):
        w = np.zeros((4, 4), np.float32)
        for d in range(4):
            w[d, (d + 2) % 4] = 1
        return conv(x, w)

    def points(record):
        w = np.zeros((6, 9), np.float32)
        for j, distance in enumerate((0, 6, 12)):
            w[2 * j, 0] = 1
            w[2 * j + 1, 1] = 1
            w[2 * j, 2] = distance
            w[2 * j + 1, 3] = distance
        centers = dense(record, w)
        distance = node("abs", sub(rc6, centers))
        w = np.zeros((3, 6), np.float32)
        for j in range(3):
            w[j, 2 * j:2 * j + 2] = -4
        return clip(conv(distance, w, [2, 2, 2]))

    def select_max(priority, eligible):
        ranked = mul(priority, eligible)
        maximum = node("gmax", maxdir(ranked))
        chosen = relu(scale(node("abs", sub(priority, maximum)), -1, 4, 1))
        return mul(chosen, eligible)

    def make_record(selected, board, removed, priority):
        pooled = node("gmean", cat(
            selected, mul(selected, row), mul(selected, col),
            mul(selected, board), mul(selected, removed),
            mul(selected, priority)
        ))
        w = np.zeros((10, 24), np.float32)
        w[0, 4:8] = 1
        w[1, 8:12] = 1
        w[4, 12:16] = 1
        w[5, 16:20] = 1
        w[6, 20:24] = 1
        w[7, 4:8] = 64
        w[7, 8:12] = 1
        w[8, 4:8] = 64
        w[8, 8:12] = 1
        w[9, :4] = 1
        for d, (dy, dx) in enumerate(directions):
            w[2, d] = dy
            w[3, d] = dx
            w[7, d] = 12 * (64 * dy + dx) + ((d + 2) % 4) * 4096
            w[8, d] = d * 4096
        result = dense(pooled, w * 4096)
        return vs(result, 0, 9), vs(result, 9, 10)

    zero1 = cvec([0])
    zero9 = cvec(np.zeros(9))
    remembered = vs("mem", 0, 9)
    old_pending = vs("mem", 9, 10)
    old_ban = vs("mem", 10, 11)
    old_ban_code = vs("mem", 11, 12)
    last_inverse = vs("mem", 12, 13)
    recent_jump = vs("mem", 13, 14)

    # Exact local sprite recognition.
    W = np.zeros((8, 16, 7, 7), np.float32)
    B = np.zeros(8, np.float32)

    def put(out, colour, y, x, amount=1):
        W[out, colour, y + 3, x + 3] += amount

    circle = [
        (y, x) for y in range(-1, 3) for x in range(-1, 3)
        if not (y in (-1, 2) and x in (-1, 2))
    ]
    corners = [(-1, -1), (-1, 2), (2, -1), (2, 2)]
    for out, colour in enumerate((14, 8, 9, 2)):
        for y, x in circle:
            put(out, colour, y, x)
        for y, x in corners:
            put(out, colour, y, x, -1)
        B[out] = -11

    purple_pattern = [
        [15, 15, 15, 15], [15, 0, 7, 15],
        [15, 7, 7, 15], [15, 15, 15, 15]
    ]
    for y in range(4):
        for x in range(4):
            put(4, purple_pattern[y][x], y - 2, x - 1)
    B[4] = -15

    for y in range(-2, 4):
        for x in range(-2, 4):
            if y in (-2, 3) or x in (-2, 3):
                put(5, 0, y, x)
    for y, x in corners:
        put(5, 1, y, x)
    B[5] = -23

    for y in (-2, 3):
        for x in (-2, 3):
            put(6, 11, y, x)
    B[6] = -3

    for y, x in circle:
        put(7, 2, y, x, -1 if y in (0, 1) and x in (0, 1) else 1)
    B[7] = -7

    detected = clip(conv("x0", W, B))
    green, red, blue, grey, purple, floor0, pad, ring = [
        sl(detected, i, i + 1) for i in range(8)
    ]
    live = mx(green, red)
    all_pegs = mx(live, blue)
    all_visible = mx(all_pegs, grey)
    floor = mx(mx(floor0, all_visible), purple)
    anchors = mx(floor, pad)
    board0 = add(add(green, scale(red, 2)), scale(blue, 3))
    land = mul(mx(floor, pad), inv(purple))
    actual_support = mx(all_pegs, purple)
    has_blue = node("gmax", blue)
    has_special = node("gmax", mx(red, blue))

    # Recover the modulo-six board lattice at arbitrary screen offsets.
    thresholds = list(range(-1, 65))
    w = np.zeros((132, 2), np.float32)
    b = np.zeros(132, np.float32)
    for axis in range(2):
        for j, t in enumerate(thresholds):
            w[axis * 66 + j, axis] = 1
            b[axis * 66 + j] = -t
    ramps = relu(conv(pix, w, b))
    w = np.zeros((12, 132), np.float32)
    for axis in range(2):
        for i in range(64):
            for t, coefficient in ((i - 1, 1), (i, -2), (i + 1, 1)):
                w[axis * 6 + i % 6, axis * 66 + t + 1] += coefficient
    phases = clip(scale(conv(ramps, w), 2, 12, -0.5))
    phase_vector = node("gmax", mul(phases, anchors))
    aligned = mul(phases, phase_vector)
    w = np.zeros((2, 12), np.float32)
    w[0, :6] = 1
    w[1, 6:] = 1
    aligned = conv(aligned, w)
    lattice = mul(sl(aligned, 0, 1), sl(aligned, 1, 2))

    rail = mx(mul(sl("x0", 5, 6), lattice), pad)
    rail6 = shifts(rail)
    pad6 = shifts(pad)
    purple6 = shifts(purple)
    land6, land12 = shifts(land, True)
    peg6, peg12 = shifts(all_pegs, True)
    green6 = shifts(green)
    board6 = shifts(board0)

    w = np.zeros((4, 2), np.float32)
    w[:, 0] = 64
    w[:, 1] = 1
    codes = conv(pix, w, [0, 4096, 8192, 12288])
    reverse_mask = mul(
        relu(scale(node("abs", sub(codes, last_inverse)), -1, 4, 1)),
        recent_jump
    )

    # Relaxed transport/jump geometry, rendezvous, and visible exits.
    walk = mx(land, rail)
    walk6, walk12 = shifts(walk, True)
    support = mx(actual_support, rail)
    support = mx(support, mul(land, has_special))
    support6 = shifts(support)
    jump_edges = mul(mul(walk, support6), walk12)
    rail_edges = mul(rail, rail6)

    offpad = inv(pad)
    green_fixed = mul(green, offpad)
    has_fixed = node("gmax", green_fixed)
    targets = mx(green_fixed, mul(green, vn(has_fixed)))
    target6 = shifts(targets)
    direct_seed = maxdir(mul(mul(walk, target6), walk12))
    meeting_seed = maxdir(mul(mul(walk, target6), opposite(walk6)))
    green_count = vscale(node("gmean", green), 4096)
    multiple_green = vclip(vscale(green_count, 1, b=-1))
    target_seed = mul(mx(direct_seed, meeting_seed), multiple_green)

    right_exit = mul(walk, clip(scale(col, 1, b=-57)))
    left_exit = mul(walk, clip(scale(col, -1, b=4)))
    top_exit = mul(walk, clip(scale(row, -1, b=4)))
    bottom_exit = mul(walk, clip(scale(row, 1, b=-59)))
    frontier = mx(
        scale(right_exit, 40),
        scale(mx(left_exit, mx(top_exit, bottom_exit)), 12)
    )
    value_init = mx(scale(target_seed, 64), frontier)

    outer = graph
    graph = []
    value_state = fresh("value_state")
    v6, v12 = shifts(value_state, True)
    jump_values = mul(relu(scale(v12, 1, 4, -1)), jump_edges)
    transport_values = mul(relu(scale(v6, 1, 4, -1)), rail_edges)
    value_new = mx(value_state, maxdir(mx(jump_values, transport_values)))
    body = graph
    graph = outer
    value_loop = fresh("value_loop")
    graph.append({
        "id": value_loop, "op": "loop", "T": 24,
        "state": [{"id": value_state, "init": value_init}],
        "body": body, "next": {value_state: value_new}
    })
    value = value_loop + "." + value_state
    value6, value12 = shifts(value, True)
    delta_value = sub(value12, value)

    # Delete-relaxed reachability. Two channels distinguish any movable
    # piece from a green piece. A jump requires both predecessor and support.
    any_init = scale(all_pegs, 64)
    green_init = scale(green, 64)
    purple_value = scale(purple, 64)

    outer = graph
    graph = []
    reach_any = fresh("reach_any")
    reach_green = fresh("reach_green")
    support_value = mx(reach_any, purple_value)
    support_reach6 = shifts(support_value)
    _, any12 = shifts(reach_any, True)
    _, green12 = shifts(reach_green, True)
    any_candidate = mul(
        relu(scale(add(any12, support_reach6), 1, 4, -65)), land
    )
    green_candidate = mul(
        relu(scale(add(green12, support_reach6), 1, 4, -65)), land
    )
    next_any = mx(reach_any, maxdir(any_candidate))
    next_green = mx(reach_green, maxdir(green_candidate))
    body = graph
    graph = outer
    reach_loop = fresh("reach_loop")
    graph.append({
        "id": reach_loop, "op": "loop", "T": 22,
        "state": [
            {"id": reach_any, "init": any_init},
            {"id": reach_green, "init": green_init}
        ],
        "body": body,
        "next": {reach_any: next_any, reach_green: next_green}
    })
    any_reach = reach_loop + "." + reach_any
    green_reach = reach_loop + "." + reach_green
    final_support6 = shifts(mx(any_reach, purple_value))

    empty = mul(land, inv(all_pegs))
    empty6, empty12 = shifts(empty, True)
    dependency_seed = mx(
        maxdir(mul(mul(land, green6), empty12)),
        maxdir(mul(mul(land, green6), opposite(empty6)))
    )
    dependency_seed = mul(dependency_seed, inv(all_pegs))
    goal_eligible = mul(dependency_seed, clip(green_reach))
    location_code = conv(pix, [[64, 1]])
    goal_priority = add(scale(green_reach, 8192),
                        scale(location_code, -1, b=4096))
    goal_max = node("gmax", mul(goal_priority, goal_eligible))
    goal_init = mul(
        relu(scale(node("abs", sub(goal_priority, goal_max)), -1, b=1)),
        goal_eligible
    )
    goal_exists = node("gmax", goal_init)
    one_vector = cvec([1])

    # Backchain the relaxed dependency graph to a currently executable jump.
    outer = graph
    graph = []
    dependency_goal = fresh("dependency_goal")
    dependency_kind = fresh("dependency_kind")
    dependency_done = fresh("dependency_done")
    dependency_record = fresh("dependency_record")
    dependency_success = fresh("dependency_success")

    _, goal12 = shifts(dependency_goal, True)
    source_value = add(mul(green_reach, dependency_kind),
                       mul(any_reach, vn(dependency_kind)))
    candidate_value = relu(
        scale(add(source_value, final_support6), 1, 4, -65)
    )
    eligible = mul(mul(goal12, clip(candidate_value)), land12)
    priority = add(scale(candidate_value, 32768, 4),
                   scale(codes, -1, 4, 32767))
    chosen = select_max(priority, eligible)
    removed = mul(
        relu(scale(node("abs", sub(board0, board6)), -1, 4, 1)),
        inv(clip(scale(board6, 1, 4, -2)))
    )
    record, found = make_record(chosen, board0, removed, priority)
    pp = points(record)
    source_point = sl(pp, 0, 1)
    middle_point = sl(pp, 1, 2)
    destination_point = sl(pp, 2, 3)

    requested_source = add(mul(green, dependency_kind),
                           mul(all_pegs, vn(dependency_kind)))
    source_present = node("gmax", mul(source_point, requested_source))
    middle_present = node("gmax", mul(middle_point, actual_support))
    destination_free = vn(node("gmax", mul(destination_point, all_pegs)))
    is_reverse = node("gmax", maxdir(mul(chosen, reverse_mask)))
    executable = vm(
        vm(vm(found, source_present), middle_present),
        vm(destination_free, vn(is_reverse))
    )
    active = vn(dependency_done)
    save = vm(active, executable)
    next_dependency_record = blend(save, record, dependency_record)
    next_dependency_success = vclip(va(dependency_success, save))

    stop = vclip(va(vn(found), vm(source_present, middle_present)))
    next_dependency_done = vclip(va(dependency_done, stop))
    new_goal = add(mul(source_point, vn(source_present)),
                   mul(middle_point, source_present))
    next_dependency_goal = add(mul(new_goal, active),
                               mul(dependency_goal, dependency_done))
    new_kind = vm(dependency_kind, vn(source_present))
    next_dependency_kind = blend(active, new_kind, dependency_kind)
    body = graph
    graph = outer
    dependency_loop = fresh("dependency_loop")
    graph.append({
        "id": dependency_loop, "op": "loop", "T": 18,
        "state": [
            {"id": dependency_goal, "init": goal_init},
            {"id": dependency_kind, "init": one_vector},
            {"id": dependency_done, "init": zero1},
            {"id": dependency_record, "init": zero9},
            {"id": dependency_success, "init": zero1}
        ],
        "body": body,
        "next": {
            dependency_goal: next_dependency_goal,
            dependency_kind: next_dependency_kind,
            dependency_done: next_dependency_done,
            dependency_record: next_dependency_record,
            dependency_success: next_dependency_success
        }
    })
    relaxed_record = dependency_loop + "." + dependency_record
    relaxed_available = vm(
        vm(dependency_loop + "." + dependency_success, goal_exists),
        has_blue
    )

    # Bounded reversible lookahead on the actual visible board.
    depth_limit = 6
    zero_stack = cvec(np.zeros(9 * depth_limit))
    initial_bound = cvec([8388608])

    def board_score(board):
        occupancy = mn(board, one)
        nonblue = mul(occupancy, clip(scale(board, -1, b=3)))
        count = node("gmean", nonblue)
        potential = node("gmean", mul(nonblue, value))
        return dense(vc(count, potential), [[-16777216, 4096]])

    initial_score = board_score(board0)
    outer = graph
    graph = []
    states = {k: fresh(k + "_state") for k in
              ("board", "stack", "bound", "depth", "first", "best", "score")}
    board = states["board"]
    stack = states["stack"]
    bound = states["bound"]
    depth = states["depth"]
    first = states["first"]
    best = states["best"]
    best_score = states["score"]

    occupancy = mn(board, one)
    mid_colour, dst_colour = shifts(board, True)
    mid_occ = mn(mid_colour, one)
    dst_occ = mn(dst_colour, one)
    midpoint_support = mx(mid_occ, purple6)
    legal = mul(mul(mul(occupancy, midpoint_support), inv(dst_occ)), land12)
    difference = node("abs", sub(board, mid_colour))
    same = mul(relu(scale(difference, -1, 4, 1)), occupancy)
    same = mul(same, inv(clip(scale(mid_colour, 1, 4, -2))))

    priority = add(scale(same, 4194304, 4),
                   scale(delta_value, 16384, 4))
    priority = add(priority, scale(codes, -1, 4, 2113535))
    below_bound = clip(sub(bound, priority))
    depth_allowed = vclip(dense(depth, [[-1]], [depth_limit]))
    at_root = vn(vclip(depth))
    top = vs(stack, 0, 9)
    inverse_code = vs(top, 7, 8)
    reverse_match = relu(
        scale(node("abs", sub(codes, inverse_code)), -1, 4, 1)
    )
    ban_match = mul(
        relu(scale(node("abs", sub(codes, old_ban_code)), -1, 4, 1)),
        vm(old_ban, at_root)
    )
    recent_reverse = mul(mul(reverse_mask, at_root), inv(same))

    eligible = mul(mul(legal, below_bound), depth_allowed)
    eligible = mul(eligible, inv(reverse_match))
    eligible = mul(eligible, inv(ban_match))
    eligible = mul(eligible, inv(recent_reverse))
    selected = select_max(priority, eligible)
    record, forward = make_record(selected, board, same, priority)

    backward = vm(vn(forward), vclip(depth))
    staying = dense(vc(forward, backward), [[-1, -1]], [1])
    effective = va(record, vm(backward, top))
    pp = points(effective)
    signed_colour = vm(
        vs(effective, 4, 5),
        dense(vc(forward, backward), [[-1, 1]])
    )
    w = np.zeros((3, 9), np.float32)
    w[1, 5] = 1
    multiplier = dense(effective, w, [1, 0, 1])
    coefficients = dense(vm(signed_colour, multiplier), np.diag([1, 1, -1]))
    new_board = add(board, sumch(mul(pp, coefficients), 3))

    pushed = vc(record, vs(stack, 0, 9 * (depth_limit - 1)))
    popped = vc(vs(stack, 9, 9 * depth_limit), zero9)
    new_stack = va(va(vm(forward, pushed), vm(backward, popped)),
                   vm(staying, stack))
    new_bound = va(
        va(vm(forward, initial_bound), vm(backward, vs(top, 6, 7))),
        vm(staying, bound)
    )
    new_depth = dense(vc(depth, forward, backward), [[1, 1, -1]])
    new_first = blend(vm(forward, at_root), record, first)
    score = board_score(new_board)
    improved = vclip(va(score, vscale(best_score, -1)))
    new_best = blend(improved, new_first, best)
    new_best_score = blend(improved, score, best_score)
    body = graph
    graph = outer
    search_loop = fresh("search_loop")
    graph.append({
        "id": search_loop, "op": "loop", "T": 1,
        "state": [
            {"id": states["board"], "init": board0},
            {"id": states["stack"], "init": zero_stack},
            {"id": states["bound"], "init": initial_bound},
            {"id": states["depth"], "init": zero1},
            {"id": states["first"], "init": zero9},
            {"id": states["best"], "init": zero9},
            {"id": states["score"], "init": initial_score}
        ],
        "body": body,
        "next": {
            states["board"]: new_board,
            states["stack"]: new_stack,
            states["bound"]: new_bound,
            states["depth"]: new_depth,
            states["first"]: new_first,
            states["best"]: new_best,
            states["score"]: new_best_score
        }
    })
    best_record = search_loop + "." + states["best"]
    fallback_record = search_loop + "." + states["first"]
    has_best = vclip(vs(best_record, 4, 5))
    has_fallback = vclip(vs(fallback_record, 4, 5))
    secondary_plan = blend(relaxed_available, relaxed_record, fallback_record)
    plan = blend(has_best, best_record, secondary_plan)

    # Three platform roles: empty landing square, movable support, passenger.
    stationary = mul(actual_support, offpad)
    stationary6 = shifts(stationary)
    fixed_green6 = shifts(green_fixed)
    source6, source12 = shifts(live, True)
    free_walk = mul(walk, inv(stationary))
    free6, free12 = shifts(free_walk, True)
    free_opposite = opposite(free6)
    value_opposite = opposite(value6)
    payload = mul(pad, actual_support)
    has_payload = node("gmax", payload)
    possible_support = mx(stationary6, mul(rail6, has_payload))
    cap32 = scale(one, 32)

    def objective(benefit, enabled):
        positive = relu(benefit)
        quality = add(mn(positive, cap32), scale(clip(positive), 100, 4))
        return mul(maxdir(mul(quality, enabled)), rail)

    empty_benefit = add(sub(value, value12), scale(fixed_green6, 64, 4))
    empty_goal = objective(empty_benefit, mul(source12, possible_support))
    support_goal = objective(sub(value_opposite, value6),
                             mul(source6, free_opposite))
    carrier_benefit = add(sub(value12, value), scale(fixed_green6, 64, 4))
    carrier_goal = objective(carrier_benefit, mul(stationary6, free12))
    capture_goal = objective(scale(sub(value_opposite, value6), 1, 4, 64),
                             mul(fixed_green6, free_opposite))
    carrier_goal = mx(carrier_goal, capture_goal)
    goals = cat(empty_goal, support_goal, carrier_goal)

    w = np.zeros((1, 1, 7, 7), np.float32)
    w[0, 0, :6, :6] = 1
    rail_area = clip(conv(rail, w))
    outer = graph
    graph = []
    route_state = fresh("route_state")
    neighbors = maxdir(near(route_state, 3), 3)
    propagated = mul(relu(scale(neighbors, 1, 3, -1)), rail_area)
    route_new = mx(route_state, propagated)
    body = graph
    graph = outer
    route_loop = fresh("route_loop")
    graph.append({
        "id": route_loop, "op": "loop", "T": 96,
        "state": [{"id": route_state, "init": goals}],
        "body": body, "next": {route_state: route_new}
    })
    routes = route_loop + "." + route_state
    route_neighbors = near(routes, 3)
    classes = cat(
        mul(pad, inv(actual_support)),
        mul(pad, mx(purple, blue)),
        mul(pad, live)
    )

    gains = []
    valid_directions = []
    for d in range(4):
        valid = mul(mul(pad, sl(rail6, d, d + 1)),
                    inv(sl(pad6, d, d + 1)))
        change = sub(sl(route_neighbors, 3 * d, 3 * d + 3), routes)
        local_gain = mul(sumch(mul(change, classes), 3), valid)
        gains.append(vscale(node("gmean", local_gain), 4096))
        valid_directions.append(node("gmax", valid))

    route_features = vc(vc(*gains), vc(*valid_directions), "aprev")
    w = np.zeros((4, 16), np.float32)
    for d, reverse_action in enumerate((2, 3, 1, 4)):
        w[d, d] = 100
        w[d, 4 + d] = 1001
        w[d, 8 + reverse_action] = -0.05
    route_logits = dense(route_features, w,
                         [-999.999, -999.996, -999.998, -999.997])

    # Selection/destination staging and recovery.
    remembered_points = points(remembered)
    remembered_destination = sl(remembered_points, 2, 3)
    pending_valid = vm(old_pending,
                       node("gmax", mul(remembered_destination, ring)))
    planned_origin = sl(points(plan), 0, 1)
    click_value = add(mul(planned_origin, vn(pending_valid)),
                      mul(remembered_destination, pending_valid))
    conv(click_value, [[1]], name="click")

    failure = node("gmax", grey)
    action_features = vc(
        route_logits, has_best, has_fallback,
        pending_valid, failure, relaxed_available
    )
    w = np.zeros((8, 9), np.float32)
    for d, action in enumerate((1, 4, 2, 3)):
        w[action, d] = 1
    w[6, 4] = 10000
    w[6, 5] = 2000
    w[6, 6] = 20000
    w[6, 8] = 9500
    w[7, 7] = 30000
    dense(action_features, w,
          [-10000, 0, 0, 0, 0, -10000, -2000, -100],
          name="logits")

    retain = vclip(va(pending_valid, failure))
    next_record = blend(retain, remembered, plan)
    next_pending = vm(vn(pending_valid), vn(failure))
    next_ban = va(
        failure,
        vm(vn(failure), vm(vscale(old_ban, 0.98), vn(pending_valid)))
    )
    next_ban_code = blend(failure, vs(remembered, 8, 9), old_ban_code)
    next_inverse = blend(pending_valid, vs(remembered, 7, 8), last_inverse)
    next_recent = blend(pending_valid, one_vector,
                        vscale(recent_jump, 0.995))
    node("vconcat", next_record, next_pending, next_ban, next_ban_code,
         next_inverse, next_recent, name="mem_next")

    # Exact static budget bookkeeping; no observations or game simulation.
    loop_costs = {}

    def account(nodes, env):
        macs = 0
        evaluations = 0
        for nd in nodes:
            evaluations += 1
            op = nd["op"]
            if op == "loop":
                local = dict(env)
                for st in nd["state"]:
                    local[st["id"]] = (
                        env[st["init"]] if "init" in st
                        else tuple(st["zeros"])
                    )
                bm, be = account(nd["body"], local)
                loop_costs[nd["id"]] = (bm, be)
                macs += nd["T"] * bm
                evaluations += nd["T"] * be
                for st in nd["state"]:
                    env[nd["id"] + "." + st["id"]] = local[
                        nd["next"][st["id"]]
                    ]
                continue

            xs = [env[i] for i in nd["in"]]
            x = xs[0]
            cost = 0
            if op == "conv":
                a = params[nd["w"]]
                shape = (a.shape[0], 64, 64)
                cost = int(np.prod(a.shape)) * 4096
            elif op == "dense":
                a = params[nd["w"]]
                shape = (a.shape[0],)
                cost = int(np.prod(a.shape))
            elif op in ("slice", "vslice"):
                shape = (nd["stop"] - nd["start"],) + x[1:]
            elif op in ("concat", "vconcat"):
                shape = (sum(z[0] for z in xs),) + x[1:]
            elif op in ("gmean", "gmax"):
                shape = (x[0],)
                cost = int(np.prod(x))
            elif op in ("add", "sub", "mul", "max2", "min2"):
                shape = (max(z[0] for z in xs), 64, 64)
                cost = int(np.prod(shape))
            elif op in ("vadd", "vmul"):
                shape = (max(z[0] for z in xs),)
                cost = shape[0]
            elif op in ("rowmax", "colmax", "rowsum", "colsum"):
                shape = x
                cost = int(np.prod(x))
            else:
                shape = x
            env[nd["id"]] = shape
            macs += cost
        return macs, evaluations

    env = {
        "x0": (16, 64, 64), "coords": (3, 64, 64),
        "mem": (14,), "aprev": (8,), "avail": (8,)
    }
    base_macs, base_evals = account(graph, env)
    body_macs, body_evals = loop_costs[search_loop]
    repetitions = min(
        64,
        1 + max(0, (1450000000 - base_macs) // body_macs),
        1 + max(0, (9500 - base_evals) // body_evals)
    )
    for nd in graph:
        if nd["id"] == search_loop:
            nd["T"] = int(repetitions)
            break

    return {"graph": graph, "memory_size": 14, "params": params}
