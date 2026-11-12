import numpy as np


def build():
    params = {}
    graph = []
    dst = graph
    channels = {
        "x0": 16, "coords": 3, "mem": 44,
        "aprev": 8, "avail": 8,
    }
    serial = 0

    def node(op, ins, out=None, name=None, **fields):
        nonlocal serial
        serial += 1
        name = name or ("n%d" % serial)
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        dst.append(nd)
        if out is None:
            if op in ("concat", "vconcat"):
                out = sum(channels[x] for x in ins)
            elif op in ("add", "sub", "mul", "min2", "max2",
                        "vadd", "vmul"):
                out = max(channels[x] for x in ins)
            else:
                out = channels[ins[0]]
        channels[name] = out
        return name

    def cv(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        key = "p%dw" % (serial + 1)
        params[key] = w
        fields = {"w": key}
        if b is not None:
            bk = "p%db" % (serial + 1)
            params[bk] = np.asarray(b, np.float32).reshape(w.shape[0])
            fields["b"] = bk
        return node("conv", [x], out=w.shape[0], name=name, **fields)

    def dense(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        key = "p%dw" % (serial + 1)
        params[key] = w
        fields = {"w": key}
        if b is not None:
            bk = "p%db" % (serial + 1)
            params[bk] = np.asarray(b, np.float32).reshape(w.shape[0])
            fields["b"] = bk
        return node("dense", [x], out=w.shape[0], name=name, **fields)

    def sc(x, a=1.0, b=0.0):
        c = channels[x]
        return cv(x, np.eye(c, dtype=np.float32) * a,
                  np.full(c, b, np.float32))

    def vsc(x, a=1.0, b=0.0):
        c = channels[x]
        return dense(x, np.eye(c, dtype=np.float32) * a,
                     np.full(c, b, np.float32))

    def sl(x, a, b):
        return node("slice", [x], out=b-a, start=a, stop=b)

    def vs(x, a, b):
        return node("vslice", [x], out=b-a, start=a, stop=b)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def cat(*xs):
        return node("concat", xs)

    def vcat(*xs):
        return node("vconcat", xs)

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def vsub(a, b):
        return va(a, vsc(b, -1.0))

    def vabs(x):
        return va(node("vrelu", [x]),
                  node("vrelu", [vsc(x, -1.0)]))

    def vclip(x):
        return vsub(
            node("vrelu", [x]),
            node("vrelu", [vsc(x, 1.0, -1.0)])
        )

    def total(*xs):
        z = xs[0]
        for x in xs[1:]:
            z = add(z, x)
        return z

    def minimum(*xs):
        z = xs[0]
        for x in xs[1:]:
            z = node("min2", [z, x])
        return z

    one = sl("coords", 2, 3)
    zero = sc(one, 0.0)
    full = sc(one, 0.0, 1000.0)

    def clip(x):
        return node("min2", [node("relu", [x]), one])

    def inv(x):
        return sc(x, -1.0, 1.0)

    def shift(x, dy=0, dx=0):
        c = channels[x]
        if c > 2:
            return cat(*[
                shift(sl(x, j, min(j + 2, c)), dy, dx)
                for j in range(0, c, 2)
            ])
        w = np.zeros((c, c, 7, 7), np.float32)
        for j in range(c):
            w[j, j, 3 + dy // 2, 3 + dx // 2] = 1.0
        return cv(cv(x, w), w)

    # Player centre and gravity-dependent camera phase.
    w = np.zeros((1, 16, 5, 5), np.float32)
    for dy in (-2, -1, 0, 1):
        w[0, 9, 2 + dy, 2] = 1.0
    player = node("relu", [cv("x0", w, [-3.0])])

    def sample_player(x):
        return vsc(node("gmean", [mul(x, player)]), 4096.0)

    pc = node("gmax", [mul("coords", player)])
    pc_pixels = vsc(pc, 63.0)
    uraw = dense(pc, [[63.0, 0.0, 0.0]], [-34.0])
    up = vclip(uraw)
    down = vsc(up, -1.0, 1.0)

    # Tile-centre lattice for the two camera phases.
    specs = (
        [(1, x) for x in range(3, 64, 6)] +
        [(0, y) for y in range(3, 60, 6)] +
        [(0, y) for y in range(5, 60, 6)]
    )
    w = np.zeros((len(specs), 3), np.float32)
    b = np.zeros(len(specs), np.float32)
    for j, (axis, centre) in enumerate(specs):
        w[j, axis] = 63.0
        b[j] = -centre
    bank = node("abs", [cv("coords", w, b)])
    bank = node("relu", [sc(bank, -4.0, 1.0)])
    bank = node("min2", [sc(bank, 2.0), one])

    w = np.zeros((3, len(specs)), np.float32)
    w[0, :11] = 1.0
    w[1, 11:21] = 1.0
    w[2, 21:31] = 1.0
    lattice_parts = cv(bank, w)
    xs = sl(lattice_parts, 0, 1)
    ys = add(mul(sl(lattice_parts, 1, 2), up),
             mul(sl(lattice_parts, 2, 3), down))
    lattice = mul(xs, ys)
    pixel_id = cv("coords", [[4032.0, 63.0, 1.0]])

    def colour(c):
        return mul(sl("x0", c, c + 1), lattice)

    empty = colour(10)
    wall = colour(5)
    red = colour(8)
    ordinary = colour(14)
    cyan = colour(12)
    goal = colour(7)

    w = np.zeros((1, 16, 3, 3), np.float32)
    w[0, 12, 1, 1] = 1.0
    w[0, 12, 0, 1] = 1.0
    solid_cyan = mul(node("relu", [cv("x0", w, [-1.0])]),
                     lattice)
    ghost = sub(cyan, solid_cyan)

    # Distinguish expansion blocks from the similarly coloured spikes.
    w = np.zeros((1, 16, 5, 5), np.float32)
    for dy in (-2, 0, 2):
        w[0, 15, 2 + dy, 2] = 1.0
    yellow = mul(node("relu", [cv("x0", w, [-2.0])]), lattice)

    breakable = total(ordinary, solid_cyan, yellow)
    solid = total(wall, red, breakable)
    passable = total(empty, ghost, breakable, player, goal)
    falling_space = add(empty, ghost)

    red_target = node("gmax", [mul(red, pixel_id)])
    goal_seen = node("gmax", [goal])

    # Six planning channels: each gravity direction with zero, one,
    # or two available reversals.
    red_count = vsc(node("gmean", [red]), 4096.0)
    any_red = vclip(red_count)
    two_red = vclip(vsc(red_count, 1.0, -1.0))
    count_gates = [
        vsc(any_red, -1.0, 1.0),
        vsub(any_red, two_red),
        two_red,
    ]
    current_gate = vcat(*[
        g for count_gate in count_gates
        for g in (vm(up, count_gate), vm(down, count_gate))
    ])

    def current(x):
        return cv(mul(x, current_gate),
                  np.ones((1, 6), np.float32))

    def repeat_gravity(x):
        return cat(x, x, x)

    def vertical(x):
        return cat(*[
            shift(sl(x, j, j + 1), -6 if j % 2 == 0 else 6, 0)
            for j in range(6)
        ])

    def consume_switch(x):
        w = np.zeros((6, 6), np.float32)
        w[2, 1] = 1.0
        w[3, 0] = 1.0
        w[4, 3] = 1.0
        w[5, 2] = 1.0
        return cv(x, w)

    # Landing-row history based on three-row terrain signatures.
    w = np.zeros((1, len(specs)), np.float32)
    for j in range(11):
        w[0, j] = float(2 ** j)
    column_bits = cv(bank, w)
    row_code = node("rowsum", [mul(wall, column_bits)])
    hash_a = sc(row_code, 1.0, 1.0)
    hash_b = sc(add(shift(row_code, -6, 0),
                    sc(shift(row_code, 6, 0), 2048.0)),
                1.0, 1.0)

    old_a = vs("mem", 8, 24)
    old_b = vs("mem", 24, 40)
    da = node("abs", [sub(hash_a, old_a)])
    db = node("abs", [sub(hash_b, old_b)])
    matches = node("relu", [sc(add(da, db), -1.0, 1.0)])
    visit_count = cv(matches, np.ones((1, 16), np.float32))

    here_a = sample_player(hash_a)
    here_b = sample_player(hash_b)
    row_difference = va(
        vabs(vsub(here_a, vs(old_a, 0, 1))),
        vabs(vsub(here_b, vs(old_b, 0, 1)))
    )
    new_row = vclip(row_difference)
    old_row = vsc(new_row, -1.0, 1.0)
    proposed_a = vcat(here_a, vs(old_a, 0, 15))
    proposed_b = vcat(here_b, vs(old_b, 0, 15))
    next_a = va(vm(new_row, proposed_a), vm(old_row, old_a))
    next_b = va(vm(new_row, proposed_b), vm(old_row, old_b))

    old_pc = vs("mem", 40, 43)
    position_difference = vabs(vsub(pc_pixels, old_pc))
    position_difference = dense(position_difference, [[1.0, 1.0, 0.0]])
    unchanged = vsc(vclip(va(position_difference, row_difference)),
                    -1.0, 1.0)
    repeat_click_context = vm(unchanged, vs("aprev", 6, 7))

    # Expansion-block reachability and first prerequisite click.
    dirs = [(-6, 0), (6, 0), (0, -6), (0, 6)]
    neighbour_valid = {d: shift(lattice, *d) for d in dirs}
    current_support = add(
        mul(shift(player, 6, 0), up),
        mul(shift(player, -6, 0), down)
    )
    growth_sources = mul(yellow, inv(current_support))
    growth_init = cat(
        sc(inv(growth_sources), 1000.0),
        mul(pixel_id, growth_sources)
    )
    growth_penalties = {}
    for d in dirs:
        penalty = total(
            sc(inv(empty), 1000.0),
            sc(inv(neighbour_valid[d]), 1000.0),
            one
        )
        growth_penalties[d] = cat(penalty, zero)

    channels["growth_state"] = 2
    old_dst = dst
    dst = []
    best = "growth_state"
    for d in dirs:
        candidate = add(shift("growth_state", *d), growth_penalties[d])
        improvement = sub(sl(best, 0, 1), sl(candidate, 0, 1))
        choose = clip(sc(improvement, 32.0, -0.1))
        best = add(best, mul(choose, sub(candidate, best)))
    growth_body = dst
    dst = old_dst
    dst.append({
        "id": "Grow",
        "op": "loop",
        "T": 20,
        "state": [{"id": "growth_state", "init": growth_init}],
        "body": growth_body,
        "next": {"growth_state": best},
    })
    channels["Grow.growth_state"] = 2

    growth_distance = sl("Grow.growth_state", 0, 1)
    growth_target = sl("Grow.growth_state", 1, 2)
    reachable_growth = mul(empty, clip(sc(growth_distance, -1.0, 21.0)))
    optional_support = add(ghost, reachable_growth)
    optional_cost = add(
        ghost,
        mul(reachable_growth, sc(growth_distance, 1.0, 0.35))
    )
    optional_target = add(
        mul(ghost, pixel_id),
        mul(reachable_growth, growth_target)
    )

    features = cat(
        solid, falling_space, breakable,
        optional_support, optional_cost, optional_target,
        mul(breakable, pixel_id)
    )
    su = shift(features, -6, 0)
    sd = shift(features, 6, 0)
    support = [
        repeat_gravity(cat(sl(su, j, j + 1), sl(sd, j, j + 1)))
        for j in range(7)
    ]
    s_solid, s_air, s_break, s_opt, s_cost, s_target, s_bid = support

    cap6 = cat(full, full, full, full, full, full)
    stand_allowed = mul(clip(total(s_solid, s_opt, player)), passable)
    stand_barrier = sc(inv(stand_allowed), 1000.0)
    position_barrier = sc(inv(passable), 1000.0)
    stop_extra = add(
        s_cost,
        sc(inv(clip(add(s_solid, s_opt))), 1000.0)
    )
    fall_extra = sc(inv(s_air), 1000.0, 0.05)
    break_extra = sc(inv(s_break), 1000.0, 1.0)
    flip_extra = cv(
        one, np.zeros((6, 1), np.float32),
        [1000.0, 1000.0, 3.0, 3.0, 3.0, 3.0]
    )

    neighbour_break = {}
    neighbour_bid = {}
    walk_extra = {}
    for label, dx in (("L", -6), ("R", 6)):
        bp = shift(breakable, 0, dx)
        pp = shift(passable, 0, dx)
        neighbour_break[label] = bp
        neighbour_bid[label] = shift(mul(pixel_id, breakable), 0, dx)
        walk_extra[label] = total(one, bp, sc(inv(pp), 1000.0))

    row = sc(sl("coords", 0, 1), 63.0)
    col = sc(sl("coords", 1, 2), 63.0)
    player_row = vs(pc_pixels, 0, 1)
    player_col = vs(pc_pixels, 1, 2)

    # Exact clear-corridor finishing circuit.
    # It is enabled only when every horizontal step has real support,
    # and the entire post-reversal vertical route is already clear.
    goal_position = vsc(node("gmax", [mul("coords", goal)]), 63.0)
    goal_row = vs(goal_position, 0, 1)
    goal_col = vs(goal_position, 1, 2)

    def interval_mask(coordinate, endpoint_a, endpoint_b):
        distances = add(
            node("abs", [sub(coordinate, endpoint_a)]),
            node("abs", [sub(coordinate, endpoint_b)])
        )
        span = vabs(vsub(endpoint_a, endpoint_b))
        return clip(sc(sub(distances, span), -2.0, 1.5))

    horizontal_interval = interval_mask(col, player_col, goal_col)
    vertical_interval = interval_mask(row, player_row, goal_row)
    horizontal_route = mul(
        mul(lattice, node("rowmax", [player])),
        horizontal_interval
    )
    vertical_route = mul(
        mul(lattice, node("colmax", [goal])),
        vertical_interval
    )
    already_open = total(empty, ghost, player, goal)
    safe_walk_tile = mul(already_open, current(s_solid))
    horizontal_bad = node("gmax", [
        mul(horizontal_route, inv(safe_walk_tile))
    ])
    vertical_bad = node("gmax", [
        mul(vertical_route, inv(already_open))
    ])
    horizontal_clear = vsc(vclip(vsc(horizontal_bad, 4.0)), -1.0, 1.0)
    vertical_clear = vsc(vclip(vsc(vertical_bad, 4.0)), -1.0, 1.0)

    goal_dy = vsub(goal_row, player_row)
    goal_behind = va(
        vm(up, vclip(vsc(goal_dy, 1.0, -0.5))),
        vm(down, vclip(vsc(goal_dy, -1.0, -0.5)))
    )
    finish_valid = vm(
        vm(vm(goal_seen, any_red), goal_behind),
        vm(horizontal_clear, vertical_clear)
    )
    goal_dx = vsub(goal_col, player_col)
    finish_left = vclip(vsc(goal_dx, -1.0, -0.5))
    finish_right = vclip(vsc(goal_dx, 1.0, -0.5))
    finish_aligned = vsc(va(finish_left, finish_right), -1.0, 1.0)
    direct_left = vm(finish_valid, finish_left)
    direct_right = vm(finish_valid, finish_right)
    direct_flip = vm(finish_valid, finish_aligned)

    # Exploration objectives: new safe heights, with unseen falls costly.
    height_change = node("abs", [sub(row, player_row)])
    different_height = clip(sc(height_change, 1.0, -5.0))
    safe_frontier = mul(lattice, mul(different_height, inv(goal)))
    safe_exploration_cost = total(
        sc(visit_count, 70.0, 26.0),
        sc(height_change, -0.25),
        dense(goal_seen, [[500.0]])
    )
    safe_terminal = add(
        full,
        mul(safe_frontier, sc(safe_exploration_cost, 1.0, -1000.0))
    )
    goal_terminal = sc(inv(goal), 1000.0)
    standing_terminal = minimum(goal_terminal, safe_terminal)

    top_edge = mul(lattice, clip(sc(row, -1.0, 6.0)))
    bottom_edge = mul(lattice, clip(sc(row, 1.0, -54.0)))
    fall_frontier = repeat_gravity(
        mul(cat(top_edge, bottom_edge), falling_space)
    )
    risky_cost = total(
        sc(visit_count, 70.0, 280.0),
        dense(goal_seen, [[350.0]])
    )
    risky_terminal = add(
        cap6,
        mul(fall_frontier, sc(risky_cost, 1.0, -1000.0))
    )
    falling_terminal = minimum(goal_terminal, risky_terminal)

    def planning_step(f):
        vf = vertical(f)
        left = add(shift(f, 0, -6), walk_extra["L"])
        right = add(shift(f, 0, 6), walk_extra["R"])
        remove = add(vf, break_extra)
        flip = add(consume_switch(f), flip_extra)

        v = minimum(left, right, remove, flip, standing_terminal)
        v = node("max2", [v, stand_barrier])
        v = node("min2", [v, cap6])

        stop = add(v, stop_extra)
        fall = add(vf, fall_extra)
        nf = minimum(stop, fall, falling_terminal)
        nf = node("max2", [nf, position_barrier])
        nf = node("min2", [nf, cap6])
        return {
            "L": left, "R": right, "B": remove, "G": flip,
            "V": v, "stop": stop, "fall": fall, "next": nf,
        }

    channels["fall_state"] = 6
    old_dst = dst
    dst = []
    result = planning_step("fall_state")
    plan_body = dst
    dst = old_dst
    dst.append({
        "id": "Plan",
        "op": "loop",
        "T": 64,
        "state": [{"id": "fall_state", "init": cap6}],
        "body": plan_body,
        "next": {"fall_state": result["next"]},
    })
    channels["Plan.fall_state"] = 6
    f_final = "Plan.fall_state"
    policy = planning_step(f_final)

    # Carry first construction prerequisites back through automatic falls.
    use_optional = mul(
        s_opt,
        clip(sc(sub(f_final, policy["stop"]), 1000.0, 1.0))
    )
    use_terminal = clip(
        sc(sub(f_final, falling_terminal), 1000.0, 1.0)
    )
    carry_requirement = mul(
        s_air, mul(inv(use_optional), inv(use_terminal))
    )
    immediate_requirement = mul(s_target, use_optional)

    channels["requirement_state"] = 6
    old_dst = dst
    dst = []
    req_next = add(
        immediate_requirement,
        mul(carry_requirement, vertical("requirement_state"))
    )
    req_body = dst
    dst = old_dst
    dst.append({
        "id": "Requirements",
        "op": "loop",
        "T": 12,
        "state": [{"id": "requirement_state", "zeros": [6, 64, 64]}],
        "body": req_body,
        "next": {"requirement_state": req_next},
    })
    channels["Requirements.requirement_state"] = 6
    requirements = "Requirements.requirement_state"

    def prefer_requirement(req, otherwise):
        return add(req, mul(inv(clip(req)), otherwise))

    current_req = current(requirements)
    targets = []
    for label, dx in (("L", -6), ("R", 6)):
        req = shift(current_req, 0, dx)
        targets.append(add(
            neighbour_bid[label],
            mul(inv(neighbour_break[label]), req)
        ))
    targets.append(prefer_requirement(
        current(vertical(requirements)), current(s_bid)
    ))
    targets.append(prefer_requirement(
        current(consume_switch(requirements)), red_target
    ))

    costs = cat(*(current(policy[k]) for k in ("L", "R", "B", "G")))
    target_maps = cat(*targets)
    cost_vector = sample_player(costs)
    target_vector = sample_player(target_maps)

    click_candidates = []
    simple_scores = []
    for j in range(4):
        cost = vs(cost_vector, j, j + 1)
        target = vs(target_vector, j, j + 1)
        has_target = vclip(target)
        distance = node("abs", [sub(pixel_id, target)])
        click_base = va(
            vsc(cost, -1.0, 0.15),
            vsc(has_target, 10000.0, -10000.0)
        )
        click_candidates.append(add(sc(distance, -10.0), click_base))
        if j < 2:
            simple_scores.append(va(
                vsc(cost, -1.0),
                vsc(has_target, -10000.0)
            ))

    click_map = click_candidates[0]
    for candidate in click_candidates[1:]:
        click_map = node("max2", [click_map, candidate])

    previous_click = vs("mem", 43, 44)
    same_click_pixel = clip(sc(
        node("abs", [sub(pixel_id, previous_click)]), -1.0, 1.0
    ))
    toggle_penalty = sc(
        mul(mul(same_click_pixel, cyan), repeat_click_context), 45.0
    )
    click_map = sub(click_map, toggle_penalty)

    direct_click_map = add(
        sc(node("abs", [sub(pixel_id, red_target)]), -10.0),
        vsc(direct_flip, 10000.0, -8000.0)
    )
    click_map = node("max2", [click_map, direct_click_map], name="click")
    click_score = node("gmax", [click_map])

    peak_mask = clip(sc(
        node("abs", [sub(click_map, click_score)]), -1000.0, 1.0
    ))
    next_click = node("gmax", [mul(pixel_id, peak_mask)])

    scores = vcat(*(simple_scores + [click_score]))
    w = np.zeros((8, 3), np.float32)
    b = np.full(8, -3000.0, np.float32)
    w[3, 0] = 1.0
    w[4, 1] = 1.0
    w[6, 2] = 1.0
    b[3] = b[4] = b[6] = 0.0
    b[0] = -2000.0
    b[7] = -950.0
    base_logits = dense(scores, w, b)

    w = np.zeros((8, 2), np.float32)
    w[3, 0] = 2000.0
    w[4, 1] = 2000.0
    finish_logits = dense(vcat(direct_left, direct_right), w)
    base_logits = va(base_logits, finish_logits)

    old_action_memory = vs("mem", 0, 8)
    previous_undo = vs("aprev", 7, 8)
    not_undo = vsc(previous_undo, -1.0, 1.0)
    next_action_memory = va(
        vm("aprev", not_undo),
        vm(old_action_memory, previous_undo)
    )
    undo_context = vm(old_action_memory, previous_undo)
    w = np.zeros((8, 8), np.float32)
    for a in (3, 4, 6):
        w[a, a] = -40.0
    undo_penalty = dense(undo_context, w)
    node("vadd", [base_logits, undo_penalty], name="logits")

    node("vconcat", [
        next_action_memory, next_a, next_b, pc_pixels, next_click
    ], name="mem_next")

    return {
        "graph": graph,
        "memory_size": 44,
        "params": params,
    }
