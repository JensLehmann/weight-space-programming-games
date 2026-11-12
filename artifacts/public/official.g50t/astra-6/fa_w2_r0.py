import numpy as np


def build():
    P = {}
    G = []
    shape = {
        "x0": ("s", 16),
        "coords": ("s", 3),
        "mem": ("v", 213),
        "aprev": ("v", 8),
        "avail": ("v", 8),
    }
    serial = [0]
    pserial = [0]

    def param(a):
        name = "p%d" % pserial[0]
        pserial[0] += 1
        P[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, *inputs, out=None, **fields):
        name = out
        if name is None:
            name = "n%d" % serial[0]
            serial[0] += 1
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        G.append(nd)

        if op in ("conv", "dense"):
            shape[name] = (
                "s" if op == "conv" else "v",
                int(P[fields["w"]].shape[0]),
            )
        elif op in ("slice", "vslice"):
            shape[name] = (
                shape[inputs[0]][0],
                fields["stop"] - fields["start"],
            )
        elif op in ("concat", "vconcat"):
            shape[name] = (
                "s" if op == "concat" else "v",
                sum(shape[x][1] for x in inputs),
            )
        elif op in ("gmax", "gmean"):
            shape[name] = ("v", shape[inputs[0]][1])
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shape[name] = ("s", max(shape[x][1] for x in inputs))
        elif op in ("vadd", "vmul"):
            shape[name] = ("v", max(shape[x][1] for x in inputs))
        else:
            shape[name] = shape[inputs[0]]
        return name

    def linear(x, w, b=None, out=None):
        w = np.asarray(w, np.float32)
        spatial = shape[x][0] == "s"
        if spatial and w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": param(w)}
        if b is not None:
            b = np.asarray(b, np.float32)
            if b.ndim == 0:
                b = np.full(w.shape[0], float(b), np.float32)
            fields["b"] = param(b)
        return node("conv" if spatial else "dense", x, out=out, **fields)

    def affine(x, scale=1.0, bias=0.0, out=None):
        c = shape[x][1]
        w = np.eye(c, dtype=np.float32) * np.asarray(scale, np.float32)
        return linear(x, w, bias, out=out)

    def sl(x, start, stop):
        return node(
            "slice" if shape[x][0] == "s" else "vslice",
            x, start=start, stop=stop
        )

    def cat(*xs, out=None):
        return node(
            "concat" if shape[xs[0]][0] == "s" else "vconcat",
            *xs, out=out
        )

    def add(a, b):
        return node(
            "vadd" if shape[a][0] == shape[b][0] == "v" else "add",
            a, b
        )

    def mul(a, b):
        return node(
            "vmul" if shape[a][0] == shape[b][0] == "v" else "mul",
            a, b
        )

    def sub(a, b):
        if shape[a][0] == shape[b][0] == "v":
            return add(a, affine(b, -1.0))
        return node("sub", a, b)

    def relu(x):
        return node("relu" if shape[x][0] == "s" else "vrelu", x)

    def absolute(x):
        if shape[x][0] == "s":
            return node("abs", x)
        return add(relu(x), relu(affine(x, -1.0)))

    def maximum(a, b):
        return node("max2", a, b)

    def pool(x):
        return node("gmax", x)

    def total(xs):
        z = xs[0]
        for x in xs[1:]:
            z = add(z, x)
        return z

    one = sl("coords", 2, 3)
    vone = pool(one)

    def clip(x):
        if shape[x][0] == "s":
            return node("min2", relu(x), one)
        lo = relu(x)
        hi = relu(affine(x, 1.0, -1.0))
        return sub(lo, hi)

    def inv(x):
        return affine(x, -1.0, 1.0)

    def color(c):
        return sl("x0", c, c + 1)

    # Cross-correlations sampling the four neighbours six pixels away.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    w = np.zeros((4, 1, 7, 7), np.float32)
    w2 = np.zeros((4, 4, 7, 7), np.float32)
    for d, (dy, dx) in enumerate(directions):
        w[d, 0, 3 + 3 * dy, 3 + 3 * dx] = 1
        w2[d, d, 3 + 3 * dy, 3 + 3 * dx] = 1
    shift_a = param(w)
    shift_b = param(w2)

    def shift6(x):
        return node(
            "conv", node("conv", x, w=shift_a), w=shift_b
        )

    def max_groups(x, c=1):
        a = sl(x, 0, c)
        b = sl(x, c, 2 * c)
        d = sl(x, 2 * c, 3 * c)
        e = sl(x, 3 * c, 4 * c)
        return maximum(maximum(a, b), maximum(d, e))

    def value_loop(prefix, seed, terrain, portals, steps=96):
        nonlocal G
        outer = G
        G = []
        state = prefix + "_state"
        shape[state] = ("s", 1)

        neighbour = max_groups(shift6(state))
        walking = relu(affine(neighbour, 1.0, -1.0))
        portal_value = pool(mul(state, portals))
        jumping = mul(portals, affine(portal_value, 1.0, -1.0))
        nxt = mul(
            terrain,
            maximum(seed, maximum(walking, jumping))
        )

        body = G
        G = outer
        G.append({
            "id": prefix,
            "op": "loop",
            "T": steps,
            "state": [{"id": state, "init": seed}],
            "body": body,
            "next": {state: nxt},
        })
        result = prefix + "." + state
        shape[result] = ("s", 1)
        return result

    def reach_loop(prefix, seed, terrain, steps=64):
        nonlocal G
        outer = G
        G = []
        state = prefix + "_state"
        shape[state] = ("s", 1)
        spread = max_groups(shift6(state))
        nxt = mul(terrain, clip(add(state, spread)))
        body = G
        G = outer
        G.append({
            "id": prefix,
            "op": "loop",
            "T": steps,
            "state": [{"id": state, "init": seed}],
            "body": body,
            "next": {state: nxt},
        })
        result = prefix + "." + state
        shape[result] = ("s", 1)
        return result

    # Hard, rounding-tolerant basis for the 10 x 10 movement lattice.
    w = np.zeros((20, 3), np.float32)
    b = np.zeros(20, np.float32)
    for i in range(10):
        w[i, 0] = 63
        w[10 + i, 1] = 63
        b[i] = b[10 + i] = -(4 + 6 * i)
    axes = clip(affine(absolute(linear("coords", w, b)), -4, 2))
    w = np.zeros((100, 20), np.float32)
    for r in range(10):
        for c in range(10):
            w[10 * r + c, r] = 1
            w[10 * r + c, 10 + c] = 1
    basis = relu(linear(axes, w, -1))
    lattice = linear(basis, np.ones((1, 100), np.float32))

    pack_w = param(np.eye(100, dtype=np.float32) * 4096)

    def unpack(v):
        return linear(
            mul(basis, v), np.ones((1, 100), np.float32)
        )

    def pack(x):
        return node(
            "dense", node("gmean", mul(basis, x)), w=pack_w
        )

    xy = sl("coords", 0, 2)
    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)

    def point_at(v):
        delta = absolute(sub(xy, v))
        p = clip(linear(delta, [[-252, -252]], 2))
        return mul(p, lattice)

    old_cache = unpack(sl("mem", 0, 100))
    spent = unpack(sl("mem", 100, 200))
    old_count = sl("mem", 200, 201)
    old_mode = sl("mem", 201, 202)
    old_target = sl("mem", 202, 204)
    old_player = sl("mem", 204, 206)
    old_goal_mode = sl("mem", 206, 207)
    initialized = sl("mem", 207, 208)
    old_enemy = sl("mem", 208, 210)
    old_enemy_present = sl("mem", 210, 211)
    old_path_count = sl("mem", 211, 212)
    phase = sl("mem", 212, 213)

    # Recognize full 5 x 5 avatars, including the contrasting centre.
    w = np.zeros((3, 16, 7, 7), np.float32)
    for o, c in enumerate((9, 2, 14)):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy or dx:
                    w[o, c, 3 + dy, 3 + dx] = 1
        w[o, 5, 3, 3] = 1
    actors = mul(clip(linear("x0", w, -24)), lattice)
    player = sl(actors, 0, 1)
    enemy = sl(actors, 2, 3)

    # The goal has four coloured corners and a coloured centre.
    w = np.zeros((1, 16, 7, 7), np.float32)
    for dy, dx in ((-3, -3), (-3, 3), (3, -3), (3, 3), (0, 0)):
        w[0, 9, 3 + dy, 3 + dx] = 1
    goal = mul(clip(linear("x0", w, -4)), lattice)

    colours = cat(color(8), color(11), color(15))
    w3 = np.zeros((3, 3, 3, 3), np.float32)
    w5 = np.zeros((3, 3, 5, 5), np.float32)
    for c in range(3):
        w3[c, c] = 1
        w5[c, c] = 1
    core = clip(linear(colours, w3, -8))
    large = clip(linear(colours, w5, -15))
    small = mul(relu(sub(core, large)), lattice)
    observed_types = linear(small, [[1, 2, 3]])
    cache = maximum(old_cache, observed_types)

    def cached_type(t):
        return clip(affine(absolute(affine(cache, 1, -t)), -2, 2))

    normal_plate = cached_type(1)
    toggle_plate = cached_type(2)
    remote_plate = cached_type(3)
    plates = cat(normal_plate, toggle_plate, remote_plate)

    door8 = mul(sl(large, 0, 1), lattice)
    door11 = mul(sl(large, 1, 2), lattice)
    doors = clip(add(door8, door11))

    w = np.zeros((1, 1, 7, 7), np.float32)
    for dy, dx in ((-3, -3), (-3, 3), (3, -3), (3, 3)):
        w[0, 0, 3 + dy, 3 + dx] = 1
    portals = mul(clip(linear(color(15), w, -3)), lattice)

    # Wide painted regions are floor; thin wire backing is not floor.
    w = np.ones((1, 16), np.float32)
    w[0, 0] = 0
    nonzero = linear("x0", w)
    floor = mul(
        clip(linear(nonzero, np.ones((1, 1, 5, 5), np.float32), -24)),
        lattice
    )

    # A retracted door is normally in its narrow wire recess.
    # Perpendicular floor support identifies the blocking position.
    door_colour = linear(colours, [[1, 1, 0]])
    w = np.zeros((4, 1, 7, 7), np.float32)
    for d, (dy, dx) in enumerate(directions):
        w[d, 0, 3 + 3 * dy, 3 + 3 * dx] = 1
    toes = linear(door_colour, w)
    fn = shift6(floor)
    vertical = add(sl(toes, 0, 1), sl(toes, 1, 2))
    horizontal = add(sl(toes, 2, 3), sl(toes, 3, 4))
    support = add(
        mul(vertical, add(sl(fn, 2, 3), sl(fn, 3, 4))),
        mul(horizontal, add(sl(fn, 0, 1), sl(fn, 1, 2)))
    )
    closed_shape = clip(support)
    retracted = mul(doors, inv(closed_shape))
    structural_floor = mul(floor, inv(retracted))

    player_xy = pool(mul(xy, player))
    enemy_xy = pool(mul(xy, enemy))
    enemy_present = pool(enemy)

    previous_dirs = sl("aprev", 1, 5)
    previous_was_move = linear(previous_dirs, [[1, 1, 1, 1]])
    displacement = absolute(sub(player_xy, old_player))
    moved = clip(affine(linear(displacement, [[1, 1]]), 1000, -0.5))
    failed = mul(mul(previous_was_move, inv(moved)), initialized)
    path_count = add(old_path_count, mul(previous_was_move, moved))

    ui = clip(affine(row, -63, 7))
    capacity = clip(affine(
        node("gmean", mul(ui, color(1))), 4096, -0.5
    ))
    reset_flag = mul(inv(initialized), sl("avail", 0, 1))

    # Recording controller.
    unspent = inv(spent)
    normal_here = pool(mul(player, mul(normal_plate, unspent)))
    remote_here = pool(mul(player, mul(remote_plate, unspent)))
    start_remote = mul(
        mul(mul(remote_here, capacity), inv(old_goal_mode)),
        inv(old_mode)
    )
    mode = clip(add(old_mode, start_remote))
    target_xy = add(
        mul(old_target, inv(start_remote)),
        mul(player_xy, start_remote)
    )
    target_point = point_at(target_xy)
    at_target = pool(mul(target_point, player))
    count = add(old_count, mode)
    remote_record = mul(
        mul(mode, at_target), clip(affine(count, 1, -31))
    )
    bounce = mul(mul(mode, at_target), inv(remote_record))
    normal_record = mul(
        mul(mul(normal_here, capacity), inv(old_goal_mode)),
        inv(mode)
    )

    # Patrol velocity estimate, with reversal at a floor boundary.
    velocity = sub(enemy_xy, old_enemy)
    step = 6.0 / 63.0
    velocity = affine(
        sub(relu(affine(velocity, 1, step)),
            relu(affine(velocity, 1, -step))),
        1, -step
    )
    velocity_gate = mul(
        mul(enemy_present, old_enemy_present),
        mul(previous_was_move, moved)
    )
    velocity = mul(velocity, velocity_gate)
    forward_enemy = mul(point_at(add(enemy_xy, velocity)), enemy_present)
    backward_enemy = mul(point_at(sub(enemy_xy, velocity)), enemy_present)
    forward_valid = pool(mul(forward_enemy, floor))
    predicted_enemy = add(
        mul(forward_enemy, forward_valid),
        mul(backward_enemy, inv(forward_valid))
    )
    danger = clip(add(enemy, predicted_enemy))
    legal_floor = mul(mul(floor, inv(doors)), inv(danger))

    # Optimistic goal distances provide switch relevance.
    relaxed = value_loop(
        "relaxed", affine(goal, 128), structural_floor, portals
    )

    # Restore remembered switch interiors under overlapping avatars.
    patch = clip(linear(plates, w5))
    wire_mask = clip(add(colours, patch))

    importance_sources = cat(
        mul(door8, closed_shape),
        mul(door11, closed_shape),
        portals
    )
    importance_sources = mul(importance_sources, relaxed)
    spread7 = np.zeros((3, 3, 7, 7), np.float32)
    for c in range(3):
        spread7[c, c] = 1
    importance_seed = linear(importance_sources, spread7)
    importance_seed = node(
        "min2", importance_seed, affine(one, 0, 128)
    )
    importance_seed = mul(importance_seed, wire_mask)

    paid_sources = mul(plates, spent)
    paid_sources = affine(paid_sources, [1, 0, 1])
    paid_seed = mul(clip(linear(paid_sources, w5)), wire_mask)

    # Simultaneously propagate relevance and replay assignments
    # within each of the three wire colours.
    w = np.zeros((12, 3, 3, 3), np.float32)
    for d, (dy, dx) in enumerate(directions):
        for c in range(3):
            w[3 * d + c, c, 1 + dy, 1 + dx] = 1
    wire_shift = param(w)
    w = np.zeros((3, 3, 3, 3), np.float32)
    for c in range(3):
        w[c, c, 1, 1] = 1
        for dy, dx in directions:
            w[c, c, 1 + dy, 1 + dx] = 1
    wire_sum = param(w)

    outer = G
    G = []
    shape["wire_score"] = ("s", 3)
    shape["wire_paid"] = ("s", 3)
    score_neighbours = max_groups(
        node("conv", "wire_score", w=wire_shift), 3
    )
    next_score = mul(
        maximum("wire_score", score_neighbours), wire_mask
    )
    next_paid = mul(
        clip(node("conv", "wire_paid", w=wire_sum)), wire_mask
    )
    body = G
    G = outer
    G.append({
        "id": "wires",
        "op": "loop",
        "T": 128,
        "state": [
            {"id": "wire_score", "init": importance_seed},
            {"id": "wire_paid", "init": paid_seed},
        ],
        "body": body,
        "next": {"wire_score": next_score, "wire_paid": next_paid},
    })
    shape["wires.wire_score"] = ("s", 3)
    shape["wires.wire_paid"] = ("s", 3)

    virtual_normal = mul(
        mul(door8, closed_shape), sl("wires.wire_paid", 0, 1)
    )
    normal_future = mul(
        structural_floor, clip(add(inv(doors), virtual_normal))
    )

    # Patrols can activate toggle switches in their reachable component.
    enemy_reach = reach_loop(
        "patrol_reach", enemy, normal_future, 64
    )
    toggle_seed = mul(toggle_plate, enemy_reach)
    toggle_wire_mask = sl(wire_mask, 1, 2)
    toggle_seed = mul(
        clip(linear(toggle_seed, np.ones((1, 1, 5, 5), np.float32))),
        toggle_wire_mask
    )

    w = np.zeros((1, 1, 3, 3), np.float32)
    w[0, 0, 1, 1] = 1
    for dy, dx in directions:
        w[0, 0, 1 + dy, 1 + dx] = 1
    toggle_sum = param(w)

    outer = G
    G = []
    shape["toggle_signal"] = ("s", 1)
    toggle_next = mul(
        clip(node("conv", "toggle_signal", w=toggle_sum)),
        toggle_wire_mask
    )
    body = G
    G = outer
    G.append({
        "id": "toggle_wires",
        "op": "loop",
        "T": 128,
        "state": [{"id": "toggle_signal", "init": toggle_seed}],
        "body": body,
        "next": {"toggle_signal": toggle_next},
    })
    shape["toggle_wires.toggle_signal"] = ("s", 1)
    virtual_toggle = mul(
        mul(door11, closed_shape), "toggle_wires.toggle_signal"
    )
    future_floor = mul(
        structural_floor,
        clip(total([inv(doors), virtual_normal, virtual_toggle]))
    )

    portal_signal = sl("wires.wire_paid", 2, 3)
    portal_signal = clip(linear(
        portal_signal, np.ones((1, 1, 7, 7), np.float32)
    ))
    active_portals = mul(portals, portal_signal)

    # Multi-source navigation: reachable goal outranks switch targets.
    relevance = linear(
        mul("wires.wire_score", plates), [[1, 1, 1]]
    )
    candidates = mul(
        add(mul(add(normal_plate, remote_plate), capacity), toggle_plate),
        unspent
    )
    phase_bias = mul(
        add(affine(col, 20), affine(row, -10)), phase
    )
    switch_score = add(affine(relevance, 0.25, 96), phase_bias)
    switch_seed = mul(candidates, switch_score)
    ordinary_seed = maximum(affine(goal, 512), switch_seed)
    navigation_seed = add(
        mul(ordinary_seed, inv(mode)),
        mul(affine(target_point, 1024), mode)
    )
    navigation = value_loop(
        "navigation", navigation_seed, future_floor, active_portals
    )
    player_value = pool(mul(navigation, player))

    # If a full set of recordings provides no target, cycle the slots.
    cycle = mul(
        mul(mul(inv(clip(player_value)), inv(capacity)), inv(mode)),
        clip(path_count)
    )
    cycle = mul(cycle, inv(reset_flag))
    record = mul(
        clip(total([normal_record, remote_record, cycle])),
        inv(reset_flag)
    )

    q = pool(mul(player, shift6(navigation)))
    fallback = pool(mul(player, shift6(relaxed)))
    q = add(q, affine(fallback, 0.01))
    legal_actions = pool(mul(player, shift6(legal_floor)))
    failure_penalty = mul(previous_dirs, failed)

    reverse = np.zeros((4, 4), np.float32)
    reverse[0, 1] = reverse[1, 0] = 1
    reverse[2, 3] = reverse[3, 2] = 1
    bounce_preference = mul(linear(previous_dirs, reverse), bounce)

    features = cat(
        q, legal_actions, failure_penalty, bounce_preference,
        record, reset_flag
    )
    w = np.zeros((8, 18), np.float32)
    b = np.full(8, -20000, np.float32)
    b[6:] = -1000000
    for d in range(4):
        w[1 + d, d] = 1
        w[1 + d, 4 + d] = 20000
        w[1 + d, 8 + d] = -5000
        w[1 + d, 12 + d] = 4000
    w[5, 16] = 60000
    w[0, 17] = 120000
    linear(features, w, b, out="logits")
    affine(one, 0, 0, out="click")

    # Persistent switch map, assignments, and compact controller state.
    mark = mul(
        player,
        clip(add(toggle_plate, mul(add(normal_plate, remote_plate), record)))
    )
    next_spent = mul(
        mul(maximum(spent, mark), inv(cycle)), inv(reset_flag)
    )
    next_mode = mul(mul(mode, inv(record)), inv(reset_flag))
    next_count = mul(count, next_mode)
    next_path_count = mul(
        mul(path_count, inv(record)), inv(reset_flag)
    )
    next_phase = add(
        mul(phase, inv(cycle)), mul(inv(phase), cycle)
    )
    goal_mode = mul(
        mul(clip(affine(player_value, 1, -256)), inv(mode)),
        inv(reset_flag)
    )

    cat(
        pack(cache),
        pack(next_spent),
        next_count,
        next_mode,
        target_xy,
        player_xy,
        goal_mode,
        vone,
        enemy_xy,
        enemy_present,
        next_path_count,
        next_phase,
        out="mem_next"
    )

    return {"graph": G, "memory_size": 213, "params": P}
