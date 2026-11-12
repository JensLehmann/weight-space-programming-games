import numpy as np


def build():
    graph, params = [], {}
    current = graph
    serial = 0
    pserial = 0
    pcache, ncaches, definitions = {}, {}, {}

    def uid(prefix="n"):
        nonlocal serial
        serial += 1
        return "%s_%d" % (prefix, serial)

    def parameter(a):
        nonlocal pserial
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key in pcache:
            return pcache[key]
        pserial += 1
        name = "p_%d" % pserial
        params[name] = a
        pcache[key] = name
        return name

    def node(op, inputs, out=None, **fields):
        cache = ncaches.setdefault(id(current), {})
        key = (op, tuple(inputs), tuple(sorted(fields.items())))
        if out is None and key in cache:
            return cache[key]
        name = out or uid(op)
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        current.append(nd)
        definitions[name] = nd
        if out is None:
            cache[key] = name
        return name

    def conv(x, w, b=None, out=None):
        f = {"w": parameter(w)}
        if b is not None:
            f["b"] = parameter(b)
        return node("conv", [x], out=out, **f)

    def dense(x, w, b=None, out=None):
        f = {"w": parameter(w)}
        if b is not None:
            f["b"] = parameter(b)
        return node("dense", [x], out=out, **f)

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def cat(xs):
        return node("concat", xs)

    def relu(x):
        return node("relu", [x])

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def maximum(a, b):
        return node("max2", [a, b])

    def minimum(a, b):
        return node("min2", [a, b])

    def scale(x, c, channels=1, bias=None):
        src = definitions.get(x)
        if src is not None and src["op"] == "conv":
            old_w = params[src["w"]]
            if old_w.shape[0] == channels:
                w = old_w * np.float32(c)
                b = np.zeros(channels, np.float32)
                if "b" in src:
                    b += params[src["b"]] * np.float32(c)
                if bias is not None:
                    b += np.float32(bias)
                return conv(src["in"][0], w, b)
        w = np.zeros((channels, channels, 1, 1), np.float32)
        for i in range(channels):
            w[i, i, 0, 0] = c
        b = None if bias is None else np.full(channels, bias, np.float32)
        return conv(x, w, b)

    def sum_maps(xs):
        z = xs[0]
        for x in xs[1:]:
            z = add(z, x)
        return z

    def max_maps(xs):
        z = xs[0]
        for x in xs[1:]:
            z = maximum(z, x)
        return z

    ones = sl("coords", 2, 3)
    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    xy = sl("coords", 0, 2)

    def complement(x):
        return sub(ones, x)

    def cap01(x):
        return minimum(relu(x), ones)

    directions = [(-3, 0), (3, 0), (0, -3), (0, 3)]
    unit_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def shift_kernel(c, offsets):
        w = np.zeros((len(offsets), c, 7, 7), np.float32)
        for i, (dy, dx) in enumerate(offsets):
            w[i, 0 if c == 1 else i, dy + 3, dx + 3] = 1
        return w

    def four_shift(x):
        return conv(x, shift_kernel(1, directions))

    def shift_scalar(x, dy, dx):
        return conv(x, shift_kernel(1, [(dy, dx)]))

    def shift_six(x, dy, dx):
        return shift_scalar(
            shift_scalar(x, dy // 2, dx // 2), dy // 2, dx // 2)

    def nine_shifts(x):
        rows = [shift_six(x, -6, 0), x, shift_six(x, 6, 0)]
        result = []
        for r in rows:
            result.extend([
                shift_six(r, 0, -6), r, shift_six(r, 0, 6)])
        return result

    def center_for(v):
        return -6 if v < -3 else (6 if v > 3 else 0)

    def long_kernel(point_lists):
        w = np.zeros((len(point_lists), 9, 7, 7), np.float32)
        centers = [-6, 0, 6]
        for o, points in enumerate(point_lists):
            for dy, dx in points:
                cy, cx = center_for(dy), center_for(dx)
                iy, ix = dy - cy, dx - cx
                if abs(iy) <= 3 and abs(ix) <= 3:
                    j = centers.index(cy) * 3 + centers.index(cx)
                    w[o, j, iy + 3, ix + 3] += 1
        return w

    specs = [(3, 3, 1, 1), (6, 3, 2, 1),
             (3, 6, 1, 2), (6, 6, 2, 2)]
    footprints = []
    for h, width, ay, ax in specs:
        footprints.append({
            (y - ay, x - ax)
            for y in range(h) for x in range(width)
        })
    footprints.append({
        (y, x) for y in range(-4, 5) for x in range(-4, 5)
        if abs(y) <= 1 or abs(x) <= 1
    })
    bomb_fp = footprints[3]

    def footprint_sum(x, footprint, bias=0.0, reverse=False):
        pts = [(-y, -z) if reverse else (y, z)
               for y, z in footprint]
        if all(abs(y) <= 3 and abs(z) <= 3 for y, z in pts):
            w = np.zeros((1, 1, 7, 7), np.float32)
            for y, z in pts:
                w[0, 0, y + 3, z + 3] += 1
            return conv(x, w, [bias])

        w = np.zeros((5, 1, 7, 7), np.float32)
        for y, z in pts:
            if abs(y) <= 3 and abs(z) <= 3:
                w[0, 0, y + 3, z + 3] += 1
            elif y == -4:
                w[1, 0, 0, z + 3] += 1
            elif y == 4:
                w[2, 0, 6, z + 3] += 1
            elif z == -4:
                w[3, 0, y + 3, 0] += 1
            elif z == 4:
                w[4, 0, y + 3, 6] += 1
        partial = conv(x, w)
        finish = np.zeros((1, 5, 3, 3), np.float32)
        finish[0, 0, 1, 1] = 1
        finish[0, 1, 0, 1] = 1
        finish[0, 2, 2, 1] = 1
        finish[0, 3, 1, 0] = 1
        finish[0, 4, 1, 2] = 1
        return conv(partial, finish, [bias])

    zero_plane = sl("x0", 0, 1)
    w = np.zeros((2, 16, 1, 1), np.float32)
    w[0, [0, 14], 0, 0] = 1
    w[1, [0, 4, 5], 0, 0] = 1
    yellow_pixels = conv("x0", w)

    pieces, selected = [], []
    for h, width, ay, ax in specs:
        w = np.zeros((1, 2, 7, 7), np.float32)
        for y in range(h):
            for x in range(width):
                border = y in (0, h - 1) or x in (0, width - 1)
                w[0, 0 if border else 1, y - ay + 3, x - ax + 3] = 2
        base = relu(conv(yellow_pixels, w, [1 - 2 * h * width]))
        edges = [
            [(0, x) for x in range(width)],
            [(h - 1, x) for x in range(width)],
            [(y, 0) for y in range(h)],
            [(y, width - 1) for y in range(h)],
        ]
        ew = np.zeros((4, 1, 7, 7), np.float32)
        for d, edge in enumerate(edges):
            for y, x in edge[1:-1]:
                ew[d, 0, y - ay + 3, x - ax + 3] += 1.0 / (len(edge) - 2)
            for y, x in (edge[0], edge[-1]):
                ew[d, 0, y - ay + 3, x - ax + 3] -= 0.5
        mixed = relu(conv(zero_plane, ew))
        penalty = conv(mixed, np.full((1, 4, 1, 1), 100, np.float32))
        p = relu(sub(base, penalty))
        pieces.append(p)
        selected.append(mul(p, zero_plane))

    eleven = sl("x0", 11, 12)
    w = np.zeros((1, 1, 7, 7), np.float32)
    n = 0
    for y in range(-3, 4):
        for x in range(-3, 4):
            if abs(y) <= 1 or abs(x) <= 1:
                w[0, 0, y + 3, x + 3] = 2
                n += 1
            else:
                w[0, 0, y + 3, x + 3] = -2
    pieces.append(relu(conv(eleven, w, [1 - 2 * n])))

    twelve = sl("x0", 12, 13)
    thirteen = sl("x0", 13, 14)
    bomb_pixels = add(twelve, thirteen)
    bomb = relu(scale(footprint_sum(bomb_pixels, bomb_fp),
                      2, bias=-71))

    # Goal borders, allowing movable sprites to occlude some border pixels.
    goal_points = []
    for h, width, ay, ax in specs:
        gh, gw = h + 2, width + 2
        gay, gax = ay + 1, ax + 1
        goal_points.append([
            (y - gay, x - gax)
            for y in range(gh) for x in range(gw)
            if y in (0, gh - 1) or x in (0, gw - 1)
        ])
    cross_goal = []
    for y in range(11):
        for x in range(11):
            border = (
                (y in (0, 10) and 3 <= x <= 7)
                or (x in (0, 10) and 3 <= y <= 7)
                or (y in (3, 7) and (x <= 3 or x >= 7))
                or (x in (3, 7) and (y <= 3 or y >= 7))
            )
            if border:
                cross_goal.append((y - 5, x - 5))
    goal_points.append(cross_goal)
    gw = long_kernel(goal_points)
    four = sl("x0", 4, 5)
    visible_count = conv(cat(nine_shifts(four)), gw)
    aw = np.zeros((1, 16, 1, 1), np.float32)
    aw[0, [0, 4, 11, 12, 13, 14], 0, 0] = 1
    acceptable = conv("x0", aw)
    covered_match = relu(conv(
        cat(nine_shifts(acceptable)), 2 * gw,
        [1 - 2 * len(p) for p in goal_points]))
    evidence = cap01(conv(
        visible_count, np.eye(5, dtype=np.float32)[:, :, None, None],
        [1 - max(6, (len(p) + 2) // 3) for p in goal_points]))
    detected_goals = mul(covered_match, evidence)

    # Four coordinate slots for each of the five goal shapes.
    # Each 15-entry slot stores five rows, five columns, five presence flags.
    remembered = []
    for slot in range(4):
        r = vs("mem", 15 * slot, 15 * slot + 5)
        c = vs("mem", 15 * slot + 5, 15 * slot + 10)
        e = vs("mem", 15 * slot + 10, 15 * slot + 15)
        dr = node("abs", [sub(row, r)])
        dc = node("abs", [sub(col, c)])
        exact = relu(scale(add(dr, dc), -1024, 5, bias=1))
        remembered.append(mul(exact, e))
    goal_tensor = maximum(detected_goals, max_maps(remembered))

    # Parallel deterministic coordinate extraction for all five shapes.
    rank_w = np.zeros((1, 3, 1, 1), np.float32)
    rank_w[0, :, 0, 0] = [1, 1.0 / 128, 1]
    rank = conv("coords", rank_w)
    loop_id = uid("goal_memory")
    rem_id, hist_id = uid("remaining"), uid("history")
    body = []
    old_current = current
    current = body
    ranked = mul(rem_id, rank)
    best_rank = node("gmax", [ranked])
    delta = node("abs", [sub(ranked, best_rank)])
    winner = mul(rem_id, relu(scale(delta, -16384, 5, bias=1)))
    wr = node("gmax", [mul(winner, row)])
    wc = node("gmax", [mul(winner, col)])
    we = node("gmax", [winner])
    item = node("vconcat", [wr, wc, we])
    old_hist = vs(hist_id, 0, 45)
    new_hist = node("vconcat", [item, old_hist])
    new_rem = relu(sub(rem_id, winner))
    current = old_current
    current.append({
        "id": loop_id,
        "op": "loop",
        "T": 4,
        "state": [
            {"id": rem_id, "init": goal_tensor},
            {"id": hist_id, "zeros": [60]},
        ],
        "body": body,
        "next": {rem_id: new_rem, hist_id: new_hist},
    })
    goal_memory = loop_id + "." + hist_id
    goals = [sl(goal_tensor, i, i + 1) for i in range(5)]

    unfilled, tasks = [], []
    for i in range(5):
        filled = mul(goals[i], pieces[i])
        uf = relu(sub(goals[i], filled))
        unfilled.append(uf)
        remaining = node("gmax", [uf])
        task = mul(pieces[i], remaining)
        if i < 4:
            task = mul(task, complement(selected[i]))
        tasks.append(task)

    wall_pixels = sub(sl("x0", 2, 3), ones)
    gulch_pixels = sl("x0", 15, 16)
    fixed_movable = add(eleven, bomb_pixels)
    wall_valid, gulch_flags = [], []
    clear_floor, nav_real_valid = [], []
    for i, fp in enumerate(footprints):
        blocked = footprint_sum(wall_pixels, fp, bias=len(fp))
        valid = relu(sub(ones, blocked))
        gf = cap01(footprint_sum(gulch_pixels, fp))
        wall_valid.append(valid)
        gulch_flags.append(gf)
        if i < 4:
            cf = mul(valid, complement(gf))
            object_free = relu(sub(
                ones, footprint_sum(fixed_movable, fp)))
            clear_floor.append(cf)
            nav_real_valid.append(mul(cf, object_free))

    task_cost = scale(ones, 2)

    def value_field(seed, valid, cost, iterations):
        nonlocal current
        seed = mul(seed, valid)
        lid, sid = uid("vf"), uid("state")
        body = []
        old = current
        current = body
        neighbors = four_shift(sid)
        best = max_maps([sl(neighbors, d, d + 1) for d in range(4)])
        propagated = relu(sub(best, cost))
        nxt = mul(maximum(seed, propagated), valid)
        current = old
        current.append({
            "id": lid, "op": "loop", "T": iterations,
            "state": [{"id": sid, "init": seed}],
            "body": body, "next": {sid: nxt},
        })
        return lid + "." + sid

    task_values = [
        value_field(scale(unfilled[i], 100),
                    wall_valid[i], task_cost, 40)
        for i in range(5)
    ]

    def landing_values(value, valid, gulch):
        nonlocal current
        value4 = cat([value] * 4)
        neighbor_valid = four_shift(valid)
        neighbor_blocked = scale(neighbor_valid, -1, 4, bias=1)
        stop_value = mul(neighbor_blocked, value4)
        ordinary_value = mul(value4, complement(gulch))
        shift_w = shift_kernel(4, directions)

        def phase(initial, iterations, tail):
            nonlocal current
            lid, sid = uid("slide"), uid("state")
            body = []
            old = current
            current = body
            shifted = conv(sid, shift_w)
            moved = add(mul(shifted, neighbor_valid), stop_value)
            nxt = add(mul(moved, gulch), ordinary_value) if tail else moved
            current = old
            current.append({
                "id": lid, "op": "loop", "T": iterations,
                "state": [{"id": sid, "init": initial}],
                "body": body, "next": {sid: nxt},
            })
            return lid + "." + sid

        return phase(phase(value4, 12, True), 6, False)

    push_scores = []
    for i in range(5):
        landed = landing_values(
            task_values[i], wall_valid[i], gulch_flags[i])
        improvement = relu(scale(
            sub(landed, task_values[i]), 1, 4, bias=-0.25))
        flag = minimum(scale(improvement, 4, 4), ones)
        score = add(scale(flag, 180, 4), improvement)
        push_scores.append(mul(score, tasks[i]))

    bomb_score = scale(bomb, 45)
    push_scores.append(cat([bomb_score] * 4))
    push_footprints = footprints + [bomb_fp]

    def contact_offsets(target_fp, pusher_fp, direction, reach):
        cy, cx = direction
        occupied = {
            (ty - py, tx - px)
            for ty, tx in target_fp for py, px in pusher_fp
        }
        result = set()
        for distance in range(1, reach + 1):
            for y, x in occupied:
                q = (y - cy * distance, x - cx * distance)
                if q not in occupied:
                    result.add(q)
        return result

    def contact_transform(sources, target_fps, direction, reach=3):
        cy, cx = direction
        shifted_inputs = []
        for source in sources:
            middle = shift_six(source, 6 * cy, 6 * cx)
            if cy:
                shifted_inputs.extend([
                    shift_six(middle, 0, -6),
                    middle, shift_six(middle, 0, 6)])
            else:
                shifted_inputs.extend([
                    shift_six(middle, -6, 0),
                    middle, shift_six(middle, 6, 0)])
        w = np.zeros((4, 3 * len(sources), 7, 7), np.float32)
        centers = [-6, 0, 6]
        for p in range(4):
            for t, target_fp in enumerate(target_fps):
                offsets = contact_offsets(
                    target_fp, footprints[p], direction, reach)
                for dy, dx in offsets:
                    sy, sx = -dy, -dx
                    orth_center = center_for(sx if cy else sy)
                    if cy:
                        iy, ix = sy - 6 * cy, sx - orth_center
                    else:
                        iy, ix = sy - orth_center, sx - 6 * cx
                    if abs(iy) <= 3 and abs(ix) <= 3:
                        channel = 3 * t + centers.index(orth_center)
                        w[p, channel, iy + 3, ix + 3] += 1
        return conv(cat(shifted_inputs), w)

    contact_by_dir = []
    for d in range(4):
        sources = [sl(s, d, d + 1) for s in push_scores]
        contact_by_dir.append(
            contact_transform(sources, push_footprints, unit_dirs[d]))

    push_subgoals, push_direction_maps = [], []
    for p in range(4):
        fc = four_shift(clear_floor[p])
        ds = [
            mul(sl(contact_by_dir[d], p, p + 1),
                sl(fc, d, d + 1))
            for d in range(4)
        ]
        push_direction_maps.append(ds)
        push_subgoals.append(max_maps(ds))

    route_seeds = [
        maximum(scale(unfilled[p], 260), push_subgoals[p])
        for p in range(4)
    ]

    # Infer burst direction from the charging edge.
    bomb_edges = np.zeros((4, 2, 7, 7), np.float32)
    edge_pairs = [
        ([(3, x) for x in range(-2, 4)],
         [(-2, x) for x in range(-2, 4)]),
        ([(-2, x) for x in range(-2, 4)],
         [(3, x) for x in range(-2, 4)]),
        ([(y, 3) for y in range(-2, 4)],
         [(y, -2) for y in range(-2, 4)]),
        ([(y, -2) for y in range(-2, 4)],
         [(y, 3) for y in range(-2, 4)]),
    ]
    for d, (charged, unchanged) in enumerate(edge_pairs):
        for y, x in charged:
            bomb_edges[d, 0, y + 3, x + 3] = 2
        for y, x in unchanged:
            bomb_edges[d, 1, y + 3, x + 3] = 2
    known = mul(relu(conv(
        cat([twelve, thirteen]), bomb_edges,
        np.full(4, -23, np.float32))), bomb)
    known_sum = conv(known, np.ones((1, 4, 1, 1), np.float32))
    unknown = relu(sub(bomb, known_sum))

    blast_masks = []
    for d, (dy, dx) in enumerate(unit_dirs):
        source = add(sl(known, d, d + 1), unknown)
        blast_masks.append(cap01(contact_transform(
            [source], [bomb_fp], (-dy, -dx), reach=6)))

    real_routes = []
    directional_shift = shift_kernel(4, directions)
    for p in range(4):
        after = cat([task_values[p]] * 4)
        for _ in range(3):
            after = conv(after, directional_shift)
        gain = relu(scale(
            sub(after, task_values[p]), 1, 4, bias=-0.5))
        flag = minimum(scale(gain, 2, 4), ones)
        benefit = add(scale(flag, 183, 4), gain)
        blast = max_maps([
            mul(sl(blast_masks[d], p, p + 1),
                sl(benefit, d, d + 1))
            for d in range(4)
        ])
        seed = maximum(route_seeds[p], blast)
        real_routes.append(value_field(
            seed, nav_real_valid[p], ones, 44))

    selected_anchor = cap01(sum_maps(selected))
    selected_xy = node("gmax", [mul(selected_anchor, xy)])
    selected_exists = node("gmax", [selected_anchor])
    node("vconcat",
         [goal_memory, selected_xy, selected_exists],
         out="mem_next")

    old_xy = vs("mem", 60, 62)
    old_valid = vs("mem", 62, 63)
    distance = node("abs", [sub(xy, old_xy)])
    previous_position = relu(conv(
        distance, np.full((1, 2, 1, 1), -63, np.float32), [1]))
    visit_penalty = scale(mul(previous_position, old_valid), 0.4)
    future_visit_penalty = four_shift(visit_penalty)

    action_fields = []
    for p in range(4):
        walk = relu(sub(four_shift(real_routes[p]), ones))
        walk = mul(walk, four_shift(nav_real_valid[p]))
        walk = sub(walk, future_visit_penalty)
        quality = maximum(walk, cat(push_direction_maps[p]))
        action_fields.append(mul(quality, selected[p]))
    action_quality = node("gmax", [sum_maps(action_fields)])

    click_map = max_maps([
        mul(relu(sub(pieces[p], selected[p])), real_routes[p])
        for p in range(4)
    ])
    conv(click_map, np.ones((1, 1, 1, 1), np.float32), out="click")
    click_quality = node("gmax", [click_map])
    head = node("vconcat", [action_quality, click_quality])

    w = np.zeros((8, 5), np.float32)
    b = np.full(8, -1000, np.float32)
    for d in range(4):
        w[d + 1, d] = 1
        b[d + 1] = 0
    w[6, 4] = 1
    b[6] = -5
    dense(head, w, b, out="logits")

    # Static dead-code removal; no observations or game execution are used.
    def retain_needed(nodes, wanted):
        live, kept, defined = set(wanted), [], set()
        for nd in nodes:
            defined.add(nd["id"])
            if nd["op"] == "loop":
                for s in nd["state"]:
                    defined.add(nd["id"] + "." + s["id"])
        for nd in reversed(nodes):
            if nd["op"] == "loop":
                finals = {
                    nd["id"] + "." + s["id"] for s in nd["state"]
                }
                if not (finals & live):
                    continue
                body, external = retain_needed(
                    nd["body"], set(nd["next"].values()))
                state_ids = {s["id"] for s in nd["state"]}
                live.update(external - state_ids)
                for s in nd["state"]:
                    if "init" in s:
                        live.add(s["init"])
                new_nd = dict(nd)
                new_nd["body"] = body
                kept.append(new_nd)
            elif nd["id"] in live:
                live.update(nd["in"])
                kept.append(nd)
        return list(reversed(kept)), live - defined

    graph, _ = retain_needed(graph, {"logits", "click", "mem_next"})
    used_params = set()

    def counts(nodes):
        structural, evaluations = 0, 0
        for nd in nodes:
            structural += 1
            evaluations += 1
            for field in ("w", "b"):
                if field in nd:
                    used_params.add(nd[field])
            if nd["op"] == "loop":
                ns, ne = counts(nd["body"])
                structural += ns
                evaluations += nd["T"] * ne
        return structural, evaluations

    ns, ne = counts(graph)
    params = {k: v for k, v in params.items() if k in used_params}
    assert ns <= 1000
    assert ne <= 10000
    assert sum(v.size for v in params.values()) <= 150000

    return {"graph": graph, "memory_size": 63, "params": params}
