import numpy as np


def build():
    graph = []
    params = {}
    target = [graph]
    serial = [0]
    pserial = [0]

    def unique(prefix="n"):
        name = "%s%d" % (prefix, serial[0])
        serial[0] += 1
        return name

    def node(op, inputs, **fields):
        name = unique()
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        target[0].append(nd)
        return name

    def parameter(a):
        name = "p%d" % pserial[0]
        pserial[0] += 1
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def conv(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], **fields)

    def dense(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], **fields)

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vsl(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

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

    def relu(x):
        return node("relu", [x])

    def point(x, weight, bias=0.0, channels=1):
        w = np.eye(channels, dtype=np.float32).reshape(
            channels, channels, 1, 1
        ) * weight
        return conv(x, w, np.full(channels, bias, dtype=np.float32))

    def clip01(x):
        return sub(relu(x), relu(point(x, 1.0, -1.0)))

    def sum_channels(x, count):
        return conv(x, np.ones((1, count, 1, 1), dtype=np.float32))

    def global_max(x):
        return node("gmax", [x])

    def one_minus_vector(x):
        return dense(x, [[-1.0]], [1.0])

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    one = sl("coords", 2, 3)

    board_mask = mul(
        mul(
            clip01(point(row, 63.0, -2.0)),
            clip01(point(row, -63.0, 40.0)),
        ),
        mul(
            clip01(point(col, 63.0, -12.0)),
            clip01(point(col, -63.0, 62.0)),
        ),
    )

    player = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 0, 0, 1],
    ], dtype=np.int32)

    goal = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ], dtype=np.int32)

    # Goal response is at goal sprite top-left + (2, 2).
    gw = np.zeros((4, 16, 7, 7), dtype=np.float32)
    gb = np.zeros(4, dtype=np.float32)
    for r in range(4):
        mask = np.rot90(goal, -r)
        gb[r] = -float(mask.sum()) + 0.5
        for y in range(6):
            for x in range(6):
                gw[r, 11, y + 1, x + 1] = (
                    1.0 if mask[y, x] else -1.0
                )
    gd = mul(
        point(relu(conv("x0", gw, gb)), 2.0, channels=4),
        board_mask,
    )

    # Scale-1 player response is at player sprite top-left.
    pw = np.zeros((4, 16, 7, 7), dtype=np.float32)
    pb = np.zeros(4, dtype=np.float32)
    for r in range(4):
        mask = np.rot90(player, -r)
        pb[r] = -float(mask.sum()) + 0.5
        for y in range(4):
            for x in range(4):
                pw[r, 11, y + 3, x + 3] = (
                    1.0 if mask[y, x] else -1.0
                )
    p1 = mul(
        point(relu(conv("x0", pw, pb)), 2.0, channels=4),
        board_mask,
    )

    # Scale-2 template split into parts fitting the maximum kernel.
    p2w = np.zeros((16, 16, 7, 7), dtype=np.float32)
    p2b = np.zeros(16, dtype=np.float32)
    for r in range(4):
        mask = np.repeat(
            np.repeat(np.rot90(player, -r), 2, axis=0),
            2,
            axis=1,
        )
        p2b[4 * r] = -float(mask.sum()) + 0.5
        for y in range(8):
            for x in range(8):
                value = 1.0 if mask[y, x] else -1.0
                if y < 7 and x < 7:
                    part, ky, kx = 0, y, x
                elif y < 7:
                    part, ky, kx = 1, y, 6
                elif x < 7:
                    part, ky, kx = 2, 6, x
                else:
                    part, ky, kx = 3, 6, 6
                p2w[4 * r + part, 11, ky, kx] = value

    pieces = conv("x0", p2w, p2b)
    join = np.zeros((4, 16, 3, 3), dtype=np.float32)
    for r in range(4):
        join[r, 4 * r + 0, 1, 1] = 1.0
        join[r, 4 * r + 1, 1, 2] = 1.0
        join[r, 4 * r + 2, 2, 1] = 1.0
        join[r, 4 * r + 3, 2, 2] = 1.0
    p2 = mul(
        point(relu(conv(pieces, join)), 2.0, channels=4),
        board_mask,
    )

    gv = global_max(gd)
    p1v = global_max(p1)
    p2v = global_max(p2)
    pv = node("vadd", [p1v, p2v])
    scale2 = dense(p2v, np.ones((1, 4), dtype=np.float32))

    p1map = sum_channels(p1, 4)
    p2map = sum_channels(p2, 4)
    gmap = sum_channels(gd, 4)

    shift_p2 = np.zeros((1, 1, 7, 7), dtype=np.float32)
    shift_p2[0, 0, 6, 6] = 1.0
    start = add(p1map, conv(p2map, shift_p2))

    shift_goal = np.zeros((1, 1, 3, 3), dtype=np.float32)
    shift_goal[0, 0, 2, 2] = 1.0
    goal_pos = conv(gmap, shift_goal)

    # Relative rotation as command bits.
    pair_input = vcat(gv, pv)
    pair_w = np.zeros((16, 8), dtype=np.float32)
    pair_b = np.full(16, -1.0, dtype=np.float32)
    for g in range(4):
        for p in range(4):
            i = 4 * g + p
            pair_w[i, g] = 1.0
            pair_w[i, 4 + p] = 1.0
    pairs = node("vrelu", [dense(pair_input, pair_w, pair_b)])

    rot_w = np.zeros((3, 16), dtype=np.float32)
    same_w = np.zeros((1, 16), dtype=np.float32)
    for g in range(4):
        for p in range(4):
            delta = (g - p) % 4
            code = (0, 5, 7, 6)[delta]
            for bit in range(3):
                rot_w[bit, 4 * g + p] = (code >> bit) & 1
            if g == p:
                same_w[0, 4 * g + p] = 1.0
    rot_bits = dense(pairs, rot_w)
    same_rotation = dense(pairs, same_w)
    need_rotation = one_minus_vector(same_rotation)

    # Detect horizontal and vertical instruction marks.
    mw = np.zeros((2, 16, 5, 5), dtype=np.float32)
    for d in (-1, 0, 1):
        for color in (1, 5):
            mw[0, color, 2, 2 + d] = 1.0
            mw[1, color, 2 + d, 2] = 1.0
        mw[0, 0, 1, 2 + d] = 1.0
        mw[0, 0, 3, 2 + d] = 1.0
        mw[1, 0, 2 + d, 1] = 1.0
        mw[1, 0, 2 + d, 3] = 1.0
    mw[0, 0, 2, 0] = 1.0
    mw[0, 0, 2, 4] = 1.0
    mw[1, 0, 0, 2] = 1.0
    mw[1, 0, 4, 2] = 1.0

    marks = point(
        relu(conv("x0", mw, [-10.5, -10.5])),
        2.0,
        channels=2,
    )
    marks = mul(
        marks,
        mul(
            mul(
                clip01(point(row, 63.0, -32.0)),
                clip01(point(row, -63.0, 50.0)),
            ),
            clip01(point(col, 63.0, -12.0)),
        ),
    )
    mark = sum_channels(marks, 2)

    first_col = dense(
        global_max(mul(mark, point(col, -1.0, 1.0))),
        [[-63.0]],
        [63.0],
    )
    first_row = dense(
        global_max(mul(mark, point(row, -1.0, 1.0))),
        [[-63.0]],
        [63.0],
    )
    last_col = dense(global_max(mul(mark, col)), [[63.0]])
    slots = dense(vcat(first_col, last_col), [[-0.2, 0.2]], [1.0])

    slot = sub(
        point(col, 63.0 / 5.0), dense(first_col, [[1.0 / 5.0]])
    )
    bit_index = sub(
        point(row, 63.0 / 3.0), dense(first_row, [[1.0 / 3.0]])
    )

    # A player position is blocked if its 4x4 bounding box
    # overlaps a visible color-6 obstacle.
    wall = sl("x0", 6, 7)
    wall_w = np.zeros((1, 1, 7, 7), dtype=np.float32)
    wall_w[0, 0, 3:7, 3:7] = 1.0
    blocked = clip01(conv(wall, wall_w))
    safe_area = mul(
        mul(
            clip01(point(row, 63.0, -2.0)),
            clip01(point(row, -63.0, 39.0)),
        ),
        mul(
            clip01(point(col, 63.0, -12.0)),
            clip01(point(col, -63.0, 61.0)),
        ),
    )
    safe = mul(safe_area, sub(one, blocked))

    # Four shifts by four pixels, using two shifts by two.
    # Directions: right, up, left, down.
    directions = [(0, 2), (-2, 0), (0, -2), (2, 0)]

    def shifted_four(x, channels):
        w1 = np.zeros((4 * channels, channels, 5, 5), np.float32)
        w2 = np.zeros(
            (4 * channels, 4 * channels, 5, 5), np.float32
        )
        for d, (dy, dx) in enumerate(directions):
            ky, kx = 2 - dy, 2 - dx
            for c in range(channels):
                i = d * channels + c
                w1[i, c, ky, kx] = 1.0
                w2[i, i, ky, kx] = 1.0
        z = conv(x, w1)
        z = conv(z, w2)
        return [
            sl(z, d * channels, (d + 1) * channels)
            for d in range(4)
        ]

    def bfs(seed, channels, iterations):
        # Values start at 64 and decrease by one per grid move.
        loop_id = unique("loop")
        state_id = unique("state")
        body = []
        outer = target[0]
        target[0] = body

        neighbours = shifted_four(state_id, channels)
        best = neighbours[0]
        for candidate in neighbours[1:]:
            best = node("max2", [best, candidate])
        best = relu(point(best, 1.0, -1.0, channels=channels))
        best = mul(best, safe)
        updated = node("max2", [state_id, best])

        target[0] = outer
        outer.append({
            "id": loop_id,
            "op": "loop",
            "T": iterations,
            "state": [{"id": state_id, "init": seed}],
            "body": body,
            "next": {state_id: updated},
        })
        return loop_id + "." + state_id

    seeds = point(cat(goal_pos, start), 64.0, channels=2)
    distances = bfs(seeds, 2, 64)
    goal_dist = sl(distances, 0, 1)
    start_dist = sl(distances, 1, 2)
    goal_at_start = global_max(mul(goal_dist, start))

    # Detect visible checkerboard checkpoints.
    ck_w = np.zeros((1, 16, 7, 7), dtype=np.float32)
    for y in range(4):
        for x in range(4):
            ck_w[0, 11, 3 + y, 3 + x] = (
                1.0 if (x + y) % 2 == 0 else -1.0
            )
    checkpoints = mul(
        point(relu(conv("x0", ck_w, [-7.5])), 2.0),
        board_mask,
    )

    # Restrict checkpoints to those reachable within the program and
    # strictly closer to the goal.
    ck_budget = sub(mul(one, slots), scale2)
    reachable_ck = clip01(point(add(start_dist, ck_budget), 1.0, -63.0))
    closer_ck = clip01(
        point(sub(goal_dist, goal_at_start), 2.0, -1.0)
    )
    candidates = mul(checkpoints, mul(reachable_ck, closer_ck))
    has_candidate = global_max(candidates)

    ck_score = mul(
        candidates,
        add(
            add(point(goal_dist, 100.0), start_dist),
            add(point(col, 0.5), point(row, 0.01)),
        ),
    )
    best_ck_score = global_max(ck_score)
    best_ck = mul(
        candidates,
        clip01(point(sub(ck_score, best_ck_score), 4000.0, 1.0)),
    )

    # Reserve slots for shrinking and for the final rotation.
    goal_budget = sub(sub(mul(one, slots), scale2), need_rotation)
    goal_reachable = clip01(
        point(add(mul(one, goal_at_start), goal_budget), 1.0, -63.0)
    )
    goal_reachable_v = global_max(goal_reachable)
    not_goal_reachable = one_minus_vector(goal_reachable_v)
    choose_ck_v = node("vmul", [not_goal_reachable, has_candidate])
    choose_goal_v = one_minus_vector(choose_ck_v)

    ck_seed = point(best_ck, 64.0)
    ck_dist = bfs(ck_seed, 1, 6)
    waypoint_dist = add(
        mul(goal_dist, choose_goal_v),
        mul(ck_dist, choose_ck_v),
    )
    waypoint = add(
        mul(goal_pos, choose_goal_v),
        mul(best_ck, choose_ck_v),
    )
    distance_at_start = global_max(mul(waypoint_dist, start))
    steps_to_waypoint = dense(distance_at_start, [[-1.0]], [64.0])

    # Unroll six greedy moves along the distance field.
    position = start
    step_choices = []
    for _ in range(6):
        next_positions = shifted_four(position, 1)
        scores = []
        for candidate in next_positions:
            scores.append(global_max(mul(candidate, waypoint_dist)))
        score_vec = vcat(*scores)

        diff_w = np.zeros((12, 4), dtype=np.float32)
        diff_b = np.zeros(12, dtype=np.float32)
        k = 0
        groups = []
        for i in range(4):
            group = []
            for j in range(4):
                if i == j:
                    continue
                diff_w[k, j] = 1.0
                diff_w[k, i] = -1.0
                diff_b[k] = 0.5 if j < i else -0.5
                group.append(k)
                k += 1
            groups.append(group)
        violations = node(
            "vrelu", [dense(score_vec, diff_w, diff_b)]
        )
        select_w = np.zeros((4, 12), dtype=np.float32)
        for i, group in enumerate(groups):
            for j in group:
                select_w[i, j] = -2.0
        selected = node(
            "vrelu",
            [dense(violations, select_w, np.ones(4, np.float32))],
        )

        at_waypoint = global_max(mul(position, waypoint))
        moving = one_minus_vector(at_waypoint)
        selected = node("vmul", [selected, moving])
        step_choices.append(selected)

        moved = None
        for d in range(4):
            part = mul(next_positions[d], vsl(selected, d, d + 1))
            moved = part if moved is None else add(moved, part)
        position = add(moved, mul(position, at_waypoint))

    scale_map = mul(one, scale2)
    t = sub(slot, scale_map)
    first_slot = relu(point(node("abs", [slot]), -1.0, 1.0))
    shrink = mul(first_slot, scale_map)

    # Directions are right=2, up=33, left=1, down=3.
    command_w = np.zeros((6, 4), dtype=np.float32)
    for d, code in enumerate((2, 33, 1, 3)):
        for bit in range(6):
            command_w[bit, d] = (code >> bit) & 1

    bits = [point(one, 0.0) for _ in range(6)]
    for i, choice in enumerate(step_choices):
        slot_i = relu(point(
            node("abs", [point(t, 1.0, -float(i))]),
            -1.0,
            1.0,
        ))
        code_bits = dense(choice, command_w)
        for b in range(6):
            bits[b] = add(
                bits[b], mul(slot_i, vsl(code_bits, b, b + 1))
            )

    rotation_slot = mul(
        relu(point(
            node("abs", [sub(t, steps_to_waypoint)]),
            -1.0,
            1.0,
        )),
        choose_goal_v,
    )
    for b in range(3):
        bits[b] = add(
            bits[b], mul(rotation_slot, vsl(rot_bits, b, b + 1))
        )

    # Shrink command 9 has bits 0 and 3 set.
    bits[0] = add(bits[0], shrink)
    bits[3] = add(bits[3], shrink)
    desired_bits = cat(*bits)

    selectors_w = np.ones((6, 1, 1, 1), dtype=np.float32)
    selectors_b = -np.arange(6, dtype=np.float32)
    distances_to_bits = node(
        "abs", [conv(bit_index, selectors_w, selectors_b)]
    )
    select_w = -np.eye(6, dtype=np.float32).reshape(6, 6, 1, 1)
    selectors = relu(
        conv(distances_to_bits, select_w, np.ones(6, np.float32))
    )
    desired = sum_channels(mul(desired_bits, selectors), 6)

    color5 = sl("x0", 5, 6)
    mismatch = node("abs", [sub(desired, color5)])
    toggle_score = mul(mark, mismatch)

    color9 = sl("x0", 9, 10)
    button_region = mul(
        mul(
            clip01(point(row, 63.0, -52.0)),
            clip01(point(row, -63.0, 63.0)),
        ),
        clip01(point(col, 63.0, -18.0)),
    )
    button_score = mul(
        mul(color9, button_region),
        point(col, 0.01, 0.25),
    )

    graph.append({
        "id": "click",
        "op": "add",
        "in": [toggle_score, button_score],
    })

    logits_w = np.zeros((8, 8), dtype=np.float32)
    logits_b = np.zeros(8, dtype=np.float32)
    logits_b[6] = 1.0
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": ["avail"],
        "w": parameter(logits_w),
        "b": parameter(logits_b),
    })

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params,
    }
