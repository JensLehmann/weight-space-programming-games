import numpy as np


def build():
    graph = []
    params = {}
    nodes = graph
    serial = 0
    pserial = 0

    def node(op, inputs=None, **fields):
        nonlocal serial
        serial += 1
        name = "n%d" % serial
        nd = {"id": name, "op": op, "in": list(inputs or [])}
        nd.update(fields)
        nodes.append(nd)
        return name

    def param(a):
        nonlocal pserial
        pserial += 1
        name = "p%d" % pserial
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def conv(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("conv", [x], **fields)

    def conv_named(x, wname):
        return node("conv", [x], w=wname)

    def dense(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", [x], **fields)

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vsl(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def cat(*xs):
        return node("concat", xs)

    def vcat(*xs):
        return node("vconcat", xs)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def mx(a, b):
        return node("max2", [a, b])

    def mn(a, b):
        return node("min2", [a, b])

    def relu(a):
        return node("relu", [a])

    def gmax(a):
        return node("gmax", [a])

    def scale(a, k, channels=1):
        w = np.eye(channels, dtype=np.float32)[:, :, None, None] * k
        return conv(a, w)

    def vaff(a, k, bias=0.0):
        return dense(a, [[k]], [bias])

    def vsub(a, b):
        return dense(vcat(a, b), [[1.0, -1.0]])

    def vmax(a, b):
        return node("vadd", [node("vrelu", [vsub(a, b)]), b])

    def vclip(a):
        positive = node("vrelu", [a])
        remaining = node("vrelu", [vaff(positive, -1.0, 1.0)])
        return vaff(remaining, -1.0, 1.0)

    def clip(a):
        return mn(relu(a), ones)

    def inv(a):
        return sub(ones, a)

    def reduce_max(x, count):
        z = sl(x, 0, 1)
        for i in range(1, count):
            z = mx(z, sl(x, i, i + 1))
        return z

    def color(c):
        return sl("x0", c, c + 1)

    ones = sl("coords", 2, 3)
    onev = gmax(ones)

    # Detect and repaint the exit in each rotation.
    base_goal = np.array([
        [9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 9, 9, 9, 9],
    ], dtype=np.int32)
    goal_templates = [np.rot90(base_goal, -k) for k in range(4)]
    gw = np.zeros((4, 16, 7, 7), np.float32)
    for k, t in enumerate(goal_templates):
        for y in range(t.shape[0]):
            for x in range(t.shape[1]):
                gw[k, int(t[y, x]), 3 + y - 2, 3 + x - 2] = 1.0
    gd = relu(conv("x0", gw, np.full(4, -29.5, np.float32)))
    gd = conv(gd, np.eye(4, dtype=np.float32)[:, :, None, None] * 2.0)

    paint = np.zeros((1, 4, 7, 7), np.float32)
    for k, t in enumerate(goal_templates):
        for y in range(t.shape[0]):
            for x in range(t.shape[1]):
                paint[0, k, 3 + 2 - y, 3 + 2 - x] += 1.0
    goal_pixels = clip(conv(gd, paint))

    # State coordinates denote player top-left plus (1, 1).
    def footprint_sum(image, size):
        w = np.zeros((1, 1, 7, 7), np.float32)
        for y in range(size):
            for x in range(size):
                w[0, 0, 3 + y - 1, 3 + x - 1] = 1.0
        return conv(image, w)

    goal2 = clip(footprint_sum(goal_pixels, 2))
    goal4 = clip(footprint_sum(goal_pixels, 4))

    # Player templates: up, down, left, right for each size.
    pw = np.zeros((8, 16, 7, 7), np.float32)
    pb = np.zeros(8, np.float32)
    for si, size in enumerate((2, 4)):
        for d in range(4):
            idx = 4 * si + d
            pb[idx] = -(size * size - 0.5)
            for y in range(size):
                for x in range(size):
                    if d == 0:
                        col = 9 if y < size // 2 else 10
                    elif d == 1:
                        col = 10 if y < size // 2 else 9
                    elif d == 2:
                        col = 9 if x < size // 2 else 10
                    else:
                        col = 10 if x < size // 2 else 9
                    pw[idx, col, 3 + y - 1, 3 + x - 1] = 1.0
    pd = relu(conv("x0", pw, pb))
    pd = conv(pd, np.eye(8, dtype=np.float32)[:, :, None, None] * 2.0)
    small_raw = sl(pd, 0, 4)
    big_dir = sl(pd, 4, 8)
    big_player = reduce_max(big_dir, 4)
    big_present = vclip(gmax(big_player))
    no_big = vaff(big_present, -1.0, 1.0)
    small_dir = mul(mul(small_raw, inv(goal2)), no_big)
    small_player = reduce_max(small_dir, 4)

    # Detect resize, fire, and teleport icons.
    iw = np.zeros((3, 16, 7, 7), np.float32)
    ib = np.array([-3.5, -2.5, -2.5], np.float32)
    for dy, dx in [(-3, 0), (0, -3), (0, 3), (3, 0)]:
        iw[0, 15, 3 + dy, 3 + dx] = 1.0
    for dy in (-3, 0, 3):
        iw[1, 6, 3 + dy, 3] = 1.0
    for dy in (-1, 2):
        iw[1, 6, 3 + dy, 3] = -1.0
    for dy, dx in [(0, 0), (0, 3), (3, 3)]:
        iw[2, 11, 3 + dy, 3 + dx] = 1.0
    iw[2, 11, 6, 3] = -1.0
    icons = relu(conv("x0", iw, ib))
    icons = conv(icons, np.eye(3, dtype=np.float32)[:, :, None, None] * 2.0)
    icon_values = gmax(icons)
    can_resize = vclip(vsl(icon_values, 0, 1))
    can_fire = vclip(vsl(icon_values, 1, 2))
    can_teleport = vclip(vsl(icon_values, 2, 3))

    # Detect and repaint the 4x4 fire targets.
    tw = np.zeros((2, 16, 7, 7), np.float32)
    for kind, inside in enumerate((13, 12)):
        for y in range(4):
            for x in range(4):
                col = 6 if y in (0, 3) or x in (0, 3) else inside
                tw[kind, col, 3 + y - 1, 3 + x - 1] = 1.0
    target_detect = relu(conv("x0", tw, [-15.5, -15.5]))
    target_detect = conv(
        target_detect, np.eye(2, dtype=np.float32)[:, :, None, None] * 2.0)
    target_anchor = reduce_max(target_detect, 2)
    target_paint = np.zeros((1, 1, 7, 7), np.float32)
    for y in range(4):
        for x in range(4):
            target_paint[0, 0, 3 + 1 - y, 3 + 1 - x] = 1.0
    target_pixels = clip(conv(target_anchor, target_paint))

    # Directional fire rays stop at visible walls and doors.
    ray_block = add(add(color(5), color(6)), add(color(12), color(13)))
    ray_free = inv(clip(ray_block))
    ray_source = conv(target_pixels, np.ones((4, 1, 1, 1), np.float32))
    ray_w = np.zeros((4, 4, 3, 3), np.float32)
    for d, (dy, dx) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        ray_w[d, d, 1 + dy, 1 + dx] = 1.0
    ray_w_name = param(ray_w)

    old_nodes = nodes
    ray_body = []
    nodes = ray_body
    rstep = conv_named("ray_state", ray_w_name)
    rstep = mul(rstep, ray_free)
    rnext = mx(ray_source, rstep)
    nodes = old_nodes
    graph.append({
        "id": "ray_loop",
        "op": "loop",
        "T": 64,
        "state": [{"id": "ray_state", "init": ray_source}],
        "body": ray_body,
        "next": {"ray_state": rnext},
    })
    rays = "ray_loop.ray_state"

    def ray_at_player(size):
        w = np.zeros((4, 4, 7, 7), np.float32)
        offsets = [
            (-1, -1), (size - 2, -1),
            (-1, -1), (-1, size - 2),
        ]
        for d, (dy, dx) in enumerate(offsets):
            w[d, d, 3 + dy, 3 + dx] = 1.0
        return conv(rays, w)

    shot_small = mul(ray_at_player(2), can_fire)
    shot_big = mul(ray_at_player(4), can_fire)

    # Detect small and large teleport destinations.
    mw = np.zeros((2, 16, 7, 7), np.float32)
    for dy, dx in [(-2, -2), (-2, 1), (1, -2), (1, 1)]:
        mw[0, 11, 3 + dy, 3 + dx] = 1.0
    for dy in (-1, 0):
        for dx in (-1, 0):
            mw[0, 15, 3 + dy, 3 + dx] = 1.0
    for dy, dx in [(-1, -1), (-1, 2), (2, -1), (2, 2)]:
        mw[1, 11, 3 + dy, 3 + dx] = 1.0
    for dy in (0, 1):
        for dx in (0, 1):
            mw[1, 15, 3 + dy, 3 + dx] = -1.0
    markers = relu(conv("x0", mw, [-7.5, -3.5]))
    markers = conv(
        markers, np.eye(2, dtype=np.float32)[:, :, None, None] * 2.0)
    small_marker = sl(markers, 0, 1)
    big_marker = sl(markers, 1, 2)

    # Visible approximation to collision geometry.
    obstacle = goal_pixels
    for c in (4, 5, 6, 12, 13):
        obstacle = add(obstacle, color(c))
    obstacle = clip(obstacle)

    def passable(size):
        inp = cat(obstacle, ones)
        w = np.zeros((1, 2, 7, 7), np.float32)
        for y in range(size):
            for x in range(size):
                w[0, 0, 3 + y - 1, 3 + x - 1] = -1.0
                w[0, 1, 3 + y - 1, 3 + x - 1] = 1.0
        p = relu(conv(inp, w, [-(size * size - 0.5)]))
        return clip(scale(p, 2.0))

    pass_small = passable(2)
    pass_big = passable(4)

    # Shifts for two- and four-pixel directional actions.
    sw = np.zeros((4, 1, 5, 5), np.float32)
    sw2 = np.zeros((4, 4, 5, 5), np.float32)
    for d, (dy, dx) in enumerate(((-2, 0), (2, 0), (0, -2), (0, 2))):
        sw[d, 0, 2 + dy, 2 + dx] = 1.0
        sw2[d, d, 2 + dy, 2 + dx] = 1.0
    sw_name = param(sw)
    sw2_name = param(sw2)

    def shift2(x):
        return conv_named(x, sw_name)

    def shift4_from_shift2(x):
        return conv_named(x, sw2_name)

    small_step_ok = shift2(pass_small)
    big_two_ok = shift2(pass_big)
    big_four_ok = shift4_from_shift2(big_two_ok)
    big_fallback_ok = mul(inv(big_four_ok), big_two_ok)
    small_goal = shift2(goal2)
    big_goal = mul(inv(big_four_ok), shift2(goal4))

    # Expansion candidates in the simulator's offset order.
    offsets = [(dx, dy) for dx in range(-2, 1) for dy in range(-2, 1)]
    ew = np.zeros((9, 1, 5, 5), np.float32)
    for i, (dx, dy) in enumerate(offsets):
        ew[i, 0, 2 + dy, 2 + dx] = 1.0
    ew_name = param(ew)
    expansion_candidates = conv_named(pass_big, ew_name)
    remaining = ones
    expansion_gates = []
    for i in range(9):
        possible = sl(expansion_candidates, i, i + 1)
        expansion_gates.append(mul(remaining, possible))
        remaining = mul(remaining, inv(possible))
    expansion_gates = cat(*expansion_gates)
    collapse9_name = param(np.ones((1, 9, 1, 1), np.float32))

    def teleport_value(value_field, marker):
        v = gmax(mul(value_field, marker))
        v = node("vmul", [v, can_teleport])
        return vaff(v, 0.90)

    def expansion_value(big_field):
        candidates = conv_named(big_field, ew_name)
        candidates = mul(candidates, expansion_gates)
        z = conv_named(candidates, collapse9_name)
        return mul(scale(z, 0.90), can_resize)

    def shrink_value(small_field):
        return mul(scale(small_field, 0.90), can_resize)

    def move_values(small_field, big_field):
        small_shift = shift2(small_field)
        small_move = mul(small_shift, small_step_ok)
        small_move = scale(small_move, 0.97, channels=4)

        big_shift2 = shift2(big_field)
        big_shift4 = shift4_from_shift2(big_shift2)
        big_move = mx(
            mul(big_shift4, big_four_ok),
            mul(big_shift2, big_fallback_ok))
        big_move = scale(big_move, 0.97, channels=4)
        return small_move, big_move

    small_static = mx(
        reduce_max(small_goal, 4),
        scale(reduce_max(shot_small, 4), 0.72))
    big_static = mx(
        reduce_max(big_goal, 4),
        scale(reduce_max(shot_big, 4), 0.72))

    old_nodes = nodes
    value_body = []
    nodes = value_body
    smove, bmove = move_values("small_state", "big_state")
    snext = mx(small_static, reduce_max(smove, 4))
    bnext = mx(big_static, reduce_max(bmove, 4))
    snext = mx(snext, expansion_value("big_state"))
    bnext = mx(bnext, shrink_value("small_state"))
    snext = mx(snext, teleport_value("small_state", small_marker))
    bnext = mx(bnext, teleport_value("big_state", big_marker))
    nodes = old_nodes
    graph.append({
        "id": "value_loop",
        "op": "loop",
        "T": 96,
        "state": [
            {"id": "small_state", "zeros": [1, 64, 64]},
            {"id": "big_state", "zeros": [1, 64, 64]},
        ],
        "body": value_body,
        "next": {"small_state": snext, "big_state": bnext},
    })
    small_value = "value_loop.small_state"
    big_value = "value_loop.big_state"

    # Remember whether a direction action has established internal facing.
    moved_previous = dense("aprev", [[0, 1, 1, 1, 1, 0, 0, 0]])
    moved = vclip(node("vadd", ["mem", moved_previous]))
    graph.append({
        "id": "mem_next",
        "op": "vadd",
        "in": [moved, vaff(onev, 0.0)],
    })

    smove, bmove = move_values(small_value, big_value)
    smove = mx(smove, small_goal)
    bmove = mx(bmove, big_goal)

    sfacing = mul(small_dir, moved)
    bfacing = mul(big_dir, moved)
    sorient = scale(mul(shot_small, inv(sfacing)), 0.72, channels=4)
    borient = scale(mul(shot_big, inv(bfacing)), 0.72, channels=4)
    smove = mx(smove, sorient)
    bmove = mx(bmove, borient)
    move_scores = gmax(
        mx(mul(smove, small_player), mul(bmove, big_player)))

    fire_here = mx(mul(shot_small, sfacing), mul(shot_big, bfacing))
    fire_score = vaff(gmax(reduce_max(fire_here, 4)), 0.72)

    resize_here = mx(
        mul(expansion_value(big_value), small_player),
        mul(shrink_value(small_value), big_player))
    resize_score = gmax(resize_here)

    stp = teleport_value(small_value, small_marker)
    btp = teleport_value(big_value, big_marker)
    teleport_here = mx(mul(small_player, stp), mul(big_player, btp))
    teleport_score = gmax(teleport_here)

    def biased_score(score, epsilon):
        present = vclip(vaff(score, 1000.0))
        return node("vadd", [score, vaff(present, epsilon)])

    rs = biased_score(resize_score, 0.00003)
    ts = biased_score(teleport_score, 0.00002)
    fs = biased_score(fire_score, 0.00001)
    spell_score = vmax(rs, vmax(ts, fs))

    def winner(a, b, c):
        difference = vsub(a, vmax(b, c))
        return vclip(vaff(difference, 100000.0, 0.5))

    choose_resize = winner(rs, ts, fs)
    choose_teleport = winner(ts, rs, fs)
    choose_fire = winner(fs, rs, ts)

    # One-pixel masks at the centres of the nine spell-panel cells.
    coord_w = np.zeros((18, 3, 1, 1), np.float32)
    coord_b = np.zeros(18, np.float32)
    for r in range(3):
        for c in range(3):
            i = 3 * r + c
            coord_w[2 * i, 0, 0, 0] = 63.0
            coord_b[2 * i] = -(50 + 5 * r)
            coord_w[2 * i + 1, 1, 0, 0] = 63.0
            coord_b[2 * i + 1] = -(25 + 5 * c)
    distances = node("abs", [conv("coords", coord_w, coord_b)])
    slot_w = np.zeros((9, 18, 1, 1), np.float32)
    for i in range(9):
        slot_w[i, 2 * i, 0, 0] = -2.0
        slot_w[i, 2 * i + 1, 0, 0] = -2.0
    slots = relu(conv(distances, slot_w, np.ones(9, np.float32)))

    selected = gmax(mul(slots, color(14)))
    selected = node("vrelu", [
        dense(selected, np.eye(9, dtype=np.float32) * 2.0,
              -np.ones(9, np.float32))
    ])

    # Columns are resize, teleport, and fire patterns.
    patterns = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 0],
        [0, 0, 0],
        [1, 0, 1],
        [0, 0, 0],
    ], dtype=np.float32)
    desired = dense(vcat(choose_resize, choose_teleport, choose_fire), patterns)
    not_desired = dense(
        desired, -np.eye(9, dtype=np.float32), np.ones(9, np.float32))
    not_selected = dense(
        selected, -np.eye(9, dtype=np.float32), np.ones(9, np.float32))
    wrong = node("vmul", [selected, not_desired])
    missing = node("vmul", [desired, not_selected])
    priorities = dense(
        vcat(wrong, missing),
        np.concatenate([
            2.0 * np.eye(9, dtype=np.float32),
            np.eye(9, dtype=np.float32),
        ], axis=1))
    click_channels = mul(slots, priorities)
    graph.append({
        "id": "click",
        "op": "conv",
        "in": [click_channels],
        "w": param(np.ones((1, 9, 1, 1), np.float32)),
    })

    head = vcat(move_scores, spell_score)
    hw = np.zeros((8, 5), np.float32)
    for d in range(4):
        hw[1 + d, d] = 1.0
    hw[6, 4] = 1.0
    hb = np.array([-5, 0, 0, 0, 0, -5, 0, -5], np.float32)
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [head],
        "w": param(hw),
        "b": param(hb),
    })

    return {"graph": graph, "memory_size": 1, "params": params}
