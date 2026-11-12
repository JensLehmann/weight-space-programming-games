import math
import numpy as np


def build():
    N, E = 6, 8
    Q = N * E
    graph, params = [], {}
    parameter_cache, constants, expression_caches = {}, {}, {}
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "aprev": (8,),
        "avail": (8,),
    }
    destination = graph
    serial = 0

    def parameter(value):
        a = np.asarray(value, np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def emit(op, inputs, name=None, **fields):
        nonlocal serial
        inputs = list(inputs)
        if op in ("add", "mul", "min2", "max2", "vadd", "vmul"):
            inputs = sorted(inputs)
        key = (op, tuple(inputs), tuple(sorted(fields.items())))
        local_cache = expression_caches.setdefault(id(destination), {})
        top_cache = expression_caches.setdefault(id(graph), {})
        if name is None:
            if key in top_cache:
                return top_cache[key]
            if key in local_cache:
                return local_cache[key]
            serial += 1
            name = "n%d" % serial

        nd = {"id": name, "op": op, "in": inputs}
        nd.update(fields)
        destination.append(nd)
        local_cache[key] = name

        ss = [shapes[x] for x in inputs]
        if op == "dense":
            shape = (params[fields["w"]].shape[0],)
        elif op == "conv":
            shape = (params[fields["w"]].shape[0], 64, 64)
        elif op in ("gmax", "gmean"):
            shape = (ss[0][0],)
        elif op in ("concat", "vconcat"):
            shape = (sum(s[0] for s in ss),) + ss[0][1:]
        elif op in ("slice", "vslice"):
            shape = (fields["stop"] - fields["start"],) + ss[0][1:]
        elif op in ("add", "sub", "mul", "min2", "max2"):
            shape = (max(s[0] for s in ss), 64, 64)
        elif op in ("vadd", "vmul"):
            shape = (max(s[0] for s in ss),)
        else:
            shape = ss[0]
        shapes[name] = shape
        return name

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("dense", [x], name, **fields)

    def conv(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("conv", [x], name, **fields)

    def pointwise(x, w, b=None):
        w = np.asarray(w, np.float32)
        return conv(x, w[:, :, None, None], b)

    def constant(value):
        nonlocal destination
        a = np.asarray(value, np.float32).reshape(-1)
        key = (a.shape, a.tobytes())
        if key not in constants:
            saved = destination
            destination = graph
            constants[key] = dense(
                "aprev", np.zeros((len(a), 8), np.float32), a
            )
            destination = saved
        return constants[key]

    zero, one = constant(0), constant(1)

    def vs(x, a, b):
        return emit("vslice", [x], start=a, stop=b)

    def ss(x, a, b):
        return emit("slice", [x], start=a, stop=b)

    def cat(*xs):
        return emit("vconcat", xs)

    def add(a, b):
        return emit("vadd", [a, b])

    def mul(a, b):
        return emit("vmul", [a, b])

    def affine(x, gain=1, bias=0):
        n = shapes[x][0]
        return dense(
            x, gain * np.eye(n, dtype=np.float32),
            np.full(n, bias, np.float32) if bias != 0 else None
        )

    def neg(x):
        return affine(x, -1)

    def sub(a, b):
        return add(a, neg(b))

    def scale(x, k):
        return mul(x, constant(k))

    def relu(x):
        return emit("vrelu", [x])

    def absolute(x):
        n = shapes[x][0]
        identity = np.eye(n, dtype=np.float32)
        split = dense(x, np.concatenate((identity, -identity), axis=0))
        positive_parts = relu(split)
        return dense(
            positive_parts, np.concatenate((identity, identity), axis=1)
        )

    def clip(x):
        n = shapes[x][0]
        identity = np.eye(n, dtype=np.float32)
        split = dense(
            x,
            np.concatenate((identity, identity), axis=0),
            np.concatenate((
                np.zeros(n, np.float32), -np.ones(n, np.float32)
            ))
        )
        positive_parts = relu(split)
        return dense(
            positive_parts, np.concatenate((identity, -identity), axis=1)
        )

    def equal(a, b):
        return clip(affine(absolute(sub(a, b)), -2, 1.5))

    def positive(x):
        return clip(affine(x, 4, -2))

    def inverse(x):
        return affine(x, -1, 1)

    def total(x):
        return dense(x, np.ones((1, shapes[x][0]), np.float32))

    def blend(old, new, gate):
        return add(mul(old, inverse(gate)), mul(new, gate))

    def sa(a, b):
        return emit("add", [a, b])

    def sd(a, b):
        return emit("sub", [a, b])

    def sm(a, b):
        return emit("mul", [a, b])

    def sr(x):
        return emit("relu", [x])

    def sab(x):
        return emit("abs", [x])

    def sc(x):
        return emit("min2", [sr(x), one])

    def maximum(x):
        return emit("gmax", [x])

    def mean(x):
        return emit("gmean", [x])

    def spatial_zero_test(x):
        return sc(sm(sd(constant(0.75), sab(x)), constant(2)))

    def cycle(n):
        w = np.zeros((n, n), np.float32)
        for i in range(n):
            w[(i + 1) % n, i] = 1
        return w

    repeat_w = np.zeros((Q, N), np.float32)
    endpoint_repeat_w = np.zeros((Q, E), np.float32)
    endpoint_reduce_w = np.zeros((E, Q), np.float32)
    for i in range(Q):
        repeat_w[i, i // E] = 1
        endpoint_repeat_w[i, i % E] = 1
        endpoint_reduce_w[i % E, i] = 1

    def repeat_piece(x):
        return dense(x, repeat_w)

    def repeat_endpoint(x):
        return dense(x, endpoint_repeat_w)

    def select_scalar(x, selector):
        return total(mul(x, selector))

    def select_endpoints(x, selector):
        return dense(mul(x, repeat_piece(selector)), endpoint_reduce_w)

    def index_mask(index, n):
        differences = dense(
            index, np.ones((n, 1), np.float32),
            -np.arange(n, dtype=np.float32)
        )
        return equal(differences, zero)

    def rotation_basis(r):
        r0 = relu(affine(r, -1, 1))
        r1 = relu(affine(absolute(affine(r, 1, -1)), -1, 1))
        r2 = relu(affine(absolute(affine(r, 1, -2)), -1, 1))
        r3 = relu(affine(r, 1, -2))
        return r0, r1, r2, r3

    def rotate_xy(x, y, w, h, r):
        r0, r1, r2, r3 = rotation_basis(r)
        u = sub(affine(h, 1, -1), y)
        v = sub(affine(w, 1, -1), x)
        xx = add(
            add(mul(r0, x), mul(r1, u)),
            add(mul(r2, v), mul(r3, y))
        )
        yy = add(
            add(mul(r0, y), mul(r1, x)),
            add(mul(r2, u), mul(r3, v))
        )
        return xx, yy

    specification = [
        ("w", N), ("h", N), ("ox", N), ("oy", N),
        ("cx", N), ("cy", N), ("locked", N),
        ("ex", Q), ("ey", Q), ("ev", Q),
        ("px", N), ("py", N), ("pr", N), ("ar", N),
        ("seed", 16), ("full", N),
        ("count", 1), ("phase", 1), ("sel", 1),
        ("piece", N), ("retry", 1),
    ]
    memory_size = sum(n for _, n in specification)
    shapes["mem"] = (memory_size,)
    memory = {}
    offset = 0
    for key, length in specification:
        memory[key] = vs("mem", offset, offset + length)
        offset += length

    inspecting = equal(memory["phase"], zero)
    planning = equal(memory["phase"], one)
    executing = equal(memory["phase"], constant(2))

    # Perception: one display pixel per logical board cell.
    comb_w = np.zeros((40, 3), np.float32)
    comb_b = np.zeros(40, np.float32)
    for i in range(20):
        comb_w[i, 0] = 63
        comb_w[i + 20, 1] = 63
        comb_b[i] = comb_b[i + 20] = -(3 + 3 * i)

    distances = pointwise("coords", comb_w, comb_b)
    tents = sc(sm(sd(constant(0.75), sab(distances)), constant(2)))
    comb_reduce = np.zeros((2, 40), np.float32)
    comb_reduce[0, :20] = 1
    comb_reduce[1, 20:] = 1
    comb = pointwise(tents, comb_reduce)
    grid = sm(ss(comb, 0, 1), ss(comb, 1, 2))

    xy_w = np.zeros((2, 3), np.float32)
    xy_w[0, 1] = 21
    xy_w[1, 0] = 21
    xy = pointwise("coords", xy_w, [-1, -1])
    gx, gy = ss(xy, 0, 1), ss(xy, 1, 2)
    rank = pointwise(xy, [[-1, -20]], [400])

    corner = sr(pointwise("coords", [[-63, -63, 0]], [1]))
    background_colour = maximum(sm("x0", corner))
    background = pointwise(
        sm("x0", background_colour), np.ones((1, 16), np.float32)
    )
    foreground = sd(one, background)
    foreground_cells = sm(foreground, grid)

    black = sm(ss("x0", 0, 1), foreground_cells)
    grey = sm(ss("x0", 4, 5), foreground_cells)
    grey_present = maximum(grey)
    black_present = maximum(black)

    body_w = np.ones((1, 16), np.float32)
    body_w[0, [0, 3, 4, 8]] = 0
    coloured = sm(pointwise("x0", body_w), foreground_cells)
    selected_body = sa(
        sm(black, inverse(grey_present)),
        sm(coloured, grey_present)
    )
    selected_valid = maximum(selected_body)

    endpoints = sm(
        sa(ss("x0", 8, 9), ss("x0", 3, 4)),
        foreground_cells
    )
    neighbours = sc(conv(
        selected_body, np.ones((1, 1, 7, 7), np.float32)
    ))
    selected_ports = sm(endpoints, neighbours)
    selected_shape = sc(sa(
        sa(selected_body, selected_ports),
        sm(black, grey_present)
    ))

    maxx = maximum(sm(selected_shape, gx))
    maxy = maximum(sm(selected_shape, gy))
    minx = sub(
        constant(19), maximum(sm(selected_shape, sd(constant(19), gx)))
    )
    miny = sub(
        constant(19), maximum(sm(selected_shape, sd(constant(19), gy)))
    )
    width = affine(sub(maxx, minx), 1, 1)
    height = affine(sub(maxy, miny), 1, 1)

    def pick(mask):
        score = sm(mask, rank)
        best = maximum(score)
        winner = sm(
            mask, sc(sa(constant(2), sm(sd(score, best), constant(1000))))
        )
        return (
            winner, maximum(sm(winner, gx)),
            maximum(sm(winner, gy)), maximum(mask)
        )

    _, body_x, body_y, _ = pick(selected_body)
    click_local_x = sub(body_x, minx)
    click_local_y = sub(body_y, miny)

    # Extract a row-major list of the selected piece's endpoints.
    extraction_states = [
        {"id": "extract_remaining", "init": selected_ports},
        {"id": "extract_x", "zeros": [E]},
        {"id": "extract_y", "zeros": [E]},
        {"id": "extract_valid", "zeros": [E]},
        {"id": "extract_index", "init": constant([1] + [0] * (E - 1))},
    ]
    for state in extraction_states:
        shapes[state["id"]] = (
            shapes[state["init"]] if "init" in state
            else tuple(state["zeros"])
        )

    extraction_body = []
    destination = extraction_body
    winner, port_x, port_y, port_exists = pick("extract_remaining")
    new_ex = add(
        "extract_x",
        mul("extract_index", mul(sub(port_x, minx), port_exists))
    )
    new_ey = add(
        "extract_y",
        mul("extract_index", mul(sub(port_y, miny), port_exists))
    )
    new_ev = add("extract_valid", mul("extract_index", port_exists))
    new_remaining = sr(sd("extract_remaining", winner))
    new_index = dense("extract_index", cycle(E))
    destination = graph

    graph.append({
        "id": "Extract", "op": "loop", "T": E,
        "state": extraction_states, "body": extraction_body,
        "next": {
            "extract_remaining": new_remaining,
            "extract_x": new_ex, "extract_y": new_ey,
            "extract_valid": new_ev, "extract_index": new_index,
        },
    })
    for state in extraction_states:
        shapes["Extract." + state["id"]] = shapes[state["id"]]

    need_variant = mul(
        selected_valid, mul(grey_present, black_present)
    )
    slot = index_mask(memory["count"], N)
    capture = mul(
        inspecting,
        mul(total(slot), mul(selected_valid, inverse(need_variant)))
    )
    write_slot = mul(slot, capture)
    data = dict(memory)

    observations = {
        "w": width, "h": height, "ox": minx, "oy": miny,
        "cx": click_local_x, "cy": click_local_y,
        "px": minx, "py": miny, "pr": zero, "ar": zero,
    }
    for key, observation in observations.items():
        data[key] = blend(memory[key], observation, write_slot)

    endpoint_write = repeat_piece(write_slot)
    for key, observation in [
        ("ex", "Extract.extract_x"),
        ("ey", "Extract.extract_y"),
        ("ev", "Extract.extract_valid"),
    ]:
        data[key] = blend(
            memory[key], repeat_endpoint(observation), endpoint_write
        )

    large_piece = positive(affine(total("Extract.extract_valid"), 1, -3))
    stack_seen = clip(add(need_variant, mul(grey_present, large_piece)))
    lock_gate = mul(inspecting, mul(slot, clip(add(capture, need_variant))))
    lock_here = select_scalar(memory["locked"], slot)
    data["locked"] = blend(
        memory["locked"], clip(add(lock_here, stack_seen)), lock_gate
    )

    active = positive(data["w"])
    number_active = total(active)
    selected_slot = index_mask(memory["sel"], N)
    synchronize = mul(executing, mul(selected_valid, selected_slot))
    data["ox"] = blend(data["ox"], minx, synchronize)
    data["oy"] = blend(data["oy"], miny, synchronize)

    # Find an uninspected piece outside the stored bounding rectangles.
    left = sa(sd(gx, data["ox"]), one)
    right = sd(add(data["ox"], data["w"]), gx)
    top = sa(sd(gy, data["oy"]), one)
    bottom = sd(add(data["oy"], data["h"]), gy)
    rectangles = emit("min2", [left, right])
    rectangles = emit("min2", [rectangles, top])
    rectangles = emit("min2", [rectangles, bottom])
    rectangles = sm(sc(rectangles), active)
    occupied = sc(pointwise(rectangles, np.ones((1, N), np.float32)))
    uninspected = sm(foreground_cells, sd(one, occupied))
    another_piece = maximum(uninspected)

    inspection_click = sa(
        sm(sm(uninspected, rank), selected_valid),
        sm(sm(foreground_cells, rank), inverse(selected_valid))
    )
    finish_inspection = mul(capture, inverse(another_piece))

    # Planner: spatial positions enumerate translations and channels
    # enumerate four rotations.
    x4 = pointwise(xy, np.tile([[1, 0]], (4, 1)))
    y4 = pointwise(xy, np.tile([[0, 1]], (4, 1)))
    xbig = pointwise(xy, np.tile([[1, 0]], (4 * Q, 1)))
    ybig = pointwise(xy, np.tile([[0, 1]], (4 * Q, 1)))

    target_repeat = np.tile(np.eye(Q, dtype=np.float32), (4, 1))
    offset_repeat = np.zeros((4 * Q, 4), np.float32)
    match_reduce = np.zeros((4, 4 * Q), np.float32)
    for r in range(4):
        offset_repeat[r * Q:(r + 1) * Q, r] = 1
        match_reduce[r, r * Q:(r + 1) * Q] = 1

    seed_initial = constant([1] + [0] * 15)
    piece_initial = constant([1] + [0] * (N - 1))
    initial_seed = blend(
        memory["seed"], seed_initial, equal(total(memory["seed"]), zero)
    )
    initial_piece = blend(
        memory["piece"], piece_initial, equal(total(memory["piece"]), zero)
    )
    initial_found = mul(
        positive(number_active),
        equal(total(mul(memory["full"], active)), number_active)
    )

    planner_states = [
        {"id": "plan_x", "init": data["px"]},
        {"id": "plan_y", "init": data["py"]},
        {"id": "plan_r", "init": data["pr"]},
        {"id": "plan_seed", "init": initial_seed},
        {"id": "plan_full", "init": memory["full"]},
        {"id": "plan_piece", "init": initial_piece},
        {"id": "plan_endpoint", "init": constant([1] + [0] * (E - 1))},
        {"id": "plan_score", "zeros": [4, 64, 64]},
        {"id": "plan_found", "init": initial_found},
        {"id": "plan_retry", "init": memory["retry"]},
    ]
    for state in planner_states:
        shapes[state["id"]] = (
            shapes[state["init"]] if "init" in state
            else tuple(state["zeros"])
        )

    planner_body = []
    destination = planner_body
    piece = "plan_piece"

    local_x = select_endpoints(data["ex"], piece)
    local_y = select_endpoints(data["ey"], piece)
    local_valid = select_endpoints(data["ev"], piece)
    sw = select_scalar(data["w"], piece)
    sh = select_scalar(data["h"], piece)
    locked = select_scalar(data["locked"], piece)
    oldx = select_scalar("plan_x", piece)
    oldy = select_scalar("plan_y", piece)
    oldr = select_scalar("plan_r", piece)
    port_count = total(local_valid)
    piece_active = select_scalar(active, piece)

    worldx, worldy = rotate_xy(
        data["ex"], data["ey"],
        repeat_piece(data["w"]), repeat_piece(data["h"]),
        repeat_piece("plan_r")
    )
    worldx = add(worldx, repeat_piece("plan_x"))
    worldy = add(worldy, repeat_piece("plan_y"))
    other_valid = mul(data["ev"], inverse(repeat_piece(piece)))
    other_valid4 = dense(other_valid, target_repeat)

    sx = total(mul(local_x, "plan_endpoint"))
    sy = total(mul(local_y, "plan_endpoint"))
    valid_port = total(mul(local_valid, "plan_endpoint"))
    u = sub(affine(sh, 1, -1), sy)
    v = sub(affine(sw, 1, -1), sx)
    offsets_x = cat(sx, u, v, sy)
    offsets_y = cat(sy, sx, u, v)

    differences_x = add(
        dense(offsets_x, offset_repeat),
        dense(neg(worldx), target_repeat)
    )
    differences_y = add(
        dense(offsets_y, offset_repeat),
        dense(neg(worldy), target_repeat)
    )
    distance = sa(
        sab(sa(xbig, differences_x)),
        sab(sa(ybig, differences_y))
    )
    matches = sm(
        sc(sm(sd(constant(0.75), distance), constant(2))),
        other_valid4
    )
    occupancy = pointwise(matches, match_reduce)
    port_score = sd(
        occupancy, sm(sr(sd(occupancy, one)), constant(3))
    )
    accumulated = sa("plan_score", sm(port_score, valid_port))

    widths = cat(sw, sh, sw, sh)
    heights = cat(sh, sw, sh, sw)
    bounds = emit("min2", [sa(x4, one), sa(y4, one)])
    bounds = emit("min2", [
        bounds, sa(sd(sub(constant(20), widths), x4), one)
    ])
    bounds = emit("min2", [
        bounds, sa(sd(sub(constant(20), heights), y4), one)
    ])
    rot_allowed = cat(one, inverse(locked), inverse(locked), inverse(locked))
    allowed = sm(sm(sc(bounds), grid), rot_allowed)
    valid_score = sd(
        accumulated, sm(sd(one, allowed), constant(1000))
    )

    old_rotation = cat(*rotation_basis(oldr))
    old_location = sm(
        spatial_zero_test(sd(x4, oldx)),
        spatial_zero_test(sd(y4, oldy))
    )
    old_location = sm(sm(old_location, old_rotation), grid)
    old_scores = maximum(sd(
        valid_score, sm(sd(one, old_location), constant(2000))
    ))
    old_matches = total(mul(old_scores, old_rotation))
    old_complete = equal(old_matches, port_count)

    seed_w = np.zeros((6, 16), np.float32)
    for j in range(16):
        seed_w[0, j] = (7 * j + 3) % 20
        seed_w[1, j] = (11 * j + 9) % 20
        for r in range(4):
            seed_w[r + 2, j] = 2.8 * math.sin((j + 1) * (r + 2))
    tie_values = dense("plan_seed", seed_w)
    refx, refy = vs(tie_values, 0, 1), vs(tie_values, 1, 2)
    rotation_tie = vs(tie_values, 2, 6)
    tie_distance = sa(sab(sd(gx, refx)), sab(sd(gy, refy)))
    tie = sa(
        sd(rotation_tie, tie_distance),
        sm(rank, constant(0.001))
    )
    preference = sa(sm(valid_score, constant(100)), tie)

    shake = clip(add("plan_retry", vs("plan_seed", 0, 1)))
    preserve = mul(old_complete, inverse(shake))
    preference = sa(
        preference, sm(old_location, scale(preserve, 50))
    )
    preference = sd(
        preference, sm(old_location, scale(shake, 220))
    )

    best_map = ss(preference, 0, 1)
    for r in range(1, 4):
        best_map = emit("max2", [best_map, ss(preference, r, r + 1)])
    best_value = maximum(best_map)
    pose_winner = sc(sa(
        constant(2), sm(sd(preference, best_value), constant(10000))
    ))
    winner_rotation = maximum(pose_winner)
    newx = total(maximum(sm(pose_winner, x4)))
    newy = total(maximum(sm(pose_winner, y4)))
    newr = dense(winner_rotation, [[0, 1, 2, 3]])
    new_matches = total(maximum(sm(accumulated, pose_winner)))

    anchor = vs(piece, 0, 1)
    newx = blend(newx, oldx, anchor)
    newy = blend(newy, oldy, anchor)
    newr = blend(newr, oldr, anchor)
    new_matches = blend(new_matches, old_matches, anchor)

    complete = mul(piece_active, equal(new_matches, port_count))
    unchanged = mul(
        mul(equal(newx, oldx), equal(newy, oldy)),
        equal(newr, oldr)
    )
    last_endpoint = vs("plan_endpoint", E - 1, E)
    update = mul(last_endpoint, mul(planning, inverse("plan_found")))
    move_update = mul(update, mul(piece_active, inverse(anchor)))
    move_slot = mul(piece, move_update)

    next_px = blend("plan_x", newx, move_slot)
    next_py = blend("plan_y", newy, move_slot)
    next_pr = blend("plan_r", newr, move_slot)

    proposed_full = add(
        mul(mul("plan_full", unchanged), inverse(piece)),
        mul(piece, complete)
    )
    next_full = blend("plan_full", proposed_full, update)
    solved = mul(
        positive(number_active),
        equal(total(mul(next_full, active)), number_active)
    )
    next_found = clip(add("plan_found", mul(update, solved)))
    next_piece = blend(piece, dense(piece, cycle(N)), update)
    next_seed = blend(
        "plan_seed", dense("plan_seed", cycle(16)), update
    )
    retry_used = mul(update, mul(piece_active, inverse(anchor)))
    next_retry = mul("plan_retry", inverse(retry_used))
    next_endpoint = dense("plan_endpoint", cycle(E))
    next_score = sm(accumulated, inverse(last_endpoint))
    destination = graph

    graph.append({
        "id": "Plan", "op": "loop", "T": E,
        "state": planner_states, "body": planner_body,
        "next": {
            "plan_x": next_px, "plan_y": next_py, "plan_r": next_pr,
            "plan_seed": next_seed, "plan_full": next_full,
            "plan_piece": next_piece, "plan_endpoint": next_endpoint,
            "plan_score": next_score, "plan_found": next_found,
            "plan_retry": next_retry,
        },
    })
    for state in planner_states:
        shapes["Plan." + state["id"]] = shapes[state["id"]]

    for key, state in [
        ("px", "plan_x"), ("py", "plan_y"), ("pr", "plan_r"),
        ("seed", "plan_seed"), ("full", "plan_full"),
        ("piece", "plan_piece"), ("retry", "plan_retry"),
    ]:
        data[key] = "Plan." + state
    planned_solution = "Plan.plan_found"

    # Execution controller.
    reached = mul(
        mul(equal(data["ox"], data["px"]), equal(data["oy"], data["py"])),
        equal(data["ar"], data["pr"])
    )
    unfinished = mul(active, inverse(reached))
    all_reached = equal(total(unfinished), zero)
    upper = np.triu(np.ones((N, N), np.float32), 1)
    target = mul(unfinished, equal(dense(unfinished, upper), zero))
    target_index = dense(target, np.arange(N, dtype=np.float32)[None, :])
    target_exists = positive(total(target))
    same_selection = equal(target_index, memory["sel"])
    ready = mul(executing, mul(target_exists, same_selection))

    target_x = select_scalar(data["px"], target)
    target_y = select_scalar(data["py"], target)
    actual_x = select_scalar(data["ox"], target)
    actual_y = select_scalar(data["oy"], target)
    target_r = select_scalar(data["pr"], target)
    actual_r = select_scalar(data["ar"], target)
    rotate_needed = inverse(equal(target_r, actual_r))
    dx, dy = sub(target_x, actual_x), sub(target_y, actual_y)
    right, left = positive(dx), positive(neg(dx))
    down, up = positive(dy), positive(neg(dy))

    execute_rotate = mul(ready, rotate_needed)
    translate_ready = mul(ready, inverse(rotate_needed))
    execute_right = mul(translate_ready, right)
    execute_left = mul(translate_ready, left)
    vertical_ready = mul(translate_ready, inverse(clip(add(right, left))))
    execute_down = mul(vertical_ready, down)
    execute_up = mul(vertical_ready, up)
    rotate_action = clip(add(
        mul(inspecting, need_variant), execute_rotate
    ))

    simple_flags = [
        execute_up, execute_down, execute_left, execute_right, rotate_action
    ]
    click_action = inverse(total(cat(*simple_flags)))
    flags = cat(zero, *simple_flags, click_action, zero)
    bias = np.zeros(8, np.float32)
    bias[0] = bias[7] = -1000
    dense(flags, 10 * np.eye(8, dtype=np.float32), bias, "logits")

    local_click_x, local_click_y = rotate_xy(
        data["cx"], data["cy"], data["w"], data["h"], data["ar"]
    )
    world_click_x = select_scalar(add(data["ox"], local_click_x), target)
    world_click_y = select_scalar(add(data["oy"], local_click_y), target)
    click_distance = sa(
        sab(sd(gx, world_click_x)), sab(sd(gy, world_click_y))
    )
    execution_click = sd(sm(grid, constant(100)), click_distance)
    select_gate = mul(
        executing, mul(target_exists, inverse(same_selection))
    )
    click_map = sa(
        sm(inspection_click, inspecting),
        sm(execution_click, select_gate)
    )
    conv(click_map, np.ones((1, 1, 1, 1), np.float32), name="click")

    actual_rotation = add(data["ar"], mul(target, execute_rotate))
    data["ar"] = sub(
        actual_rotation,
        scale(relu(affine(actual_rotation, 1, -3)), 4)
    )

    retry_arrangement = mul(executing, all_reached)
    data["phase"] = sub(
        add(
            memory["phase"],
            add(finish_inspection, mul(planning, planned_solution))
        ),
        retry_arrangement
    )
    data["count"] = add(memory["count"], capture)
    next_inspection_selection = add(
        memory["count"], mul(capture, another_piece)
    )
    selection = blend(
        memory["sel"], next_inspection_selection,
        mul(inspecting, inverse(need_variant))
    )
    data["sel"] = blend(selection, target_index, select_gate)
    data["retry"] = blend(data["retry"], one, retry_arrangement)
    data["full"] = mul(data["full"], inverse(retry_arrangement))
    emit(
        "vconcat", [data[key] for key, _ in specification],
        name="mem_next"
    )

    # Dead-code elimination; no game states are evaluated here.
    def prune(nodes, wanted):
        needed = set(wanted)
        kept = []
        for nd in reversed(nodes):
            if nd["op"] == "loop":
                prefix = nd["id"] + "."
                if not any(x.startswith(prefix) for x in needed):
                    continue
                body, body_needed = prune(nd["body"], nd["next"].values())
                nd = dict(nd)
                nd["body"] = body
                local_ids = {s["id"] for s in nd["state"]}
                local_ids.update(b["id"] for b in body)
                needed.update(x for x in body_needed if x not in local_ids)
                needed.update(
                    s["init"] for s in nd["state"] if "init" in s
                )
                kept.append(nd)
            elif nd["id"] in needed:
                needed.update(nd["in"])
                kept.append(nd)
        return list(reversed(kept)), needed

    graph, _ = prune(graph, {"logits", "click", "mem_next"})

    def node_macs(nd):
        op = nd["op"]
        if op == "conv":
            return params[nd["w"]].size * 4096
        if op == "dense":
            return params[nd["w"]].size
        if op in ("add", "sub", "mul", "max2", "min2", "vadd", "vmul"):
            return int(np.prod(shapes[nd["id"]]))
        if op in ("rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"):
            return int(np.prod(shapes[nd["in"][0]]))
        return 0

    def costs(nodes):
        macs = evaluations = definitions = 0
        for nd in nodes:
            definitions += 1
            if nd["op"] == "loop":
                bm, be, bd = costs(nd["body"])
                macs += nd["T"] * bm
                evaluations += 1 + nd["T"] * be
                definitions += bd
            else:
                macs += node_macs(nd)
                evaluations += 1
        return macs, evaluations, definitions

    plan_node = next(nd for nd in graph if nd["id"] == "Plan")
    body_macs, body_evals, _ = costs(plan_node["body"])
    plan_node["T"] = 0
    base_macs, base_evals, _ = costs(graph)
    maximum_iterations = min(
        64,
        (1500000000 - base_macs) // max(1, body_macs),
        (10000 - base_evals) // max(1, body_evals),
    )
    plan_node["T"] = int(maximum_iterations // E * E)
    assert plan_node["T"] >= E, "Planning scan exceeds runtime budget"

    used = set()

    def collect(nodes):
        for nd in nodes:
            if nd["op"] == "loop":
                collect(nd["body"])
            else:
                used.update(nd[k] for k in ("w", "b") if k in nd)

    collect(graph)
    params = {k: v for k, v in params.items() if k in used}

    def spatial_channels(nodes):
        count = 0
        for nd in nodes:
            if nd["op"] == "loop":
                count += spatial_channels(nd["body"])
                for state in nd["state"]:
                    shape = shapes[state["id"]]
                    if len(shape) == 3:
                        count += shape[0]
            elif len(shapes[nd["id"]]) == 3:
                count += shapes[nd["id"]][0]
        return count

    def check_shapes(nodes):
        for nd in nodes:
            if nd["op"] == "loop":
                for state in nd["state"]:
                    shape = shapes[state["id"]]
                    assert shape[0] <= (512 if len(shape) == 3 else 4096)
                check_shapes(nd["body"])
            else:
                shape = shapes[nd["id"]]
                assert shape[0] <= (512 if len(shape) == 3 else 4096)

    macs, evaluations, definitions = costs(graph)
    parameter_count = sum(a.size for a in params.values())
    channels = 19 + spatial_channels(graph)
    assert memory_size <= 256, ("memory", memory_size)
    assert parameter_count <= 150000, ("parameters", parameter_count)
    assert macs <= 1500000000, ("MACs", macs)
    assert evaluations <= 10000, ("evaluations", evaluations)
    assert definitions <= 1000, ("nodes", definitions)
    assert channels <= 4096, ("spatial channels", channels)
    assert all(np.isfinite(a).all() for a in params.values())
    check_shapes(graph)

    return {
        "graph": graph,
        "memory_size": memory_size,
        "params": params,
    }
