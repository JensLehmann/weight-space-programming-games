import numpy as np


def build():
    P = {}
    graph = []
    sink = graph
    memory_size = 159
    shapes = {
        "x0": ("s", 16),
        "coords": ("s", 3),
        "aprev": ("v", 8),
        "avail": ("v", 8),
        "mem": ("v", memory_size),
    }
    serial = [0]

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def param(a):
        name = fresh("p")
        P[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, ins, name=None, **fields):
        name = fresh() if name is None else name
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        sink.append(nd)

        if op in ("conv", "dense"):
            shapes[name] = (
                "s" if op == "conv" else "v",
                P[fields["w"]].shape[0],
            )
        elif op in ("gmax", "gmean"):
            shapes[name] = ("v", shapes[ins[0]][1])
        elif op in ("concat", "vconcat"):
            shapes[name] = (
                shapes[ins[0]][0],
                sum(shapes[z][1] for z in ins),
            )
        elif op in ("slice", "vslice"):
            shapes[name] = (
                shapes[ins[0]][0],
                fields["stop"] - fields["start"],
            )
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shapes[name] = ("s", max(shapes[z][1] for z in ins))
        elif op in ("vadd", "vmul"):
            shapes[name] = ("v", max(shapes[z][1] for z in ins))
        else:
            shapes[name] = shapes[ins[0]]
        return name

    def linear(x, w, b=None, name=None):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 1:
            w = w[None, :]
        spatial = shapes[x][0] == "s"
        if spatial:
            w = w[:, :, None, None]
        fields = {"w": param(w)}
        if b is not None:
            b = np.asarray(b, dtype=np.float32)
            if b.ndim == 0:
                b = np.full(w.shape[0], b, dtype=np.float32)
            fields["b"] = param(b)
        return node("conv" if spatial else "dense", [x], name, **fields)

    def affine(x, a=1.0, b=0.0, name=None):
        c = shapes[x][1]
        return linear(x, np.eye(c, dtype=np.float32) * a, b, name)

    def const(values):
        a = np.asarray(values, dtype=np.float32).reshape(-1)
        return linear("aprev", np.zeros((len(a), 8), np.float32), a)

    def sl(x, start, stop):
        return node(
            "slice" if shapes[x][0] == "s" else "vslice",
            [x], start=start, stop=stop,
        )

    def cat(*xs):
        return node(
            "concat" if shapes[xs[0]][0] == "s" else "vconcat", xs
        )

    def add(a, b, name=None):
        op = "add" if "s" in (shapes[a][0], shapes[b][0]) else "vadd"
        return node(op, [a, b], name)

    def mul(a, b, name=None):
        op = "mul" if "s" in (shapes[a][0], shapes[b][0]) else "vmul"
        return node(op, [a, b], name)

    def sub(a, b):
        if "s" in (shapes[a][0], shapes[b][0]):
            return node("sub", [a, b])
        return add(a, affine(b, -1))

    def relu(x):
        return node("relu" if shapes[x][0] == "s" else "vrelu", [x])

    one = const([1.0])

    def clip(x):
        r = relu(x)
        if shapes[x][0] == "s":
            return node("min2", [r, one])
        return sub(r, relu(affine(x, 1, -1)))

    def inv(x):
        return affine(x, -1, 1)

    def maximum(a, b):
        return node("max2", [a, b])

    def minimum(a, b):
        return node("min2", [a, b])

    def gm(x):
        return node("gmax", [x])

    def total_channels(x):
        return linear(x, np.ones((1, shapes[x][1]), np.float32))

    def shift(x, dy, dx):
        c = shapes[x][1]
        w = np.zeros((c, c, 7, 7), np.float32)
        for k in range(c):
            w[k, k, 3 + dy, 3 + dx] = 1
        return node("conv", [x], w=param(w))

    def loop(T, initial, body_fn):
        nonlocal sink
        lid = fresh("loop")
        states = {}
        definitions = []
        for key, value in initial.items():
            sid = fresh("state")
            states[key] = sid
            if isinstance(value, str):
                definitions.append({"id": sid, "init": value})
                shapes[sid] = shapes[value]
            else:
                kind, count = value
                dims = [count, 64, 64] if kind == "s" else [count]
                definitions.append({"id": sid, "zeros": dims})
                shapes[sid] = (kind, count)

        outer = sink
        body = []
        sink = body
        nxt = body_fn(states)
        sink = outer
        sink.append({
            "id": lid,
            "op": "loop",
            "T": T,
            "state": definitions,
            "body": body,
            "next": {states[k]: nxt[k] for k in states},
        })
        result = {}
        for k, sid in states.items():
            result[k] = lid + "." + sid
            shapes[result[k]] = shapes[sid]
        return result

    # Memory layout:
    # 0: initialization flag; 1: click cursor; 2: stall counter;
    # 3:12: axis-offset cycle; 12:156: selected-piece template;
    # 156:158: template origin; 158: template-valid flag.
    old_done = sl("mem", 0, 1)
    old_cursor = sl("mem", 1, 2)
    old_stale = sl("mem", 2, 3)
    old_trials = sl("mem", 3, 12)
    old_template = sl("mem", 12, 156)
    old_origin = sl("mem", 156, 158)
    old_valid = sl("mem", 158, 159)
    previous_move = linear("aprev", [[0, 1, 1, 1, 1, 0, 0, 0]])
    previous_click = sl("aprev", 6, 7)

    # Coordinates and logical cell centres.
    pixel = linear("coords", [[63, 0, 0], [0, 63, 0]])
    py, px = sl(pixel, 0, 1), sl(pixel, 1, 2)
    index = linear(pixel, [[64, 1]])

    w = np.zeros((42, 3), np.float32)
    w[:21, 0] = 63
    w[21:, 1] = 63
    b = -np.tile(1 + 3 * np.arange(21, dtype=np.float32), 2)
    lattice_dist = node("abs", [linear("coords", w, b)])
    lattice_hats = clip(affine(lattice_dist, -3, 1.5))
    w = np.zeros((2, 42), np.float32)
    w[0, :21] = 1
    w[1, 21:] = 1
    lattice_axes = linear(lattice_hats, w)
    lattice = mul(sl(lattice_axes, 0, 1), sl(lattice_axes, 1, 2))

    # Read each cell's base colour and its interface marker separately.
    w = np.zeros((8, 16, 3, 3), np.float32)
    original_colours = [
        c for c in range(16) if c not in (4, 9, 10, 11)
    ]
    w[0, original_colours, 0, 0] = 1
    w[1, 10, 0, 0] = 1
    w[2, original_colours + [4], 0, 0] = 1
    w[3, 11, 1, 1] = 1
    w[4, 0, 1, 1] = 1
    w[5, [0, 9, 11], 1, 1] = 1
    w[6, 5, 1, 1] = 1
    w[7, 10, 1, 1] = 1
    sampled = mul(node("conv", ["x0"], w=param(w)), lattice)

    original = sl(sampled, 0, 1)
    axis_base = sl(sampled, 1, 2)
    covered = sl(sampled, 2, 3)
    target = sl(sampled, 3, 4)
    selected = sl(sampled, 4, 5)
    movable_marker = sl(sampled, 5, 6)
    fixed_marker = sl(sampled, 6, 7)
    fixed_axis_marker = sl(sampled, 7, 8)
    unselected_marker = mul(sl("x0", 9, 10), lattice)

    # Find axes from long runs, allowing partial occlusion by pieces.
    axis_y_map = clip(affine(node("rowsum", [axis_base]), 1, -5))
    axis_x_map = clip(affine(node("colsum", [axis_base]), 1, -5))
    has_x, has_y = gm(axis_x_map), gm(axis_y_map)
    axis_x = gm(mul(px, axis_x_map))
    axis_y = gm(mul(py, axis_y_map))
    x_only = mul(mul(axis_base, axis_x_map), inv(axis_y_map))
    y_only = mul(mul(axis_base, axis_y_map), inv(axis_x_map))
    free_x = mul(has_x, inv(gm(mul(x_only, fixed_axis_marker))))
    free_y = mul(has_y, inv(gm(mul(y_only, fixed_axis_marker))))
    selected_x = gm(mul(x_only, selected))
    selected_y = gm(mul(y_only, selected))
    axis_selected = clip(add(selected_x, selected_y))

    # Fresh extraction. A remembered click supplies a seed when targets
    # hide all of the selected object's usual selection markers.
    seed = mul(mul(original, selected), inv(axis_selected))
    clicked_point = clip(affine(
        node("abs", [sub(index, old_cursor)]), -3, 1.5
    ))
    clicked_seed_gate = mul(
        mul(previous_click, old_done), inv(axis_selected)
    )
    clicked_seed = mul(
        mul(clicked_point, original), clicked_seed_gate
    )
    seed = maximum(seed, clicked_seed)
    flood_domain = mul(
        mul(original, inv(unselected_marker)), inv(fixed_marker)
    )
    fw = np.zeros((1, 1, 7, 7), np.float32)
    for yy, xx in [(3, 3), (0, 3), (6, 3), (3, 0), (3, 6)]:
        fw[0, 0, yy, xx] = 1
    flood_weight = param(fw)

    def flood_body(s):
        spread = node("conv", [s["a"]], w=flood_weight)
        return {"a": minimum(spread, flood_domain)}

    fresh_active = loop(32, {"a": seed}, flood_body)["a"]

    # Analytic encoder/decoder for a 12-by-12 logical-cell template.
    # Each of its 144 channels is a spatial indicator of one template cell.
    template_axis_w = np.zeros((24, 2, 1, 1), np.float32)
    template_axis_w[:12, 0, 0, 0] = 1
    template_axis_w[12:, 1, 0, 0] = 1
    template_axis_b = -np.tile(
        3 * np.arange(12, dtype=np.float32), 2
    )
    template_join_w = np.zeros((144, 24, 1, 1), np.float32)
    for r in range(12):
        for c in range(12):
            template_join_w[12 * r + c, r, 0, 0] = -3
            template_join_w[12 * r + c, 12 + c, 0, 0] = -3
    taw = param(template_axis_w)
    tab = param(template_axis_b)
    tjw = param(template_join_w)
    tjb = param(np.full(144, 1.5, np.float32))

    def template_points(origin):
        relative = sub(pixel, origin)
        distances = node("conv", [relative], w=taw, b=tab)
        distances = node("abs", [distances])
        return clip(node("conv", [distances], w=tjw, b=tjb))

    movement_delta = linear(
        "aprev",
        [
            [0, -3, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, -3, 3, 0, 0, 0],
        ],
    )
    predicted_origin = add(old_origin, movement_delta)
    predicted_points = template_points(predicted_origin)
    predicted_active = clip(
        total_channels(mul(predicted_points, old_template))
    )

    # Trust the translated template only when it remains consistent with
    # both the originals and all visible selected-piece markers.
    missing_original = gm(relu(sub(predicted_active, original)))
    unexplained_seed = gm(relu(sub(seed, predicted_active)))
    cache_error = clip(add(missing_original, unexplained_seed))
    use_cache = mul(
        mul(old_valid, previous_move),
        mul(inv(cache_error), inv(axis_selected)),
    )
    active = add(
        mul(predicted_active, use_cache),
        mul(fresh_active, inv(use_cache)),
    )
    active_exists = gm(active)
    other_originals = clip(sub(original, active))

    def bounds(mask):
        hi = gm(mul(pixel, mask))
        lo = affine(
            gm(mul(affine(pixel, -1, 64), mask)), -1, 64
        )
        return lo, hi

    active_lo, active_hi = bounds(active)
    target_lo, target_hi = bounds(target)
    extent = sub(active_hi, active_lo)
    extent_y, extent_x = sl(extent, 0, 1), sl(extent, 1, 2)
    target_mid = affine(add(target_lo, target_hi), 0.5)
    encoded_template = gm(mul(template_points(active_lo), active))

    # Initialize axes near the centre of the target geometry. A short
    # deterministic offset cycle provides recovery from stalled layouts.
    stale = mul(affine(old_stale, 1, 1), inv(previous_move))
    trigger = mul(clip(affine(stale, 1, -5)), old_done)

    first = np.zeros(9, np.float32)
    first[0] = 1
    missing_trial = clip(inv(total_channels(old_trials)))
    trials = add(
        old_trials, linear(missing_trial, first[:, None])
    )
    rotate = np.zeros((9, 9), np.float32)
    for i in range(9):
        rotate[(i + 1) % 9, i] = 1
    trial_next = add(
        trials, mul(sub(linear(trials, rotate), trials), trigger)
    )

    xy_trials = [
        (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]
    offsets_x = linear(
        trial_next, [[3 * p[0] for p in xy_trials]]
    )
    offsets_y = linear(
        trial_next, [[3 * p[1] for p in xy_trials]]
    )
    offsets_1d = linear(
        trial_next, [[0, -3, 3, -6, 6, -9, 9, -12, 12]]
    )
    two_free = mul(free_x, free_y)
    offsets_x = add(
        mul(offsets_x, two_free),
        mul(offsets_1d, inv(two_free)),
    )
    offsets_y = add(
        mul(offsets_y, two_free),
        mul(offsets_1d, inv(two_free)),
    )
    desired = add(target_mid, cat(offsets_y, offsets_x))
    desired = add(
        sub(
            relu(affine(desired, 1, -1)),
            relu(affine(desired, 1, -61)),
        ),
        one,
    )
    desired_y, desired_x = sl(desired, 0, 1), sl(desired, 1, 2)
    delta_axis_x = sub(desired_x, axis_x)
    delta_axis_y = sub(desired_y, axis_y)

    def vector_abs(x):
        return add(relu(x), relu(affine(x, -1)))

    need_x = mul(
        free_x, clip(affine(vector_abs(delta_axis_x), 1, -1.6))
    )
    need_y = mul(
        free_y, clip(affine(vector_abs(delta_axis_y), 1, -1.6))
    )
    choose_x = need_x
    choose_y = mul(inv(need_x), need_y)
    any_need = clip(add(need_x, need_y))
    done_now = mul(old_done, inv(trigger))
    initializing = mul(inv(done_now), any_need)
    done_next = inv(initializing)

    # Reflection through spatial row/column transfer.
    def point_mask(coordinate, location):
        distance = node("abs", [sub(coordinate, location)])
        return clip(affine(distance, -3, 1.5))

    def reflect(image, axis, horizontal_coordinate):
        coordinate = px if horizontal_coordinate else py
        reduce_op = "rowmax" if horizontal_coordinate else "colmax"
        twice_axis = affine(axis, 2)
        channels = shapes[image][1]

        def body(s):
            src = point_mask(coordinate, s["t"])
            line = node(reduce_op, [mul(image, src)])
            destination = sub(twice_axis, s["t"])
            dst = point_mask(coordinate, destination)
            return {
                "t": affine(s["t"], 1, 3),
                "out": maximum(s["out"], mul(line, dst)),
            }

        return loop(
            21,
            {"t": one, "out": ("s", channels)},
            body,
        )["out"]

    # Independently reconstruct active and other-object coverage.
    originals_pair = cat(active, other_originals)
    reflected_x_pair = reflect(originals_pair, axis_x, True)
    reflected_y_four = reflect(
        cat(originals_pair, reflected_x_pair), axis_y, False
    )
    reflected_y_pair = sl(reflected_y_four, 0, 2)
    reflected_xy_pair = sl(reflected_y_four, 2, 4)

    bad_x_pair = gm(relu(sub(reflected_x_pair, covered)))
    bad_y_pair = gm(relu(sub(reflected_y_pair, covered)))
    allow_x_pair = mul(has_x, inv(clip(bad_x_pair)))
    allow_y_pair = mul(has_y, inv(clip(bad_y_pair)))
    allow_x = sl(allow_x_pair, 0, 1)
    allow_y = sl(allow_y_pair, 0, 1)

    coverage_pair = maximum(
        originals_pair, mul(reflected_x_pair, allow_x_pair)
    )
    coverage_pair = maximum(
        coverage_pair, mul(reflected_y_pair, allow_y_pair)
    )
    coverage_pair = maximum(
        coverage_pair,
        mul(reflected_xy_pair, mul(allow_x_pair, allow_y_pair)),
    )
    other_coverage = sl(coverage_pair, 1, 2)
    residual = mul(target, inv(other_coverage))

    # Fold residual targets into the active piece's translation space.
    rx = reflect(residual, axis_x, True)
    off_x = clip(
        affine(node("abs", [sub(px, axis_x)]), 2, -1)
    )
    folded_x = add(
        residual, mul(mul(rx, off_x), allow_x)
    )
    ry = reflect(folded_x, axis_y, False)
    off_y = clip(
        affine(node("abs", [sub(py, axis_y)]), 2, -1)
    )
    folded = add(
        folded_x, mul(mul(ry, off_y), allow_y)
    )

    # Recurrent correlation scores all translations simultaneously.
    phase0 = np.zeros(12, np.float32)
    phase0[0] = 1
    phase_init = const(phase0)
    phase_rotate = np.zeros((12, 12), np.float32)
    for i in range(12):
        phase_rotate[(i + 1) % 12, i] = 1

    def correlation_body(s):
        dx = linear(
            s["phase"], [3 * np.arange(12, dtype=np.float32)]
        )
        source = add(active_lo, cat(s["dy"], dx))
        distance = node("abs", [sub(pixel, source)])
        query = clip(linear(distance, [[-3, -3]], 1.5))
        coefficient = gm(mul(query, active))
        score = add(s["score"], mul(s["field"], coefficient))

        wrap = sl(s["phase"], 11, 12)
        next_column = shift(s["field"], 0, 3)
        next_row = shift(s["row"], 3, 0)
        row = add(
            s["row"], mul(sub(next_row, s["row"]), wrap)
        )
        field = add(
            next_column, mul(sub(next_row, next_column), wrap)
        )
        return {
            "phase": linear(s["phase"], phase_rotate),
            "dy": add(s["dy"], affine(wrap, 3)),
            "row": row,
            "field": field,
            "score": score,
        }

    scores = loop(
        144,
        {
            "phase": phase_init,
            "dy": ("v", 1),
            "row": folded,
            "field": folded,
            "score": ("s", 1),
        },
        correlation_body,
    )["score"]

    # Board bounds and non-straddling reflection quadrants.
    end_y, end_x = add(py, extent_y), add(px, extent_x)
    overflow = add(
        relu(affine(end_y, 1, -61)),
        relu(affine(end_x, 1, -61)),
    )
    boundary_bad = clip(affine(overflow, 1, -0.1))
    cross_x = clip(affine(
        minimum(sub(axis_x, px), sub(end_x, axis_x)), 1, -0.1
    ))
    cross_y = clip(affine(
        minimum(sub(axis_y, py), sub(end_y, axis_y)), 1, -0.1
    ))
    crossing = clip(add(
        mul(cross_x, allow_x), mul(cross_y, allow_y)
    ))
    quadrant_possible = gm(mul(
        lattice, inv(clip(add(boundary_bad, crossing)))
    ))
    invalid = add(
        boundary_bad, mul(crossing, quadrant_possible)
    )
    scores = sub(scores, affine(invalid, 1000))
    scores = sub(scores, affine(inv(lattice), 10000))

    # Choose coverage first, then distance, then row-major order.
    best_score = gm(scores)
    best = clip(affine(sub(scores, best_score), 2, 1.1))
    distance = linear(
        node("abs", [sub(pixel, active_lo)]),
        [[1 / 3, 1 / 3]],
    )
    proximity = mul(affine(distance, -1, 128), best)
    nearest = clip(
        affine(sub(proximity, gm(proximity)), 2, 1.1)
    )
    rank = mul(affine(index, -1, 4096), nearest)
    winner = clip(affine(sub(rank, gm(rank)), 2, 1.1))
    destination = gm(mul(pixel, winner))
    displacement = sub(destination, active_lo)
    dy, dx = sl(displacement, 0, 1), sl(displacement, 1, 2)

    def positive_motion(d):
        return clip(affine(d, 1, -0.5))

    solve_dirs = cat(
        positive_motion(affine(dy, -1)),
        positive_motion(dy),
        positive_motion(affine(dx, -1)),
        positive_motion(dx),
    )
    solve_dirs = mul(solve_dirs, active_exists)
    solve_moving = clip(total_channels(solve_dirs))
    solve_click = inv(solve_moving)

    # Cycle among other movable originals after settling the active piece.
    candidates = mul(
        mul(original, movable_marker), inv(active)
    )
    candidate_exists = gm(candidates)
    after_cursor = clip(
        affine(sub(index, old_cursor), 2, -1)
    )
    after_cursor = mul(after_cursor, candidates)
    piece_click = sub(
        add(
            affine(candidates, 10000),
            affine(after_cursor, 5000),
        ),
        index,
    )
    click_winner = clip(
        affine(sub(piece_click, gm(piece_click)), 2, 1.1)
    )
    new_cursor = gm(mul(index, click_winner))

    # Axis initialization and recovery.
    use_x = mul(choose_x, selected_x)
    use_y = mul(choose_y, selected_y)
    init_dirs = cat(
        mul(positive_motion(affine(delta_axis_y, -1)), use_y),
        mul(positive_motion(delta_axis_y), use_y),
        mul(positive_motion(affine(delta_axis_x, -1)), use_x),
        mul(positive_motion(delta_axis_x), use_x),
    )
    init_click = inv(clip(add(use_x, use_y)))
    axis_click = add(
        mul(x_only, choose_x), mul(y_only, choose_y)
    )
    axis_click = sub(affine(axis_click, 10000), index)

    controls = add(
        mul(cat(init_dirs, init_click), initializing),
        mul(cat(solve_dirs, solve_click), inv(initializing)),
    )
    w = np.zeros((8, 5), np.float32)
    for i in range(4):
        w[i + 1, i] = 10
    w[6, 4] = 10
    linear(
        controls, w,
        [-100, 0, 0, 0, 0, -20, 0, -100],
        name="logits",
    )
    add(
        mul(axis_click, initializing),
        mul(piece_click, inv(initializing)),
        name="click",
    )

    update_cursor = mul(
        mul(solve_click, inv(initializing)), candidate_exists
    )
    cursor_next = add(
        old_cursor,
        mul(sub(new_cursor, old_cursor), update_cursor),
    )
    stale_next = mul(stale, inv(trigger))
    node(
        "vconcat",
        [
            done_next, cursor_next, stale_next, trial_next,
            encoded_template, active_lo, active_exists,
        ],
        name="mem_next",
    )

    return {
        "graph": graph,
        "memory_size": memory_size,
        "params": P,
    }
