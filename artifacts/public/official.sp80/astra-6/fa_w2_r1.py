import numpy as np
from collections import Counter


def build():
    P, parameter_cache = {}, {}
    graph = []
    current = graph
    serial = [0]
    shapes = {
        "x0": (True, 16), "coords": (True, 3),
        "mem": (False, 1), "aprev": (False, 8), "avail": (False, 8),
    }

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p" + str(len(P))
            P[name] = a
            parameter_cache[key] = name
        return parameter_cache[key]

    def N(op, ins, name=None, **kw):
        name = name or fresh()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(kw)
        current.append(nd)
        if op in ("conv", "dense"):
            shapes[name] = (op == "conv", P[kw["w"]].shape[0])
        elif op in ("concat", "vconcat"):
            shapes[name] = (op == "concat", sum(shapes[z][1] for z in ins))
        elif op in ("slice", "vslice"):
            shapes[name] = (op == "slice", kw["stop"] - kw["start"])
        elif op in ("gmax", "gmean"):
            shapes[name] = (False, shapes[ins[0]][1])
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shapes[name] = (True, max(shapes[z][1] for z in ins))
        elif op in ("vadd", "vmul"):
            shapes[name] = (False, max(shapes[z][1] for z in ins))
        else:
            shapes[name] = shapes[ins[0]]
        return name

    def lin(x, w=1.0, b=None, name=None):
        spatial, c = shapes[x]
        w = np.asarray(w, np.float32)
        if w.ndim == 0:
            if float(w) == 1 and b is None and name is None:
                return x
            w = np.eye(c, dtype=np.float32) * w
        elif w.ndim == 1:
            w = w[None, :]
        o = w.shape[0]
        if spatial:
            w = w.reshape(o, c, 1, 1)
        kw = {"w": parameter(w)}
        if b is not None:
            b = np.asarray(b, np.float32)
            if b.ndim == 0:
                b = np.full(o, b, np.float32)
            kw["b"] = parameter(b)
        return N("conv" if spatial else "dense", [x], name=name, **kw)

    def conv(x, w, b=None):
        kw = {"w": parameter(w)}
        if b is not None:
            b = np.asarray(b, np.float32)
            if b.ndim == 0:
                b = np.full(np.asarray(w).shape[0], b, np.float32)
            kw["b"] = parameter(b)
        return N("conv", [x], **kw)

    def sl(x, a, b):
        if a == 0 and b == shapes[x][1]:
            return x
        return N("slice" if shapes[x][0] else "vslice",
                 [x], start=a, stop=b)

    def cat(xs):
        if len(xs) == 1:
            return xs[0]
        return N("concat" if shapes[xs[0]][0] else "vconcat", xs)

    def add(a, b):
        return N("add" if shapes[a][0] or shapes[b][0] else "vadd", [a, b])

    def mul(a, b):
        return N("mul" if shapes[a][0] or shapes[b][0] else "vmul", [a, b])

    def sub(a, b):
        if shapes[a][0] or shapes[b][0]:
            return N("sub", [a, b])
        return add(a, lin(b, -1.0))

    def maximum(a, b):
        if shapes[a][0] or shapes[b][0]:
            return N("max2", [a, b])
        return add(b, N("vrelu", [sub(a, b)]))

    def minimum(a, b):
        return N("min2", [a, b])

    one, rr, cc = sl("coords", 2, 3), sl("coords", 0, 1), sl("coords", 1, 2)

    def clip(x):
        if shapes[x][0]:
            return minimum(N("relu", [x]), one)
        return lin(N("vrelu", [
            lin(N("vrelu", [x]), -1.0, 1.0)]), -1.0, 1.0)

    def hard(x):
        return clip(lin(x, 2.0, -0.5))

    def total(x):
        if shapes[x][1] == 1:
            return x
        return lin(x, np.ones((1, shapes[x][1]), np.float32))

    def fields(xs):
        return total(cat(xs))

    def constant(v):
        return lin("avail", np.zeros((1, 8), np.float32), v)

    def loop(initial, count, fn):
        nonlocal current
        lid, sid = fresh("L"), fresh("s")
        shapes[sid] = shapes[initial]
        outer = current
        current = []
        nxt = fn(sid)
        body = current
        current = outer
        current.append({
            "id": lid, "op": "loop", "T": count,
            "state": [{"id": sid, "init": initial}],
            "body": body, "next": {sid: nxt},
        })
        out = lid + "." + sid
        shapes[out] = shapes[initial]
        return out

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]
    perpendicular = [[2, 3], [2, 3], [0, 1], [0, 1]]
    turns = [(d, p) for d in range(4) for p in perpendicular[d]]

    def shift(x, dy, dx):
        k = max(3, 2 * max(abs(dy), abs(dx)) + 1)
        w = np.zeros((1, 1, k, k), np.float32)
        w[0, 0, k // 2 + dy, k // 2 + dx] = 1
        return conv(x, w)

    def ahead(x):
        w = np.zeros((4, 1, 3, 3), np.float32)
        for d, (dy, dx) in enumerate(directions):
            w[d, 0, 1 + dy, 1 + dx] = 1
        return conv(x, w)

    def select(x, indices):
        w = np.zeros((len(indices), shapes[x][1]), np.float32)
        for i, j in enumerate(indices):
            w[i, j] = 1
        return lin(x, w)

    def turn_gates(a):
        return minimum(
            select(a, [d for d, p in turns]),
            select(a, [opposite[p] for d, p in turns]))

    # Logical-cell sampling. Nine copies share each forward propagation loop.
    w = np.zeros((1, 16, 7, 7), np.float32)
    for dy, dx in [(0, 0), (0, 3), (3, 0), (3, 3)]:
        w[0, 6, 3 + dy, 3 + dx] = 1
    scale4 = N("gmax", [N("relu", [conv("x0", w, -3.0)])])
    scale3 = lin(scale4, -1.0, 1.0)

    def triangle(axis, positions):
        positions = np.asarray(positions, np.float32)
        z = lin(axis, np.full((len(positions), 1), 63, np.float32),
                -positions)
        return N("relu", [lin(N("abs", [z]), -1.0, 1.0)])

    sample_r = add(
        mul(triangle(rr, 1 + 3 * np.arange(21)), scale3),
        mul(triangle(rr, 2 + 4 * np.arange(21)), scale4))
    sample_c = add(
        mul(triangle(cc, 1 + 3 * np.arange(21)), scale3),
        mul(triangle(cc, 2 + 4 * np.arange(21)), scale4))
    raw_r = triangle(rr, np.arange(63))
    raw_c = triangle(cc, np.arange(63))
    repeat = np.concatenate([np.eye(21, dtype=np.float32)] * 3, axis=1)
    dest_r, dest_c = lin(raw_r, repeat), lin(raw_c, repeat)
    qy = lin(dest_r, np.arange(21, dtype=np.float32))
    qx = lin(dest_c, np.arange(21, dtype=np.float32))

    group_w = np.zeros((3, 63), np.float32)
    for k in range(3):
        group_w[k, 21 * k:21 * (k + 1)] = 1
    rg, cg = lin(raw_r, group_w), lin(raw_c, group_w)
    tiles = hard(mul(
        select(rg, [k // 3 for k in range(9)]),
        select(cg, [k % 3 for k in range(9)])))
    tile_active = total(tiles)

    v3 = np.zeros((1, 21), np.float32)
    v4 = v3.copy()
    v3[0, 1:21], v4[0, :16] = 1, 1
    vr = add(mul(lin(dest_r, v3), scale3), mul(lin(dest_r, v4), scale4))
    vc = add(mul(lin(dest_c, v3), scale3), mul(lin(dest_c, v4), scale4))
    valid = hard(mul(vr, vc))

    def compact(f):
        z = N("colmax", [mul(f, sample_r)])
        z = total(mul(z, dest_r))
        z = N("rowmax", [mul(z, sample_c)])
        return hard(total(mul(z, dest_c)))

    original, packed = [], []
    for colors in ([6], [4], [8], [9], [15], [11, 13], [1]):
        w = np.zeros((1, 16), np.float32)
        w[0, list(colors)] = 1
        f = lin("x0", w)
        original.append(f)
        packed.append(compact(f))

    source, emitter, bars0, selected, mirrors0, cups, walls0 = packed
    orientation = N("gmax", [
        mul(source, select(ahead(emitter), opposite))])
    fallback = N("vrelu", [
        lin(orientation, -np.ones((1, 4), np.float32), 1.0)])
    orientation = add(orientation, lin(
        fallback, np.array([[0], [1], [0], [0]], np.float32)))
    orientation8 = select(orientation, [d for d, p in turns])
    vertical = lin(orientation, [1, 1, 0, 0])
    horizontal = lin(orientation, [0, 0, 1, 1])
    lateral_orientation = cat([vertical, horizontal])

    def lateral(x, sign):
        w = np.zeros((2, 1, 3, 3), np.float32)
        w[0, 0, 1, 1 + sign] = 1
        w[1, 0, 1 + sign, 1] = 1
        return total(mul(conv(x, w), lateral_orientation))

    def forward(x):
        return total(mul(ahead(x), orientation))

    def spread(x):
        return add(mul(N("rowmax", [x]), vertical),
                   mul(N("colmax", [x]), horizontal))

    def extents(q, reduction):
        active = N(reduction, [selected])
        left = N(reduction, [mul(selected, lin(q, -1.0, 32.0))])
        right = N(reduction, [mul(selected, q)])
        gl = N("gmax", [left])
        return active, sub(gl, left), lin(add(right, gl), 1.0, -32.0), \
            lin(gl, -1.0, 32.0)

    row_active, xlo, xhi, xmin = extents(qx, "rowmax")
    col_active, ylo, yhi, ymin = extents(qy, "colmax")

    wv = np.zeros((1, 1, 3, 3), np.float32)
    wh = wv.copy()
    wv[0, 0, 0, 1] = wv[0, 0, 2, 1] = 1
    wh[0, 0, 1, 0] = wh[0, 0, 1, 2] = 1
    is_mirror = N("gmax", [
        clip(minimum(conv(selected, wv), conv(selected, wh)))])
    is_bar = lin(is_mirror, -1.0, 1.0)
    has_mirror = maximum(is_mirror, N("gmax", [mirrors0]))
    straight = lin(has_mirror, -1.0, 1.0)

    ca = ahead(cups)
    mouths = minimum(ca, minimum(
        select(ca, [p[0] for p in perpendicular]),
        select(ca, [p[1] for p in perpendicular])))
    mouth_map = total(mouths)
    cup_branch = sub(ca, mouths)
    outside = sub(one, valid)
    walls = clip(add(walls0, outside))
    wall_ahead = ahead(walls0)

    source_projection = add(
        mul(N("colmax", [source]), vertical),
        mul(N("rowmax", [source]), horizontal))
    mp = cat([sl(N("colmax", [mouths]), 0, 2),
              sl(N("rowmax", [mouths]), 2, 4)])
    primary_projection = total(mul(mp, orientation))

    selected_turns = mul(
        mul(turn_gates(ahead(selected)), sub(one, selected)), orientation8)
    ow = np.zeros((4, 8), np.float32)
    for t, (d, p) in enumerate(turns):
        ow[p, t] = 1
    outgoing = lin(N("gmax", [selected_turns]), ow)
    mirror_projection = total(mul(mp, outgoing))

    def distance(seed):
        seed = mul(clip(seed), valid)

        def body(s):
            a = ahead(s)
            z = maximum(maximum(sl(a, 0, 1), sl(a, 1, 2)),
                        maximum(sl(a, 2, 3), sl(a, 3, 4)))
            z = N("relu", [lin(z, 1.0, -1.0 / 32.0)])
            return mul(maximum(z, seed), valid)

        return loop(seed, 24, body)

    target_distance = distance(mouth_map)
    source_distance = distance(source_projection)
    primary_distance = distance(primary_projection)
    mirror_distance = distance(mirror_projection)

    top_band = clip(conv(spread(source), np.ones((1, 1, 3, 3), np.float32)))
    cup_margin = clip(conv(cups, np.ones((1, 1, 3, 3), np.float32)))
    forbidden = clip(fields([walls, top_band, cup_margin]))

    legal = []
    for dy, dx in directions:
        collision = N("gmax", [
            mul(shift(selected, -dy, -dx), forbidden)])
        legal.append(lin(collision, -1.0, 1.0))
    legal = cat(legal)

    def pool(f):
        return lin(N("gmean", [mul(f, tiles)]), 4096.0)

    def unpool(v):
        return total(mul(tiles, v))

    cup_count, original_area = pool(mouth_map), pool(selected)

    # A tiny phase-dependent preference breaks otherwise flat rearrangements.
    clicks = add("mem", sl("aprev", 6, 7))
    counter = sub(clicks, lin(N("vrelu", [lin(clicks, 1.0, -11.0)]), 12.0))
    lin(counter, 1.0, name="mem_next")
    phase = lin(clip(lin(counter, 1.0, -5.0)), -2.0, 1.0)
    upstream = lin(
        cat([qy, qx]),
        np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], np.float32),
        np.array([0, 20, 0, 20], np.float32))
    upstream = mul(total(mul(upstream, orientation)), phase)

    # Small backward planner: recognize other bar lengths and propagate routes
    # through hypothetical horizontal translations at their observed rows.
    zero = lin(one, 0.0)
    run = loop(zero, 20, lambda s: mul(
        bars0, lin(lateral(s, -1), 1.0, 1.0)))
    run_end = mul(bars0, sub(one, lateral(bars0, 1)))
    allowed = {}
    span = forbidden
    shifted_forbidden = forbidden
    for length in range(1, 7):
        if length > 1:
            shifted_forbidden = lateral(shifted_forbidden, 1)
            span = maximum(span, shifted_forbidden)
        if length >= 2:
            match = N("relu", [
                lin(N("abs", [lin(run, 1.0, -float(length))]), -1.0, 1.0)])
            present = spread(mul(match, run_end))
            allowed[length] = mul(present, sub(one, span))

    safe_seed = clip(add(mouth_map, outside))
    passable = sub(one, clip(add(cups, walls0)))
    boundary_left = sub(one, lateral(one, -1))
    boundary_right = sub(one, lateral(one, 1))

    def backward_body(s):
        below = forward(s)
        left = add(lateral(below, -1), boundary_left)
        right = below
        choices = {}
        for length in range(1, 7):
            right = lateral(right, 1)
            if length >= 2:
                quality = N("relu", [
                    lin(minimum(left, right), 1.0, -0.025)])
                choices[length] = mul(quality, allowed[length])
        suffix = expanded = choices[6]
        for length in range(5, 0, -1):
            if length >= 2:
                suffix = maximum(suffix, choices[length])
            expanded = maximum(suffix, lateral(expanded, -1))
        propagated = mul(maximum(below, expanded), passable)
        return maximum(safe_seed, maximum(s, propagated))

    safe_routes = loop(safe_seed, 24, backward_body)
    safe_left = add(lateral(safe_routes, -1), boundary_left)
    safe_right = add(lateral(safe_routes, 1), boundary_right)

    advance_w = np.zeros((4, 4, 3, 3), np.float32)
    branch_w = np.zeros((4, 4, 3, 3), np.float32)
    turn_w = np.zeros((4, 8, 3, 3), np.float32)
    for d, (dy, dx) in enumerate(directions):
        advance_w[d, d, 1 - dy, 1 - dx] = 1
        for p in perpendicular[d]:
            py, px = directions[p]
            branch_w[d, d, 1 - py, 1 - px] = 1
    for t, (d, p) in enumerate(turns):
        py, px = directions[p]
        turn_w[p, t, 1 - py, 1 - px] = 1
    seeds = mul(source, orientation)

    diagonal_w = np.zeros((4, 1, 3, 3), np.float32)
    for d, (dy, dx) in enumerate(directions):
        for p in perpendicular[d]:
            py, px = directions[p]
            diagonal_w[d, 0, 1 + dy + py, 1 + dx + px] = 1

    specs = [
        ("x", 0, True), ("y", -1, True), ("y", 1, True),
        ("x", -1, True), ("x", 1, True),
    ]
    specs += [("x", x, False) for x in range(0, 21, 2)]
    specs += [("y", y, False) for y in range(0, 21, 2)]

    def interval(diff, lo, hi, active):
        return mul(hard(lin(minimum(sub(diff, lo), sub(hi, diff)),
                            1.0, 1.0)), active)

    def notch(s):
        w = np.zeros((1, 1, 3, 3), np.float32)
        for dy, dx in directions:
            w[0, 0, 1 + dy, 1 + dx] = 1
        return mul(clip(conv(s, w, -1.0)), sub(one, s))

    tiny_cost = constant(0.0001)
    movement_cost = add(lin(straight, 0.02), lin(has_mirror, 0.0001))
    budget_fallback = {movement_cost: tiny_cost}
    best_groups, win_groups = [], []

    for group_index in range(3):
        batch = specs[9 * group_index:9 * (group_index + 1)]
        constants = np.array([s[1] for s in batch], np.float32)
        axes = np.array([s[0] == "x" for s in batch], np.float32)
        rx = np.array([s[2] and s[0] == "x" for s in batch], np.float32)
        ry = np.array([s[2] and s[0] == "y" for s in batch], np.float32)
        anchor = lin(tiles, constants)
        if np.any(rx):
            anchor = add(anchor, mul(lin(tiles, rx), xmin))
        if np.any(ry):
            anchor = add(anchor, mul(lin(tiles, ry), ymin))
        axis_x = lin(tiles, axes)
        axis_y = sub(tile_active, axis_x)
        sx = interval(sub(qx, anchor), xlo, xhi, row_active)
        sy = interval(sub(qy, anchor), ylo, yhi, col_active)
        candidate = add(mul(sx, axis_x), mul(sy, axis_y))
        cb = mul(candidate, is_bar)
        cm = mul(candidate, is_mirror)

        bars = clip(add(bars0, cb))
        mirrors = clip(add(mirrors0, cm))
        empty = sub(one, clip(fields([bars, mirrors, cups, walls, emitter])))
        ba, ma = ahead(bars), ahead(mirrors)
        tg = turn_gates(ma)
        ms0 = select(ma, [p[0] for p in perpendicular])
        ms1 = select(ma, [p[1] for p in perpendicular])
        mirror_branch = sub(ma, minimum(ma, maximum(ms0, ms1)))
        branch_gate = clip(add(add(ba, cup_branch), mirror_branch))

        def fluid_body(f, bg=branch_gate, gates=tg, free=empty):
            adv = conv(f, advance_w)
            br = conv(mul(f, bg), branch_w)
            repeated = select(f, [d for d, p in turns])
            tr = conv(mul(repeated, gates), turn_w)
            incoming = mul(add(add(adv, br), tr), free)
            return maximum(f, minimum(incoming, one))

        flow = loop(mul(seeds, empty), 64, fluid_body)
        filled = total(minimum(flow, mouths))
        spill = total(mul(flow, wall_ahead))
        coverage = mul(distance(clip(total(flow))), mouth_map)
        spill_distance = mul(spill, sub(one, target_distance))

        endpoints = N("relu", [
            sub(sub(conv(cb, diagonal_w), lin(ahead(cb), 2.0)),
                lin(cb, 2.0))])
        endpoints = total(mul(endpoints, orientation))
        endpoint_promise = mul(endpoints, primary_distance)
        input_promise = mul(cb, source_distance)
        mirror_promise = mul(notch(cm), add(source_distance, mirror_distance))
        height_promise = mul(cb, upstream)

        fluid_metric = lin(cat([
            filled, coverage, spill, spill_distance, endpoint_promise,
            input_promise, mirror_promise, height_promise
        ]), np.array([8, 2, -2, -4, 0.08, 0.01, 0.5, 0.00025], np.float32))

        left_edge = mul(cb, sub(one, lateral(cb, -1)))
        right_edge = mul(cb, sub(one, lateral(cb, 1)))
        routing_quality = add(mul(left_edge, safe_left),
                              mul(right_edge, safe_right))
        planning_metric = fields([
            lin(routing_quality, 100.0), lin(input_promise, 0.02),
            lin(fluid_metric, 0.01)])
        metric = add(mul(fluid_metric, has_mirror),
                     mul(planning_metric, straight))
        budget_fallback[metric] = fluid_metric

        defect = add(sub(cup_count, pool(filled)), pool(spill))
        wins = clip(lin(defect, -2.0, 1.0))
        lost = N("vrelu", [sub(original_area, pool(candidate))])
        bad = clip(add(lost, pool(mul(candidate, forbidden))))
        score = add(add(pool(metric), lin(wins, 1000.0)), lin(bad, -2000.0))
        dx, dy = sub(anchor, xmin), sub(anchor, ymin)
        travel = add(mul(N("abs", [dx]), axis_x),
                     mul(N("abs", [dy]), axis_y))
        score_map = sub(unpool(score), mul(travel, movement_cost))

        dm = cat([
            mul(hard(lin(dy, -1.0)), axis_y), mul(hard(dy), axis_y),
            mul(hard(lin(dx, -1.0)), axis_x), mul(hard(dx), axis_x)])
        same = mul(sub(one, total(dm)), tile_active)
        flags = cat([same, mul(dm, legal)])
        masked = sub(score_map, lin(flags, -4000.0, 4000.0))
        best_groups.append(N("gmax", [masked]))
        win_groups.append(N("gmax", [mul(unpool(wins), same)]))

    best = maximum(maximum(best_groups[0], best_groups[1]), best_groups[2])
    solved = maximum(maximum(win_groups[0], win_groups[1]), win_groups[2])
    moves = sub(sl(best, 1, 5), sl(best, 0, 1))

    clickable = add(original[2], original[4])
    rank = lin("coords", np.array(
        [4032.0 / 4095.0, 63.0 / 4095.0, 0], np.float32))
    selected_rank = N("gmax", [mul(original[3], rank)])
    after = clip(lin(sub(rank, selected_rank), 10000.0, -0.1))
    lin(cat([clickable, after, rank]), [1000, 10, -1], name="click")
    click_logit = lin(N("gmax", [clickable]), 200.0005, -200.0)
    pour_logit = lin(solved, 10000.0, -100.0)
    N("vconcat", [
        constant(-10000), moves, pour_logit, click_logit, constant(-10000)
    ], name="logits")

    # Algebraic graph compiler. It does not evaluate observations or game states.
    def compile_graph(replacements, aggressive):
        def resolve(name):
            while name in replacements:
                name = replacements[name]
            return name

        refs = Counter()

        def count_refs(nodes):
            for nd in nodes:
                if nd["op"] == "loop":
                    for s in nd["state"]:
                        if "init" in s:
                            refs[resolve(s["init"])] += 1
                    for v in nd["next"].values():
                        refs[resolve(v)] += 1
                    count_refs(nd["body"])
                else:
                    for v in nd["in"]:
                        refs[resolve(v)] += 1

        count_refs(graph)

        class Scope:
            def __init__(self, nodes, external=None, states=()):
                self.defs = {n["id"]: n for n in nodes if n["op"] != "loop"}
                self.loops = {}
                for n in nodes:
                    if n["op"] == "loop":
                        for s in n["state"]:
                            self.loops[n["id"] + "." + s["id"]] = n
                self.external = external
                self.states = set(states)
                self.out = []
                self.alias = {}
                self.cse = {}
                self.expr_cache = {}

            def affine(self, name):
                n = self.defs.get(name)
                if n is None:
                    return False
                op = n["op"]
                if op == "conv":
                    return P[n["w"]].shape[-1] == 1
                if op in ("dense", "concat", "vconcat", "slice", "vslice", "vadd"):
                    return True
                if op in ("add", "sub"):
                    return all(shapes[x][0] == shapes[name][0] for x in n["in"])
                return False

            def leaf(self, name):
                c = shapes[name][1]
                return {name: np.eye(c, dtype=np.float32)}, np.zeros(c, np.float32)

            def expression(self, name, root=False):
                name = resolve(name)
                key = (name, root)
                if key in self.expr_cache:
                    return self.expr_cache[key]
                expand = self.affine(name)
                if expand and not root and refs[name] > 1:
                    nd = self.defs[name]
                    small = max([shapes[name][1]] +
                                [shapes[x][1] for x in nd["in"]]) <= 8
                    expand = aggressive and small
                if not expand:
                    answer = self.leaf(name)
                    self.expr_cache[key] = answer
                    return answer

                nd = self.defs[name]
                op = nd["op"]
                es = [self.expression(x) for x in nd["in"]]
                if op in ("conv", "dense"):
                    w = P[nd["w"]].reshape(shapes[name][1], -1)
                    d, b = es[0]
                    out = {k: w @ v for k, v in d.items()}
                    bias = w @ b
                    if "b" in nd:
                        bias = bias + P[nd["b"]]
                elif op in ("slice", "vslice"):
                    a, z = nd["start"], nd["stop"]
                    out = {k: v[a:z] for k, v in es[0][0].items()}
                    bias = es[0][1][a:z]
                elif op in ("concat", "vconcat"):
                    c = shapes[name][1]
                    out, bias = {}, np.zeros(c, np.float32)
                    offset = 0
                    for d, b in es:
                        n = len(b)
                        bias[offset:offset + n] = b
                        for k, v in d.items():
                            if k not in out:
                                out[k] = np.zeros((c, v.shape[1]), np.float32)
                            out[k][offset:offset + n] += v
                        offset += n
                else:
                    c = shapes[name][1]
                    out, bias = {}, np.zeros(c, np.float32)
                    for i, (d, b) in enumerate(es):
                        sign = -1 if op == "sub" and i == 1 else 1
                        bias += sign * b
                        for k, v in d.items():
                            if v.shape[0] == 1 and c != 1:
                                v = np.repeat(v, c, axis=0)
                            if k in out:
                                out[k] = out[k] + sign * v
                            else:
                                out[k] = sign * v.copy()
                out = {k: v for k, v in out.items() if np.any(v != 0)}
                answer = (out, np.asarray(bias, np.float32))
                self.expr_cache[key] = answer
                return answer

            def ordinary(self, op, ins, **kw):
                key = (op, tuple(ins), tuple(sorted(kw.items())))
                if key in self.cse:
                    return self.cse[key]
                name = fresh("z")
                nd = {"id": name, "op": op, "in": list(ins)}
                nd.update(kw)
                self.out.append(nd)
                self.cse[key] = name
                return name

            def emit(self, name):
                name = resolve(name)
                if name in self.alias:
                    return self.alias[name]
                if name in self.states:
                    return name
                if name in self.loops:
                    nd = self.loops[name]
                    new_states = []
                    for s in nd["state"]:
                        t = dict(s)
                        if "init" in t:
                            t["init"] = self.emit(t["init"])
                        new_states.append(t)
                    child = Scope(nd["body"], self.emit,
                                  [s["id"] for s in nd["state"]])
                    nxt = {s: child.emit(v) for s, v in nd["next"].items()}
                    self.out.append({
                        "id": nd["id"], "op": "loop", "T": nd["T"],
                        "state": new_states, "body": child.out, "next": nxt
                    })
                    for s in nd["state"]:
                        outname = nd["id"] + "." + s["id"]
                        self.alias[outname] = outname
                    return self.alias[name]
                if name not in self.defs:
                    return self.external(name) if self.external else name

                nd = self.defs[name]
                if not self.affine(name):
                    ins = [self.emit(x) for x in nd["in"]]
                    kw = {k: v for k, v in nd.items()
                          if k not in ("id", "op", "in", "comment")}
                    result = self.ordinary(nd["op"], ins, **kw)
                else:
                    d, b = self.expression(name, True)
                    spatial, c = shapes[name]
                    if not d:
                        inp = "coords" if spatial else "avail"
                        ins = [self.emit(inp)]
                        w = np.zeros((c, 3 if spatial else 8), np.float32)
                    else:
                        ins = [self.emit(k) for k in d]
                        w = np.concatenate(list(d.values()), axis=1)
                    bias_zero = not np.any(b != 0)
                    identity = (w.shape[0] == w.shape[1] and bias_zero and
                                np.array_equal(w, np.eye(w.shape[0],
                                                        dtype=np.float32)))
                    if identity:
                        result = ins[0] if len(ins) == 1 else self.ordinary(
                            "concat" if spatial else "vconcat", ins)
                    else:
                        sliced = False
                        if len(ins) == 1 and bias_zero:
                            columns = np.argmax(w, axis=1)
                            test = np.zeros_like(w)
                            test[np.arange(c), columns] = 1
                            if (np.array_equal(w, test) and
                                    np.array_equal(columns,
                                                   np.arange(columns[0],
                                                             columns[0] + c))):
                                result = self.ordinary(
                                    "slice" if spatial else "vslice", ins,
                                    start=int(columns[0]), stop=int(columns[0] + c))
                                sliced = True
                        if not sliced:
                            inp = ins[0] if len(ins) == 1 else self.ordinary(
                                "concat" if spatial else "vconcat", ins)
                            kw = {"w": parameter(
                                w[:, :, None, None] if spatial else w)}
                            if not bias_zero:
                                kw["b"] = parameter(b)
                            result = self.ordinary(
                                "conv" if spatial else "dense", [inp], **kw)
                self.alias[name] = result
                return result

        scope = Scope(graph)
        outputs = {name: scope.emit(name)
                   for name in ("logits", "click", "mem_next")}
        for name, source_name in outputs.items():
            spatial, c = shapes[name]
            w = np.eye(c, dtype=np.float32)
            if spatial:
                w = w[:, :, None, None]
            scope.out.append({
                "id": name, "op": "conv" if spatial else "dense",
                "in": [source_name], "w": parameter(w)
            })

        used = set()

        def collect(nodes):
            for nd in nodes:
                if nd["op"] == "loop":
                    collect(nd["body"])
                else:
                    for key in ("w", "b"):
                        if key in nd:
                            used.add(nd[key])

        collect(scope.out)
        return {
            "graph": scope.out, "memory_size": 1,
            "params": {k: P[k] for k in used}
        }

    def within_budget(artifact):
        pars = artifact["params"]
        env = {
            "x0": (True, 16), "coords": (True, 3),
            "mem": (False, 1), "aprev": (False, 8), "avail": (False, 8)
        }
        totals = {"nodes": 0, "evals": 0, "mac": 0, "channels": 19}
        dimensions_ok = [True]

        def walk(nodes, multiplier=1):
            for nd in nodes:
                totals["nodes"] += 1
                totals["evals"] += multiplier
                op = nd["op"]
                if op == "loop":
                    for s in nd["state"]:
                        if "init" in s:
                            shape = env[s["init"]]
                        else:
                            z = s["zeros"]
                            shape = (len(z) == 3, z[0])
                        env[s["id"]] = shape
                        if shape[0]:
                            totals["channels"] += shape[1]
                    walk(nd["body"], multiplier * nd["T"])
                    for sid, value in nd["next"].items():
                        env[nd["id"] + "." + sid] = env[value]
                        if env[value][0]:
                            totals["channels"] += env[value][1]
                    continue
                xs = [env[x] for x in nd["in"]]
                mac = 0
                if op in ("conv", "dense"):
                    w = pars[nd["w"]]
                    shape = (op == "conv", w.shape[0])
                    mac = w.size * (4096 if op == "conv" else 1)
                elif op in ("concat", "vconcat"):
                    shape = (op == "concat", sum(x[1] for x in xs))
                elif op in ("slice", "vslice"):
                    shape = (op == "slice", nd["stop"] - nd["start"])
                elif op in ("gmax", "gmean"):
                    shape = (False, xs[0][1])
                    mac = xs[0][1] * 4096
                elif op in ("add", "sub", "mul", "max2", "min2"):
                    shape = (True, max(x[1] for x in xs))
                    mac = shape[1] * 4096
                elif op in ("vadd", "vmul"):
                    shape = (False, max(x[1] for x in xs))
                    mac = shape[1]
                else:
                    shape = xs[0]
                    if op in ("rowmax", "colmax", "rowsum", "colsum"):
                        mac = shape[1] * 4096
                env[nd["id"]] = shape
                if shape[0]:
                    totals["channels"] += shape[1]
                    dimensions_ok[0] &= shape[1] <= 512
                else:
                    dimensions_ok[0] &= shape[1] <= 4096
                totals["mac"] += multiplier * mac

        walk(artifact["graph"])
        return (
            dimensions_ok[0] and totals["nodes"] <= 1000 and
            totals["evals"] <= 10000 and totals["mac"] <= 1500000000 and
            totals["channels"] <= 4096 and
            sum(v.size for v in pars.values()) <= 150000 and
            all(np.isfinite(v).all() for v in pars.values())
        )

    # Compiler choices depend only on static graph resource counts.
    artifact = compile_graph({}, False)
    if not within_budget(artifact):
        artifact = compile_graph({}, True)
    if not within_budget(artifact):
        artifact = compile_graph(budget_fallback, False)
    if not within_budget(artifact):
        artifact = compile_graph(budget_fallback, True)
    assert within_budget(artifact)
    return artifact
