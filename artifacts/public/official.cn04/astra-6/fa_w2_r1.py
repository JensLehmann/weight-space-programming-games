import numpy as np


def build():
    N, E = 6, 7
    L, K = N * E, 4 * N * E

    fields = [
        ("tx", L), ("ty", L),
        ("w", N), ("h", N), ("lock", N),
        ("cx", N), ("cy", N), ("cr", N),
        ("sx", N), ("sy", N), ("sr", N), ("sd", N),
        ("ct", N + 1), ("opened", L),
        ("depth", 1), ("found", 1),
        ("stage", 1), ("count", 1), ("grow", 1), ("active", 1),
        ("targetx", N), ("targety", N), ("feasible", 1),
        ("bx", L), ("boot", 1),
    ]
    M = sum(size for _, size in fields)

    graph, params = [], {}
    shapes = {
        "x0": (16, 64, 64), "coords": (3, 64, 64),
        "mem": (M,), "aprev": (8,), "avail": (8,),
    }
    destination = graph
    serial = [0]
    parameter_cache, constant_cache = {}, {}
    global_cache, local_cache = {}, {}

    def parameter(value):
        a = np.asarray(value, np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def emit(op, inputs, name=None, **kwargs):
        key = (op, tuple(inputs), tuple(sorted(kwargs.items())))
        if name is None:
            if key in global_cache:
                return global_cache[key]
            if destination is not graph and key in local_cache:
                return local_cache[key]
            name = "n%d" % serial[0]
            serial[0] += 1
            if destination is graph:
                global_cache[key] = name
            else:
                local_cache[key] = name

        node = {"id": name, "op": op, "in": list(inputs)}
        node.update(kwargs)
        destination.append(node)
        ss = [shapes[x] for x in inputs]
        if op == "dense":
            shape = (params[kwargs["w"]].shape[0],)
        elif op == "conv":
            shape = (params[kwargs["w"]].shape[0], 64, 64)
        elif op in ("slice", "vslice"):
            shape = (kwargs["stop"] - kwargs["start"],) + ss[0][1:]
        elif op in ("concat", "vconcat"):
            shape = (sum(s[0] for s in ss),) + ss[0][1:]
        elif op in ("gmax", "gmean"):
            shape = (ss[0][0],)
        elif op in ("vadd", "vmul"):
            shape = (max(s[0] for s in ss),)
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shape = (max(s[0] for s in ss), 64, 64)
        else:
            shape = ss[0]
        shapes[name] = shape
        return name

    def dense(x, w, b=None, name=None):
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return emit("dense", [x], name=name, **kw)

    def conv(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return emit("conv", [x], name=name, **kw)

    one = dense("aprev", np.zeros((1, 8), np.float32), [1])

    def const(value):
        nonlocal destination
        a = np.asarray(value, np.float32).reshape(-1)
        key = (a.shape, a.tobytes())
        if key not in constant_cache:
            saved = destination
            destination = graph
            constant_cache[key] = dense(one, a[:, None])
            destination = saved
        return constant_cache[key]

    def sl(x, start, stop):
        op = "vslice" if len(shapes[x]) == 1 else "slice"
        return emit(op, [x], start=start, stop=stop)

    def cat(*xs):
        return emit("vconcat", xs)

    def add(a, b):
        return emit("vadd", [a, b])

    def mul(a, b):
        return emit("vmul", [a, b])

    def affine(x, a=1, b=0):
        if a == 1 and b == 0:
            return x
        n = shapes[x][0]
        if n <= 8:
            return dense(
                x, np.eye(n, dtype=np.float32) * a,
                np.full(n, b, np.float32) if b != 0 else None)
        y = x if a == 1 else mul(x, const(a))
        return y if b == 0 else add(y, const(b))

    def neg(x):
        return affine(x, -1)

    def sub(a, b):
        return add(a, neg(b))

    def inv(x):
        return affine(x, -1, 1)

    def step(x, threshold=0):
        return emit("vsigmoid", [affine(x, 256, -256 * threshold)])

    def equal(a, b):
        d = sub(a, b)
        return emit("vsigmoid", [affine(mul(d, d), -1024, 64)])

    def total(x):
        return dense(x, np.ones((1, shapes[x][0]), np.float32))

    def blend(old_value, new_value, gate):
        return add(mul(old_value, inv(gate)), mul(new_value, gate))

    def first(mask):
        n = shapes[mask][0]
        preceding = dense(
            mask, np.tril(np.ones((n, n), np.float32), -1))
        return emit("vrelu", [sub(mask, preceding)])

    def vmax(a, b):
        return add(emit("vrelu", [sub(a, b)]), b)

    def vmin(a, b):
        return sub(a, emit("vrelu", [sub(a, b)]))

    def sand(a, b):
        return emit("mul", [a, b])

    def sadd(a, b):
        return emit("add", [a, b])

    def ssub(a, b):
        return emit("sub", [a, b])

    def saff(x, a=1, b=0):
        if a == 1 and b == 0:
            return x
        n = shapes[x][0]
        return conv(
            x, np.eye(n, dtype=np.float32) * a,
            np.full(n, b, np.float32) if b != 0 else None)

    def sstep(x, threshold=0):
        return emit("sigmoid", [saff(x, 256, -256 * threshold)])

    def ssum(x):
        return conv(x, np.ones((1, shapes[x][0]), np.float32))

    def gmax(x):
        return emit("gmax", [x])

    zero, unit = const(0), const(1)
    ids = const(np.arange(1, N + 1))
    endpoint_indices = const(np.arange(K))
    depth_indices = const(np.arange(N + 1))
    row_indices = const(np.arange(E))

    # Fixed linear routing maps.
    R = np.zeros((L, N), np.float32)
    A = np.zeros((E, L), np.float32)
    D4 = np.zeros((K, L), np.float32)
    OR = np.zeros((N, L), np.float32)
    for i in range(N):
        for e in range(E):
            j = i * E + e
            R[j, i] = 1
            A[e, j] = 1
            OR[i, j] = 1
            D4[4*j:4*j+4, j] = 1
    B = np.tile(np.eye(L, dtype=np.float32), (E, 1))
    C = np.repeat(np.eye(E, dtype=np.float32), L, axis=0)

    old = {}
    offset = 0
    for key, size in fields:
        old[key] = sl("mem", offset, offset + size)
        offset += size

    # Canonical camera coordinates and cell-centre sampling.
    xy = conv(
        "coords", np.array([[0, 21, 0], [21, 0, 0]], np.float32),
        [-1, -1])
    X, Y = sl(xy, 0, 1), sl(xy, 1, 2)

    lw = np.zeros((40, 3), np.float32)
    lb = np.zeros(40, np.float32)
    for i in range(20):
        lw[i, 0] = 21
        lw[20+i, 1] = 21
        lb[i] = lb[20+i] = -1-i
    distance = emit("abs", [conv("coords", lw, lb)])
    lines = emit("sigmoid", [saff(distance, -1024, 102.4)])
    line_sum = np.zeros((2, 40), np.float32)
    line_sum[0, :20] = 1
    line_sum[1, 20:] = 1
    line_hits = conv(lines, line_sum)
    lattice = sand(sl(line_hits, 0, 1), sl(line_hits, 1, 2))

    background = step(emit("gmean", ["x0"]), 0.4)
    fg_planes = emit("relu", [ssub("x0", background)])
    foreground = ssum(fg_planes)
    c0, c4 = sl(fg_planes, 0, 1), sl(fg_planes, 4, 5)
    ports = sadd(sl(fg_planes, 8, 9), sl(fg_planes, 3, 4))
    grey = gmax(sand(c4, lattice))

    adjacency = np.zeros((1, 1, 7, 7), np.float32)
    for dy, dx in [(0, 0), (-3, 0), (3, 0), (0, -3), (0, 3)]:
        adjacency[0, 0, 3+dy, 3+dx] = 1
    near_black = sstep(conv(c0, adjacency), 0.5)
    selected_ports = sand(
        ports, sadd(grey, sand(near_black, inv(grey))))
    selected = sadd(
        sand(ssub(foreground, c4), grey),
        sand(sadd(c0, sand(ports, near_black)), inv(grey)))
    selected = sand(selected, lattice)
    selected_ports = sand(selected_ports, lattice)
    active_exists = gmax(selected)

    xmin = affine(gmax(sand(selected, saff(X, -1, 21))), -1, 21)
    ymin = affine(gmax(sand(selected, saff(Y, -1, 21))), -1, 21)
    xmax = affine(gmax(sand(selected, saff(X, 1, 1))), 1, -1)
    ymax = affine(gmax(sand(selected, saff(Y, 1, 1))), 1, -1)
    width = affine(sub(xmax, xmin), 1, 1)
    height = affine(sub(ymax, ymin), 1, 1)
    marker = mul(grey, gmax(sand(c0, selected)))
    body11 = mul(grey, gmax(sand(sl(fg_planes, 11, 12), selected)))

    # Sequential connector enumeration, using a short independent loop.
    priority = ssub(saff(Y, -21, 1000), X)
    port_body = []
    destination = port_body
    local_cache = {}
    shapes["port_remaining"] = (1, 64, 64)
    shapes["port_values"] = (3 * E,)

    port_score = sand("port_remaining", priority)
    port_best = gmax(port_score)
    port_peak = sand(
        "port_remaining", sstep(ssub(port_score, port_best), -0.1))
    port_valid_now = gmax(port_peak)
    port_x_now = affine(
        gmax(sand(port_peak, saff(X, 1, 1))), 1, -1)
    port_y_now = affine(
        gmax(sand(port_peak, saff(Y, 1, 1))), 1, -1)
    port_remaining_next = ssub("port_remaining", port_peak)

    shift = np.zeros((3*E, 3*E), np.float32)
    inject = np.zeros((3*E, 3), np.float32)
    for group in range(3):
        for j in range(E - 1):
            shift[group*E+j, group*E+j+1] = 1
        inject[group*E+E-1, group] = 1
    port_values_next = add(
        dense("port_values", shift),
        dense(cat(port_x_now, port_y_now, port_valid_now), inject))

    destination = graph
    port_loop = {
        "id": "enumerate_ports", "op": "loop", "T": E,
        "state": [
            {"id": "port_remaining", "init": selected_ports},
            {"id": "port_values", "zeros": [3 * E]},
        ],
        "body": port_body,
        "next": {
            "port_remaining": port_remaining_next,
            "port_values": port_values_next,
        },
    }
    graph.append(port_loop)
    shapes["enumerate_ports.port_remaining"] = (1, 64, 64)
    shapes["enumerate_ports.port_values"] = (3 * E,)
    gathered = "enumerate_ports.port_values"
    pv = sl(gathered, 2*E, 3*E)
    missing = const(np.full(E, -100))
    local_x = blend(missing, sub(sl(gathered, 0, E), xmin), pv)
    local_y = blend(missing, sub(sl(gathered, E, 2*E), ymin), pv)
    inferred_lock = mul(body11, step(total(pv), 2.5))

    # Save one certified body pixel per local row for later clicking.
    body_pixels = emit("relu", [ssub(selected, selected_ports)])
    row_distance = emit(
        "abs", [ssub(ssub(Y, ymin), row_indices)])
    row_selector = emit("sigmoid", [saff(row_distance, -256, 64)])
    row_body = sand(body_pixels, row_selector)
    body_row_score = gmax(sand(row_body, saff(X, -1, 21)))
    body_row_valid = step(body_row_score, 0.5)
    body_local_x = blend(
        missing, sub(affine(body_row_score, -1, 21), xmin),
        body_row_valid)

    inventory = inv(old["stage"])
    growing = mul(mul(inventory, active_exists), marker)
    capture = mul(mul(inventory, active_exists), inv(marker))
    capture = mul(capture, inv(step(old["count"], N - 0.5)))
    slot = mul(equal(ids, affine(old["count"], 1, 1)), capture)
    slot_ports = dense(slot, R)

    data = dict(old)
    data["tx"] = blend(old["tx"], dense(local_x, A.T), slot_ports)
    data["ty"] = blend(old["ty"], dense(local_y, A.T), slot_ports)
    data["bx"] = blend(old["bx"], dense(body_local_x, A.T), slot_ports)
    data["w"] = blend(old["w"], width, slot)
    data["h"] = blend(old["h"], height, slot)
    lock_value = step(add(old["grow"], inferred_lock), 0.5)
    data["lock"] = blend(old["lock"], lock_value, slot)
    data["cx"] = blend(old["cx"], xmin, slot)
    data["cy"] = blend(old["cy"], ymin, slot)
    data["cr"] = blend(old["cr"], zero, slot)
    data["count"] = add(old["count"], capture)
    data["grow"] = mul(
        step(add(old["grow"], growing), 0.5), inv(capture))
    present = step(data["w"], 0.5)

    def rectangles(px, py, ww, hh, valid):
        a = sstep(ssub(X, px), -0.25)
        b = sstep(ssub(add(px, ww), X), 0.25)
        c = sstep(ssub(Y, py), -0.25)
        d = sstep(ssub(add(py, hh), Y), 0.25)
        return sand(sand(sand(a, b), sand(c, d)), valid)

    inventory_boxes = rectangles(
        data["cx"], data["cy"], data["w"], data["h"], present)
    outside_known = saff(sstep(ssum(inventory_boxes), 0.5), -1, 1)
    next_pixels = sand(sand(foreground, lattice), outside_known)
    has_next = gmax(next_pixels)
    full_inventory = step(data["count"], N - 0.5)
    finish_inventory = mul(
        capture, step(add(inv(has_next), full_inventory), 0.5))
    data["stage"] = add(old["stage"], finish_inventory)

    # Identify the sprite actually selected by the previous click.
    matches = mul(
        mul(equal(old["cx"], xmin), equal(old["cy"], ymin)),
        step(old["w"], 0.5))
    matched_id = total(mul(first(matches), ids))
    clicked = sl("aprev", 6, 7)
    recover = mul(
        mul(mul(clicked, old["stage"]), active_exists),
        step(total(matches), 0.5))
    active_id = blend(old["active"], matched_id, recover)
    active_id = blend(active_id, data["count"], finish_inventory)
    active_hot = mul(
        equal(ids, active_id), mul(old["stage"], active_exists))
    data["cx"] = blend(data["cx"], xmin, active_hot)
    data["cy"] = blend(data["cy"], ymin, active_hot)

    # Prefer an immovable-orientation object as the search root.
    locked_root = first(mul(data["lock"], present))
    root = add(
        locked_root,
        mul(first(present), inv(step(total(locked_root), 0.5))))

    previous_correct = mul(
        mul(equal(data["cx"], old["targetx"]),
            equal(data["cy"], old["targety"])),
        equal(data["cr"], old["sr"]))
    previous_done = inv(step(total(mul(
        present, inv(previous_correct))), 0.5))
    reject = mul(old["found"], step(
        add(previous_done, inv(old["feasible"])), 0.5))
    rejected_counter = blend(
        old["ct"], const(K),
        mul(equal(depth_indices, old["depth"]), reject))

    port_valid = mul(step(data["tx"], -0.5), dense(present, R))
    object_count = total(present)
    initial = {
        "sx": blend(old["sx"], const(np.zeros(N)), finish_inventory),
        "sy": blend(old["sy"], const(np.zeros(N)), finish_inventory),
        "sr": blend(old["sr"], const(np.zeros(N)), finish_inventory),
        "sd": blend(old["sd"], root, finish_inventory),
        "ct": blend(rejected_counter, const(np.zeros(N+1)),
                    finish_inventory),
        "opened": blend(
            old["opened"], mul(port_valid, dense(root, R)),
            finish_inventory),
        "depth": blend(old["depth"], unit, finish_inventory),
        "found": mul(
            mul(old["found"], inv(reject)), inv(finish_inventory)),
    }

    nonzero_rotation = const(np.tile([0, 1, 1, 1], L))
    rotation_allowed = inv(mul(
        dense(data["lock"], D4 @ R), nonzero_rotation))

    def rotation_masks(r):
        n = shapes[r][0]
        d = dense(
            r, np.tile(np.eye(n, dtype=np.float32), (4, 1)),
            -np.repeat(np.arange(4, dtype=np.float32), n))
        q = emit("vsigmoid", [affine(mul(d, d), -1024, 64)])
        return [sl(q, i*n, (i+1)*n) for i in range(4)]

    def rotation_coefficients(w, h, r):
        q0, q1, q2, q3 = rotation_masks(r)
        aa, bb = sub(q0, q2), sub(q3, q1)
        wm, hm = affine(w, 1, -1), affine(h, 1, -1)
        ox = add(mul(q1, hm), mul(q2, wm))
        oy = add(mul(q2, hm), mul(q3, wm))
        return aa, bb, ox, oy

    # Persistent depth-first connector alignment search.
    search_body = []
    destination = search_body
    local_cache = {}
    S = {}
    for key, src in initial.items():
        S[key] = "state_" + key
        shapes[S[key]] = shapes[src]

    run = mul(data["stage"], inv(S["found"]))
    unused_objects = inv(step(S["sd"], 0.5))
    unused_ports = mul(port_valid, dense(unused_objects, R))
    depth_hot = equal(depth_indices, S["depth"])
    search_counter = total(mul(S["ct"], depth_hot))
    later = step(sub(endpoint_indices, search_counter), -0.5)
    candidates = mul(
        mul(dense(unused_ports, D4), rotation_allowed), later)
    chosen = first(candidates)
    anchor_exists = step(total(S["opened"]), 0.5)
    has_candidate = mul(total(chosen), anchor_exists)
    backing = inv(has_candidate)

    chosen_endpoint = dense(chosen, D4.T)
    chosen_object = dense(chosen_endpoint, OR)
    chosen_local = dense(chosen_endpoint, A)
    chosen_rotation = dense(
        chosen, np.tile(np.arange(4, dtype=np.float32), L)[None, :])
    chosen_index = dense(
        chosen, np.arange(K, dtype=np.float32)[None, :])

    back_object = mul(equal(S["sd"], S["depth"]), present)
    process_object = add(
        mul(chosen_object, has_candidate), mul(back_object, backing))
    process_ports = dense(process_object, R)
    process_rotation = add(
        mul(chosen_rotation, has_candidate),
        mul(total(mul(S["sr"], back_object)), backing))

    aa, bb, ox, oy = rotation_coefficients(
        data["w"], data["h"], S["sr"])
    world_x = add(
        add(mul(data["tx"], dense(aa, R)),
            mul(data["ty"], dense(bb, R))),
        dense(add(ox, S["sx"]), R))
    world_y = add(
        add(mul(data["tx"], dense(neg(bb), R)),
            mul(data["ty"], dense(aa, R))),
        dense(add(oy, S["sy"]), R))

    anchor = first(S["opened"])
    anchor_x = total(mul(anchor, world_x))
    anchor_y = total(mul(anchor, world_y))
    px = dense(mul(data["tx"], process_ports), A)
    py = dense(mul(data["ty"], process_ports), A)
    pvalid = dense(mul(port_valid, process_ports), A)
    pw = total(mul(data["w"], process_object))
    ph = total(mul(data["h"], process_object))
    pa, pb, pox, poy = rotation_coefficients(
        pw, ph, process_rotation)
    rel_x = add(add(mul(px, pa), mul(py, pb)), pox)
    rel_y = add(add(mul(px, neg(pb)), mul(py, pa)), poy)

    forward_x = sub(anchor_x, total(mul(rel_x, chosen_local)))
    forward_y = sub(anchor_y, total(mul(rel_y, chosen_local)))
    process_x = add(
        mul(forward_x, has_candidate),
        mul(total(mul(S["sx"], back_object)), backing))
    process_y = add(
        mul(forward_y, has_candidate),
        mul(total(mul(S["sy"], back_object)), backing))
    trial_x, trial_y = add(rel_x, process_x), add(rel_y, process_y)

    existing = mul(
        mul(port_valid, dense(step(S["sd"], 0.5), R)),
        inv(process_ports))
    dx = sub(dense(trial_x, C), dense(world_x, B))
    dy = sub(dense(trial_y, C), dense(world_y, B))
    squared_distance = add(mul(dx, dx), mul(dy, dy))
    coincident = step(neg(squared_distance), -0.25)
    coincident = mul(
        mul(coincident, dense(pvalid, C)), dense(existing, B))
    matched_old = dense(coincident, B.T)
    matched_new = dense(coincident, C.T)

    hits_closed = step(
        total(mul(matched_old, inv(S["opened"]))), 0.5)
    excess = sub(
        sub(add(total(S["opened"]), affine(total(pvalid), 2)),
            affine(total(matched_new), 2)),
        total(unused_ports))
    impossible = step(excess, 0.5)
    accept = mul(
        mul(mul(has_candidate, inv(hits_closed)), inv(impossible)), run)
    remove = mul(mul(backing, step(S["depth"], 1.5)), run)

    new_local_open = sub(pvalid, matched_new)
    scattered_open = mul(dense(new_local_open, A.T), process_ports)
    forward_open = add(sub(S["opened"], matched_old), scattered_open)
    backward_open = mul(
        add(S["opened"], matched_old), inv(process_ports))
    next_open = blend(
        blend(S["opened"], forward_open, accept), backward_open, remove)

    changed = mul(process_object, add(accept, remove))
    next_sd = add(
        mul(S["sd"], inv(changed)),
        mul(mul(process_object, accept), affine(S["depth"], 1, 1)))
    place_gate = mul(process_object, accept)
    next_sx = blend(S["sx"], process_x, place_gate)
    next_sy = blend(S["sy"], process_y, place_gate)
    next_sr = blend(S["sr"], process_rotation, place_gate)
    next_depth = sub(add(S["depth"], accept), remove)

    counter_delta = sub(
        affine(chosen_index, 1, 1), search_counter)
    next_ct = add(S["ct"], mul(
        mul(depth_hot, mul(has_candidate, run)), counter_delta))
    clear_next = mul(
        equal(depth_indices, affine(S["depth"], 1, 1)), accept)
    next_ct = mul(next_ct, inv(clear_next))
    solved = mul(
        inv(step(total(next_open), 0.5)),
        equal(next_depth, object_count))
    next_found = add(S["found"], mul(run, solved))

    next_values = {
        "sx": next_sx, "sy": next_sy, "sr": next_sr, "sd": next_sd,
        "ct": next_ct, "opened": next_open,
        "depth": next_depth, "found": next_found,
    }
    destination = graph
    search_loop = {
        "id": "search", "op": "loop", "T": 1,
        "state": [{"id": S[k], "init": initial[k]} for k in initial],
        "body": search_body,
        "next": {S[k]: next_values[k] for k in initial},
    }
    graph.append(search_loop)
    result = {}
    for key in initial:
        name = "search." + S[key]
        shapes[name] = shapes[initial[key]]
        result[key] = name
        data[key] = name

    def grouped_max(v, groups, length):
        while length > 1:
            new_length = (length + 1) // 2
            wa = np.zeros(
                (groups * new_length, groups * length), np.float32)
            wb = np.zeros_like(wa)
            for g in range(groups):
                for j in range(new_length):
                    wa[g*new_length+j, g*length+2*j] = 1
                    wb[g*new_length+j,
                       g*length+min(2*j+1, length-1)] = 1
            v = vmax(dense(v, wa), dense(v, wb))
            length = new_length
        return v

    # Translate the relative solution into the board.
    odd = add(
        equal(result["sr"], const(1)),
        equal(result["sr"], const(3)))
    target_w = blend(data["w"], data["h"], odd)
    target_h = blend(data["h"], data["w"], odd)
    extents = cat(
        neg(result["sx"]), neg(result["sy"]),
        add(result["sx"], target_w), add(result["sy"], target_h))
    absent = inv(present)
    extents = sub(
        extents, affine(cat(absent, absent, absent, absent), 100))
    bounds = grouped_max(extents, 4, N)
    lower_x, lower_y = sl(bounds, 0, 1), sl(bounds, 1, 2)
    upper_x = affine(sl(bounds, 2, 3), -1, 20)
    upper_y = affine(sl(bounds, 3, 4), -1, 20)
    root_x = total(mul(data["cx"], root))
    root_y = total(mul(data["cy"], root))
    shift_x = vmax(lower_x, vmin(root_x, upper_x))
    shift_y = vmax(lower_y, vmin(root_y, upper_y))
    target_x = add(result["sx"], shift_x)
    target_y = add(result["sy"], shift_y)
    feasible = mul(
        step(sub(upper_x, lower_x), -0.5),
        step(sub(upper_y, lower_y), -0.5))
    data["targetx"], data["targety"] = target_x, target_y
    data["feasible"] = feasible

    # Feedback execution controller.
    correct = mul(
        mul(equal(data["cx"], target_x), equal(data["cy"], target_y)),
        equal(data["cr"], result["sr"]))
    unfinished = mul(present, inv(correct))
    wanted = first(unfinished)
    any_unfinished = step(total(unfinished), 0.5)
    wanted_id = total(mul(wanted, ids))
    executing = mul(mul(data["stage"], result["found"]), feasible)
    selected_wanted = mul(equal(wanted_id, active_id), active_exists)
    work = mul(mul(executing, any_unfinished), selected_wanted)
    execution_click = mul(
        mul(executing, any_unfinished), inv(selected_wanted))

    desired_r = total(mul(wanted, result["sr"]))
    current_r = total(mul(wanted, data["cr"]))
    rotate_needed = inv(equal(desired_r, current_r))
    rotate_action = mul(work, rotate_needed)
    translating = mul(work, inv(rotate_needed))
    delta_x = total(mul(wanted, sub(target_x, data["cx"])))
    delta_y = total(mul(wanted, sub(target_y, data["cy"])))
    x_positive = step(delta_x, 0.5)
    x_negative = step(neg(delta_x), 0.5)
    right = mul(translating, x_positive)
    left = mul(translating, x_negative)
    vertical = mul(
        translating, inv(step(add(x_positive, x_negative), 0.5)))
    down = mul(vertical, step(delta_y, 0.5))
    up = mul(vertical, step(neg(delta_y), 0.5))
    up = add(up, mul(executing, inv(any_unfinished)))
    rotate_action = add(rotate_action, growing)

    startup_reset = inv(old["boot"])
    action_features = cat(
        startup_reset, up, down, left, right, rotate_action)
    aw = np.zeros((8, 6), np.float32)
    aw[0, 0] = 3000
    for i in range(5):
        aw[i+1, i+1] = 100
    ab = np.array(
        [-1000, -1, -1, -1, -1, -1, 0, -1000], np.float32)
    dense(action_features, aw, ab, name="logits")

    # Transform the saved body pixels into current display coordinates.
    wanted_ports = dense(wanted, R)
    click_local_x = dense(mul(data["bx"], wanted_ports), A)
    click_valid = step(click_local_x, -0.5)
    wanted_w = total(mul(data["w"], wanted))
    wanted_h = total(mul(data["h"], wanted))
    ca, cb, cox, coy = rotation_coefficients(
        wanted_w, wanted_h, current_r)
    click_world_x = add(
        add(add(mul(click_local_x, ca), mul(row_indices, cb)), cox),
        total(mul(data["cx"], wanted)))
    click_world_y = add(
        add(add(mul(click_local_x, neg(cb)), mul(row_indices, ca)), coy),
        total(mul(data["cy"], wanted)))
    click_distance = sadd(
        emit("abs", [ssub(X, click_world_x)]),
        emit("abs", [ssub(Y, click_world_y)]))
    certified_points = ssum(sand(
        emit("sigmoid", [saff(click_distance, -256, 64)]),
        click_valid))

    current_odd = add(
        equal(data["cr"], const(1)),
        equal(data["cr"], const(3)))
    current_w = blend(data["w"], data["h"], current_odd)
    current_h = blend(data["h"], data["w"], current_odd)
    current_boxes = rectangles(
        data["cx"], data["cy"], current_w, current_h, present)
    wanted_box = ssum(sand(current_boxes, wanted))
    box_count = ssum(current_boxes)
    unique = sstep(saff(box_count, -1, 1.5))
    quality = sadd(
        sadd(saff(selected, -5, 15), saff(unique, 100)),
        saff(certified_points, 50))
    execution_map = sand(
        sand(sand(sand(wanted_box, foreground), lattice), quality),
        priority)
    inventory_click = mul(
        mul(inventory, inv(growing)), has_next)
    inventory_map = sand(next_pixels, priority)
    emit("add", [
        sand(inventory_map, inventory_click),
        sand(execution_map, execution_click)
    ], name="click")

    rotate_gate = mul(wanted, mul(work, rotate_needed))
    advanced_rotation = affine(data["cr"], 1, 1)
    advanced_rotation = sub(
        advanced_rotation, affine(step(advanced_rotation, 3.5), 4))
    data["cr"] = blend(data["cr"], advanced_rotation, rotate_gate)
    inventory_active = add(data["count"], has_next)
    active_next = blend(active_id, inventory_active, inventory_click)
    active_next = blend(active_next, data["count"], finish_inventory)
    data["active"] = blend(active_next, wanted_id, execution_click)

    # Clear all working memory on the initial reset, but retain its flag.
    working_memory = cat(*(data[k] for k, _ in fields if k != "boot"))
    working_memory = mul(working_memory, old["boot"])
    emit("vconcat", [working_memory, unit], name="mem_next")

    # Static dead-code elimination for both independent loops.
    loop_external = {}
    for loop in (port_loop, search_loop):
        body_index = {nd["id"]: nd for nd in loop["body"]}
        body_live, external = set(), set()

        def visit_body(name):
            if name in body_live:
                return
            nd = body_index.get(name)
            if nd is None:
                external.add(name)
                return
            body_live.add(name)
            for src in nd["in"]:
                visit_body(src)

        for src in loop["next"].values():
            visit_body(src)
        loop["body"] = [
            nd for nd in loop["body"] if nd["id"] in body_live]
        state_ids = {s["id"] for s in loop["state"]}
        loop_external[loop["id"]] = external - state_ids

    index = {nd["id"]: nd for nd in graph}
    live = set()

    def visit(name):
        if "." in name:
            name = name.split(".", 1)[0]
        if name in live or name not in index:
            return
        live.add(name)
        nd = index[name]
        if nd["op"] == "loop":
            for state in nd["state"]:
                if "init" in state:
                    visit(state["init"])
            for src in loop_external[name]:
                visit(src)
        else:
            for src in nd["in"]:
                visit(src)

    for output in ("logits", "click", "mem_next"):
        visit(output)
    graph = [nd for nd in graph if nd["id"] in live]

    all_nodes = []
    for nd in graph:
        all_nodes.append(nd)
        if nd["op"] == "loop":
            all_nodes.extend(nd["body"])
    used_parameters = {
        nd[key] for nd in all_nodes
        for key in ("w", "b") if key in nd
    }
    params = {k: v for k, v in params.items() if k in used_parameters}

    def node_macs(nd):
        op = nd["op"]
        if op == "loop":
            return nd["T"] * sum(node_macs(x) for x in nd["body"])
        if op == "conv":
            return int(params[nd["w"]].size) * 4096
        if op == "dense":
            return int(params[nd["w"]].size)
        if op in ("add", "sub", "mul", "max2", "min2", "vadd", "vmul"):
            return int(np.prod(shapes[nd["id"]]))
        if op in ("rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"):
            return int(np.prod(shapes[nd["in"][0]]))
        return 0

    def node_evaluations(nd):
        if nd["op"] == "loop":
            return 1 + nd["T"] * len(nd["body"])
        return 1

    outside = [nd for nd in graph if nd is not search_loop]
    outside_macs = sum(node_macs(nd) for nd in outside)
    outside_evaluations = sum(node_evaluations(nd) for nd in outside)
    search_body = search_loop["body"]
    body_macs = sum(node_macs(nd) for nd in search_body)
    evaluation_limit = (
        10000 - outside_evaluations - 1) // len(search_body)
    mac_limit = (
        1500000000 - outside_macs) // max(body_macs, 1)
    search_loop["T"] = int(min(256, evaluation_limit, mac_limit))

    parameter_count = sum(a.size for a in params.values())
    spatial_channels = 19
    for nd in all_nodes:
        if nd["op"] != "loop" and len(shapes[nd["id"]]) == 3:
            spatial_channels += shapes[nd["id"]][0]
        elif nd["op"] == "loop":
            for state in nd["state"]:
                if len(shapes[state["id"]]) == 3:
                    spatial_channels += shapes[state["id"]][0]

    assert M <= 256, ("memory", M)
    assert parameter_count <= 150000, ("parameters", parameter_count)
    assert len(all_nodes) <= 1000, ("nodes", len(all_nodes))
    assert spatial_channels <= 4096, ("channels", spatial_channels)
    assert search_loop["T"] >= 1, ("unrolling", search_loop["T"])
    assert sum(node_macs(nd) for nd in graph) <= 1500000000
    assert sum(node_evaluations(nd) for nd in graph) <= 10000
    assert all(len(s) != 3 or s[0] <= 512 for s in shapes.values())
    assert all(len(s) != 1 or s[0] <= 4096 for s in shapes.values())

    return {"graph": graph, "memory_size": M, "params": params}
