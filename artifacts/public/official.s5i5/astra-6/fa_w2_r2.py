import numpy as np


def build():
    P, cache, graph = {}, {}, []
    stack = [graph]
    shapes = {
        "x0": (16, 64, 64), "coords": (3, 64, 64),
        "avail": (8,), "aprev": (8,), "mem": (216,)
    }
    serial = [0]

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def par(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p" + str(len(P))
            cache[key] = name
            P[name] = a
        return cache[key]

    def emit(op, ins, name=None, **kw):
        name = name or fresh()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(kw)
        stack[-1].append(nd)
        ss = [shapes[x] for x in ins]
        if op == "conv":
            sh = (P[kw["w"]].shape[0], 64, 64)
        elif op == "dense":
            sh = (P[kw["w"]].shape[0],)
        elif op in ("concat", "vconcat"):
            sh = (sum(s[0] for s in ss),) + ss[0][1:]
        elif op in ("slice", "vslice"):
            sh = (kw["stop"] - kw["start"],) + ss[0][1:]
        elif op in ("gmax", "gmean"):
            sh = (ss[0][0],)
        elif op in ("add", "sub", "mul", "max2", "min2"):
            sh = (max(s[0] for s in ss), 64, 64)
        elif op in ("vadd", "vmul"):
            sh = (max(s[0] for s in ss),)
        else:
            sh = ss[0]
        shapes[name] = sh
        return name

    def dense(x, w, b=None, name=None):
        kw = {"w": par(w)}
        if b is not None:
            kw["b"] = par(b)
        return emit("dense", [x], name=name, **kw)

    def conv(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        kw = {"w": par(w)}
        if b is not None:
            kw["b"] = par(b)
        return emit("conv", [x], name=name, **kw)

    zero = dense("avail", np.zeros((1, 8)), [0])
    constants = {}

    def C(value):
        a = np.asarray(value, np.float32).reshape(-1)
        key = (a.shape, a.tobytes())
        if key not in constants:
            stack.append(graph)
            constants[key] = dense(zero, np.zeros((len(a), 1)), a)
            stack.pop()
        return constants[key]

    def va(a, b):
        return emit("vadd", [a, b])

    def vm(a, b):
        return emit("vmul", [a, b])

    def vn(a):
        return vm(a, C(-1))

    def vs(a, b):
        return va(a, vn(b))

    def vr(a):
        return emit("vrelu", [a])

    def af(a, scale=1, bias=0):
        n = shapes[a][0]
        if n <= 24:
            return dense(a, np.eye(n, dtype=np.float32) * scale,
                         np.full(n, bias, np.float32))
        y = a if scale == 1 else vm(a, C(scale))
        return y if bias == 0 else va(y, C(bias))

    def ab(a):
        return va(vr(a), vr(vn(a)))

    def clip(a):
        if shapes[a][0] <= 24:
            return af(vr(af(vr(a), -1, 1)), -1, 1)
        return vs(vr(a), vr(va(a, C(-1))))

    def eq(a, b, tolerant=False):
        return clip(af(ab(vs(a, b)), -1, 2 if tolerant else 1))

    def vmax(a, b):
        return va(b, vr(vs(a, b)))

    def total(a):
        return dense(a, np.ones((1, shapes[a][0]), np.float32))

    def vc(xs):
        return emit("vconcat", xs)

    def sl(a, start, stop):
        return emit("slice" if len(shapes[a]) == 3 else "vslice",
                    [a], start=start, stop=stop)

    def sa(a, b):
        return emit("add", [a, b])

    def sm(a, b):
        return emit("mul", [a, b])

    def ss(a, b):
        return emit("sub", [a, b])

    def sr(a):
        return emit("relu", [a])

    def sab(a):
        return emit("abs", [a])

    def saf(a, scale=1, bias=0):
        n = shapes[a][0]
        if n <= 2:
            return conv(a, np.eye(n, dtype=np.float32) * scale,
                        np.full(n, bias, np.float32))
        y = a if scale == 1 else sm(a, C(scale))
        return y if bias == 0 else sa(y, C(bias))

    def sclip(a):
        return emit("min2", [sr(a), C(1)])

    def gm(a):
        return emit("gmax", [a])

    def sample(mask, field):
        return af(emit("gmean", [sm(mask, field)]), 4096)

    def loop(T, initial, body_fn):
        lid = fresh("L")
        states, ids = [], []
        for init in initial:
            sid = fresh("s")
            ids.append(sid)
            if isinstance(init, str):
                states.append({"id": sid, "init": init})
                shapes[sid] = shapes[init]
            else:
                states.append({"id": sid, "zeros": list(init)})
                shapes[sid] = tuple(init)
        body = []
        stack.append(body)
        nxt = body_fn(ids)
        stack.pop()
        nd = {
            "id": lid, "op": "loop", "T": int(T),
            "state": states, "body": body,
            "next": dict(zip(ids, nxt))
        }
        stack[-1].append(nd)
        outs = []
        for sid in ids:
            out = lid + "." + sid
            shapes[out] = shapes[sid]
            outs.append(out)
        return outs, nd

    X = conv("coords", [[0, 63, 0]])
    Y = conv("coords", [[63, 0, 0]])
    priority = conv("coords", [[-4032, -63, 4096]])

    colors = [6, 7, 8, 9, 10, 11, 12, 14]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    R, U, B = 10, 4, 2
    N = R + U + B
    A = 3 * len(colors)

    # Colored anchor blocks, including markers drawn over their centres.
    aw = np.zeros((32, 16, 5, 5), np.float32)
    clabel, xlabel, ylabel = [], [], []
    for ci, color_id in enumerate(colors):
        for di, (dxi, dyi) in enumerate(directions):
            k = 4 * ci + di
            for yy in (-1, 1):
                for xx in (-1, 1):
                    cc = 3 if xx * dxi + yy * dyi == -1 else color_id
                    aw[k, cc, yy + 2, xx + 2] = 1
            aw[k, color_id, 2, 2] = 1
            aw[k, 13, 2, 2] = 1
            clabel.append(color_id)
            xlabel.append(dxi)
            ylabel.append(dyi)
    anchors = sr(conv("x0", aw, np.full(32, -4)))
    anchor_any = conv(anchors, np.ones((1, 32), np.float32))
    color_field = conv(anchors, [clabel])
    dx_field = conv(anchors, [xlabel])
    dy_field = conv(anchors, [ylabel])

    def extract(mask, count):
        first = np.zeros(count, np.float32)
        first[0] = 1
        shift = np.zeros((count, count), np.float32)
        for i in range(count):
            shift[(i + 1) % count, i] = 1

        def body(st):
            remaining, result, index = st
            ranked = sm(remaining, priority)
            chosen = sm(
                sr(saf(sab(ss(ranked, gm(ranked))), -2, 1)), remaining)
            return [
                ss(remaining, chosen),
                sa(result, sm(chosen, index)),
                dense(index, shift)
            ]

        out, _ = loop(count, [mask, (count, 64, 64), C(first)], body)
        return out[1]

    rod_seeds = extract(anchor_any, R)
    rex = gm(rod_seeds)
    rc = sample(rod_seeds, color_field)
    rdx = sample(rod_seeds, dx_field)
    rdy = sample(rod_seeds, dy_field)
    sx, sy = sample(rod_seeds, X), sample(rod_seeds, Y)

    tile_colors = colors + [1]
    tw = np.zeros((len(tile_colors), 16, 3, 3), np.float32)
    for k, color_id in enumerate(tile_colors):
        for yy in (0, 2):
            for xx in (0, 2):
                tw[k, color_id, yy, xx] = 1
                tw[k, 3, yy, xx] = 1
    tiles = sr(conv("x0", tw, np.full(len(tile_colors), -3)))
    tile_code = conv(tiles, [tile_colors])
    allowed = sm(sr(saf(sab(ss(tile_code, rc)), -1, 1)), rex)

    offx, offy = ss(X, sx), ss(Y, sy)
    forward = sa(sm(offx, rdx), sm(offy, rdy))
    lateral = sab(ss(sm(offx, rdy), sm(offy, rdx)))
    ray = sm(sclip(sa(forward, C(1))), sclip(saf(lateral, -4, 2)))
    allowed = sm(allowed, ray)

    cross = np.zeros((1, 1, 7, 7), np.float32)
    for yy, xx in ((3, 3), (0, 3), (6, 3), (3, 0), (3, 6)):
        cross[0, 0, yy, xx] = 1

    def flood(seeds, permitted, count):
        def body(st):
            spread = emit("concat", [
                conv(sl(st[0], i, i + 1), cross) for i in range(count)
            ])
            return [emit("min2", [spread, permitted])]
        out, nd = loop(21, [seeds], body)
        return out[0], nd

    rods, rod_flood = flood(rod_seeds, allowed, R)

    def geometry(components, existence):
        xmin = vm(af(gm(sm(components, saf(X, -1, 64))), -1, 64),
                  existence)
        xmax = gm(sm(components, X))
        ymin = vm(af(gm(sm(components, saf(Y, -1, 64))), -1, 64),
                  existence)
        ymax = gm(sm(components, Y))
        xx = af(va(xmin, xmax), 0.5)
        yy = af(va(ymin, ymax), 0.5)
        ww = vm(af(vs(xmax, xmin), 1, 3), existence)
        hh = vm(af(vs(ymax, ymin), 1, 3), existence)
        return xx, yy, ww, hh

    rcx, rcy, rw, rh = geometry(rods, rex)
    rlen = vmax(rw, rh)
    endpoint_x = va(rcx, vm(rdx, af(rlen, 0.5, 1.5)))
    endpoint_y = va(rcy, vm(rdy, af(rlen, 0.5, 1.5)))

    # Neutral bars are found at colored-rod endpoints, not by arbitrary
    # connected components. Their axis is perpendicular to their parent.
    endpoint_dist = sa(sab(ss(X, endpoint_x)), sab(ss(Y, endpoint_y)))
    endpoint_mask = sm(sclip(saf(endpoint_dist, -4, 2)), rex)
    neutral_pixels = sl("x0", 1, 2)
    neutral_contacts = sm(endpoint_mask, neutral_pixels)
    contact_any = sclip(conv(neutral_contacts, np.ones((1, R))))
    horizontal_field = conv(sm(neutral_contacts, ab(rdy)),
                            np.ones((1, R)))
    neutral_seeds = extract(contact_any, U)
    uex = gm(neutral_seeds)
    usx, usy = sample(neutral_seeds, X), sample(neutral_seeds, Y)
    horizontal = clip(sample(neutral_seeds, horizontal_field))
    vertical = vs(uex, horizontal)
    neutral_lateral = sa(
        sm(sab(ss(Y, usy)), horizontal),
        sm(sab(ss(X, usx)), vertical))
    neutral_allowed = sm(
        sm(sclip(saf(neutral_lateral, -4, 2)), uex),
        sl(tiles, len(colors), len(colors) + 1))
    neutrals, neutral_flood = flood(neutral_seeds, neutral_allowed, U)
    ucx, ucy, uw, uh = geometry(neutrals, uex)

    dot_w = np.zeros((1, 16, 3, 3), np.float32)
    dot_w[0, 13, 1, 1] = 1
    for yy in (0, 2):
        for xx in (0, 2):
            dot_w[0, 13, yy, xx] = -1
    dot_map = sr(conv("x0", dot_w))
    goal_w = np.zeros((1, 16, 3, 3), np.float32)
    for yy, xx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        goal_w[0, 13, yy, xx] = 1
    goal_map = sr(conv("x0", goal_w, [-3]))

    dot_seeds, goal_seeds = extract(dot_map, B), extract(goal_map, B)
    dex, gex = gm(dot_seeds), gm(goal_seeds)
    dcx, dcy = sample(dot_seeds, X), sample(dot_seeds, Y)
    gcx, gcy = sample(goal_seeds, X), sample(goal_seeds, Y)

    zR, zU, zB = C(np.zeros(R)), C(np.zeros(U)), C(np.zeros(B))
    exists = vc([rex, uex, dex])
    solid = vc([rex, uex, zB])
    marker = vc([zR, zU, dex])
    neutral = vc([zR, uex, zB])
    dx, dy = vc([rdx, zU, zB]), vc([rdy, zU, zB])
    color = vc([rc, uex, zB])
    cx, cy = vc([rcx, ucx, dcx]), vc([rcy, ucy, dcy])
    width = vc([rw, uw, af(dex, 3)])
    height = vc([rh, uh, af(dex, 3)])
    adx, ady = ab(dx), ab(dy)
    rod = va(adx, ady)
    length = vmax(width, height)
    half_minus, half_plus = af(length, 0.5, -1.5), af(length, 0.5, 1.5)
    bx, by = vs(cx, vm(dx, half_minus)), vs(cy, vm(dy, half_minus))
    endx, endy = va(cx, vm(dx, half_plus)), va(cy, vm(dy, half_plus))

    er = np.zeros((N * N, N), np.float32)
    ec = np.zeros_like(er)
    reduce_child = np.zeros((N, N * N), np.float32)
    for i in range(N):
        for j in range(N):
            p = i * N + j
            er[p, i] = 1
            ec[p, j] = 1
            reduce_child[j, p] = 1

    def rows(v):
        return dense(v, er)

    def cols(v):
        return dense(v, ec)

    def receive(v):
        return dense(v, reduce_child)

    def close_pair(ax, ay, bx_, by_):
        return clip(af(va(ab(vs(ax, bx_)), ab(vs(ay, by_))), -2, 2))

    def inside(px, py, ox, oy, ww, hh):
        ux = vr(va(vs(ab(vs(px, ox)), af(ww, 0.5)), C(0.25)))
        uy = vr(va(vs(ab(vs(py, oy)), af(hh, 0.5)), C(0.25)))
        return clip(af(va(ux, uy), -4, 1.5))

    rod_link = close_pair(rows(endx), rows(endy), cols(bx), cols(by))
    neutral_link = inside(
        rows(endx), rows(endy), cols(cx), cols(cy),
        cols(width), cols(height))
    dot_link = inside(
        cols(cx), cols(cy), rows(cx), rows(cy),
        rows(width), rows(height))
    links = va(
        vm(va(vm(rod_link, cols(rod)),
              vm(neutral_link, cols(neutral))), rows(rod)),
        vm(vm(dot_link, cols(marker)), rows(solid)))
    links = vm(vm(links, cols(exists)),
               C((1 - np.eye(N)).reshape(-1)))

    reach_rows, closure_nodes = [], []
    for i in range(N):
        initial = np.zeros(N, np.float32)
        initial[i] = 1

        def closure_body(st):
            return [clip(va(st[0], receive(vm(links, rows(st[0])))))]

        out, nd = loop(5, [C(initial)], closure_body)
        reach_rows.append(out[0])
        closure_nodes.append(nd)
    descendants = vc(reach_rows)

    def descend(v):
        return receive(vm(descendants, rows(v)))

    pdx = receive(vm(links, rows(dx)))
    pdy = receive(vm(links, rows(dy)))
    skip_turn = clip(vs(vm(pdy, dx), vm(pdx, dy)))
    can_shrink = clip(af(length, 1, -3.5))
    bx_minus_by, bx_plus_by = vs(bx, by), va(bx, by)
    twice_bx, twice_by = af(bx, 2), af(by, 2)
    h_minus_w, w_minus_h = vs(height, width), vs(width, height)

    glyphs = [
        [[1, 1, 1], [1, 0, 0], [1, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 0, 1], [1, 0, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    ]
    ui_w = np.zeros((5 * len(colors), 16, 3, 3), np.float32)
    ui_mix = np.zeros((A, 5 * len(colors)), np.float32)
    action_colors, grows, shrinks, turns = [], [], [], []
    for i, c in enumerate(colors):
        for g, template in enumerate(glyphs):
            for yy in range(3):
                for xx in range(3):
                    ui_w[5 * i + g,
                         c if template[yy][xx] else 4, yy, xx] = 1
        ui_mix[3 * i, 5 * i] = 1
        ui_mix[3 * i, 5 * i + 2] = 1
        ui_mix[3 * i + 1, 5 * i + 1] = 1
        ui_mix[3 * i + 1, 5 * i + 3] = 1
        ui_mix[3 * i + 2, 5 * i + 4] = 1
        action_colors += [c, c, c]
        shrinks += [1, 0, 0]
        grows += [0, 1, 0]
        turns += [0, 0, 1]
    ui = conv(sr(conv("x0", ui_w, np.full(5 * len(colors), -8))),
              ui_mix)
    ui_exists = gm(ui)

    wall = sl("x0", 15, 16)
    dilation = conv(wall, np.ones((1, 1, 3, 3), np.float32))
    free = ss(C(1), emit("min2", [dilation, C(1)]))
    for edge in (X, Y, saf(X, -1, 63), saf(Y, -1, 63)):
        free = sm(free, sclip(edge))

    wave_w = np.zeros((4 * B, B, 3, 3), np.float32)
    for d, (yy, xx) in enumerate(((0, 1), (2, 1), (1, 0), (1, 2))):
        for g in range(B):
            wave_w[d * B + g, g, yy, xx] = 1

    def wave_body(st):
        shifted = conv(st[0], wave_w)
        a = emit("max2", [sl(shifted, 0, B), sl(shifted, B, 2 * B)])
        b = emit("max2", [sl(shifted, 2 * B, 3 * B),
                          sl(shifted, 3 * B, 4 * B)])
        advanced = sm(sr(ss(emit("max2", [a, b]), C(1 / 256))), free)
        return [emit("max2", [advanced, goal_seeds])]

    wave, wave_node = loop(128, [goal_seeds], wave_body)
    distance = sa(sab(ss(X, gcx)), sab(ss(Y, gcy)))
    fallback = sm(sr(saf(distance, -1 / 2048, 0.12)), gex)
    value_field = emit("max2", [wave[0], fallback])
    value_field = sa(value_field, sm(goal_seeds, C(0.02)))
    assignment_w = np.array(
        [[1, 0, 0, 1], [0, 1, 1, 0]], dtype=np.float32)

    def endpoint_value(points):
        vals = [gm(sm(sl(points, i, i + 1), value_field))
                for i in range(B)]
        choices = dense(vc(vals), assignment_w)
        return vmax(sl(choices, 0, 1), sl(choices, 1, 2))

    c2 = vm(color, color)
    c3 = vm(c2, color)
    hx = af(va(c2, af(color, 3)), 3, 21)
    hy = va(c3, af(color, 5, 11))
    hw, hh = af(c2, 11, 13), af(c2, 7, 17)

    def fingerprint(xx, yy, ww, hhh, ddx, ddy):
        terms = va(va(vm(xx, hx), vm(yy, hy)),
                   va(vm(ww, hw), vm(hhh, hh)))
        orient = af(va(vm(ddx, hw), vm(ddy, hh)), 5)
        return total(va(terms, orient))

    current_hash = fingerprint(cx, cy, width, height, dx, dy)
    history = sl("mem", 0, 64)
    action_history = sl("mem", 64, 128)
    bad_history = sl("mem", 128, 192)
    usage = sl("mem", 192, 216)
    same_config = eq(history, current_hash, tolerant=True)

    first64 = np.zeros(64, np.float32)
    first64[0] = 1
    last_noop = eq(sl(history, 0, 1), current_hash, tolerant=True)
    bad_effective = clip(va(bad_history, vm(C(first64), last_noop)))

    first_action = np.zeros(A, np.float32)
    first_action[0] = 1
    action_shift = np.zeros((A, A), np.float32)
    for i in range(A):
        action_shift[(i + 1) % A, i] = 1
    action_numbers = np.arange(1, A + 1, dtype=np.float32)[None, :]

    def candidate_body(st):
        index, scores = st
        cc = dense(index, [action_colors])
        grow = dense(index, [grows])
        shrink = dense(index, [shrinks])
        turn = dense(index, [turns])
        number = dense(index, action_numbers)
        selected = vm(eq(color, cc), rod)
        delta = vm(selected, af(vs(grow, vm(shrink, can_shrink)), 3))
        lx, ly = vm(delta, dx), vm(delta, dy)

        rotating = vm(selected, turn)
        s2 = vm(rotating, skip_turn)
        s1 = vs(rotating, s2)
        r1, r2 = descend(s1), descend(s2)
        qx = va(lx, va(vm(s1, bx_minus_by), vm(s2, twice_bx)))
        qy = va(ly, va(vm(s1, bx_plus_by), vm(s2, twice_by)))
        linear = af(va(r1, af(r2, 2)), -1, 1)
        nx = vs(va(va(vm(linear, cx), vm(r1, cy)), descend(qx)),
                af(lx, 0.5))
        ny = vs(va(vs(vm(linear, cy), vm(r1, cx)), descend(qy)),
                af(ly, 0.5))
        nw = va(width, va(vm(delta, adx), vm(r1, h_minus_w)))
        nh = va(height, va(vm(delta, ady), vm(r1, w_minus_h)))
        ndx = va(vm(linear, dx), vm(r1, dy))
        ndy = vs(vm(linear, dy), vm(r1, dx))

        ax, ay = sab(ss(X, nx)), sab(ss(Y, ny))
        ix = sclip(sm(ss(af(nw, 0.5), ax), C(4)))
        iy = sclip(sm(ss(af(nh, 0.5), ay), C(4)))
        bodies = sm(sm(ix, iy), solid)
        occupancy = conv(bodies, np.ones((1, N), np.float32))
        overlap = gm(sr(ss(occupancy, C(1))))
        wall_hit = gm(sm(occupancy, wall))
        ox = vr(va(ab(af(nx, 1, -31.5)), af(nw, 0.5, -32)))
        oy = vr(va(ab(af(ny, 1, -31.5)), af(nh, 0.5, -32)))
        outside = total(vm(va(ox, oy), solid))
        collision = va(va(overlap, wall_hit), outside)

        point_distance = sl(sa(ax, ay), R + U, N)
        points = sm(sclip(saf(point_distance, -4, 2)), dex)
        value = endpoint_value(points)
        predicted_hash = fingerprint(nx, ny, nw, nh, ndx, ndy)
        tabu = total(eq(history, predicted_hash, tolerant=True))
        unchanged = eq(current_hash, predicted_hash, tolerant=True)
        edge_matches = vm(same_config, eq(action_history, number))
        repeated = total(edge_matches)
        bad = total(vm(edge_matches, bad_effective))
        present = total(vm(index, ui_exists))
        count = total(vm(index, usage))
        area = total(vm(vm(nw, nh), solid))

        score = value
        score = va(score, af(area, -0.00004))
        score = va(score, af(count, -0.00002))
        score = va(score, af(tabu, -0.003))
        score = va(score, af(unchanged, -5))
        score = va(score, af(collision, -20))
        score = va(score, af(repeated, -0.04))
        score = va(score, af(bad, -100))
        score = va(score, af(present, 1000, -1000))
        return [dense(index, action_shift), va(scores, vm(index, score))]

    candidates, candidate_node = loop(
        A, [C(first_action), (A,)], candidate_body)
    scores = candidates[1]

    raw_click = conv(sm(ui, scores), np.ones((1, A)))
    support = conv(ui, np.full((1, A), 10000), [-10000])
    click = emit("add", [raw_click, support], name="click")
    maxima = sr(saf(sab(ss(click, gm(click))), -1000000, 1))
    ranked = sm(maxima, priority)
    chosen = sm(maxima, sr(saf(sab(ss(ranked, gm(ranked))), -2, 1)))
    used = gm(sm(ui, chosen))
    chosen_number = dense(used, action_numbers)
    usage_next = va(usage, used)

    history_shift = np.zeros((64, 64), np.float32)
    history_shift[1:, :63] = np.eye(63, dtype=np.float32)
    insert = np.zeros((64, 1), np.float32)
    insert[0, 0] = 1
    history_next = va(dense(history, history_shift),
                      dense(current_hash, insert))
    actions_next = va(dense(action_history, history_shift),
                      dense(chosen_number, insert))
    bad_next = dense(bad_effective, history_shift)
    emit("vconcat", [history_next, actions_next, bad_next, usage_next],
         name="mem_next")

    logits_bias = np.zeros(8, np.float32)
    logits_bias[6] = 10
    dense("avail", np.zeros((8, 8), np.float32),
          logits_bias, name="logits")

    # Static resource accounting; this does not execute the network.
    def evaluations(nodes):
        return sum(
            nd["T"] * evaluations(nd["body"])
            if nd["op"] == "loop" else 1 for nd in nodes
        )

    def macs(nodes):
        result = 0
        for nd in nodes:
            op = nd["op"]
            if op == "loop":
                result += nd["T"] * macs(nd["body"])
            elif op == "conv":
                result += int(np.prod(P[nd["w"]].shape)) * 4096
            elif op == "dense":
                result += int(np.prod(P[nd["w"]].shape))
            elif op in ("add", "sub", "mul", "max2", "min2",
                        "vadd", "vmul"):
                result += int(np.prod(shapes[nd["id"]]))
            elif op in ("rowmax", "colmax", "rowsum", "colsum",
                        "gmax", "gmean"):
                result += int(np.prod(shapes[nd["in"][0]]))
        return result

    def shorten(nd, minimum):
        while nd["T"] > minimum and (
                evaluations(graph) > 9800 or macs(graph) > 1490000000):
            nd["T"] -= 1

    shorten(wave_node, 48)
    for nd in closure_nodes:
        shorten(nd, 4)
    shorten(wave_node, 1)

    return {"graph": graph, "memory_size": 216, "params": P}
