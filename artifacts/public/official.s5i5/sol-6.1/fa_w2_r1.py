import numpy as np


def build():
    P = {}
    G = []
    current = G
    serial = [0]
    pcache = {}
    caches = {}
    expressions = {}
    constants = {}
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "avail": (8,),
        "aprev": (8,),
    }

    colors = [0, 6, 7, 8, 9, 10, 11, 12, 14]
    C = len(colors)
    N = 2 * C + 2
    A = 3 * C
    M = A + 1
    shapes["mem"] = (M,)

    def param(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in pcache:
            name = "p%d" % len(P)
            pcache[key] = name
            P[name] = a
        return pcache[key]

    def nd(op, ins, name=None, **fields):
        key = (op, tuple(ins), tuple(sorted(fields.items())))
        cache = caches.setdefault(id(current), {})
        global_cache = caches.setdefault(id(G), {})
        if name is None:
            if key in global_cache:
                return global_cache[key]
            if key in cache:
                return cache[key]
            name = "n%d" % serial[0]
            serial[0] += 1
        d = {"id": name, "op": op, "in": list(ins)}
        d.update(fields)
        current.append(d)
        ss = [shapes[x] for x in ins]
        if op == "conv":
            sh = (P[fields["w"]].shape[0], 64, 64)
        elif op == "dense":
            sh = (P[fields["w"]].shape[0],)
        elif op in ("gmax", "gmean"):
            sh = (ss[0][0],)
        elif op in ("concat", "vconcat"):
            sh = (sum(s[0] for s in ss),) + ss[0][1:]
        elif op in ("slice", "vslice"):
            sh = (fields["stop"] - fields["start"],) + ss[0][1:]
        elif op in ("add", "sub", "mul", "max2", "min2"):
            sh = (max(s[0] for s in ss), 64, 64)
        elif op in ("vadd", "vmul"):
            sh = (max(s[0] for s in ss),)
        else:
            sh = ss[0]
        shapes[name] = sh
        cache[key] = name
        return name

    def conv(x, w, b=None, name=None):
        f = {"w": param(w)}
        if b is not None:
            f["b"] = param(b)
        return nd("conv", [x], name=name, **f)

    def dense(x, w, b=None, name=None):
        f = {"w": param(w)}
        if b is not None:
            f["b"] = param(b)
        return nd("dense", [x], name=name, **f)

    def sl(x, start, stop):
        return nd(
            "vslice" if len(shapes[x]) == 1 else "slice",
            [x], start=start, stop=stop
        )

    def cat(xs):
        if len(xs) == 1:
            return xs[0]
        return nd(
            "vconcat" if len(shapes[xs[0]]) == 1 else "concat", xs
        )

    def k(value):
        nonlocal current
        value = float(value)
        if value not in constants:
            saved = current
            current = G
            z = dense("avail", np.zeros((1, 8), np.float32), [value])
            constants[value] = z
            expressions[z] = ({}, np.array([value], np.float32))
            current = saved
        return constants[value]

    def affine(terms, bias=0.0):
        """Fold only small vector expressions; never form large dense matrices."""
        n = max(shapes[x][0] for x, _ in terms)

        if n > N:
            pieces = []
            for x, coefficient in terms:
                pieces.append(
                    x if coefficient == 1
                    else nd("vmul", [x, k(coefficient)])
                )
            out = pieces[0]
            for x in pieces[1:]:
                out = nd("vadd", [out, x])
            if bias:
                out = nd("vadd", [out, k(bias)])
            return out

        def assemble(expand):
            matrices = {}
            b = np.full(n, bias, np.float32)
            for x, coefficient in terms:
                if expand and x in expressions:
                    mm, bb = expressions[x]
                elif x in expressions and not expressions[x][0]:
                    mm, bb = expressions[x]
                else:
                    mm = {x: np.eye(shapes[x][0], dtype=np.float32)}
                    bb = np.zeros(shapes[x][0], np.float32)
                if bb.size == 1 and n != 1:
                    bb = np.repeat(bb, n)
                b += np.float32(coefficient) * bb
                for root, matrix in mm.items():
                    if matrix.shape[0] == 1 and n != 1:
                        matrix = np.repeat(matrix, n, axis=0)
                    value = np.float32(coefficient) * matrix
                    if root in matrices:
                        matrices[root] = matrices[root] + value
                    else:
                        matrices[root] = value.copy()
            matrices = {
                root: matrix for root, matrix in matrices.items()
                if np.any(matrix != 0)
            }
            return matrices, b

        matrices, b = assemble(True)
        if sum(shapes[r][0] for r in matrices) > 2 * N:
            matrices, b = assemble(False)

        roots = sorted(matrices)
        if roots:
            inp = cat(roots)
            w = np.concatenate([matrices[r] for r in roots], axis=1)
            out = dense(inp, w, b)
        else:
            out = dense("avail", np.zeros((n, 8), np.float32), b)
        expressions[out] = (matrices, b)
        return out

    def va(x, y):
        return affine([(x, 1), (y, 1)])

    def vs(x, y):
        return affine([(x, 1), (y, -1)])

    def vm(x, y):
        return nd("vmul", [x, y])

    def scale(x, value):
        if len(shapes[x]) == 1:
            return affine([(x, value)])
        return nd("mul", [x, k(value)])

    def neg(x):
        return scale(x, -1)

    def vabs(x):
        return va(nd("vrelu", [x]), nd("vrelu", [neg(x)]))

    def binary(x):
        # Exact zero below the tolerance; tanh rounds to one on binary inputs.
        z = affine([(x, 32)], -8)
        return nd("vtanh", [nd("vrelu", [z])])

    def near_zero(distance):
        return binary(affine([(distance, -4)], 2))

    def sa(x, y):
        return nd("add", [x, y])

    def ss(x, y):
        return nd("sub", [x, y])

    def sm(x, y):
        return nd("mul", [x, y])

    def clip(x):
        return nd("min2", [nd("relu", [x]), k(1)])

    def total(x):
        return dense(x, np.ones((1, shapes[x][0]), np.float32))

    def state(init):
        name = "s%d" % serial[0]
        serial[0] += 1
        shapes[name] = shapes[init]
        return {"id": name, "init": init}

    def loop(states, body, nxt, T):
        name = "L%d" % serial[0]
        serial[0] += 1
        d = {
            "id": name, "op": "loop", "T": int(T),
            "state": states, "body": body, "next": nxt,
        }
        G.append(d)
        result = {}
        for s in states:
            ref = name + "." + s["id"]
            shapes[ref] = shapes[s["id"]]
            result[s["id"]] = ref
        return result, d

    # Coordinates in pixel units.
    w = np.zeros((2, 3, 1, 1), np.float32)
    w[0, 1, 0, 0] = 63
    w[1, 0, 0, 0] = 63
    xy = conv("coords", w)
    X, Y = sl(xy, 0, 1), sl(xy, 1, 2)
    rank = sa(scale(Y, 64), sa(X, k(1)))
    inverse_rank = ss(k(4097), rank)

    # Resize-control dividers and rotation-control crosses.
    uw = np.zeros((A, 16, 5, 5), np.float32)
    ub = np.empty(A, np.float32)
    for ci, color in enumerate(colors):
        h, v, r = ci, C + ci, 2 * C + ci
        for dy in (-1, 0, 1):
            uw[h, 3, 2 + dy, 2] = 1
        uw[h, 4, 2, 1] = 1
        uw[h, 4, 2, 3] = 1
        uw[h, color, 1, 0] = 1
        uw[h, color, 1, 4] = 1
        ub[h] = -6
        uw[v] = uw[h].transpose(0, 2, 1)
        ub[v] = -6
        for dy, dx in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            uw[r, color, 2 + dy, 2 + dx] = 1
        for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            uw[r, 2, 2 + dy, 2 + dx] = 1
        ub[r] = -8

    ui = nd("relu", [conv("x0", uw, ub)])
    uh, uv, ur = sl(ui, 0, C), sl(ui, C, 2 * C), sl(ui, 2 * C, A)
    sum_colors = np.ones((1, C, 1, 1), np.float32)
    line_h = np.zeros((1, 1, 7, 7), np.float32)
    line_h[0, 0, 3, :] = 1
    line_v = line_h.transpose(0, 1, 3, 2).copy()
    square7 = np.ones((1, 1, 7, 7), np.float32)
    square5 = np.ones((1, 1, 5, 5), np.float32)
    harea = conv(conv(conv(uh, sum_colors), line_h), square7)
    varea = conv(conv(conv(uv, sum_colors), line_v), square7)
    rarea = conv(conv(ur, sum_colors), square5)
    ui_area = clip(sa(sa(harea, varea), rarea))

    color_w = np.zeros((C, 16, 1, 1), np.float32)
    for ci, color in enumerate(colors):
        color_w[ci, color, 0, 0] = 1
    rod_colors = sm(conv("x0", color_w), ss(k(1), ui_area))

    # Pivot orientation: down, right, up, left.
    pw = np.zeros((4 * C, 16, 3, 3), np.float32)
    check_w = np.zeros((4, 1, 3, 3), np.float32)
    for d in range(4):
        if d in (0, 2):
            offset = -1 if d == 0 else 1
            base_positions = [(offset, dx) for dx in (-1, 0, 1)]
            color_positions = [(0, dx) for dx in (-1, 0, 1)]
        else:
            offset = -1 if d == 1 else 1
            base_positions = [(dy, offset) for dy in (-1, 0, 1)]
            color_positions = [(dy, 0) for dy in (-1, 0, 1)]
        for yy, xx in base_positions:
            check_w[d, 0, 1 + yy, 1 + xx] = 1
        for ci, color in enumerate(colors):
            out = d * C + ci
            for yy, xx in base_positions:
                pw[out, 3, 1 + yy, 1 + xx] += 1
                pw[out, 13, 1 + yy, 1 + xx] += 1
            for yy, xx in color_positions:
                pw[out, color, 1 + yy, 1 + xx] += 1
                pw[out, 13, 1 + yy, 1 + xx] += 1

    raw_orientation = nd("relu", [
        conv("x0", pw, np.full(4 * C, -5, np.float32))
    ])
    checks = clip(conv(
        sl("x0", 3, 4), check_w, np.full(4, -1, np.float32)
    ))
    check_replicate = np.zeros((4 * C, 4, 1, 1), np.float32)
    for d in range(4):
        check_replicate[d * C:(d + 1) * C, d, 0, 0] = 1
    orientation = sm(raw_orientation, conv(checks, check_replicate))

    merge_w = np.zeros((C, 4 * C, 1, 1), np.float32)
    replicate_w = np.zeros((4 * C, C, 1, 1), np.float32)
    horizontal_w = np.zeros_like(merge_w)
    vertical_w = np.zeros_like(merge_w)
    dx_w = np.zeros((C, 4 * C), np.float32)
    dy_w = np.zeros_like(dx_w)
    for d in range(4):
        for ci in range(C):
            merge_w[ci, d * C + ci, 0, 0] = 1
            replicate_w[d * C + ci, ci, 0, 0] = 1
            target = horizontal_w if d in (1, 3) else vertical_w
            target[ci, d * C + ci, 0, 0] = 1
            dx_w[ci, d * C + ci] = [0, 1, 0, -1][d]
            dy_w[ci, d * C + ci] = [1, 0, -1, 0][d]
    pivots = clip(conv(orientation, merge_w))

    def seeds(mask):
        hi = nd("gmax", [sm(mask, rank)])
        lo = nd("gmax", [sm(mask, inverse_rank)])
        first = sm(mask, clip(ss(
            k(2), scale(nd("abs", [ss(inverse_rank, lo)]), 4)
        )))
        last = sm(mask, clip(ss(
            k(2), scale(nd("abs", [ss(rank, hi)]), 4)
        )))
        distinct = binary(affine([(hi, 4), (lo, 4)], -16390))
        return first, sm(last, distinct)

    first_seed, last_seed = seeds(pivots)
    cross_c = np.zeros((C, C, 3, 3), np.float32)
    for ci in range(C):
        cross_c[ci, ci, 1, :] = 1
        cross_c[ci, ci, :, 1] = 1
    cross_one = np.zeros((1, 1, 3, 3), np.float32)
    cross_one[0, 0, 1, :] = 1
    cross_one[0, 0, :, 1] = 1

    def component(seed, allowed, kernel):
        nonlocal current
        st = state(seed)
        body = []
        current = body
        expanded = clip(conv(st["id"], kernel))
        nxt = sm(expanded, allowed)
        current = G
        result, _ = loop([st], body, {st["id"]: nxt}, 64)
        return result[st["id"]]

    def extrema(mask):
        xmax = nd("gmax", [sm(mask, X)])
        ymax = nd("gmax", [sm(mask, Y)])
        xmin = affine([
            (nd("gmax", [sm(mask, ss(k(64), X))]), -1)
        ], 64)
        ymin = affine([
            (nd("gmax", [sm(mask, ss(k(64), Y))]), -1)
        ], 64)
        return xmin, xmax, ymin, ymax

    marker_plane = sl("x0", 13, 14)

    def extract_rods(seed):
        ob = sm(orientation, conv(seed, replicate_w))
        ov = nd("gmax", [ob])
        dx, dy = dense(ov, dx_w), dense(ov, dy_w)
        bx = nd("gmax", [sm(seed, X)])
        by = nd("gmax", [sm(seed, Y)])
        present = nd("gmax", [seed])
        hs = conv(ob, horizontal_w)
        vs_ = conv(ob, vertical_w)
        strip = clip(conv(
            sa(nd("rowmax", [hs]), nd("colmax", [vs_])), cross_c
        ))
        allowed = sm(strip, clip(sa(rod_colors, marker_plane)))
        comp = component(seed, allowed, cross_c)
        xmin, xmax, ymin, ymax = extrema(comp)
        right = affine([(xmax, 1), (bx, -1)], -1)
        left = affine([(xmin, 1), (bx, -1)], 1)
        down = affine([(ymax, 1), (by, -1)], -1)
        up = affine([(ymin, 1), (by, -1)], 1)
        tx = va(bx, va(
            vm(nd("vrelu", [dx]), right),
            vm(nd("vrelu", [neg(dx)]), left)
        ))
        ty = va(by, va(
            vm(nd("vrelu", [dy]), down),
            vm(nd("vrelu", [neg(dy)]), up)
        ))
        return bx, by, tx, ty, dx, dy, present

    rfirst = extract_rods(first_seed)
    rlast = extract_rods(last_seed)

    white = sl("x0", 1, 2)
    wf, _ = seeds(white)
    whi = nd("gmax", [sm(white, rank)])
    wl = sm(white, clip(ss(
        k(2), scale(nd("abs", [ss(rank, whi)]), 4)
    )))

    def extract_passive(seed):
        comp = component(seed, white, cross_one)
        xlo, xhi, ylo, yhi = extrema(comp)
        return (
            affine([(xlo, 1)], 1),
            affine([(ylo, 1)], 1),
            affine([(xhi, 1)], -1),
            affine([(yhi, 1)], -1),
            nd("gmax", [seed]),
        )

    pfirst, plast = extract_passive(wf), extract_passive(wl)
    pdiff = k(0)
    for i in range(4):
        pdiff = va(pdiff, vabs(vs(pfirst[i], plast[i])))
    plast_present = vm(plast[4], binary(
        affine([(pdiff, 4)], -2)
    ))

    zero = k(0)
    bx = cat([rfirst[0], rlast[0], pfirst[0], plast[0]])
    by = cat([rfirst[1], rlast[1], pfirst[1], plast[1]])
    tx = cat([rfirst[2], rlast[2], pfirst[2], plast[2]])
    ty = cat([rfirst[3], rlast[3], pfirst[3], plast[3]])
    dx = cat([rfirst[4], rlast[4], zero, zero])
    dy = cat([rfirst[5], rlast[5], zero, zero])
    present = cat([rfirst[6], rlast[6], pfirst[4], plast_present])

    marker_w = np.zeros((1, 16, 3, 3), np.float32)
    marker_w[0, 13, 1, 1] = 1
    for yy, xx in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        marker_w[0, 13, yy, xx] = -1
    marker_centres = nd("relu", [conv("x0", marker_w)])

    goal_w = np.zeros((1, 16, 3, 3), np.float32)
    for yy, xx in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        goal_w[0, 13, yy, xx] = 1
    goals = nd("relu", [conv("x0", goal_w, [-3])])
    wall = sl("x0", 15, 16)
    grown_wall = clip(conv(wall, np.ones((1, 1, 3, 3), np.float32)))
    open_space = ss(k(1), grown_wall)
    goal_drive = scale(goals, 100)

    shift_w = np.zeros((4, 1, 7, 7), np.float32)
    for i, (yy, xx) in enumerate([(0, 3), (6, 3), (3, 0), (3, 6)]):
        shift_w[i, 0, yy, xx] = 1
    fs = state(goal_drive)
    body = []
    current = body
    shifted = conv(fs["id"], shift_w)
    neighbour = sl(shifted, 0, 1)
    for i in range(1, 4):
        neighbour = nd("max2", [neighbour, sl(shifted, i, i + 1)])
    spread = sm(nd("relu", [ss(neighbour, k(1))]), open_space)
    fn = nd("max2", [spread, goal_drive])
    current = G
    result, field_loop = loop([fs], body, {fs["id"]: fn}, 40)
    field = result[fs["id"]]

    # Pairwise broadcasting matrices.
    EP = np.zeros((N * N, N), np.float32)
    EC = np.zeros_like(EP)
    for i in range(N):
        for j in range(N):
            EP[i * N + j, i] = 1
            EC[i * N + j, j] = 1
    SC, SP = EC.T.copy(), EP.T.copy()

    def ep(x):
        return dense(x, EP)

    def ec(x):
        return dense(x, EC)

    pair_present = vm(ep(present), ec(present))
    offdiag = (1 - np.eye(N, dtype=np.float32)).reshape(-1)
    offdiag_v = dense(
        "avail", np.zeros((N * N, 8), np.float32), offdiag
    )
    pair_present = vm(pair_present, offdiag_v)
    expected_x = va(tx, scale(dx, 3))
    expected_y = va(ty, scale(dy, 3))
    attachment_distance = va(
        vabs(vs(ec(bx), ep(expected_x))),
        vabs(vs(ec(by), ep(expected_y)))
    )
    adjacency = vm(near_zero(attachment_distance), pair_present)

    def tree_mv(x, matrix, collapse=SC):
        return dense(vm(ep(x), matrix), collapse)

    # One loop visits all roots and records their attachment descendants.
    period = 4
    first_root = np.zeros(N, np.float32)
    first_root[0] = 1
    root_init = dense(
        "avail", np.zeros((N, 8), np.float32), first_root
    )
    closure_init = dense(
        "avail", np.zeros((N * N, 8), np.float32),
        np.zeros(N * N, np.float32)
    )
    clock_init_array = np.zeros(period, np.float32)
    clock_init_array[0] = 1
    clock_init = dense(
        "avail", np.zeros((period, 8), np.float32), clock_init_array
    )
    root_cycle = np.zeros((N, N), np.float32)
    for i in range(N):
        root_cycle[(i + 1) % N, i] = 1
    clock_cycle = np.zeros((period, period), np.float32)
    for i in range(period):
        clock_cycle[(i + 1) % period, i] = 1

    sr, sh = state(root_init), state(root_init)
    sc, st = state(closure_init), state(clock_init)
    body = []
    current = body
    reached = binary(va(sh["id"], tree_mv(sh["id"], adjacency)))
    row = vm(ep(sr["id"]), ec(reached))
    closure_next = binary(va(sc["id"], row))
    advance = sl(st["id"], period - 1, period)
    stay = affine([(advance, -1)], 1)
    rotated_root = dense(sr["id"], root_cycle)
    next_root = va(vm(sr["id"], stay), vm(rotated_root, advance))
    next_reached = va(vm(reached, stay), vm(rotated_root, advance))
    next_clock = dense(st["id"], clock_cycle)
    current = G
    result, _ = loop(
        [sr, sh, sc, st], body,
        {
            sr["id"]: next_root,
            sh["id"]: next_reached,
            sc["id"]: closure_next,
            st["id"]: next_clock,
        },
        N * period
    )
    reach = result[sc["id"]]

    def descendants(x):
        return tree_mv(x, reach)

    def boxes(bxx, byy, txx, tyy):
        hx = scale(vabs(vs(bxx, txx)), 0.5)
        hy = scale(vabs(vs(byy, tyy)), 0.5)
        mx = affine([(bxx, 0.5), (txx, 0.5)])
        my = affine([(byy, 0.5), (tyy, 0.5)])
        return vs(mx, hx), va(mx, hx), vs(my, hy), va(my, hy)

    def footprint(bounds):
        xlo, xhi, ylo, yhi = bounds
        xd = sa(nd("relu", [ss(xlo, X)]), nd("relu", [ss(X, xhi)]))
        yd = sa(nd("relu", [ss(ylo, Y)]), nd("relu", [ss(Y, yhi)]))
        return sm(clip(ss(k(2), xd)), clip(ss(k(2), yd)))

    initial_foot = footprint(boxes(bx, by, tx, ty))
    attached = vm(
        nd("gmax", [sm(initial_foot, marker_centres)]), present
    )
    useful = binary(dense(vm(reach, ec(attached)), SP))
    inactive = vs(present, useful)

    def point_field(px, py):
        distance = sa(
            nd("abs", [ss(X, px)]), nd("abs", [ss(Y, py)])
        )
        value = nd("gmax", [ss(field, scale(distance, 100))])
        return value, distance

    initial_values, initial_distance = point_field(tx, ty)
    initial_metric = total(vm(initial_values, attached))
    slack = ss(ss(field, scale(initial_distance, 0.25)), initial_values)
    per_marker_path = sm(clip(sa(k(1), scale(slack, 2))), attached)
    corridor = clip(conv(
        per_marker_path, np.ones((1, N, 1, 1), np.float32)
    ))
    corridor = clip(conv(
        corridor, np.ones((1, 1, 3, 3), np.float32)
    ))

    parent_dx, parent_dy = tree_mv(dx, adjacency), tree_mv(dy, adjacency)
    parent_exists = binary(dense(adjacency, SC))
    backward_distance = va(
        vabs(va(dy, parent_dx)), vabs(vs(parent_dy, dx))
    )
    double_turn = vm(near_zero(backward_distance), parent_exists)

    old_tabu = sl("mem", 0, A)
    old_metric = sl("mem", A, A + 1)
    stalled = binary(affine([
        (old_metric, 10), (initial_metric, -10)
    ], 1))
    tabu = vm(old_tabu, stalled)

    # Compact click maps: decrease, increase, rotate.
    hw = np.zeros((2 * C, C, 5, 5), np.float32)
    vw = np.zeros_like(hw)
    for ci in range(C):
        hw[ci, ci, 2, 4] = 1
        hw[C + ci, ci, 2, 0] = 1
        vw[ci, ci, 4, 2] = 1
        vw[C + ci, ci, 0, 2] = 1
    resize_maps = sa(conv(uh, hw), conv(uv, vw))
    action_maps = cat([resize_maps, ur])
    action_available = nd("gmax", [action_maps])

    selector_w = np.zeros((N, A), np.float32)
    for a in range(A):
        ci = a % C
        selector_w[ci, a] = 1
        selector_w[C + ci, a] = 1
    delta_w = np.zeros((1, A), np.float32)
    delta_w[0, :C] = -3
    delta_w[0, C:2 * C] = 3
    rotate_w = np.zeros((1, A), np.float32)
    rotate_w[0, 2 * C:] = 1
    shrink_w = np.zeros((1, A), np.float32)
    shrink_w[0, :C] = 1
    cycle_w = np.zeros((A, A), np.float32)
    for a in range(A):
        cycle_w[(a + 1) % A, a] = 1
    first_action = np.zeros(A, np.float32)
    first_action[0] = 1
    proposal_init = dense(
        "avail", np.zeros((A, 8), np.float32), first_action
    )
    best_hot_init = dense(
        "avail", np.zeros((A, 8), np.float32), np.zeros(A, np.float32)
    )
    proposal_state = state(proposal_init)
    score_state = state(k(-1000000))
    hot_state = state(best_hot_init)
    span = va(vabs(vs(tx, bx)), vabs(vs(ty, by)))
    shrinkable = binary(affine([(span, 4)], -2))

    body = []
    current = body
    proposal = proposal_state["id"]
    selected = vm(dense(proposal, selector_w), present)
    delta = dense(proposal, delta_w)
    rotating = dense(proposal, rotate_w)
    shrinking = dense(proposal, shrink_w)
    effective = vm(selected, affine([
        (vm(shrinking, affine([(shrinkable, -1)], 1)), -1)
    ], 1))
    own_sx = vm(vm(effective, dx), delta)
    own_sy = vm(vm(effective, dy), delta)
    all_sx, all_sy = descendants(own_sx), descendants(own_sy)
    influence = vm(descendants(effective), rotating)
    twice = vm(descendants(vm(effective, double_turn)), rotating)
    pivot_x = descendants(vm(effective, bx))
    pivot_y = descendants(vm(effective, by))

    def transform(px, py, base):
        first_dx = affine([
            (py, 1), (pivot_y, -1), (pivot_x, 1), (px, -1)
        ])
        first_dy = affine([
            (pivot_x, 1), (px, -1), (pivot_y, 1), (py, -1)
        ])
        second_dx = affine([
            (pivot_x, 1), (pivot_y, 1), (px, -1), (py, -1)
        ])
        second_dy = affine([
            (pivot_y, 1), (px, 1), (pivot_x, -1), (py, -1)
        ])
        sx = vs(all_sx, own_sx) if base else all_sx
        sy = vs(all_sy, own_sy) if base else all_sy
        nx = affine([
            (px, 1), (sx, 1),
            (vm(influence, first_dx), 1), (vm(twice, second_dx), 1)
        ])
        ny = affine([
            (py, 1), (sy, 1),
            (vm(influence, first_dy), 1), (vm(twice, second_dy), 1)
        ])
        return nx, ny

    cbx, cby = transform(bx, by, True)
    ctx, cty = transform(tx, ty, False)
    bounds = boxes(cbx, cby, ctx, cty)
    xlo, xhi, ylo, yhi = bounds

    def overlap(lo, hi):
        gap1 = nd("vrelu", [vs(ep(lo), ec(hi))])
        gap2 = nd("vrelu", [vs(ec(lo), ep(hi))])
        return binary(affine([(gap1, -1), (gap2, -1)], 3))

    pair_collision = vm(
        vm(overlap(xlo, xhi), overlap(ylo, yhi)), pair_present
    )
    pair_count = scale(total(pair_collision), 0.5)
    candidate_foot = footprint(bounds)
    wall_hit = nd("gmax", [sm(candidate_foot, wall)])
    boundary = affine([
        (nd("vrelu", [affine([(xlo, -1)], 1)]), 1),
        (nd("vrelu", [affine([(xhi, 1)], -62)]), 1),
        (nd("vrelu", [affine([(ylo, -1)], 1)]), 1),
        (nd("vrelu", [affine([(yhi, 1)], -62)]), 1),
    ])
    illegal = va(
        pair_count, total(vm(va(wall_hit, boundary), present))
    )
    legal = near_zero(illegal)
    changed = binary(total(effective))
    button_exists = binary(total(vm(proposal, action_available)))
    legal = vm(vm(legal, changed), button_exists)

    candidate_values, _ = point_field(ctx, cty)
    metric = total(vm(candidate_values, attached))
    path_area = scale(nd("gmean", [sm(candidate_foot, corridor)]), 4096)
    path_cost = scale(total(vm(path_area, inactive)), 0.04)
    candidate_span = va(vabs(vs(ctx, cbx)), vabs(vs(cty, cby)))
    size_cost = scale(total(vm(candidate_span, inactive)), 0.003)
    tabu_cost = total(vm(proposal, tabu))
    score = affine([
        (metric, 1), (path_cost, -1), (size_cost, -1),
        (tabu_cost, -1), (legal, 10000)
    ], -10000)
    choose = binary(affine([
        (score, 10000), (score_state["id"], -10000)
    ]))
    not_choose = affine([(choose, -1)], 1)
    best_score_next = va(
        vm(score_state["id"], not_choose), vm(score, choose)
    )
    best_hot_next = va(
        vm(hot_state["id"], not_choose), vm(proposal, choose)
    )
    proposal_next = dense(proposal, cycle_w)
    current = G
    result, _ = loop(
        [proposal_state, score_state, hot_state], body,
        {
            proposal_state["id"]: proposal_next,
            score_state["id"]: best_score_next,
            hot_state["id"]: best_hot_next,
        }, A
    )
    chosen = result[hot_state["id"]]

    conv(
        sm(action_maps, chosen), np.ones((1, A, 1, 1), np.float32),
        name="click"
    )
    logits_b = np.zeros(8, np.float32)
    logits_b[6] = 10
    dense(
        "avail", np.zeros((8, 8), np.float32), logits_b, name="logits"
    )
    new_tabu = va(
        scale(old_tabu, 0.8), vm(scale(chosen, 0.75), stalled)
    )
    nd("vconcat", [new_tabu, initial_metric], name="mem_next")

    # Remove construction intermediates superseded by affine folding.
    def prune(nodes, needed_outputs):
        needed = set(needed_outputs)
        kept = []
        for d in reversed(nodes):
            if d["op"] == "loop":
                exports = {
                    d["id"] + "." + s["id"] for s in d["state"]
                }
                if not (exports & needed):
                    continue
                required_body = set(d["next"].values())
                body_kept, body_external = prune(d["body"], required_body)
                d = dict(d)
                d["body"] = body_kept
                local_states = {s["id"] for s in d["state"]}
                needed.update(body_external - local_states)
                for s in d["state"]:
                    if "init" in s:
                        needed.add(s["init"])
                kept.append(d)
            elif d["id"] in needed:
                kept.append(d)
                needed.update(d["in"])
        kept.reverse()
        defined = {d["id"] for d in kept}
        return kept, needed - defined

    G, _ = prune(G, {"click", "logits", "mem_next"})

    def evaluations(nodes):
        count = 0
        for d in nodes:
            if d["op"] == "loop":
                count += 1 + d["T"] * evaluations(d["body"])
            else:
                count += 1
        return count

    # Allocate the remaining evaluation budget to distance propagation.
    for d in G:
        if d["id"] == field_loop["id"]:
            body_cost = evaluations(d["body"])
            other_cost = evaluations(G) - d["T"] * body_cost
            d["T"] = max(1, min(40, (10000 - other_cost) // body_cost))
            break

    used_params = set()

    def collect(nodes):
        for d in nodes:
            if d["op"] == "loop":
                collect(d["body"])
            else:
                if "w" in d:
                    used_params.add(d["w"])
                if "b" in d:
                    used_params.add(d["b"])

    collect(G)
    P = {name: value for name, value in P.items() if name in used_params}

    return {"graph": G, "memory_size": M, "params": P}
