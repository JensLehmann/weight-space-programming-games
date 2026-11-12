import numpy as np


def build():
    P, G = {}, []
    serial = [0]
    shifts = {}

    def node(op, ins=None, graph=None, name=None, **kw):
        graph = G if graph is None else graph
        if name is None:
            name = "n%d" % serial[0]
            serial[0] += 1
        d = {"id": name, "op": op}
        if ins is not None:
            d["in"] = list(ins)
        d.update(kw)
        graph.append(d)
        return name

    def par(a):
        name = "p%d" % len(P)
        P[name] = np.asarray(a, np.float32)
        return name

    def conv(x, w, b=None, graph=None, name=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return node("conv", [x], graph, name, **fields)

    def dense(x, w, b=None, graph=None, name=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return node("dense", [x], graph, name, **fields)

    def lin(x, w, b=None, graph=None):
        w = np.asarray(w, np.float32)
        return conv(x, w[:, :, None, None], b, graph)

    def sl(x, a, b, graph=None):
        return node("slice", [x], graph, start=a, stop=b)

    def vsl(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def cat(xs, graph=None):
        return node("concat", xs, graph)

    def vc(xs):
        return node("vconcat", xs)

    def maximum(xs, graph=None):
        z = xs[0]
        for x in xs[1:]:
            z = node("max2", [z, x], graph)
        return z

    def plus(xs, graph=None):
        z = xs[0]
        for x in xs[1:]:
            z = node("add", [z, x], graph)
        return z

    def shift(x, channels, dy, dx, graph=None):
        k = 2 * max(abs(dy), abs(dx)) + 1
        if k == 1:
            return x
        key = (channels, dy, dx)
        if key not in shifts:
            w = np.zeros((channels, channels, k, k), np.float32)
            for c in range(channels):
                w[c, c, k // 2 + dy, k // 2 + dx] = 1
            shifts[key] = par(w)
        return node("conv", [x], graph, w=shifts[key])

    def ray_shift(x, dy, dx, graph):
        # Two blocks reduce MACs for the long ray-propagation loop.
        a = shift(sl(x, 0, 2, graph), 2, dy, dx, graph)
        b = shift(sl(x, 2, 4, graph), 2, dy, dx, graph)
        return cat([a, b], graph)

    one = sl("coords", 2, 3)
    zero = lin(one, [[0]])

    def clip(x, graph=None):
        return node("min2", [
            node("relu", [x], graph), one
        ], graph)

    def inv(x, graph=None):
        return node("sub", [one, x], graph)

    def constant(value, length=1):
        return dense(
            "avail", np.zeros((length, 8), np.float32),
            np.full(length, value, np.float32)
        )

    minus1 = constant(-1, 4)
    minus4 = constant(-4, 4)
    minus5 = constant(-5, 4)

    colours = {
        c: sl("x0", c, c + 1)
        for c in (3, 5, 6, 9, 10, 11, 12, 13, 14, 15)
    }

    def expand(x, height, width=None, origin=-1):
        width = height if width is None else width
        w = np.zeros((1, 1, 7, 7), np.float32)
        for y in range(origin, origin + height):
            for xx in range(origin, origin + width):
                w[0, 0, 3 - y, 3 - xx] = 1
        return clip(conv(x, w))

    def footprint(x, size, channels, bias=0, sign=1):
        w = np.zeros((channels, channels, 7, 7), np.float32)
        for c in range(channels):
            for y in range(-1, size - 1):
                for xx in range(-1, size - 1):
                    w[c, c, 3 + y, 3 + xx] = sign
        return conv(x, w, np.full(channels, bias, np.float32))

    # Exact door detectors, independent of player orientation and size.
    door_template = np.array([
        [9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 9, 9, 9, 9],
    ], np.int32)
    wd = np.zeros((4, 16, 7, 7), np.float32)
    door_shapes = []
    for r in range(4):
        template = np.rot90(door_template, -r)
        h, w = template.shape
        door_shapes.append((h, w))
        for y in range(h):
            for x in range(w):
                wd[r, template[y, x], y + 1, x + 1] = 1
    door_anchors = node("relu", [conv("x0", wd, [-29] * 4)])
    door_parts = [
        expand(sl(door_anchors, r, r + 1), h, w, -2)
        for r, (h, w) in enumerate(door_shapes)
    ]
    goal = maximum(door_parts)

    # Player anchors are upper-left + (1,1).
    wp = np.zeros((8, 16, 7, 7), np.float32)
    bp = np.zeros(8, np.float32)
    for si, size in enumerate((2, 4)):
        for direction in range(4):
            o = 4 * si + direction
            for y in range(size):
                for x in range(size):
                    if direction == 0:
                        c = 9 if y < size // 2 else 10
                    elif direction == 1:
                        c = 10 if y < size // 2 else 9
                    elif direction == 2:
                        c = 9 if x < size // 2 else 10
                    else:
                        c = 10 if x < size // 2 else 9
                    wp[o, c, y + 2, x + 2] = 1
            bp[o] = 1 - size * size
    player_matches = node("relu", [conv("x0", wp, bp)])

    no_door_big = inv(clip(footprint(goal, 4, 1)))
    big_orientations = node("mul", [
        sl(player_matches, 4, 8), no_door_big
    ])
    pb = lin(big_orientations, np.ones((1, 4), np.float32))
    big_area = expand(pb, 4)

    small_exclusion = maximum([
        clip(footprint(goal, 2, 1)),
        clip(footprint(big_area, 2, 1))
    ])
    small_orientations = node("mul", [
        sl(player_matches, 0, 4), inv(small_exclusion)
    ])
    ps = lin(small_orientations, np.ones((1, 4), np.float32))
    orientations = cat([small_orientations, big_orientations])
    players = cat([ps, pb])
    player_sizes = node("gmax", [players])
    facing = dense(
        node("gmax", [orientations]),
        np.concatenate([np.eye(4), np.eye(4)], axis=1)
    )

    # Detect each fire switch and its corresponding removable gate.
    wt = np.zeros((2, 16, 7, 7), np.float32)
    for o, interior in enumerate((13, 12)):
        for y in range(4):
            for x in range(4):
                c = 6 if y in (0, 3) or x in (0, 3) else interior
                wt[o, c, y + 2, x + 2] = 1
    switches = node("relu", [conv("x0", wt, [-15, -15])])
    ta = expand(sl(switches, 0, 1), 4)
    tb = expand(sl(switches, 1, 2), 4)
    remove_a = maximum([ta, colours[13]])
    remove_b = maximum([tb, colours[12]])

    # Spell availability is read from UI icons.
    ui_near = clip(conv(colours[3], np.ones((1, 1, 7, 7))))
    icon_pixels = node("mul", [
        cat([colours[15], colours[11], colours[6]]), ui_near
    ])
    spells_available = node("gmax", [icon_pixels])
    have_resize = vsl(spells_available, 0, 1)
    have_teleport = vsl(spells_available, 1, 2)
    have_fire = vsl(spells_available, 2, 3)

    # Small teleport pads.
    ws = np.zeros((1, 16, 7, 7), np.float32)
    for y in (-1, 0):
        for x in (-1, 0):
            ws[0, 15, y + 3, x + 3] = 1
    for y in range(-2, 2):
        for x in range(-2, 2):
            if y in (-1, 0) and x in (-1, 0):
                continue
            ws[0, 15, y + 3, x + 3] = -4
    small_pads = node("mul", [
        node("relu", [conv("x0", ws, [-3])]), inv(ui_near)
    ])

    def large_marker(highlight):
        w = np.zeros((1, 16, 7, 7), np.float32)
        for y in (-1, 2):
            for x in (-1, 2):
                w[0, 11, y + 3, x + 3] = 1
        count = 4
        if highlight:
            for y in (-2, 3):
                for x in (-2, 3):
                    w[0, 11, y + 3, x + 3] = 1
            count += 4
        for y in (0, 1):
            for x in (0, 1):
                for c in (9, 10, 15):
                    w[0, c, y + 3, x + 3] = -8
        return node("mul", [
            node("relu", [conv("x0", w, [1 - count])]),
            inv(ui_near)
        ])

    regular_pads = large_marker(False)
    highlighted_pads = large_marker(True)
    no_highlight = dense(
        node("gmax", [highlighted_pads]), [[-1]], [1]
    )
    large_pads = maximum([
        highlighted_pads,
        node("mul", [regular_pads, no_highlight])
    ])

    # The frame around a small teleport pad is a tangible obstacle.
    frame_source = maximum([small_pads, ps])
    wf = np.zeros((1, 1, 7, 7), np.float32)
    for y in (-2, 1):
        for x in (-2, 1):
            wf[0, 0, 3 - y, 3 - x] = 1
    small_frame = node("mul", [
        clip(conv(frame_source, wf)), colours[11]
    ])

    # Four hypothetical worlds: current, A removed, B removed, both removed.
    wbase = np.zeros((1, 16), np.float32)
    for c in (0, 1, 3, 4, 5, 6, 7, 8, 12, 13):
        wbase[0, c] = 1
    base_wall = plus([lin("x0", wbase), goal, small_frame])
    world_transform = np.array([
        [1, 0, 0],
        [1, -1, 0],
        [1, 0, -1],
        [1, -1, -1],
    ], np.float32)
    walls = lin(
        cat([base_wall, remove_a, remove_b]), world_transform
    )

    wbounds = np.array([
        [126, 0, 0], [0, 126, 0],
        [-126, 0, 0], [0, -126, 0]
    ], np.float32)
    bounds = clip(conv(
        "coords", wbounds[:, :, None, None], [-1, -1, 123, 123]
    ))
    small_bound = node("min2", [sl(bounds, 0, 1), sl(bounds, 1, 2)])
    big_bound = node("min2", [
        small_bound,
        node("min2", [sl(bounds, 2, 3), sl(bounds, 3, 4)])
    ])
    free_s = node("mul", [
        node("relu", [footprint(walls, 2, 4, 1, -1)]),
        small_bound
    ])
    free_b = node("mul", [
        node("relu", [footprint(walls, 4, 4, 1, -1)]),
        big_bound
    ])
    seed_s = lin(
        clip(footprint(goal, 2, 1)), np.full((4, 1), 1000)
    )
    seed_b = lin(
        clip(footprint(goal, 4, 1)), np.full((4, 1), 1000)
    )

    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    free4, blocked4 = [], []
    for dy, dx in directions:
        f = shift(shift(free_b, 4, dy, dx), 4, dy, dx)
        free4.append(f)
        blocked4.append(inv(f))

    # Ordered growth offsets: the engine chooses the first fitting one.
    growth = []
    already = zero
    for dx in (-2, -1, 0):
        for dy in (-2, -1, 0):
            fits = shift(free_b, 4, dy, dx)
            selected = node("mul", [fits, inv(already)])
            selected = node("mul", [selected, have_resize])
            growth.append((dy, dx, selected))
            already = node("max2", [already, fits])

    # Fire rays ignore the traversable firing bars.
    wray = np.zeros((1, 16), np.float32)
    for c in (0, 1, 3, 5, 7, 8):
        wray[0, c] = 1
    ray_base = plus([lin("x0", wray), remove_a, remove_b])
    ray_walls = lin(
        cat([ray_base, remove_a, remove_b]), world_transform
    )
    ray_free = clip(inv(ray_walls))
    sa = node("mul", [
        lin(ta, [[1], [0], [1], [0]]), have_fire
    ])
    sb = node("mul", [
        lin(tb, [[1], [1], [0], [0]]), have_fire
    ])

    ray_body, ray_states, ray_next = [], [], {}
    ray_names = {}
    for typ, source in (("a", sa), ("b", sb)):
        for d, (dy, dx) in enumerate(
            [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ):
            sid = "ray_%s_%d" % (typ, d)
            ray_names[typ, d] = sid
            ray_states.append({"id": sid, "init": source})
            z = ray_shift(sid, dy, dx, ray_body)
            z = node("min2", [z, ray_free], ray_body)
            z = node("max2", [z, source], ray_body)
            ray_next[sid] = z
    node("loop", name="R", T=64, state=ray_states,
         body=ray_body, next=ray_next)

    shot, shot_any = {}, {}
    for size_name, size in (("s", 2), ("b", 4)):
        for typ in ("a", "b"):
            maps = []
            for d in range(4):
                if d in (0, 2):
                    dy, dx = -1, -1
                elif d == 1:
                    dy, dx = size - 2, -1
                else:
                    dy, dx = -1, size - 2
                z = shift("R." + ray_names[typ, d], 4, dy, dx)
                shot[size_name, typ, d] = z
                maps.append(z)
            shot_any[size_name, typ] = maximum(maps)

    # Configuration-space Bellman propagation.
    body = []

    def nav_value(v, big, graph):
        n2 = [shift(v, 4, dy, dx, graph) for dy, dx in directions]
        if not big:
            return maximum(n2, graph)
        candidates = []
        for d, (dy, dx) in enumerate(directions):
            n4 = shift(n2[d], 4, dy, dx, graph)
            candidates.append(node("add", [
                node("mul", [n4, free4[d]], graph),
                node("mul", [n2[d], blocked4[d]], graph)
            ], graph))
        return maximum(candidates, graph)

    ns = node("add", [nav_value("vs", False, body), minus1], body)
    nb = node("add", [nav_value("vb", True, body), minus1], body)

    growth_terms = [
        node("mul", [
            shift("vb", 4, dy, dx, body), selected
        ], body)
        for dy, dx, selected in growth
    ]
    grow_value = node("add", [plus(growth_terms, body), minus5], body)
    shrink_value = node("mul", [
        node("add", ["vs", minus5], body), have_resize
    ], body)
    ns = node("max2", [ns, grow_value], body)
    nb = node("max2", [nb, shrink_value], body)

    for state_name, pads in (("vs", small_pads), ("vb", large_pads)):
        best = node("gmax", [
            node("mul", [state_name, pads], body)
        ], body)
        best = node("vmul", [best, have_teleport], body)
        best = node("vadd", [best, minus4], body)
        if state_name == "vs":
            ns = node("max2", [ns, best], body)
        else:
            nb = node("max2", [nb, best], body)

    wa = np.zeros((4, 4), np.float32)
    wb = np.zeros((4, 4), np.float32)
    wa[0, 1] = wa[2, 3] = 1
    wb[0, 2] = wb[1, 3] = 1
    ba = [-4, -1000, -4, -1000]
    bb = [-4, -4, -1000, -1000]
    for size_name, state_name in (("s", "vs"), ("b", "vb")):
        current = ns if size_name == "s" else nb
        for typ, w, b in (("a", wa, ba), ("b", wb, bb)):
            future = lin(state_name, w, b, body)
            future = node("mul", [
                future, shot_any[size_name, typ]
            ], body)
            current = node("max2", [current, future], body)
        if size_name == "s":
            ns = current
        else:
            nb = current

    ns = node("max2", [
        node("mul", [node("relu", [ns], body), free_s], body),
        seed_s
    ], body)
    nb = node("max2", [
        node("mul", [node("relu", [nb], body), free_b], body),
        seed_b
    ], body)
    node("loop", name="V", T=32,
         state=[{"id": "vs", "init": seed_s},
                {"id": "vb", "init": seed_b}],
         body=body, next={"vs": ns, "vb": nb})

    vs, vb = "V.vs", "V.vb"
    vsc, vbc = sl(vs, 0, 1), sl(vb, 0, 1)

    move_values = []
    for d, (dy, dx) in enumerate(directions):
        sv = shift(vsc, 1, dy, dx)
        b2 = shift(vbc, 1, dy, dx)
        b4 = shift(b2, 1, dy, dx)
        bv = plus([
            node("mul", [b4, sl(free4[d], 0, 1)]),
            node("mul", [b2, sl(blocked4[d], 0, 1)])
        ])
        move_values.append(node("gmax", [plus([
            node("mul", [sv, ps]), node("mul", [bv, pb])
        ])]))
    qmove = dense(vc(move_values), np.eye(4), [-1] * 4)

    fire_values = []
    for d in range(4):
        size_terms = []
        for size_name, values, player in (
            ("s", vs, ps), ("b", vb, pb)
        ):
            terms = []
            for typ, future_world in (("a", 1), ("b", 2)):
                terms.append(node("mul", [
                    sl(values, future_world, future_world + 1),
                    sl(shot[size_name, typ, d], 0, 1)
                ]))
            size_terms.append(node("mul", [plus(terms), player]))
        fire_values.append(node("gmax", [plus(size_terms)]))
    fire_potential = vc(fire_values)
    qturn = dense(fire_potential, np.eye(4), [-5] * 4)
    qdirections = node("gmax", [node("max2", [
        node("mul", [one, qmove]), node("mul", [one, qturn])
    ])])
    qfire = dense(
        node("vmul", [fire_potential, facing]),
        np.ones((1, 4)), [-4]
    )

    actual_growth = plus([
        node("mul", [
            shift(vbc, 1, dy, dx), sl(selected, 0, 1)
        ])
        for dy, dx, selected in growth
    ])
    resize_value = node("gmax", [plus([
        node("mul", [actual_growth, ps]),
        node("mul", [vsc, pb])
    ])])
    resize_value = node("vmul", [resize_value, have_resize])
    qresize = dense(resize_value, [[1]], [-5])

    tele_s = node("vmul", [
        node("gmax", [node("mul", [vsc, small_pads])]),
        vsl(player_sizes, 0, 1)
    ])
    tele_b = node("vmul", [
        node("gmax", [node("mul", [vbc, large_pads])]),
        vsl(player_sizes, 1, 2)
    ])
    tele_value = node("vmul", [
        node("vadd", [tele_s, tele_b]), have_teleport
    ])
    qtele = dense(tele_value, [[1]], [-4])
    qspells = vc([qresize, qtele, qfire])
    spell_maps = node("mul", [one, qspells])
    qclick = node("gmax", [maximum([
        sl(spell_maps, 0, 1), sl(spell_maps, 1, 2),
        sl(spell_maps, 2, 3)
    ])])

    outw = np.zeros((8, 5), np.float32)
    for d in range(4):
        outw[d + 1, d] = 1
    outw[6, 4] = 1
    outb = np.full(8, -10000, np.float32)
    outb[1:5] = 0
    outb[6] = 0.25
    dense(vc([qdirections, qclick]), outw, outb, name="logits")

    # Select a spell with hard piecewise-linear comparisons.
    pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
    dw = np.zeros((6, 3), np.float32)
    db = np.zeros(6, np.float32)
    for r, (i, j) in enumerate(pairs):
        dw[r, i], dw[r, j] = 1, -1
        db[r] = 0.5 if i > j else -0.5
    compare = node("vrelu", [dense(qspells, dw, db)])
    excess = node("vrelu", [
        dense(compare, np.eye(6), [-1] * 6)
    ])
    compare = dense(
        vc([compare, excess]),
        np.concatenate([np.eye(6), -np.eye(6)], axis=1)
    )
    wf1, wf2 = np.zeros((3, 6)), np.zeros((3, 6))
    for i in range(3):
        wf1[i, 2 * i] = 1
        wf2[i, 2 * i + 1] = 1
    winner = node("vmul", [
        dense(compare, wf1), dense(compare, wf2)
    ])

    # Read the spell grid and click the first mismatching cell.
    centres = [
        (50 + 5 * r, 25 + 5 * c)
        for r in range(3) for c in range(3)
    ]
    wc, bc = np.zeros((18, 3)), np.zeros(18)
    for i, (y, x) in enumerate(centres):
        wc[2 * i, 0], wc[2 * i + 1, 1] = 63, 63
        bc[2 * i], bc[2 * i + 1] = -y, -x
    distances = node("abs", [
        conv("coords", wc[:, :, None, None], bc)
    ])
    wm = np.zeros((9, 18))
    for i in range(9):
        wm[i, 2 * i:2 * i + 2] = -2
    click_centres = node("relu", [
        lin(distances, wm, np.ones(9))
    ])
    active = node("gmax", [node("mul", [
        click_centres,
        plus([colours[14], colours[6], colours[11], colours[15]])
    ])])

    patterns = np.zeros((9, 3), np.float32)
    patterns[[1, 3, 5, 7], 0] = 1
    patterns[[0, 1, 4], 1] = 1
    patterns[[1, 4, 7], 2] = 1
    desired = dense(winner, patterns)
    overlap = node("vmul", [active, desired])
    mismatch = dense(
        vc([active, desired, overlap]),
        np.concatenate([np.eye(9), np.eye(9), -2 * np.eye(9)], axis=1)
    )
    click_scores = dense(
        mismatch, 100 * np.eye(9), -0.01 * np.arange(9)
    )
    weighted = node("mul", [click_centres, click_scores])
    conv(weighted, np.ones((1, 9, 1, 1)), name="click")

    return {"graph": G, "memory_size": 0, "params": P}
