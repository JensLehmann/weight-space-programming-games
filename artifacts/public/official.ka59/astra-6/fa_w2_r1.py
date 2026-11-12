import copy
import numpy as np


def build():
    P = {}
    pcache = {}
    graph = []
    dest = [graph]
    caches = [{}]
    counter = [0]
    value_loops = []

    BIG = 4096.0
    sizes = [(3, 3), (3, 6), (6, 3), (6, 6)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def fresh(prefix="n"):
        counter[0] += 1
        return prefix + str(counter[0])

    def param(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in pcache:
            name = "p" + str(len(P))
            P[name] = a
            pcache[key] = name
        return pcache[key]

    def node(op, ins, **fields):
        key = (op, tuple(ins), tuple(sorted(fields.items())))
        if key in caches[-1]:
            return caches[-1][key]
        name = fresh()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        dest[0].append(nd)
        caches[-1][key] = name
        return name

    def conv(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("conv", [x], **fields)

    def dense(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", [x], **fields)

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vsl(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def cat(xs):
        return xs[0] if len(xs) == 1 else node("concat", xs)

    def vcat(xs):
        return xs[0] if len(xs) == 1 else node("vconcat", xs)

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

    def relu(a):
        return node("relu", [a])

    def affine(x, a=1.0, b=0.0, c=1):
        if a == 1.0 and b == 0.0:
            return x
        w = (a * np.eye(c, dtype=np.float32)).reshape(c, c, 1, 1)
        return conv(x, w, np.full(c, b, np.float32))

    def va(x, a=1.0, b=0.0, n=1):
        return dense(
            x, a * np.eye(n, dtype=np.float32),
            np.full(n, b, np.float32)
        )

    def gm(x):
        return node("gmax", [x])

    def sumc(x, c):
        return conv(x, np.ones((1, c, 1, 1), np.float32))

    def maxlist(xs):
        out = xs[0]
        for x in xs[1:]:
            out = maximum(out, x)
        return out

    def max4(x):
        return maxlist([sl(x, i, i + 1) for i in range(4)])

    one = sl("coords", 2, 3)
    xy = sl("coords", 0, 2)
    zero = conv("x0", np.zeros((1, 16, 1, 1), np.float32))
    zero4 = conv("x0", np.zeros((4, 16, 1, 1), np.float32))
    twenty = affine(one, 0.0, 20.0)

    def inv(x, c=1):
        return affine(x, -1.0, 1.0, c)

    def clip(x):
        return minimum(relu(x), one)

    def vclip(x, n):
        positive = node("vrelu", [x])
        excess = node("vrelu", [va(positive, 1.0, -1.0, n)])
        return node("vadd", [positive, va(excess, -1.0, 0.0, n)])

    def shift(x, dy, dx):
        while dy or dx:
            sy = max(-3, min(3, dy))
            sx = max(-3, min(3, dx))
            r = max(abs(sy), abs(sx))
            w = np.zeros((1, 1, 2 * r + 1, 2 * r + 1), np.float32)
            w[0, 0, r + sy, r + sx] = 1.0
            x = conv(x, w)
            dy -= sy
            dx -= sx
        return x

    def correlate(x, terms, channels=1):
        merged = {}
        for ch, y, xx, value in terms:
            key = (int(ch), int(y), int(xx))
            merged[key] = merged.get(key, 0.0) + float(value)
        merged = {k: v for k, v in merged.items() if v != 0.0}
        if not merged:
            return zero

        ymin = min(k[1] for k in merged)
        xmin = min(k[2] for k in merged)
        groups = {}
        for (ch, y, xx), value in merged.items():
            tile = ((y - ymin) // 7, (xx - xmin) // 7)
            groups.setdefault(tile, []).append((ch, y, xx, value))

        parts = []
        for tile in sorted(groups):
            ts = groups[tile]
            cy = (min(t[1] for t in ts) + max(t[1] for t in ts)) // 2
            cx = (min(t[2] for t in ts) + max(t[2] for t in ts)) // 2
            r = max(max(abs(t[1] - cy), abs(t[2] - cx)) for t in ts)
            w = np.zeros((1, channels, 2 * r + 1, 2 * r + 1), np.float32)
            for ch, y, xx, value in ts:
                w[0, ch, y - cy + r, xx - cx + r] += value
            if channels == 1 and r == 0 and w[0, 0, 0, 0] == 1.0:
                part = x
            else:
                part = conv(x, w)
            parts.append(shift(part, cy, cx))

        result = parts[0]
        for part in parts[1:]:
            result = add(result, part)
        return result

    def samples(x, points):
        return correlate(x, [(0, y, xx, 1.0) for y, xx in points])

    def splat(x, points):
        return samples(x, [(-y, -xx) for y, xx in points])

    def exact(x, points):
        return relu(affine(samples(x, points), 1.0, 1.0 - len(points)))

    def rectangle(w, h):
        return [(y, x) for y in range(h) for x in range(w)]

    def mix(xs, flags):
        return sumc(mul(cat(xs), flags), len(xs))

    wd = np.zeros((4, 1, 7, 7), np.float32)
    for d, (dy, dx) in enumerate(directions):
        wd[d, 0, 3 + 3 * dy, 3 + 3 * dx] = 1.0

    def four_shifts(x):
        return conv(x, wd)

    wu = np.zeros((4, 4, 3, 3), np.float32)
    for d, (dy, dx) in enumerate(directions):
        wu[d, d, 1 + dy, 1 + dx] = 1.0

    def advance_lanes(x):
        for _ in range(3):
            x = conv(x, wu)
        return x

    key_plane = conv(
        "coords",
        np.array([1.0, 1.0 / 64.0, 1.0], np.float32).reshape(1, 3, 1, 1)
    )

    def choose(mask, score):
        weighted = mul(mask, affine(score, 1.0, 2000.0))
        tied = relu(affine(sub(weighted, gm(weighted)), 1024.0, 1.0))
        keyed = mul(mul(tied, mask), key_plane)
        last = relu(affine(sub(keyed, gm(keyed)), 8192.0, 1.0))
        return minimum(mask, last)

    def at_point(field, point):
        return gm(mul(field, point))

    # Largest-first parsing prevents a selected large piece from
    # producing false small-piece detections.
    color = {c: sl("x0", c, c + 1)
             for c in (0, 2, 4, 5, 11, 12, 13, 14, 15)}
    edge = add(color[14], color[0])
    core = add(add(color[0], color[4]), color[5])
    features = cat([edge, core])

    pieces = [None] * 4
    occupancy = [None] * 4
    active_parts = [None] * 4
    for i in (3, 1, 2, 0):
        w, h = sizes[i]
        ys = [1] if h == 3 else [2, 3]
        xs = [1] if w == 3 else [2, 3]
        center = {(y, x) for y in ys for x in xs}
        points = rectangle(w, h)
        terms = [
            (1 if (y, x) in center else 0, y, x, 1.0)
            for y, x in points
        ]
        p = relu(affine(
            correlate(features, terms, 2), 1.0, 1.0 - w * h
        ))
        pieces[i] = p
        occupancy[i] = splat(p, points)
        active_parts[i] = mul(p, shift(color[0], ys[0], xs[0]))
        if i != 0:
            features = mul(features, inv(minimum(occupancy[i], one)))

    piece_stack = cat(pieces)
    active = sumc(cat(active_parts), 4)
    yellow_occupancy = sumc(cat(occupancy), 4)

    free_wall = inv(color[2])
    free_land = sub(free_wall, color[15])
    fits = [exact(free_wall, rectangle(w, h)) for w, h in sizes]
    lands = [exact(free_land, rectangle(w, h)) for w, h in sizes]

    cross_points = [
        (y, x) for y in range(9) for x in range(9)
        if 3 <= y < 6 or 3 <= x < 6
    ]
    cross = exact(color[11], cross_points)
    cross_fit = exact(free_wall, cross_points)
    cross_land = exact(free_land, cross_points)

    bomb_pixels = add(color[12], color[13])
    bomb = exact(bomb_pixels, rectangle(6, 6))
    total_occupancy = add(add(yellow_occupancy, color[11]), bomb_pixels)

    # Goal corner filters tolerate ordinary partial border occlusion.
    corner_w = np.zeros((4, 1, 3, 3), np.float32)
    outside = [
        [(-1, 0), (0, -1)],
        [(-1, 0), (0, 1)],
        [(1, 0), (0, -1)],
        [(1, 0), (0, 1)],
    ]
    for c in range(4):
        corner_w[c, 0, 1, 1] = 1.0
        for dy, dx in outside[c]:
            corner_w[c, 0, 1 + dy, 1 + dx] = -1.0
    corners = relu(conv(color[4], corner_w))
    goals = []
    for w, h in sizes:
        terms = [
            (0, -1, -1, 1.0), (1, -1, w, 1.0),
            (2, h, -1, 1.0), (3, h, w, 1.0)
        ]
        goals.append(clip(affine(correlate(corners, terms, 4), 1.0, -2.0)))
    goal_stack = cat(goals)
    remaining = mul(goal_stack, inv(piece_stack, 4))

    cross_goal_points = [
        (-1, 2), (-1, 6), (9, 2), (9, 6),
        (2, -1), (6, -1), (2, 9), (6, 9)
    ]
    allowed_outline = maximum(color[4], minimum(total_occupancy, one))
    cross_goal = minimum(
        exact(allowed_outline, cross_goal_points),
        clip(affine(samples(color[4], cross_goal_points), 1.0, -2.0))
    )
    cross_remaining = mul(cross_goal, inv(cross))
    cross_candidates = mul(cross, inv(cross_goal))
    cross_pending = node("vmul", [gm(cross_remaining), gm(cross)])

    top = shift(color[12], 0, 2)
    bottom = shift(color[12], 5, 2)
    left = shift(color[12], 2, 0)
    right = shift(color[12], 2, 5)
    oriented = [
        mul(bomb, relu(sub(bottom, top))),
        mul(bomb, relu(sub(top, bottom))),
        mul(bomb, relu(sub(right, left))),
        mul(bomb, relu(sub(left, right)))
    ]
    uncoloured = mul(bomb, relu(affine(
        samples(color[12], rectangle(6, 6)), -1.0, 1.0
    )))
    lower_half = clip(affine(sl("coords", 0, 1), 126.0, -63.0))
    oriented[0] = add(oriented[0], mul(uncoloured, lower_half))
    oriented[1] = add(oriented[1], mul(uncoloured, inv(lower_half)))

    piece_count = dense(
        node("gmean", [piece_stack]),
        np.full((1, 4), 4096.0, np.float32)
    )
    has_helper = vclip(va(piece_count, 1.0, -1.0), 1)

    # Memory: phase[4], locked goal shape[4], goal row/col[2], valid[1].
    old_phase = vsl("mem", 0, 4)
    old_flags = vsl("mem", 4, 8)
    old_xy = vsl("mem", 8, 10)
    old_valid = vsl("mem", 10, 11)
    missing_phase = node("vrelu", [
        dense(old_phase, -np.ones((1, 4), np.float32), np.ones(1, np.float32))
    ])
    phase = node("vadd", [
        old_phase,
        dense(missing_phase, np.array([[1], [0], [0], [0]], np.float32))
    ])
    distance_to_lock = sumc(node("abs", [sub(xy, old_xy)]), 2)
    old_marker = mul(
        relu(affine(distance_to_lock, -128.0, 1.0)), old_valid
    )
    old_done = gm(mul(
        old_marker, sumc(mul(piece_stack, old_flags), 4)
    ))
    keep_lock = node("vmul", [
        vclip(node("vadd", [old_valid, va(old_done, -1.0, 0.0)]), 1),
        va(cross_pending, -1.0, 1.0)
    ])

    # Static contact geometry.
    cross_cells = {(y // 3, x // 3) for y, x in cross_points}

    def side_offsets(w, h, hw, hh, d, is_cross=False):
        if is_cross:
            hc, wc = hh // 3, hw // 3
            dy, dx = directions[d]
            result = []
            for y in range(-hc, 4):
                for x in range(-wc, 4):
                    here = {
                        (y + yy, x + xx)
                        for yy in range(hc) for xx in range(wc)
                    }
                    there = {(yy + dy, xx + dx) for yy, xx in here}
                    if not (here & cross_cells) and (there & cross_cells):
                        result.append((3 * y, 3 * x))
            return result
        if d in (0, 1):
            y = h if d == 0 else -hh
            return [(y, x) for x in range(3 - hw, w - 2, 3)]
        x = w if d == 2 else -hw
        return [(y, x) for y in range(3 - hh, h - 2, 3)]

    def support(w, h, is_cross=False):
        return [
            minimum(samples(
                lands[0], side_offsets(w, h, 3, 3, d, is_cross)
            ), one)
            for d in range(4)
        ]

    # Value propagation models five-cell kicks and continued gulch motion.
    def object_dp(fit, land, goal, kick_support, walking=False, front=None):
        next_fit = four_shifts(fit)
        blocked = affine(next_fit, BIG, -BIG, 4)
        stop_penalty = affine(next_fit, -BIG, 0.0, 4)
        slide_penalty = affine(mul(next_fit, inv(land)), BIG, -BIG, 4)
        land_penalty = affine(land, BIG, -BIG)
        kick_penalty = affine(kick_support, BIG, -BIG - 4.0, 4)
        fit_cap = affine(fit, BIG, 0.0)
        if walking:
            walk_penalty = affine(
                mul(four_shifts(land), land), BIG, -BIG - 1.0, 4
            )
        if front is not None:
            blast_penalty = affine(front, BIG, -BIG - 5.0, 4)

        sv, sc, sk = fresh("s"), fresh("s"), fresh("s")
        sj = fresh("s") if front is not None else None
        body = []
        old = dest[0]
        dest[0] = body
        caches.append({})

        coast = relu(maximum(
            add(advance_lanes(sc), slide_penalty),
            add(sv, land_penalty)
        ))
        stopped = add(sv, stop_penalty)
        z = coast
        three_steps = None
        for t in range(5):
            z = maximum(add(advance_lanes(z), blocked), stopped)
            if t == 2:
                three_steps = z
        candidate = add(z, kick_penalty)
        if walking:
            candidate = maximum(candidate, add(four_shifts(sv), walk_penalty))
        if front is not None:
            candidate = maximum(candidate, add(three_steps, blast_penalty))
        nv = minimum(maximum(goal, max4(candidate)), fit_cap)

        caches.pop()
        dest[0] = old
        lid = fresh("L")
        states = [
            {"id": sv, "init": goal},
            {"id": sc, "zeros": [4, 64, 64]},
            {"id": sk, "zeros": [4, 64, 64]}
        ]
        nxt = {sv: nv, sc: coast, sk: z}
        if sj is not None:
            states.append({"id": sj, "zeros": [4, 64, 64]})
            nxt[sj] = three_steps
        dest[0].append({
            "id": lid, "op": "loop", "T": 1,
            "state": states, "body": body, "next": nxt
        })
        value_loops.append(lid)
        answer = {"v": lid + "." + sv, "k": lid + "." + sk}
        if sj is not None:
            answer["j"] = lid + "." + sj
        return answer

    def navigation(seed, free):
        penalty = affine(free, BIG, -BIG - 1.0)
        state = fresh("s")
        body = []
        old = dest[0]
        dest[0] = body
        caches.append({})
        neighbors = [shift(state, 3 * dy, 3 * dx) for dy, dx in directions]
        nv = maximum(seed, add(maxlist(neighbors), penalty))
        caches.pop()
        dest[0] = old
        lid = fresh("L")
        dest[0].append({
            "id": lid, "op": "loop", "T": 42,
            "state": [{"id": state, "init": seed}],
            "body": body, "next": {state: nv}
        })
        return lid + "." + state

    def rewards(end, current, point, base):
        gain = relu(sub(end, current))
        gate = minimum(gain, one)
        value = affine(minimum(gain, twenty), 2.0, base, 4)
        return mul(mul(value, gate), point)

    cd = object_dp(
        cross_fit, cross_land,
        affine(cross_remaining, 1000.0, 0.0),
        cat(support(9, 9, True))
    )
    cross_target = choose(cross_candidates, cd["v"])
    cross_rewards = rewards(
        cd["k"], at_point(cd["v"], cross_target), cross_target, 1000.0
    )

    # The smallest selectable shape can occupy every cross contact recess.
    cross_channels = [
        splat(sl(cross_rewards, d, d + 1),
              side_offsets(9, 9, 3, 3, d, True))
        for d in range(4)
    ]
    cross_terminal = mul(
        mul(cat(cross_channels), lands[0]), four_shifts(lands[0])
    )
    cross_terminals = [cross_terminal, zero4, zero4, zero4]
    assistance = [minimum(max4(cross_terminal), one), zero, zero, zero]

    goal_presence = node("vmul", [
        gm(remaining), va(cross_pending, -1.0, 1.0)
    ])
    assistance_presence = gm(cat(assistance))
    eligible = node("vmul", [
        vclip(node("vadd", [goal_presence, assistance_presence]), 4),
        gm(piece_stack)
    ])

    order0 = [3, 1, 2, 0]
    ws = np.zeros((16, 4), np.float32)
    wp = np.zeros((16, 4), np.float32)
    wo = np.zeros((4, 16), np.float32)
    for p in range(4):
        order = order0[p:] + order0[:p]
        for rank, typ in enumerate(order):
            row = 4 * p + typ
            ws[row, typ] = 1.0
            for higher in order[:rank]:
                ws[row, higher] = -1.0
            wp[row, p] = 1.0
            wo[typ, row] = 1.0
    selectors = node("vrelu", [dense(eligible, ws)])
    proposed_flags = dense(
        node("vmul", [selectors, dense(phase, wp)]), wo
    )
    unlock = va(keep_lock, -1.0, 1.0)
    flags = node("vadd", [
        node("vmul", [old_flags, keep_lock]),
        node("vmul", [proposed_flags, unlock])
    ])

    new_goal = choose(sumc(mul(remaining, flags), 4), zero)
    locked_goal = mul(
        add(mul(old_marker, keep_lock), mul(new_goal, unlock)),
        va(cross_pending, -1.0, 1.0)
    )
    lock_valid = gm(locked_goal)

    yellow_fit = mix(fits, flags)
    yellow_land = mix(lands, flags)
    yellow_goal = maximum(
        affine(locked_goal, 1000.0, 0.0),
        sumc(mul(affine(cat(assistance), 1100.0, 0.0, 4), flags), 4)
    )
    candidates = sumc(mul(
        mul(piece_stack, maximum(inv(goal_stack, 4), assistance_presence)),
        flags
    ), 4)

    supports = [support(w, h) for w, h in sizes]
    yellow_support = mul(cat([
        mix([supports[t][d] for t in range(4)], flags)
        for d in range(4)
    ]), has_helper)

    def typed(x):
        return mul(cat([x] * 4), flags)

    front_channels = []
    for d in range(4):
        terms = []
        for typ, (w, h) in enumerate(sizes):
            if d == 0:
                offsets = [(-h, x) for x in range(3 - w, 4, 3)]
            elif d == 1:
                offsets = [(6, x) for x in range(3 - w, 4, 3)]
            elif d == 2:
                offsets = [(y, -w) for y in range(3 - h, 4, 3)]
            else:
                offsets = [(y, 6) for y in range(3 - h, 4, 3)]
            terms.extend((typ, -y, -x, 1.0) for y, x in offsets)
        front_channels.append(minimum(
            correlate(typed(oriented[d]), terms, 4), one
        ))
    bomb_front = mul(cat(front_channels), yellow_land)

    yd = object_dp(
        yellow_fit, yellow_land, yellow_goal, yellow_support,
        walking=True, front=bomb_front
    )

    bias_w = np.array([
        [0, 40, -40, 0],
        [0, 0, 0, 40],
        [0, 0, 0, 0]
    ], np.float32)
    bias = sumc(mul("coords", dense(phase, bias_w)), 3)
    yellow_target = choose(candidates, add(yd["v"], bias))

    render_terms = []
    for typ, (w, h) in enumerate(sizes):
        render_terms.extend(
            (typ, -y, -x, 1.0) for y, x in rectangle(w, h)
        )
    target_occupancy = correlate(typed(yellow_target), render_terms, 4)
    others = relu(sub(total_occupancy, target_occupancy))

    own_free_shapes = []
    for i, (w, h) in enumerate(sizes):
        clear = relu(affine(samples(others, rectangle(w, h)), -1.0, 1.0))
        own_free_shapes.append(minimum(clear, lands[i]))
    own_free = mix(own_free_shapes, flags)

    kick_seed = max4(add(
        yd["k"], affine(yellow_support, BIG, -BIG - 4.0, 4)
    ))
    blast_seed = max4(add(
        yd["j"], affine(bomb_front, BIG, -BIG - 5.0, 4)
    ))
    own_seed = mul(
        relu(maximum(yellow_goal, maximum(kick_seed, blast_seed))),
        own_free
    )
    own_field = navigation(own_seed, own_free)
    yellow_value = at_point(own_field, yellow_target)

    yellow_rewards = rewards(yd["k"], yellow_value, yellow_target, 100.0)
    ordinary_rewards = mul(
        rewards(four_shifts(own_field), yellow_value, yellow_target, 99.0),
        four_shifts(own_free)
    )
    riding_rewards = mul(
        rewards(yd["j"], yellow_value, yellow_target, 100.0), bomb_front
    )
    own_rewards = maximum(ordinary_rewards, riding_rewards)

    typed_rewards = [typed(sl(yellow_rewards, d, d + 1)) for d in range(4)]
    yellow_terminals = []
    for hw, hh in sizes:
        channels = []
        for d in range(4):
            terms = []
            for typ, (w, h) in enumerate(sizes):
                terms.extend(
                    (typ, -y, -x, 1.0)
                    for y, x in side_offsets(w, h, hw, hh, d)
                )
            channels.append(correlate(typed_rewards[d], terms, 4))
        yellow_terminals.append(cat(channels))

    # Reposition upward launchers towards useful landing corridors.
    upward_landing = sl(yd["j"], 0, 1)
    potentials = []
    for w, h in sizes:
        potentials.append(maxlist([
            mul(shift(upward_landing, -h, x), shift(yellow_land, -h, x))
            for x in (0, 3)
        ]))
    potential = mix(potentials, flags)
    bomb_next_fit = four_shifts(fits[3])
    blocked = affine(bomb_next_fit, BIG, -BIG, 4)
    stopped = add(potential, affine(bomb_next_fit, -BIG, 0.0, 4))
    z = cat([potential] * 4)
    for _ in range(5):
        z = maximum(add(advance_lanes(z), blocked), stopped)

    needs_transport = node("vrelu", [va(yellow_value, -1.0, 1.0)])
    lateral_axes = dense(
        "avail", np.zeros((4, 8), np.float32),
        np.array([0, 0, 1, 1], np.float32)
    )
    bomb_rewards = mul(mul(
        rewards(z, potential, oriented[0], 750.0), lateral_axes
    ), needs_transport)

    bomb_terminals = []
    for hw, hh in sizes:
        bomb_terminals.append(cat([
            zero, zero,
            splat(sl(bomb_rewards, 2, 3), side_offsets(6, 6, hw, hh, 2)),
            splat(sl(bomb_rewards, 3, 4), side_offsets(6, 6, hw, hh, 3))
        ]))

    helper_scores = []
    target_exclusion = affine(yellow_target, 500.0, 0.0)
    for i, (w, h) in enumerate(sizes):
        obstacles = (total_occupancy if i == 0
                     else relu(sub(total_occupancy, occupancy[i])))
        clear = relu(affine(samples(obstacles, rectangle(w, h)), -1.0, 1.0))
        free = minimum(clear, lands[i])
        terminal = maximum(
            maximum(cross_terminals[i], yellow_terminals[i]),
            bomb_terminals[i]
        )
        terminal = mul(mul(terminal, lands[i]), four_shifts(lands[i]))
        terminal = mul(terminal, maximum(free, pieces[i]))
        field = navigation(max4(terminal), free)
        q = maximum(
            terminal,
            add(four_shifts(field),
                affine(four_shifts(free), BIG, -BIG - 1.0, 4))
        )
        helper_scores.append(mul(relu(sub(q, target_exclusion)), pieces[i]))

    scores = maximum(maxlist(helper_scores), own_rewards)
    selected_scores = gm(mul(scores, active))
    best_map = max4(scores)
    nonselected_map = mul(best_map, inv(active))
    click_score = gm(nonselected_map)

    out_features = vcat([selected_scores, click_score, phase])
    out_w = np.zeros((8, 9), np.float32)
    out_b = np.full(8, -10000.0, np.float32)
    for d in range(4):
        out_w[d + 1, d] = 1.0
        out_w[d + 1, 5 + d] = 0.01
        out_b[d + 1] = 0.0
    out_w[6, 4] = 1.0
    out_b[6] = -2.0
    graph.append({
        "id": "logits", "op": "dense", "in": [out_features],
        "w": param(out_w), "b": param(out_b)
    })

    click_parts = []
    for i, (w, h) in enumerate(sizes):
        cy = 1 if h == 3 else 2
        cx = 1 if w == 3 else 2
        click_parts.append(shift(mul(nonselected_map, pieces[i]), -cy, -cx))
    click_field = maxlist(click_parts)
    graph.append({
        "id": "click", "op": "conv", "in": [click_field],
        "w": param(np.ones((1, 1, 1, 1), np.float32))
    })

    unproductive = node("vrelu", [va(gm(best_map), -1.0, 1.0)])
    selected_best = gm(mul(best_map, active))
    failed_switch = node("vmul", [
        node("vrelu", [va(selected_best, -1.0, 1.0)]),
        vsl("aprev", 6, 7)
    ])
    stuck = vclip(node("vadd", [unproductive, failed_switch]), 1)
    rotation = np.zeros((4, 4), np.float32)
    for i in range(4):
        rotation[(i + 1) % 4, i] = 1.0
    change = dense(phase, rotation - np.eye(4, dtype=np.float32))
    next_phase = node("vadd", [
        phase, node("vmul", [change, stuck])
    ])
    next_flags = node("vmul", [flags, lock_valid])
    next_xy = gm(mul(xy, locked_goal))
    graph.append({
        "id": "mem_next", "op": "vconcat",
        "in": [next_phase, next_flags, next_xy, lock_valid]
    })

    # Static compilation and exact resource bookkeeping.
    def prune(nodes, wanted):
        need = set(wanted)
        defined = set()
        for nd in nodes:
            if nd["op"] == "loop":
                defined.update(nd["id"] + "." + s["id"] for s in nd["state"])
            else:
                defined.add(nd["id"])
        kept = []
        for nd in reversed(nodes):
            if nd["op"] == "loop":
                outputs = {nd["id"] + "." + s["id"] for s in nd["state"]}
                if not (outputs & need):
                    continue
                body, external = prune(nd["body"], set(nd["next"].values()))
                nd["body"] = body
                need.update(external - {s["id"] for s in nd["state"]})
                for state in nd["state"]:
                    if "init" in state:
                        need.add(state["init"])
                kept.append(nd)
            elif nd["id"] in need:
                need.update(nd["in"])
                kept.append(nd)
        return list(reversed(kept)), need - defined

    def compile_graph(replacements):
        nodes = copy.deepcopy(graph)

        def resolve(x):
            while x in replacements:
                x = replacements[x]
            return x

        def rewrite(ns):
            for nd in ns:
                if nd["op"] == "loop":
                    rewrite(nd["body"])
                    for state in nd["state"]:
                        if "init" in state:
                            state["init"] = resolve(state["init"])
                    nd["next"] = {k: resolve(v) for k, v in nd["next"].items()}
                else:
                    nd["in"] = [resolve(x) for x in nd["in"]]

        rewrite(nodes)
        return prune(nodes, {"logits", "click", "mem_next"})[0]

    initial_shapes = {
        "x0": (16, 64, 64), "coords": (3, 64, 64),
        "mem": (11,), "aprev": (8,), "avail": (8,)
    }

    def account(nodes):
        loop_costs = {}
        used_params = set()

        def visit(ns, env):
            macs = evaluations = n_nodes = channels = 0
            for nd in ns:
                op = nd["op"]
                n_nodes += 1
                if op == "loop":
                    local = dict(env)
                    for state in nd["state"]:
                        shape = (env[state["init"]] if "init" in state
                                 else tuple(state["zeros"]))
                        local[state["id"]] = shape
                        if len(shape) == 3:
                            channels += shape[0]
                    bm, be, bn, bc = visit(nd["body"], local)
                    loop_costs[nd["id"]] = (bm, be)
                    macs += nd["T"] * bm
                    evaluations += 1 + nd["T"] * be
                    n_nodes += bn
                    channels += bc
                    for state in nd["state"]:
                        shape = local[nd["next"][state["id"]]]
                        env[nd["id"] + "." + state["id"]] = shape
                        if len(shape) == 3:
                            channels += shape[0]
                    continue

                for key in ("w", "b"):
                    if key in nd:
                        used_params.add(nd[key])
                xs = [env[x] for x in nd["in"]]
                x = xs[0]
                cost = 0
                if op == "conv":
                    w = P[nd["w"]]
                    assert len(x) == 3 and w.shape[1] == x[0], "conv shape"
                    shape = (w.shape[0], 64, 64)
                    cost = int(w.size) * 4096
                elif op == "dense":
                    w = P[nd["w"]]
                    assert len(x) == 1 and w.shape[1] == x[0], "dense shape"
                    shape = (w.shape[0],)
                    cost = int(w.size)
                elif op in ("slice", "vslice"):
                    shape = (nd["stop"] - nd["start"],) + x[1:]
                elif op in ("concat", "vconcat"):
                    shape = (sum(s[0] for s in xs),) + x[1:]
                elif op in ("gmax", "gmean"):
                    shape = (x[0],)
                    cost = x[0] * 4096
                elif op in ("add", "sub", "mul", "max2", "min2"):
                    assert (xs[0][0] == xs[1][0]
                            or 1 in (xs[0][0], xs[1][0])), "broadcast"
                    shape = (max(xs[0][0], xs[1][0]), 64, 64)
                    cost = shape[0] * 4096
                elif op in ("vadd", "vmul"):
                    assert (xs[0][0] == xs[1][0]
                            or 1 in (xs[0][0], xs[1][0])), "vector broadcast"
                    shape = (max(xs[0][0], xs[1][0]),)
                    cost = shape[0]
                else:
                    shape = x
                assert shape[0] <= (512 if len(shape) == 3 else 4096)
                env[nd["id"]] = shape
                macs += cost
                evaluations += 1
                if len(shape) == 3:
                    channels += shape[0]
            return macs, evaluations, n_nodes, channels

        result = visit(nodes, dict(initial_shapes))
        return result, loop_costs, used_params

    replacements = {}
    compiled = compile_graph(replacements)
    stats, costs, used = account(compiled)

    def too_large(st, used_set):
        return (
            st[2] > 990 or st[3] + 19 > 4050
            or sum(P[k].size for k in used_set) > 149000
        )

    # Fixed-priority removal of optional topology, never weight fitting.
    if too_large(stats, used):
        replacements[front_channels[2]] = zero
        replacements[front_channels[3]] = zero
        compiled = compile_graph(replacements)
        stats, costs, used = account(compiled)

    if too_large(stats, used):
        replacements.update({x: zero4 for x in bomb_terminals})
        compiled = compile_graph(replacements)
        stats, costs, used = account(compiled)

    if too_large(stats, used):
        replacements[bomb_front] = zero4
        compiled = compile_graph(replacements)
        stats, costs, used = account(compiled)

    if too_large(stats, used):
        replacements[own_field] = yd["v"]
        compiled = compile_graph(replacements)
        stats, costs, used = account(compiled)

    live_ids = {nd["id"] for nd in compiled if nd["op"] == "loop"}
    live_value_ids = set(value_loops) & live_ids
    iter_macs = sum(costs[k][0] for k in live_value_ids)
    iter_evals = sum(costs[k][1] for k in live_value_ids)
    fixed_macs = stats[0] - iter_macs
    fixed_evals = stats[1] - iter_evals

    iterations = min(
        56,
        int((1_490_000_000 - fixed_macs) // max(1, iter_macs)),
        int((9900 - fixed_evals) // max(1, iter_evals))
    )
    iterations = max(1, iterations)
    for nd in compiled:
        if nd["id"] in live_value_ids:
            nd["T"] = iterations

    stats, costs, used = account(compiled)
    if stats[0] > 1_500_000_000 or stats[1] > 10000:
        for nd in compiled:
            if nd["op"] == "loop" and nd["id"] not in live_value_ids:
                nd["T"] = 28
        stats, costs, used = account(compiled)

    assert stats[2] <= 1000, ("nodes", stats[2])
    assert stats[3] + 19 <= 4096, ("channels", stats[3] + 19)
    assert stats[0] <= 1_500_000_000, ("MACs", stats[0])
    assert stats[1] <= 10000, ("evaluations", stats[1])
    final_params = {k: P[k] for k in used}
    assert sum(a.size for a in final_params.values()) <= 150000, "parameters"
    assert all(np.isfinite(a).all() for a in final_params.values())

    return {"graph": compiled, "memory_size": 11, "params": final_params}
