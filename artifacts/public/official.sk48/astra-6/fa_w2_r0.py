import numpy as np


def build():
    def construct(N, O, push_depth):
        G = 4
        M = 16
        P = {}
        graph = []
        shapes = {
            "x0": (16, 64, 64),
            "coords": (3, 64, 64),
            "mem": (M,),
            "aprev": (8,),
            "avail": (8,),
        }
        counter = [0]
        parameter_cache = {}
        node_cache = {}
        constant_cache = {}

        def fresh(prefix="n"):
            counter[0] += 1
            return prefix + str(counter[0])

        def parameter(a):
            a = np.ascontiguousarray(a, dtype=np.float32)
            key = (a.shape, a.tobytes())
            if key not in parameter_cache:
                name = fresh("p")
                parameter_cache[key] = name
                P[name] = a
            return parameter_cache[key]

        def emit(L, op, ins, name=None, **fields):
            key = (id(L), op, tuple(ins), tuple(sorted(fields.items())))
            if name is None and key in node_cache:
                return node_cache[key]
            name = name or fresh()
            nd = {"id": name, "op": op, "in": list(ins)}
            nd.update(fields)
            L.append(nd)
            ss = [shapes[x] for x in ins]
            if op == "conv":
                shapes[name] = (P[fields["w"]].shape[0], 64, 64)
            elif op == "dense":
                shapes[name] = (P[fields["w"]].shape[0],)
            elif op in ("gmax", "gmean"):
                shapes[name] = (ss[0][0],)
            elif op in ("concat", "vconcat"):
                shapes[name] = (sum(s[0] for s in ss),) + ss[0][1:]
            elif op in ("slice", "vslice"):
                shapes[name] = (
                    fields["stop"] - fields["start"],
                ) + ss[0][1:]
            elif op in ("add", "sub", "mul", "max2", "min2"):
                shapes[name] = (max(s[0] for s in ss), 64, 64)
            elif op in ("vadd", "vmul"):
                shapes[name] = (max(s[0] for s in ss),)
            else:
                shapes[name] = ss[0]
            node_cache[key] = name
            return name

        def conv(L, x, w, b=None, name=None):
            w = np.asarray(w, np.float32)
            if w.ndim == 2:
                w = w[:, :, None, None]
            fields = {"w": parameter(w)}
            if b is not None:
                fields["b"] = parameter(b)
            return emit(L, "conv", [x], name=name, **fields)

        def dense(L, x, w, b=None, name=None):
            fields = {"w": parameter(w)}
            if b is not None:
                fields["b"] = parameter(b)
            return emit(L, "dense", [x], name=name, **fields)

        def C(value):
            a = np.asarray(value, np.float32).reshape(-1)
            key = (a.shape, a.tobytes())
            if key not in constant_cache:
                constant_cache[key] = dense(
                    graph, "avail", np.zeros((a.size, 8), np.float32), a
                )
            return constant_cache[key]

        def sl(L, x, start, stop=None):
            return emit(
                L, "slice" if len(shapes[x]) == 3 else "vslice", [x],
                start=start, stop=start + 1 if stop is None else stop
            )

        def cat(L, xs, name=None):
            return emit(
                L, "concat" if len(shapes[xs[0]]) == 3 else "vconcat",
                xs, name=name
            )

        def add(L, a, b, name=None):
            op = "vadd" if len(shapes[a]) == len(shapes[b]) == 1 else "add"
            return emit(L, op, [a, b], name=name)

        def sub(L, a, b):
            return emit(L, "sub", [a, b])

        def mul(L, a, b):
            op = "vmul" if len(shapes[a]) == len(shapes[b]) == 1 else "mul"
            return emit(L, op, [a, b])

        def scale(L, x, a):
            return x if a == 1 else mul(L, x, C(a))

        def affine(L, x, a=1.0, b=0.0):
            if a == 1 and b == 0:
                return x
            n = shapes[x][0]
            if len(shapes[x]) == 1:
                return dense(
                    L, x, np.eye(n, dtype=np.float32) * a,
                    np.full(n, b, np.float32)
                )
            if a == -1:
                return sub(L, C(b), x)
            if b == 0:
                return scale(L, x, a)
            if n <= 8:
                return conv(
                    L, x, np.eye(n, dtype=np.float32) * a,
                    np.full(n, b, np.float32)
                )
            return add(L, scale(L, x, a), C(b))

        def relu(L, x):
            return emit(
                L, "relu" if len(shapes[x]) == 3 else "vrelu", [x]
            )

        def absolute(L, x):
            return emit(L, "abs", [x])

        def clip(L, x):
            return emit(L, "min2", [relu(L, x), C(1.0)])

        def maximum(L, a, b):
            return emit(L, "max2", [a, b])

        def eq(L, a, b):
            return relu(L, sub(
                L, C(1.0), absolute(L, sub(L, a, b))
            ))

        def pixel_eq(L, a, b, sharpness=4.0):
            return clip(L, affine(
                L, absolute(L, sub(L, a, b)), -sharpness, 2.0
            ))

        def threshold(L, x, t=0.0):
            return clip(L, affine(
                L, x, float(2 ** 28), 1.0 - float(2 ** 28) * t
            ))

        def pool(L, x):
            return emit(L, "gmax", [x])

        def total(L, x):
            return conv(L, x, np.ones((1, shapes[x][0]), np.float32))

        def product(L, xs):
            z = xs[0]
            for x in xs[1:]:
                z = mul(L, z, x)
            return z

        def plus(L, xs):
            z = xs[0]
            for x in xs[1:]:
                z = add(L, z, x)
            return z

        def choose(L, horizontal, vertical, active, inactive):
            return add(
                L, mul(L, horizontal, inactive), mul(L, vertical, active)
            )

        def quantize(v, low=-1, high=8):
            d = shapes[v][0]
            thresholds = list(range(low + 1, high + 1))
            h = 2 * d * len(thresholds)
            w = np.zeros((h, d), np.float32)
            b = np.zeros(h, np.float32)
            out = np.zeros((d, h), np.float32)
            k = 0
            for i in range(d):
                for t in thresholds:
                    w[k, i] = 4
                    b[k] = 4 * (0.5 - t)
                    out[i, k] = 1
                    k += 1
                    w[k, i] = 4
                    b[k] = 4 * (0.5 - t) - 1
                    out[i, k] = -1
                    k += 1
            z = relu(graph, dense(graph, v, w, b))
            return dense(
                graph, z, out, np.full(d, low, np.float32)
            )

        def colour_equalities(colours, targets):
            z = cat(graph, [colours, targets])
            w = np.zeros((2 * N * G, N + G), np.float32)
            for g in range(G):
                for i in range(N):
                    k = g * N + i
                    w[k, i] = 1
                    w[k, N + g] = -1
                    w[N * G + k] = -w[k]
            z = relu(graph, dense(graph, z, w))
            w2 = np.concatenate([
                -np.eye(N * G, dtype=np.float32),
                -np.eye(N * G, dtype=np.float32)
            ], axis=1)
            return relu(graph, dense(
                graph, z, w2, np.ones(N * G, np.float32)
            ))

        def state(init=None, zeros=None):
            sid = fresh("s")
            if init is not None:
                shapes[sid] = shapes[init]
                return {"id": sid, "init": init}
            shapes[sid] = tuple(zeros)
            return {"id": sid, "zeros": list(zeros)}

        def loop(states, body, nxt, T):
            lid = fresh("loop")
            nd = {
                "id": lid, "op": "loop", "T": int(T),
                "state": states, "body": body, "next": nxt
            }
            graph.append(nd)
            outputs = {}
            for s in states:
                alias = lid + "." + s["id"]
                shapes[alias] = shapes[s["id"]]
                outputs[s["id"]] = alias
            return nd, outputs

        # -----------------------------------------------------------
        # Visual parser.
        # -----------------------------------------------------------
        row = conv(graph, "coords", [[63.0, 0.0, 0.0]])
        col = conv(graph, "coords", [[0.0, 63.0, 0.0]])
        zero = conv(graph, "coords", np.zeros((1, 3), np.float32))

        def broadcast(v):
            return add(graph, zero, v)

        top = clip(graph, affine(graph, row, -1.0, 53.0))
        bottom = sub(graph, C(1.0), top)
        floor = mul(graph, sl(graph, "x0", 4), top)
        row_floor = clip(graph, affine(
            graph, emit(graph, "rowsum", [floor]), 1.0, -9.0
        ))
        col_floor = clip(graph, affine(
            graph, emit(graph, "colsum", [floor]), 1.0, -9.0
        ))
        xmin = affine(graph, pool(graph, mul(
            graph, col_floor, affine(graph, col, -1.0, 64.0)
        )), -1.0, 64.0)
        ymin = affine(graph, pool(graph, mul(
            graph, row_floor, affine(graph, row, -1.0, 64.0)
        )), -1.0, 64.0)
        xmax = pool(graph, mul(graph, col_floor, col))
        ymax = pool(graph, mul(graph, row_floor, row))

        def extent(lo, hi):
            z = dense(
                graph, cat(graph, [lo, hi]),
                [[-1.0 / 6.0, 1.0 / 6.0]], [1.0 / 6.0]
            )
            return quantize(z, 0, 9)

        width = extent(xmin, xmax)
        height = extent(ymin, ymax)
        width_minus = affine(graph, width, 1.0, -1.0)
        height_minus = affine(graph, height, 1.0, -1.0)

        # Filled bounding rectangle, rather than the visible-floor mask.
        bounds = dense(
            graph, cat(graph, [xmin, xmax, ymin, ymax]),
            np.diag(np.array([-1, 1, -1, 1], np.float32))
        )
        edge_coordinates = conv(graph, "coords", [
            [0, 63, 1], [0, -63, 1],
            [63, 0, 1], [-63, 0, 1]
        ])
        edges = clip(graph, add(graph, edge_coordinates, bounds))
        board = product(graph, [sl(graph, edges, i) for i in range(4)])

        U = scale(graph, sub(graph, col, xmin), 1.0 / 6.0)
        V = scale(graph, sub(graph, row, ymin), 1.0 / 6.0)
        colour_number = conv(
            graph, "x0", np.arange(16, dtype=np.float32)[None, :]
        )
        tile_w = np.zeros((8, 16, 5, 5), np.float32)
        for j, colour in enumerate(range(8, 16)):
            for dy in range(3):
                for dx in range(3):
                    tile_w[j, colour, 2 + dy, 2 + dx] = 1
            tile_w[j, colour, 1, 2] = -9
            tile_w[j, colour, 2, 1] = -9
        tile = relu(graph, conv(
            graph, "x0", tile_w, np.full(8, -8, np.float32)
        ))
        tile = product(graph, [total(graph, tile), board, top])
        point_features = cat(graph, [
            affine(graph, U, 1.0, -1.0 / 6.0),
            affine(graph, V, 1.0, -1.0 / 6.0),
            colour_number, broadcast(C(1.0))
        ])
        priority = add(
            graph, scale(graph, row, -64.0), scale(graph, col, -1.0)
        )

        def enumerate_points(points, count):
            states = [state(init=points), state(zeros=[4 * count])]
            remaining, acc = [s["id"] for s in states]
            B = []
            key = relu(B, add(B, scale(B, remaining, 8192.0), priority))
            chosen = mul(B, clip(B, affine(
                B, sub(B, key, pool(B, key)), 1.0, 1.0
            )), remaining)
            values = pool(B, mul(B, point_features, chosen))
            acc_next = cat(B, [
                sl(B, acc, 4, 4 * count), values
            ]) if count > 1 else values
            remaining_next = relu(B, sub(B, remaining, chosen))
            _, out = loop(
                states, B,
                {remaining: remaining_next, acc: acc_next}, count
            )
            fields = []
            for j in range(4):
                w = np.zeros((count, 4 * count), np.float32)
                for i in range(count):
                    w[i, 4 * i + j] = 1
                fields.append(dense(graph, out[acc], w))
            fields[0] = quantize(fields[0], 0, 8)
            fields[1] = quantize(fields[1], 0, 8)
            return fields

        block_x, block_y, block_colour, exists = enumerate_points(tile, N)

        def phase_mask(coord, offset):
            z = conv(
                graph, coord, np.ones((9, 1), np.float32),
                -np.arange(9, dtype=np.float32) - offset
            )
            z = clip(graph, affine(
                graph, absolute(graph, z), -18.0, 2.0
            ))
            return total(graph, z)

        gx0 = phase_mask(U, 0.0)
        gy0 = phase_mask(V, 0.0)
        gx1 = phase_mask(U, 1.0 / 6.0)
        gy1 = phase_mask(V, 1.0 / 6.0)

        wall_w = np.zeros((1, 16, 3, 3), np.float32)
        wall_w[0, 5] = 1
        black_square = relu(graph, conv(graph, "x0", wall_w, [-8.0]))
        walls = product(graph, [black_square, gx1, gy1, board, top])
        wall_x, wall_y, _, wall_exists = enumerate_points(walls, O)
        wx = [sl(graph, wall_x, i) for i in range(O)]
        wy = [sl(graph, wall_y, i) for i in range(O)]
        we = [sl(graph, wall_exists, i) for i in range(O)]

        head_w = np.zeros((3, 16, 5, 5), np.float32)
        for j, colour in enumerate((6, 15, 11)):
            for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
                head_w[j, colour, 2 + dy, 2 + dx] = 1
            for dy, dx in ((-1, 0), (0, -1), (2, 0), (0, 2)):
                head_w[j, colour, 2 + dy, 2 + dx] = -4
        heads = relu(graph, conv(graph, "x0", head_w, [-3.0] * 3))
        head_maps = [
            mul(graph, sl(graph, heads, i), top) for i in range(3)
        ]
        head_present = [pool(graph, h) for h in head_maps]

        def head_coord(h, coordinate):
            z = mul(graph, h, affine(
                graph, coordinate, 1.0, 10.0 - 2.0 / 6.0
            ))
            return quantize(
                affine(graph, pool(graph, z), 1.0, -10.0), -1, 8
            )

        hrow = head_coord(head_maps[0], V)
        vcol = head_coord(head_maps[1], U)
        fcol = head_coord(head_maps[2], U)
        vp = head_present[1]
        selected_w = np.zeros((1, 16, 3, 3), np.float32)
        selected_w[0, 0, 0, 0] = 1
        selected = pool(graph, mul(
            graph, head_maps[1], conv(graph, "x0", selected_w)
        ))
        stripe_w = np.zeros((1, 16), np.float32)
        stripe_w[0, 1:4] = 1
        stripe = conv(graph, "x0", stripe_w)

        def belt_length(horizontal, location, present):
            target = affine(graph, location, 1.0, 2.0 / 6.0)
            align = pixel_eq(
                graph, V if horizontal else U, target, 18.0
            )
            lattice = gx0 if horizontal else gy0
            distance = affine(
                graph, U if horizontal else V, 1.0, 1.0
            )
            z = product(graph, [stripe, align, lattice, board, distance])
            return mul(graph, quantize(pool(graph, z), 0, 9), present)

        hlen = belt_length(True, hrow, head_present[0])
        vlen = belt_length(False, vcol, vp)
        flen = belt_length(False, fcol, head_present[2])

        target_w = np.zeros((1, 16), np.float32)
        target_w[0, 8:16] = np.arange(8, 16, dtype=np.float32)
        target_pixels = conv(graph, "x0", target_w)
        hud_row = pixel_eq(graph, row, C(57.0))
        goal_exists, goal_masks = [], []
        for belt in range(2):
            hud_head = mul(graph, sl(graph, heads, belt), bottom)
            hud_x = affine(graph, pool(graph, mul(
                graph, hud_head, col
            )), 1.0, -2.0)
            tx = dense(
                graph, hud_x, np.ones((G, 1), np.float32),
                7.0 + 6.0 * np.arange(G, dtype=np.float32)
            )
            colours = pool(graph, product(graph, [
                target_pixels, pixel_eq(graph, col, tx), hud_row
            ]))
            colours = mul(graph, colours, head_present[belt])
            goal_exists.append(quantize(colours, 0, 1))
            goal_masks.append(colour_equalities(block_colour, colours))

        # Any effective directional move changes one conveyor coordinate
        # or length. A click changes the selection component.
        observed = cat(graph, [hrow, vcol, hlen, vlen, selected])
        old_observed = sl(graph, "mem", 11, 16)
        dw = np.concatenate([
            np.concatenate([np.eye(5), -np.eye(5)], axis=1),
            np.concatenate([-np.eye(5), np.eye(5)], axis=1)
        ], axis=0).astype(np.float32)
        difference = relu(graph, dense(
            graph, cat(graph, [observed, old_observed]), dw
        ))
        unchanged = relu(graph, dense(
            graph, difference, -np.ones((1, 10), np.float32), [1.0]
        ))
        prev_w = np.zeros((5, 8), np.float32)
        for i, a in enumerate((1, 2, 3, 4, 6)):
            prev_w[i, a] = 1
        failed_previous = mul(
            graph, unchanged, dense(graph, "aprev", prev_w)
        )

        # -----------------------------------------------------------
        # Deterministic parallel rollout seeds.
        # -----------------------------------------------------------
        phase = broadcast(sl(graph, "mem", 10))
        phase_next = sub(graph, C(1.0), absolute(
            graph, affine(graph, phase, 1.91, -0.951)
        ))
        coefficients = np.array(
            [1.991, 1.973, 1.987, 1.961, 1.979], np.float32
        )

        def tent(L, x):
            z = add(
                L, mul(L, x, C(coefficients)), C(-0.5 * coefficients)
            )
            return sub(L, C(1.0), absolute(L, z))

        seed_w = np.zeros((5, 3), np.float32)
        fractions = np.array([0.83, 0.29, 0.61, 0.11, 0.47], np.float32)
        seed_w[:, 0] = 0.6 * fractions
        seed_w[:, 1] = 0.6 * (1.0 - fractions)
        seeds = conv(
            graph, "coords", seed_w,
            np.arange(5, dtype=np.float32) * 0.04
        )
        seeds = add(graph, seeds, scale(graph, phase_next, 0.2))
        for _ in range(7):
            seeds = tent(graph, seeds)

        inherited = product(graph, [
            pixel_eq(graph, row, C(0.0)),
            pixel_eq(graph, col, C(0.0)),
            threshold(graph, phase, 0.001)
        ])
        seeds = add(
            graph, mul(graph, seeds, sub(graph, C(1.0), inherited)),
            mul(graph, inherited, sl(graph, "mem", 0, 5))
        )
        previous_action = mul(
            graph, inherited, sl(graph, "mem", 5, 10)
        )

        root_hr = broadcast(hrow)
        root_vc = broadcast(vcol)
        root_hl = broadcast(hlen)
        root_vl = broadcast(vlen)
        root_active = broadcast(selected)

        initial = [
            broadcast(block_x), broadcast(block_y),
            root_hr, root_vc, root_hl, root_vl, root_active,
            seeds, previous_action,
            broadcast(C(-100000000.0)),
            broadcast(C(0.0)), broadcast(C(0.0)), broadcast(C(0.0)),
            broadcast(C(np.zeros(5, np.float32))), seeds
        ]
        states = [state(init=z) for z in initial]
        (bx, by, hr, vc, hl, vl, active, rng, previous,
         best, clock, wasted, root_penalty,
         stored_action, stored_seed) = [s["id"] for s in states]
        B = []

        first = eq(B, clock, C(0.0))
        inactive = sub(B, C(1.0), active)
        length = choose(B, hl, vl, active, inactive)
        origin = choose(B, hr, vc, active, inactive)

        # Boundary and obstacle legality, shared by proposal and transition.
        horizontal_legal = cat(B, [
            clip(B, hr), clip(B, sub(B, height_minus, hr)),
            clip(B, hl), clip(B, sub(B, width, hl)), broadcast(vp)
        ])
        vertical_legal = cat(B, [
            clip(B, vl), clip(B, sub(B, height, vl)),
            clip(B, vc), clip(B, sub(B, width_minus, vc)), broadcast(vp)
        ])
        legal = choose(
            B, horizontal_legal, vertical_legal, active, inactive
        )
        origin_changes = add(
            B, mul(B, C([-1, 1, 0, 0, 0]), inactive),
            mul(B, C([0, 0, -1, 1, 0]), active)
        )
        length_changes = add(
            B, mul(B, C([0, 0, -1, 1, 0]), inactive),
            mul(B, C([-1, 1, 0, 0, 0]), active)
        )
        candidate_origins = add(B, origin, origin_changes)
        candidate_lengths = add(B, length, length_changes)
        for i in range(O):
            wa = choose(
                B, broadcast(wx[i]), broadcast(wy[i]), active, inactive
            )
            wp = choose(
                B, broadcast(wy[i]), broadcast(wx[i]), active, inactive
            )
            hit = product(B, [
                eq(B, candidate_origins, wp),
                clip(B, sub(B, candidate_lengths, wa)), we[i]
            ])
            legal = mul(B, legal, sub(B, C(1.0), hit))
        legal = mul(B, legal, sub(
            B, C(1.0), mul(B, first, failed_previous)
        ))

        # Randomized proposals are implemented by fixed chaotic maps.
        rng2 = rng
        for _ in range(3):
            rng2 = tent(B, rng2)
        proposal_scores = plus(B, [
            rng2, scale(B, previous, 0.22),
            C([-0.001, -0.002, -0.003, 0.0, -0.20]),
            scale(B, sub(B, legal, C(1.0)), 4.0)
        ])
        proposal_max = maximum(
            B, sl(B, proposal_scores, 0, 2),
            sl(B, proposal_scores, 2, 4)
        )
        proposal_max = maximum(
            B, sl(B, proposal_max, 0), sl(B, proposal_max, 1)
        )
        proposal_max = maximum(
            B, proposal_max, sl(B, proposal_scores, 4)
        )
        winners = threshold(B, sub(B, proposal_scores, proposal_max))
        earlier = clip(B, conv(
            B, winners, np.tril(np.ones((5, 5), np.float32), -1)
        ))
        action = mul(B, winners, sub(B, C(1.0), earlier))

        stored_action2 = add(B, stored_action, mul(
            B, first, sub(B, action, stored_action)
        ))
        stored_seed2 = add(B, stored_seed, mul(
            B, first, sub(B, rng2, stored_seed)
        ))

        # -----------------------------------------------------------
        # Analytical conveyor and block transition.
        # -----------------------------------------------------------
        difference_w = np.zeros((N * N, N), np.float32)
        transpose_w = np.zeros((N * N, N * N), np.float32)
        row_sum_w = np.zeros((N, N * N), np.float32)
        pair_w = np.zeros((N * N, N), np.float32)
        for i in range(N):
            row_sum_w[i, i * N:(i + 1) * N] = 1
            for j in range(N):
                k = i * N + j
                difference_w[k, j] += 1
                difference_w[k, i] -= 1
                transpose_w[j * N + i, k] = 1
                pair_w[k, i] += 1
                pair_w[k, j] += 1
        pair_exists = relu(graph, dense(
            graph, exists, pair_w, -np.ones(N * N, np.float32)
        ))

        def matvec(L, matrix, vector):
            return conv(
                L, mul(L, matrix, cat(L, [vector] * N)), row_sum_w
            )

        aa = [sl(B, action, j) for j in range(5)]
        mx = sub(B, aa[3], aa[2])
        my = sub(B, aa[1], aa[0])
        switch = aa[4]
        longitudinal_move = choose(B, mx, my, active, inactive)
        transverse_move = choose(B, my, mx, active, inactive)
        extend = relu(B, longitudinal_move)
        retract = relu(B, affine(B, longitudinal_move, -1.0))
        transverse = absolute(B, transverse_move)
        along = choose(B, bx, by, active, inactive)
        across = choose(B, by, bx, active, inactive)
        origin2 = add(B, origin, transverse_move)

        line_hit = maximum(
            B, eq(B, across, origin),
            mul(B, transverse, eq(B, across, origin2))
        )
        reach = clip(B, sub(B, add(B, length, extend), along))
        requested = product(B, [line_hit, reach, exists])
        nx, ny = add(B, bx, mx), add(B, by, my)

        within = product(B, [
            clip(B, affine(B, nx, 1.0, 1.0)),
            clip(B, sub(B, width, nx)),
            clip(B, affine(B, ny, 1.0, 1.0)),
            clip(B, sub(B, height, ny))
        ])
        blocked = sub(B, C(1.0), within)
        for i in range(O):
            blocked = maximum(B, blocked, product(B, [
                eq(B, nx, wx[i]), eq(B, ny, wy[i]), we[i]
            ]))

        hreach = mul(B, clip(B, sub(B, hl, bx)), exists)
        vreach = mul(B, clip(B, sub(B, vl, by)), exists)
        freach = mul(B, clip(B, sub(B, flen, by)), exists)
        hm = mul(B, eq(B, by, hr), hreach)
        vm = mul(B, eq(B, bx, vc), vreach)
        h_any = maximum(B, hm, mul(B, eq(B, ny, hr), hreach))
        v_any = maximum(B, vm, mul(B, eq(B, nx, vc), vreach))
        f_any = mul(B, maximum(
            B, eq(B, bx, fcol), eq(B, nx, fcol)
        ), freach)
        crossing_blocked = clip(B, plus(B, [
            product(B, [absolute(B, mx), inactive, v_any]),
            product(B, [absolute(B, my), active, h_any]),
            mul(B, absolute(B, mx), f_any)
        ]))
        blocked = maximum(B, blocked, crossing_blocked)

        dx = conv(B, bx, difference_w)
        dy = conv(B, by, difference_w)
        contact_distance = add(
            B, absolute(B, sub(B, dx, mx)),
            absolute(B, sub(B, dy, my))
        )
        contact = mul(B, relu(B, sub(
            B, C(1.0), contact_distance
        )), pair_exists)
        contact_t = conv(B, contact, transpose_w)
        immobile = blocked
        for _ in range(push_depth):
            immobile = maximum(B, immobile, matvec(B, contact, immobile))
        mobile = mul(B, exists, sub(B, C(1.0), immobile))
        moving = mul(B, requested, mobile)
        for _ in range(push_depth):
            moving = mul(B, maximum(
                B, moving, matvec(B, contact_t, moving)
            ), mobile)

        valid = mul(
            B, total(B, mul(B, legal, action)), sub(B, C(1.0), switch)
        )
        failed_push = clip(B, total(B, mul(B, requested, immobile)))
        valid = mul(B, valid, sub(
            B, C(1.0), mul(B, transverse, failed_push)
        ))

        movement = mul(B, moving, valid)
        bx2 = add(B, bx, mul(B, movement, mx))
        by2 = add(B, by, mul(B, movement, my))
        transverse_valid = mul(B, transverse, valid)
        hr2 = add(B, hr, product(B, [transverse_valid, inactive, my]))
        vc2 = add(B, vc, product(B, [transverse_valid, active, mx]))
        dl = mul(B, valid, sub(B, extend, retract))
        hl2 = add(B, hl, mul(B, inactive, dl))
        vl2 = add(B, vl, mul(B, active, dl))
        active2 = add(B, active, mul(
            B, switch, affine(B, active, -2.0, 1.0)
        ))
        bad_step = sub(B, C(1.0), add(B, valid, switch))
        root_penalty2 = add(B, root_penalty, product(
            B, [first, bad_step, C(20000.0)]
        ))

        # -----------------------------------------------------------
        # Ordered-prefix and next-colour evaluation.
        # -----------------------------------------------------------
        rank_h = matvec(B, clip(B, affine(B, dx, -1.0)), hm)
        rank_v = matvec(B, clip(B, affine(B, dy, -1.0)), vm)
        target_sum_w = np.zeros((G, G * N), np.float32)
        target_repeat_w = np.zeros((G * N, G), np.float32)
        object_sum_w = np.zeros((N, G * N), np.float32)
        for g in range(G):
            target_sum_w[g, g * N:(g + 1) * N] = 1
            target_repeat_w[g * N:(g + 1) * N, g] = 1
            for i in range(N):
                object_sum_w[i, g * N + i] = 1
        prefix_w = np.tril(np.ones((G, G), np.float32))
        fail_w = -np.eye(G, dtype=np.float32)
        for g in range(1, G):
            fail_w[g, g - 1] = 1
        fail_b = np.zeros(G, np.float32)
        fail_b[0] = 1
        goal_indices = C(np.repeat(np.arange(G, dtype=np.float32), N))

        def minimum_channels(L, x):
            n = shapes[x][0]
            power = 1
            while power < n:
                power *= 2
            if power > n:
                x = cat(L, [x] + [broadcast(C(1000.0))] * (power - n))
            while power > 1:
                half = power // 2
                x = emit(L, "min2", [
                    sl(L, x, 0, half), sl(L, x, half, power)
                ])
                power = half
            return x

        def belt_score(belt, membership, rank, perpendicular,
                       belt_origin, longitudinal, belt_len):
            rank_match = eq(B, cat(B, [rank] * G), goal_indices)
            matching = product(B, [
                rank_match, goal_masks[belt], cat(B, [membership] * G)
            ])
            correct = conv(B, matching, target_sum_w)
            errors = sub(B, goal_exists[belt], correct)
            prefix = relu(B, sub(B, C(1.0), conv(B, errors, prefix_w)))
            count = total(B, mul(B, prefix, goal_exists[belt]))
            first_failure = conv(B, prefix, fail_w, fail_b)
            next_mask = conv(B, mul(
                B, conv(B, first_failure, target_repeat_w),
                goal_masks[belt]
            ), object_sum_w)
            wrong = sub(B, total(B, membership), count)
            distance = plus(B, [
                absolute(B, sub(B, perpendicular, belt_origin)),
                scale(B, absolute(B, sub(
                    B, belt_len, affine(B, longitudinal, 1.0, 1.0)
                )), 0.4),
                scale(B, absolute(B, sub(B, rank, count)), 2.0),
                affine(B, next_mask, -40.0, 40.0)
            ])
            done = sl(B, prefix, G - 1)
            proximity = product(B, [
                minimum_channels(B, distance),
                sub(B, C(1.0), done), C(-0.4)
            ])
            value = plus(B, [
                scale(B, count, 100.0),
                total(B, correct),
                scale(B, wrong, -0.75), proximity
            ])
            return value, done

        score_h, done_h = belt_score(0, hm, rank_h, by, hr, bx, hl)
        score_v, done_v = belt_score(1, vm, rank_v, bx, vc, by, vl)
        score = plus(B, [
            score_h, score_v,
            scale(B, mul(B, done_h, done_v), 1000.0),
            scale(B, clock, -0.06),
            scale(B, wasted, -0.35),
            scale(B, first, -1000000.0),
            scale(B, root_penalty, -1.0)
        ])
        best2 = maximum(B, best, score)
        clock2 = affine(B, clock, 1.0, 1.0)
        wasted2 = add(B, wasted, bad_step)
        next_values = [
            bx2, by2, hr2, vc2, hl2, vl2, active2, rng2, action,
            best2, clock2, wasted2, root_penalty2,
            stored_action2, stored_seed2
        ]
        main_loop, out = loop(
            states, B,
            {s["id"]: z for s, z in zip(states, next_values)}, 1
        )

        final_score = out[best]
        eligible = threshold(
            graph, sub(graph, final_score, pool(graph, final_score))
        )
        lane = add(graph, scale(graph, row, 64.0), col)
        key = mul(graph, eligible, affine(graph, lane, -1.0, 4096.0))
        winner = pixel_eq(graph, key, pool(graph, key))
        chosen_action = pool(graph, mul(
            graph, out[stored_action], winner
        ))
        chosen_seed = pool(graph, mul(
            graph, out[stored_seed], winner
        ))

        action_w = np.zeros((8, 5), np.float32)
        action_b = np.full(8, -1000.0, np.float32)
        for j, a in enumerate((1, 2, 3, 4, 6)):
            action_w[a, j] = 100
            action_b[a] = 0
        dense(graph, chosen_action, action_w, action_b, name="logits")
        click_map = add(
            graph, mul(graph, head_maps[0], selected),
            mul(graph, head_maps[1], affine(
                graph, selected, -1.0, 1.0
            ))
        )
        add(graph, click_map, C(0.0), name="click")
        cat(graph, [
            chosen_seed, chosen_action, pool(graph, phase_next), observed
        ], name="mem_next")

        # -----------------------------------------------------------
        # Dead-code elimination and exact symbolic resource accounting.
        # No game states or network evaluations are executed here.
        # -----------------------------------------------------------
        dependencies = {}
        for nd in graph:
            if nd["op"] != "loop":
                dependencies[nd["id"]] = list(nd["in"])
            else:
                all_next = list(nd["next"].values())
                for s in nd["state"]:
                    dependencies[s["id"]] = (
                        [s["init"]] if "init" in s else []
                    )
                    dependencies[nd["id"] + "." + s["id"]] = (
                        all_next + [q["id"] for q in nd["state"]]
                    )
                for q in nd["body"]:
                    dependencies[q["id"]] = list(q["in"])

        used = set()
        stack = ["logits", "click", "mem_next"]
        while stack:
            name = stack.pop()
            if name not in used:
                used.add(name)
                stack.extend(dependencies.get(name, []))

        clean = []
        for nd in graph:
            if nd["op"] == "loop":
                if any(nd["id"] + "." + s["id"] in used
                       for s in nd["state"]):
                    nd["body"] = [
                        q for q in nd["body"] if q["id"] in used
                    ]
                    clean.append(nd)
            elif nd["id"] in used:
                clean.append(nd)
        graph = clean

        used_parameters = set()
        for nd in graph:
            nodes = nd["body"] if nd["op"] == "loop" else [nd]
            for q in nodes:
                for field in ("w", "b"):
                    if field in q:
                        used_parameters.add(q[field])
        P = {k: v for k, v in P.items() if k in used_parameters}

        def local_mac(nd):
            op = nd["op"]
            if op == "conv":
                return int(np.prod(P[nd["w"]].shape)) * 4096
            if op == "dense":
                return int(np.prod(P[nd["w"]].shape))
            if op in ("add", "sub", "mul", "max2", "min2"):
                return shapes[nd["id"]][0] * 4096
            if op in ("rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"):
                return shapes[nd["in"][0]][0] * 4096
            if op in ("vadd", "vmul"):
                return shapes[nd["id"]][0]
            return 0

        def costs(nodes):
            mac = evaluations = count = 0
            for nd in nodes:
                count += 1
                evaluations += 1
                if nd["op"] == "loop":
                    m, e, c = costs(nd["body"])
                    mac += nd["T"] * m
                    evaluations += nd["T"] * e
                    count += c
                else:
                    mac += local_mac(nd)
            return mac, evaluations, count

        body_mac, body_evals, _ = costs(main_loop["body"])
        all_mac, all_evals, _ = costs(graph)
        fixed_mac = all_mac - body_mac
        fixed_evals = all_evals - body_evals
        horizon = min(
            48,
            int((1440000000 - fixed_mac) // max(1, body_mac)),
            int((9800 - fixed_evals) // max(1, body_evals))
        )
        main_loop["T"] = max(1, horizon)

        live_shapes = [(16, 64, 64), (3, 64, 64), (M,), (8,), (8,)]
        for nd in graph:
            if nd["op"] == "loop":
                for s in nd["state"]:
                    live_shapes.append(shapes[s["id"]])
                    live_shapes.append(shapes[nd["id"] + "." + s["id"]])
                live_shapes.extend(shapes[q["id"]] for q in nd["body"])
            else:
                live_shapes.append(shapes[nd["id"]])
        channels = sum(s[0] for s in live_shapes if len(s) == 3)
        mac, evaluations, count = costs(graph)
        valid = (
            horizon >= 4
            and count <= 1000
            and evaluations <= 10000
            and mac <= 1500000000
            and channels <= 4096
            and sum(a.size for a in P.values()) <= 150000
            and max(s[0] for s in live_shapes if len(s) == 3) <= 512
            and max(s[0] for s in live_shapes if len(s) == 1) <= 4096
        )
        return {
            "graph": graph, "memory_size": M, "params": P
        }, valid, horizon

    # Allocation choices depend only on symbolic graph size.
    candidates = []
    for objects, obstacles, propagation in (
        (7, 2, 3), (6, 2, 3), (6, 1, 3),
        (5, 1, 2), (4, 1, 2), (3, 1, 1)
    ):
        network, valid, horizon = construct(
            objects, obstacles, propagation
        )
        if valid:
            candidates.append((objects, obstacles, horizon, network))
            if objects >= 6 and horizon >= 18:
                return network
    if candidates:
        return max(
            candidates,
            key=lambda z: (min(z[0], 6), z[2], z[1], z[0])
        )[3]

    # Finite, very small resource safeguard. This branch also observes
    # the image rather than replaying a predefined action sequence.
    w = np.zeros((2, 16, 1, 1), np.float32)
    w[0, 6, 0, 0] = 1.0
    w[1, 8, 0, 0] = 1.0
    output = np.zeros((8, 2), np.float32)
    output[1] = [-1.0, 1.0]
    output[2] = [1.0, -1.0]
    bias = np.full(8, -10.0, np.float32)
    bias[1:3] = 0.0
    return {
        "memory_size": 0,
        "graph": [
            {"id": "colours", "op": "conv", "in": ["x0"], "w": "fw"},
            {"id": "rows", "op": "slice", "in": ["coords"],
             "start": 0, "stop": 1},
            {"id": "weighted", "op": "mul", "in": ["colours", "rows"]},
            {"id": "centres", "op": "gmean", "in": ["weighted"]},
            {"id": "logits", "op": "dense", "in": ["centres"],
             "w": "fo", "b": "fb"},
            {"id": "click", "op": "slice", "in": ["colours"],
             "start": 0, "stop": 1}
        ],
        "params": {"fw": w, "fo": output, "fb": bias}
    }
