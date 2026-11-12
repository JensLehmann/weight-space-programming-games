import numpy as np


def build():
    def construct(advanced):
        P = {}
        G = []
        target = G
        shapes = {
            "x0": ("s", 16),
            "coords": ("s", 3),
            "mem": ("v", 15),
            "aprev": ("v", 8),
            "avail": ("v", 8),
        }
        pcache = {}
        ocache = {}
        serial = 0

        def parameter(a):
            a = np.asarray(a, dtype=np.float32)
            key = (a.shape, a.tobytes())
            if key not in pcache:
                name = "p%d" % len(P)
                P[name] = a
                pcache[key] = name
            return pcache[key]

        def node(op, xs, **fields):
            nonlocal serial
            key = (id(target), op, tuple(xs), tuple(sorted(fields.items())))
            if key in ocache:
                return ocache[key]
            name = "n%d" % serial
            serial += 1
            nd = {"id": name, "op": op, "in": list(xs)}
            nd.update(fields)
            target.append(nd)
            ocache[key] = name

            if op == "conv":
                shapes[name] = ("s", P[fields["w"]].shape[0])
            elif op == "dense":
                shapes[name] = ("v", P[fields["w"]].shape[0])
            elif op in ("slice", "vslice"):
                shapes[name] = (
                    shapes[xs[0]][0], fields["stop"] - fields["start"]
                )
            elif op in ("concat", "vconcat"):
                shapes[name] = (
                    shapes[xs[0]][0], sum(shapes[x][1] for x in xs)
                )
            elif op in ("gmax", "gmean"):
                shapes[name] = ("v", shapes[xs[0]][1])
            elif op in ("add", "sub", "mul", "max2", "min2"):
                shapes[name] = ("s", max(shapes[x][1] for x in xs))
            elif op in ("vadd", "vmul"):
                shapes[name] = ("v", max(shapes[x][1] for x in xs))
            else:
                shapes[name] = shapes[xs[0]]
            return name

        def conv(x, w, b=None):
            fields = {"w": parameter(w)}
            if b is not None:
                fields["b"] = parameter(b)
            return node("conv", [x], **fields)

        def dense(x, w, b=None):
            fields = {"w": parameter(w)}
            if b is not None:
                fields["b"] = parameter(b)
            return node("dense", [x], **fields)

        def sl(x, a, b):
            return node(
                "slice" if shapes[x][0] == "s" else "vslice",
                [x], start=a, stop=b,
            )

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

        def relu(x):
            return node("relu", [x])

        def absolute(x):
            return node("abs", [x])

        def gmax(x):
            return node("gmax", [x])

        one = sl("coords", 2, 3)

        def spatial(x):
            return x if shapes[x][0] == "s" else mul(one, x)

        def affine(x, a=1.0, b=0.0):
            x = spatial(x)
            if a == 1 and b == 0:
                return x
            c = shapes[x][1]
            return conv(
                x,
                (np.eye(c, dtype=np.float32) * np.float32(a))[
                    :, :, None, None
                ],
                np.full(c, b, np.float32),
            )

        def clip(x):
            return minimum(relu(x), one)

        def inv(x):
            return affine(x, -1, 1)

        def ge(x, t):
            return clip(affine(x, 1, 1 - t))

        def greater(x, t=0.0):
            return clip(affine(x, 128, 1 - 128 * t))

        def eq(x, value=0.0):
            return clip(affine(
                absolute(affine(x, 1, -value)), -4, 2
            ))

        def mix(a, b, s):
            return add(a, mul(sub(b, a), s))

        def total(xs):
            if len(xs) == 1:
                return xs[0]
            if len(xs) == 2:
                return add(xs[0], xs[1])
            if any(shapes[x][1] != 1 for x in xs):
                y = xs[0]
                for x in xs[1:]:
                    y = add(y, x)
                return y
            return conv(
                cat(xs), np.ones((1, len(xs), 1, 1), np.float32)
            )

        def glob(x):
            return spatial(gmax(x))

        def at(x, point):
            return glob(mul(x, point))

        def kernel(x, entries, k=7, bias=None):
            w = np.zeros((1, 1, k, k), np.float32)
            h = k // 2
            for dy, dx, v in entries:
                w[0, 0, h + dy, h + dx] += v
            b = None if bias is None else np.array([bias], np.float32)
            return conv(x, w, b)

        def detect(x, entries, threshold, k=7):
            # These templates have a maximum score equal to threshold.
            return relu(kernel(x, entries, k, 1 - threshold))

        def shift(x, dy, dx):
            while dy or dx:
                sy = max(-3, min(3, dy))
                sx = max(-3, min(3, dx))
                k = 2 * max(abs(sy), abs(sx)) + 1
                w = np.zeros((1, 1, k, k), np.float32)
                h = k // 2
                w[0, 0, h - sy, h - sx] = 1
                x = conv(x, w)
                dy -= sy
                dx -= sx
            return x

        def sample(x, dy, dx):
            return shift(x, -dy, -dx)

        def rectangle(point, h, w):
            cy, cx = h // 2, w // 2
            center = shift(point, cy, cx)
            filt = np.zeros((1, 1, 7, 7), np.float32)
            for y in range(h):
                for x in range(w):
                    filt[0, 0, 3 - (y - cy), 3 - (x - cx)] = 1
            return minimum(conv(center, filt), one)

        def top_left(x):
            return relu(kernel(
                x, [(0, 0, 1), (-1, 0, -1), (0, -1, -1)], 3
            ))

        def walk(x):
            return detect(clip(x), [
                (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)
            ], 4, 3)

        def complete4(color, player_pixels):
            starts = detect(
                clip(add(color, player_pixels)),
                [(y, x, 1) for y in range(4) for x in range(4)],
                16,
            )
            return minimum(add(color, rectangle(starts, 4, 4)), one)

        row = conv(
            "coords",
            np.array([[[[63]], [[0]], [[0]]]], np.float32),
        )
        col = conv(
            "coords",
            np.array([[[[0]], [[63]], [[0]]]], np.float32),
        )
        rank = total([affine(row, 64), col, one])
        zero = affine(one, 0)
        C = {c: sl("x0", c, c + 1) for c in
             (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)}

        play = node("colmax", [C[4]])
        ui = inv(play)
        pp = mul(C[14], play)
        player = top_left(pp)
        initialized = spatial(sl("mem", 0, 1))

        def hat(c):
            entries = [(dy, dx, 1)
                       for dy in (-1, 0, 1, 2, 3)
                       for dx in range(-3, 4)]
            entries += [(-2, dx, -1) for dx in range(-3, 4)]
            return detect(mul(C[c], ui), entries, 35)

        buttons = {
            "a": hat(8), "b": hat(9), "c": hat(6), "d": hat(15),
            "e": mul(C[2], ui), "f": mul(C[11], ui),
        }
        have = {k: glob(v) for k, v in buttons.items()}

        # Portal locations, including a player-covered endpoint.
        visible_ports = walk(mul(add(C[6], C[7]), play))
        port_count = affine(
            spatial(node("gmean", [visible_ports])), 4096
        )
        initial_ports = minimum(add(
            visible_ports, mul(player, eq(port_count, 1))
        ), one)
        initial_exists = glob(initial_ports)
        hi0 = glob(mul(rank, initial_ports))
        lo0 = mul(
            affine(
                glob(mul(affine(rank, -1, 4097), initial_ports)),
                -1, 4097,
            ),
            initial_exists,
        )
        port_lo = mix(lo0, spatial(sl("mem", 1, 2)), initialized)
        port_hi = mix(hi0, spatial(sl("mem", 2, 3)), initialized)
        port_exists = mix(
            initial_exists, spatial(sl("mem", 3, 4)), initialized
        )
        ports = mul(
            minimum(add(
                eq(sub(rank, port_lo)), eq(sub(rank, port_hi))
            ), one),
            port_exists,
        )
        port_pixels = rectangle(ports, 2, 2)
        usable_ports = mul(ports, have["c"])

        # A pickup is an overlay on a red landing, not a closed gate.
        key_entries = [(0, 1, 1), (0, 2, 1)]
        key_entries += [(1, dx, 1) for dx in range(4)]
        key_entries += [(0, 0, -1), (0, 3, -1)]
        key_entries += [(2, dx, -1) for dx in range(4)]
        key = detect(mul(C[15], play), key_entries, 6)
        key_pixels = rectangle(key, 2, 4)
        use_key = mul(glob(key), inv(have["d"]))
        goal = mix(
            top_left(mul(C[11], play)), key, use_key
        )

        base_visible = clip(total([
            mul(C[2], play), mul(C[11], play),
            port_pixels, key_pixels,
        ]))
        stable_base = complete4(base_visible, pp)

        # Restore covered pivots and transpose their blue arms.
        pivot_pixels = complete4(mul(C[13], play), pp)
        pivots = top_left(pivot_pixels)
        near_pivot = pivot_pixels
        for _ in range(3):
            near_pivot = minimum(conv(
                near_pivot, np.ones((1, 1, 7, 7), np.float32)
            ), one)
        blue = mul(complete4(mul(C[8], play), pp), near_pivot)

        arms = []
        for length in (2, 4, 6, 8):
            specs = [
                ((0, -length), (3, -length), (-length, 0), (2, 4)),
                ((0, 3 + length), (3, 3 + length),
                 (length + 2, 0), (2, 4)),
                ((-length, 0), (-length, 3),
                 (0, -length), (4, 2)),
                ((3 + length, 0), (3 + length, 3),
                 (0, length + 2), (4, 2)),
            ]
            for p, q, offset, dims in specs:
                evidence = minimum(
                    sample(blue, p[0], p[1]),
                    sample(blue, q[0], q[1]),
                )
                marker = mul(pivots, evidence)
                arms.append(rectangle(
                    shift(marker, offset[0], offset[1]),
                    dims[0], dims[1],
                ))
        alternate_blue = mix(
            blue, minimum(total(arms), one), have["a"]
        )

        def gate_states(c, enabled):
            raw = mul(C[c], play)
            if c == 15:
                raw = relu(sub(raw, key_pixels))
            pixels = complete4(raw, pp)
            opened = walk(pixels)
            diagonal = kernel(pixels, [
                (0, 0, 1), (1, 1, 1),
                (0, 1, -1), (1, 0, -1),
            ], 3)
            other = relu(affine(absolute(diagonal), 1, -1))
            return opened, mix(opened, other, enabled)

        Bgate = gate_states(9, have["b"])
        Dgate = gate_states(15, have["d"])

        phase0 = dense(
            "aprev", np.zeros((6, 8), np.float32),
            np.array([1, 0, 0, 0, 0, 0], np.float32),
        )
        phase = phase0
        anchor = zero
        saved_frow = zero
        current_F = zero
        Fplanes = [zero]
        eactual = zero
        eplan = zero
        prepare_e = zero
        loose_ones = zero
        object_planes = [zero]
        object_union = zero
        physical_object = zero
        has_object = zero
        object_height = zero

        if advanced:
            p1 = mul(C[1], play)
            p12 = mul(C[12], play)
            p5 = mul(C[5], play)
            r1 = node("rowsum", [p1])
            r12 = node("rowsum", [p12])
            r15 = node("rowsum", [add(p1, p5)])

            def strip_top(x):
                return detect(node("rowmax", [x]), [
                    (0, 0, 1), (1, 0, 1),
                    (-1, 0, -1), (2, 0, -1),
                ], 2, 5)

            ft1 = mul(strip_top(p1), eq(r1, 6))
            ft12 = mul(strip_top(p12), eq(r12, 6))
            fr1 = minimum(add(ft1, shift(ft1, 1, 0)), one)
            fr12 = minimum(add(ft12, shift(ft12, 1, 0)), one)
            fvisible = minimum(add(
                mul(p1, fr1), mul(p12, fr12)
            ), one)
            fpoint = top_left(fvisible)
            fseen = glob(fpoint)
            fcol = at(col, fpoint)
            frow = at(row, fpoint)
            flast = glob(ft12)

            erows = minimum(add(ge(r15, 14), ge(r12, 14)), one)
            evisible = mul(clip(total([p1, p5, p12])), erows)
            epoint = top_left(evisible)
            eseen = glob(epoint)
            ecol = at(col, epoint)
            erow = at(row, epoint)
            e_full = glob(mul(ge(r12, 14), erows))

            old_anchor = spatial(sl("mem", 4, 5))
            old_frow = spatial(sl("mem", 5, 6))
            seen_anchor = add(fcol, one)
            anchor = mix(
                seen_anchor, minimum(old_anchor, seen_anchor),
                ge(old_anchor, 1),
            )
            aligned = mul(
                mul(eseen, fseen),
                eq(sub(sub(frow, erow), affine(one, 0, 10))),
            )
            anchor = mix(
                anchor, minimum(anchor, add(ecol, one)), aligned
            )
            anchor = mix(anchor, affine(fcol, 1, -9), flast)
            anchor = mix(old_anchor, anchor, fseen)
            saved_frow = mix(old_frow, add(frow, one), fseen)
            f_present = mul(ge(anchor, 1), ge(saved_frow, 1))
            ax = affine(anchor, 1, -1)
            fy = affine(saved_frow, 1, -1)

            phase_visible = gmax(clip(affine(absolute(conv(
                sub(fcol, ax),
                np.ones((6, 1, 1, 1), np.float32),
                -2 * np.arange(6, dtype=np.float32),
            )), -4, 2)))
            old_phase = sl("mem", 6, 12)
            phase_prior = node("vadd", [
                node("vmul", [old_phase, gmax(initialized)]),
                node("vmul", [phase0, gmax(inv(initialized))]),
            ])
            phase = node("vadd", [
                node("vmul", [phase_visible, gmax(fseen)]),
                node("vmul", [phase_prior, gmax(inv(fseen))]),
            ])
            froot = mul(
                mul(eq(sub(col, ax)), eq(sub(row, fy))),
                f_present,
            )
            Fplanes = [
                rectangle(shift(froot, 0, 2 * i), 2, 6)
                for i in range(6)
            ]
            current_F = conv(
                mul(cat(Fplanes), phase),
                np.ones((1, 6, 1, 1), np.float32),
            )

            eroot_saved = mul(
                mul(
                    eq(sub(col, ax)),
                    eq(sub(row, affine(fy, 1, -10))),
                ),
                mul(f_present, have["e"]),
            )
            eroot = mix(epoint, eroot_saved, f_present)
            epiece = rectangle(eroot, 2, 4)
            Eplane = minimum(total([
                epiece, shift(epiece, 0, 4),
                shift(epiece, 0, 8), shift(epiece, 0, 12),
            ]), one)
            eactual = mix(
                mul(add(p1, p12), erows), Eplane, e_full
            )
            eplan = mix(eactual, Eplane, have["e"])
            prepare_e = mul(
                mul(mul(have["e"], eseen), inv(e_full)),
                affine(one, 0, 600),
            )
            loose_ones = relu(sub(
                sub(p1, mul(p1, erows)), fvisible
            ))
            remembered_frows = mul(
                minimum(add(
                    eq(sub(row, fy)),
                    eq(sub(row, affine(fy, 1, 1))),
                ), one),
                f_present,
            )

            # Detect the jaw and infer the payload's rail.
            jaw_entries = [
                (dy, dx, 1)
                for dy in (0, 1) for dx in range(4)
            ]
            jaw_entries += [
                (dy, dx, 1)
                for dy in (-1, 2) for dx in range(1, 4)
            ]
            jaw_entries += [(-1, 0, -1), (2, 0, -1)]
            jaw_color = mul(C[8], play)
            crusher = ge(total([
                kernel(jaw_color, jaw_entries),
                sample(jaw_color, -1, 4),
                sample(jaw_color, 2, 4),
            ]), 16)
            crusher_exists = glob(crusher)
            crusher_x = affine(at(col, crusher), 1, 9)
            crusher_y = affine(at(row, crusher), 1, 1)
            attached = mul(
                crusher_exists,
                inv(at(sample(C[8], -3, 3), crusher)),
            )

            raw_object = relu(sub(
                sub(p12, mul(p12, erows)),
                mul(p12, remembered_frows),
            ))
            has_object = mul(glob(raw_object), crusher_exists)

            def low_coordinate(coordinate, mask):
                return affine(
                    glob(mul(
                        affine(coordinate, -1, 63), mask
                    )),
                    -1, 63,
                )

            ox0 = low_coordinate(col, raw_object)
            oy0 = low_coordinate(row, raw_object)
            ox1 = at(col, raw_object)
            oy1 = at(row, raw_object)
            object_cx = affine(total([ox0, ox1, one]), .5)
            object_cy = affine(total([oy0, oy1, one]), .5)
            black = mul(
                mul(C[0], play), inv(eq(row, 63))
            )
            black_exists = glob(black)
            rail_x = mix(
                minimum(
                    crusher_x, affine(object_cx, 1, -12)
                ),
                affine(low_coordinate(col, black), 1, 8),
                black_exists,
            )
            rail_y = mix(
                minimum(crusher_y, object_cy),
                affine(low_coordinate(row, black), 1, 2),
                black_exists,
            )
            object_height = mul(
                ge(sub(object_cy, rail_y), 6), has_object
            )
            object_station = ge(sub(object_cx, rail_x), 6)

            left_root = mul(
                mul(
                    eq(sub(col, rail_x)),
                    eq(sub(row, rail_y)),
                ),
                has_object,
            )
            bar = rectangle(shift(left_root, -6, -1), 6, 4)
            left_top = minimum(total([
                bar, shift(bar, 6, 0),
                shift(left_root, -2, -3),
                rectangle(shift(left_root, -1, -3), 2, 2),
                shift(left_root, 1, -3),
            ]), one)
            right_top = shift(left_top, 0, 12)
            left_bottom = shift(left_top, 12, 0)
            right_bottom = shift(right_top, 12, 0)
            object_planes = [
                minimum(add(left_top, right_top), one),
                minimum(add(left_bottom, right_bottom), one),
            ]
            object_union = minimum(
                add(object_planes[0], object_planes[1]), one
            )
            physical_object = mix(
                mix(left_top, right_top, object_station),
                mix(left_bottom, right_bottom, object_station),
                object_height,
            )

        nf = len(Fplanes)
        no = len(object_planes)
        N = 8 * nf * no

        stable_plan = clip(total([
            stable_base, pivot_pixels, eplan, loose_ones,
        ]))
        stable_physical = clip(total([
            stable_base, pivot_pixels, eactual, loose_ones,
            current_F, physical_object,
        ]))
        physical_walk = maximum(
            walk(add(stable_physical, blue)),
            maximum(Bgate[0], Dgate[0]),
        )

        terrain = {}
        for f in range(nf):
            for o in range(no):
                for a in range(2):
                    terrain[f, o, a] = walk(total([
                        stable_plan, Fplanes[f], object_planes[o],
                        blue if a == 0 else alternate_blue,
                    ]))
        gate_masks = {
            (b, d): maximum(Bgate[b], Dgate[d])
            for b in range(2) for d in range(2)
        }
        masks = cat([
            maximum(
                terrain[f, o, bits & 1],
                gate_masks[(bits >> 1) & 1, (bits >> 2) & 1],
            )
            for f in range(nf)
            for o in range(no)
            for bits in range(8)
        ])

        # Shared local propagation and configuration transitions.
        neighbor_w = np.zeros((2, 2, 5, 5), np.float32)
        for c in range(2):
            for dy, dx in (
                (0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)
            ):
                neighbor_w[c, c, 2 + dy, 2 + dx] = 1

        transition_w = np.zeros((8, 24, 1, 1), np.float32)
        for bits in range(8):
            for bit in (1, 2, 4):
                transition_w[bits, bits ^ bit, 0, 0] = 1
            transition_w[bits, 8 + bits, 0, 0] = 1
            transition_w[bits, 16 + bits, 0, 0] = 1

        body = []
        target = body
        shapes["reach_state"] = ("s", N)
        shapes["distance_state"] = ("s", N)
        local = cat([
            conv(sl("reach_state", i, i + 2), neighbor_w)
            for i in range(0, N, 2)
        ])
        blocks = [
            sl("reach_state", 8 * i, 8 * (i + 1))
            for i in range(nf * no)
        ]
        transitions = []
        for f in range(nf):
            for o in range(no):
                current = blocks[no * f + o]
                other = blocks[no * f + ((o + 1) % no)]
                future = blocks[no * ((f + 1) % nf) + o]
                transitions.append(conv(
                    cat([current, other, future]),
                    transition_w,
                ))
        changed = cat(transitions)
        remote = mul(
            usable_ports, gmax(mul("reach_state", usable_ports))
        )
        # Explicit broadcast additions retain all configuration channels.
        drive = add(add(add(local, changed), remote), goal)
        reachable = minimum(mul(drive, masks), one)
        accumulated = add("distance_state", reachable)
        target = G

        G.append({
            "id": "planner", "op": "loop", "T": 1,
            "state": [
                {"id": "reach_state", "zeros": [N, 64, 64]},
                {"id": "distance_state", "zeros": [N, 64, 64]},
            ],
            "body": body,
            "next": {
                "reach_state": reachable,
                "distance_state": accumulated,
            },
        })
        shapes["planner.reach_state"] = ("s", N)
        shapes["planner.distance_state"] = ("s", N)
        D = "planner.distance_state"

        def selector(bits=0, advance=0, flip=False):
            selections = []
            for current_o in range(no):
                w = np.zeros((N, 6), np.float32)
                for f in range(nf):
                    future_o = (
                        (current_o + int(flip)) % no
                    )
                    idx = (
                        8 * (no * ((f + advance) % nf) + future_o)
                        + bits
                    )
                    w[idx, f] = 1
                selections.append(dense(phase, w))
            if no == 1:
                return selections[0]
            return node("vadd", [
                node("vmul", [
                    selections[0], gmax(inv(object_height))
                ]),
                node("vmul", [
                    selections[1], gmax(object_height)
                ]),
            ])

        current_selector = selector()

        def read(values, selection):
            return spatial(dense(
                node("vmul", [values, selection]),
                np.ones((1, N), np.float32),
            ))

        point_values = gmax(mul(D, player))
        scores = {
            "a": mul(read(point_values, selector(1)), have["a"]),
            "b": mul(read(point_values, selector(2)), have["b"]),
            "d": mul(read(point_values, selector(4)), have["d"]),
            "f": mul(
                read(point_values, selector(0, 1)),
                have["f"] if advanced else zero,
            ),
        }
        current_D = conv(
            mul(D, current_selector),
            np.ones((1, N, 1, 1), np.float32),
        )
        scores["c"] = mul(
            glob(mul(
                current_D, relu(sub(usable_ports, player))
            )),
            glob(mul(player, usable_ports)),
        )

        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        movement = []
        physical_neighbors = []
        object_neighbors = []
        object_walk = walk(object_union)
        for dy, dx in directions:
            destination = shift(player, dy, dx)
            movement.append(glob(mul(current_D, destination)))
            physical_neighbors.append(at(physical_walk, destination))
            object_neighbors.append(at(object_walk, destination))
        move_field = cat(movement)
        physical_field = cat(physical_neighbors)
        object_field = cat(object_neighbors)

        adjusted = {}
        for k, bias in (
            ("a", .08), ("b", .06), ("c", .04),
            ("d", .02), ("f", 0.0),
        ):
            adjusted[k] = add(
                scores[k], affine(ge(scores[k], 1), bias)
            )
        normal_click = total([
            mul(buttons[k], adjusted[k])
            for k in ("a", "b", "c", "d", "f")
        ] + [mul(buttons["e"], prepare_e)])
        normal_best = glob(normal_click)

        active = zero
        target_height = zero
        target_station = zero
        combined_click = normal_click

        if advanced:
            # Deterministic winner flags for four movement proposals.
            comparison_w = np.zeros((12, 4, 1, 1), np.float32)
            comparison_b = np.zeros(12, np.float32)
            groups = []
            j = 0
            for i in range(4):
                group = []
                for q in range(4):
                    if i == q:
                        continue
                    comparison_w[j, i, 0, 0] = 128
                    comparison_w[j, q, 0, 0] = -128
                    comparison_b[j] = 4 if i < q else -4
                    group.append(j)
                    j += 1
                groups.append(group)
            comparisons = clip(conv(
                move_field, comparison_w, comparison_b
            ))
            winner = cat([
                mul(
                    mul(
                        sl(comparisons, g[0], g[0] + 1),
                        sl(comparisons, g[1], g[1] + 1),
                    ),
                    sl(comparisons, g[2], g[2] + 1),
                )
                for g in groups
            ])
            sum4 = np.ones((1, 4, 1, 1), np.float32)
            winning_score = conv(mul(winner, move_field), sum4)
            py, px = at(row, player), at(col, player)
            next_y = add(py, conv(
                winner,
                np.array([-2, 2, 0, 0], np.float32).reshape(1, 4, 1, 1),
            ))
            next_x = add(px, conv(
                winner,
                np.array([0, 0, -2, 2], np.float32).reshape(1, 4, 1, 1),
            ))
            next_physical = conv(mul(winner, physical_field), sum4)
            next_object = conv(mul(winner, object_field), sum4)

            def square(c):
                entries = [
                    (dy, dx, 1)
                    for dy in range(-1, 3)
                    for dx in range(-1, 3)
                ]
                for q in range(-1, 3):
                    entries += [
                        (-2, q, -1), (3, q, -1),
                        (q, -2, -1), (q, 3, -1),
                    ]
                return detect(mul(C[c], ui), entries, 16)

            def left_right(points):
                lo = low_coordinate(col, points)
                hi = at(col, points)
                return (
                    mul(points, eq(sub(col, lo))),
                    mul(points, eq(sub(col, hi))),
                )

            control_left, control_right = left_right(square(10))
            control_up, control_down = left_right(square(9))
            grab_entries = [
                (dy, dx, 1)
                for dy in (-1, 0, 1)
                for dx in range(-3, 4)
            ]
            grab_entries += [
                (dy, dx, -1)
                for dy in (-2, 2)
                for dx in range(-3, 4)
            ]
            control_grab = detect(
                mul(C[8], ui), grab_entries, 21
            )

            old_active = spatial(sl("mem", 12, 13))
            old_height = spatial(sl("mem", 13, 14))
            old_station = spatial(sl("mem", 14, 15))
            flip_score = mul(
                read(point_values, selector(0, 0, True)),
                has_object,
            )
            flip_trigger = mul(
                mul(has_object, inv(old_active)),
                greater(sub(
                    flip_score, maximum(winning_score, normal_best)
                ), .2),
            )
            missing_trigger = mul(
                mul(
                    mul(
                        mul(has_object, inv(old_active)),
                        inv(flip_trigger),
                    ),
                    mul(inv(next_physical), next_object),
                ),
                greater(sub(winning_score, normal_best), -.1),
            )
            trigger = minimum(add(
                flip_trigger, missing_trigger
            ), one)
            height0 = mix(
                ge(sub(next_y, rail_y), 6),
                inv(object_height),
                flip_trigger,
            )
            station0 = ge(sub(
                mix(next_x, px, flip_trigger), rail_x
            ), 6)
            target_height = mix(old_height, height0, trigger)
            target_station = mix(old_station, station0, trigger)

            resting_x = add(rail_x, affine(target_station, 12))
            resting_y = add(rail_y, affine(target_height, 12))
            finished = mul(
                mul(attached, eq(sub(crusher_x, resting_x))),
                eq(sub(crusher_y, resting_y)),
            )
            active = mul(
                minimum(add(old_active, trigger), one),
                inv(finished),
            )
            desired_x = mix(object_cx, resting_x, attached)
            desired_y = mix(object_cy, resting_y, attached)
            change_row = ge(
                absolute(sub(crusher_y, desired_y)), 2
            )
            away_left = ge(sub(crusher_x, rail_x), 2)
            go_left = minimum(add(
                mul(change_row, away_left),
                mul(
                    inv(change_row),
                    ge(sub(crusher_x, desired_x), 2),
                ),
            ), one)
            go_right = mul(
                inv(change_row),
                ge(sub(desired_x, crusher_x), 2),
            )
            vertical_ready = mul(change_row, inv(away_left))
            go_up = mul(
                vertical_ready,
                ge(sub(crusher_y, desired_y), 2),
            )
            go_down = mul(
                vertical_ready,
                ge(sub(desired_y, crusher_y), 2),
            )
            grab = mul(
                mul(
                    eq(sub(crusher_x, desired_x)),
                    eq(sub(crusher_y, desired_y)),
                ),
                inv(attached),
            )
            control_map = total([
                mul(control_left, go_left),
                mul(control_right, go_right),
                mul(control_up, go_up),
                mul(control_down, go_down),
                mul(control_grab, grab),
            ])
            combined_click = add(
                normal_click,
                mul(control_map, affine(active, 1000)),
            )

        # Remember a covered translating-strip phase.
        next_phase = phase
        if advanced:
            other_best = prepare_e
            for k in ("a", "b", "c", "d"):
                other_best = maximum(other_best, adjusted[k])
            best_move = glob(move_field)
            choose_f = mul(
                mul(inv(active), have["f"]),
                greater(sub(
                    adjusted["f"],
                    maximum(other_best, affine(best_move, 1, .2)),
                ), .001),
            )
            rotation = np.zeros((6, 6), np.float32)
            for i in range(6):
                rotation[(i + 1) % 6, i] = 1
            next_phase = node("vadd", [
                node("vmul", [phase, gmax(inv(choose_f))]),
                node("vmul", [
                    dense(phase, rotation), gmax(choose_f)
                ]),
            ])

        G.append({
            "id": "click", "op": "conv", "in": [combined_click],
            "w": parameter(np.ones((1, 1, 1, 1), np.float32)),
        })
        shapes["click"] = ("s", 1)
        policy_features = vcat([
            gmax(move_field), gmax(combined_click)
        ])
        policy_w = np.zeros((8, 5), np.float32)
        policy_b = np.full(8, -1000, np.float32)
        for i in range(4):
            policy_w[i + 1, i] = 1
            policy_b[i + 1] = 0
        policy_w[6, 4] = 1
        policy_b[6] = -.2
        G.append({
            "id": "logits", "op": "dense", "in": [policy_features],
            "w": parameter(policy_w),
            "b": parameter(policy_b),
        })
        shapes["logits"] = ("v", 8)

        memory_features = vcat([
            gmax(one), gmax(port_lo), gmax(port_hi),
            gmax(port_exists), gmax(anchor), gmax(saved_frow),
            next_phase, gmax(active),
            gmax(target_height), gmax(target_station),
        ])
        G.append({
            "id": "mem_next", "op": "dense", "in": [memory_features],
            "w": parameter(np.eye(15, dtype=np.float32)),
        })
        shapes["mem_next"] = ("v", 15)

        def prune(nodes):
            producers = {nd["id"]: nd for nd in nodes}
            for nd in nodes:
                if nd["op"] == "loop":
                    for s in nd["state"]:
                        producers[nd["id"] + "." + s["id"]] = nd
            needed = set()
            stack = ["logits", "click", "mem_next"]
            while stack:
                nd = producers.get(stack.pop())
                if nd is None or nd["id"] in needed:
                    continue
                needed.add(nd["id"])
                if nd["op"] != "loop":
                    stack.extend(nd["in"])
                else:
                    local = {s["id"] for s in nd["state"]}
                    local.update(b["id"] for b in nd["body"])
                    for s in nd["state"]:
                        if "init" in s:
                            stack.append(s["init"])
                    for b in nd["body"]:
                        stack.extend(
                            x for x in b["in"] if x not in local
                        )
            return [nd for nd in nodes if nd["id"] in needed]

        G = prune(G)

        # Algebraically fold single-use linear pointwise layers.
        # This is graph compilation, not weight fitting.
        for _ in range(4):
            producers = {
                nd["id"]: nd for nd in G if nd["op"] != "loop"
            }
            uses = {}
            for nd in G:
                if nd["op"] == "loop":
                    for b in nd["body"]:
                        for x in b["in"]:
                            uses[x] = uses.get(x, 0) + 1
                    for s in nd["state"]:
                        if "init" in s:
                            x = s["init"]
                            uses[x] = uses.get(x, 0) + 1
                else:
                    for x in nd["in"]:
                        uses[x] = uses.get(x, 0) + 1

            changed_any = False
            for nd in G:
                if nd["op"] != "conv":
                    continue
                w1 = P[nd["w"]]
                if w1.shape[-1] != 1:
                    continue
                src = nd["in"][0]
                prev = producers.get(src)
                if (
                    prev is None or prev["op"] != "conv"
                    or uses.get(src, 0) != 1
                    or src in ("click", "logits", "mem_next")
                ):
                    continue
                w0 = P[prev["w"]]
                if w1.shape[1] != w0.shape[0]:
                    continue
                fused = np.einsum(
                    "ab,bcij->acij",
                    w1[:, :, 0, 0], w0,
                    optimize=True,
                ).astype(np.float32)
                b0 = P[prev["b"]] if "b" in prev else np.zeros(
                    w0.shape[0], np.float32
                )
                b1 = P[nd["b"]] if "b" in nd else np.zeros(
                    w1.shape[0], np.float32
                )
                bias = (w1[:, :, 0, 0] @ b0 + b1).astype(np.float32)
                nd["in"] = list(prev["in"])
                nd["w"] = parameter(fused)
                if np.any(bias):
                    nd["b"] = parameter(bias)
                else:
                    nd.pop("b", None)
                changed_any = True
            G = prune(G)
            if not changed_any:
                break

        def mac(nd):
            op = nd["op"]
            if op == "conv":
                return int(P[nd["w"]].size) * 4096
            if op == "dense":
                return int(P[nd["w"]].size)
            if op in (
                "add", "sub", "mul", "max2", "min2",
                "rowmax", "colmax", "rowsum", "colsum",
            ):
                return shapes[nd["id"]][1] * 4096
            if op in ("gmax", "gmean"):
                return shapes[nd["in"][0]][1] * 4096
            if op in ("vadd", "vmul"):
                return shapes[nd["id"]][1]
            return 0

        top_mac = sum(mac(nd) for nd in G if nd["op"] != "loop")
        top_evals = sum(nd["op"] != "loop" for nd in G) + 1
        loops = [nd for nd in G if nd["op"] == "loop"]
        loop_node = loops[0]
        body_mac = sum(mac(nd) for nd in loop_node["body"])
        T = min(
            128,
            int((1_480_000_000 - top_mac) // max(1, body_mac)),
            int((9800 - top_evals) // len(loop_node["body"])),
        )
        loop_node["T"] = max(1, T)

        used = set()
        all_nodes = []
        for nd in G:
            if nd["op"] == "loop":
                all_nodes.extend(nd["body"])
            else:
                all_nodes.append(nd)
        for nd in all_nodes:
            for field in ("w", "b"):
                if field in nd:
                    used.add(nd[field])
        final_params = {name: P[name] for name in used}

        node_count = len(G) + sum(
            len(nd["body"]) + len(nd["state"])
            for nd in loops
        )
        channels = 19
        for nd in all_nodes:
            if shapes[nd["id"]][0] == "s":
                channels += shapes[nd["id"]][1]
        for nd in loops:
            channels += sum(
                s["zeros"][0] for s in nd["state"]
            )
            # Conservative allowance for exposed final state tensors.
            channels += sum(
                s["zeros"][0] for s in nd["state"]
            )
        parameter_count = sum(a.size for a in final_params.values())
        total_mac = top_mac + loop_node["T"] * body_mac
        evaluations = top_evals + loop_node["T"] * len(loop_node["body"])

        valid = (
            node_count <= 1000
            and channels <= 4096
            and parameter_count <= 150000
            and total_mac <= 1500000000
            and evaluations <= 10000
            and T >= 1
            and all(np.isfinite(a).all() for a in final_params.values())
        )
        return {
            "graph": G,
            "memory_size": 15,
            "params": final_params,
        }, valid

    network, valid = construct(True)
    if valid:
        return network
    network, valid = construct(False)
    if not valid:
        raise ValueError("Internal graph budget check failed")
    return network
