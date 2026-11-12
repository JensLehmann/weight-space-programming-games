import numpy as np


def build():
    def construct(composition):
        G, P, cache, pcache = [], {}, {}, {}
        shape = {
            "x0": (16, 64, 64),
            "coords": (3, 64, 64),
            "aprev": (8,),
            "avail": (8,),
        }
        dest = [G]
        caches = [cache]
        serial = [0]
        consts = {}
        const_values = {}

        def param(a):
            a = np.asarray(a, np.float32)
            key = (a.shape, a.tobytes())
            if key not in pcache:
                name = "p%d" % len(P)
                P[name] = a
                pcache[key] = name
            return pcache[key]

        def node(op, ins, name=None, **kw):
            key = (op, tuple(ins), tuple(sorted(kw.items())))
            if name is None:
                if key in caches[-1]:
                    return caches[-1][key]
                if len(caches) > 1 and key in cache:
                    return cache[key]
                name = "n%d" % serial[0]
                serial[0] += 1

            d = dict(id=name, op=op, **kw)
            d["in"] = list(ins)
            dest[-1].append(d)
            caches[-1][key] = name
            s = [shape[x] for x in ins]

            if op in ("conv", "dense"):
                n = P[kw["w"]].shape[0]
                shape[name] = (n, 64, 64) if op == "conv" else (n,)
            elif op in ("slice", "vslice"):
                shape[name] = (kw["stop"] - kw["start"],) + s[0][1:]
            elif op in ("concat", "vconcat"):
                shape[name] = (sum(x[0] for x in s),) + s[0][1:]
            elif op in ("gmax", "gmean"):
                shape[name] = (s[0][0],)
            elif op in ("add", "sub", "mul", "max2", "min2"):
                shape[name] = (max(x[0] for x in s), 64, 64)
            elif op in ("vadd", "vmul"):
                shape[name] = (max(x[0] for x in s),)
            else:
                shape[name] = s[0]
            return name

        def conv(x, w, b=None, name=None):
            kw = {"w": param(w)}
            if b is not None:
                kw["b"] = param(b)
            return node("conv", [x], name=name, **kw)

        def dense(x, w, b=None, name=None):
            kw = {"w": param(w)}
            if b is not None:
                kw["b"] = param(b)
            return node("dense", [x], name=name, **kw)

        def sl(x, a, b):
            return node(
                "slice" if len(shape[x]) == 3 else "vslice",
                [x], start=int(a), stop=int(b)
            )

        def cat(xs):
            if len(xs) == 1:
                return xs[0]
            return node(
                "concat" if len(shape[xs[0]]) == 3 else "vconcat", xs
            )

        def vr(x):
            return node("vrelu", [x])

        def sr(x):
            return node("relu", [x])

        def va(a, b):
            return node("vadd", [a, b])

        def vm(a, b):
            return node("vmul", [a, b])

        def sa(a, b):
            return node("add", [a, b])

        def sm(a, b):
            return node("mul", [a, b])

        one_sp = sl("coords", 2, 3)
        one = node("gmax", [one_sp])

        def C(v):
            v = float(v)
            if v not in consts:
                old_d, old_c = dest[-1], caches[-1]
                dest[-1], caches[-1] = G, cache
                z = dense(
                    one, np.zeros((1, 1), np.float32),
                    np.array([v], np.float32)
                )
                consts[v] = z
                const_values[z] = v
                dest[-1], caches[-1] = old_d, old_c
            return consts[v]

        for v in (
            0, 1, -1, 2, 3, -3, 3.5, -3.5,
            7, -7, 1000, 2000, 0.01
        ):
            C(v)

        def aff(x, a=1, b=0):
            n = shape[x][0]
            if n <= 24:
                return dense(
                    x, np.eye(n, dtype=np.float32) * a,
                    np.full(n, b, np.float32)
                )
            z = x if a == 1 else vm(x, C(a))
            return z if b == 0 else va(z, C(b))

        def neg(x):
            return vm(x, C(-1))

        def sub(a, b):
            if b in const_values:
                return aff(a, 1, -const_values[b])
            if a in const_values:
                return aff(b, -1, const_values[a])
            return va(a, neg(b))

        def sums(xs):
            z = xs[0]
            for x in xs[1:]:
                z = va(z, x)
            return z

        def vabs(x):
            n = shape[x][0]
            if n <= 24:
                I = np.eye(n, dtype=np.float32)
                z = vr(dense(x, np.concatenate([I, -I], 0)))
                return dense(z, np.concatenate([I, I], 1))
            return va(vr(x), vr(neg(x)))

        def eq0(x):
            n = shape[x][0]
            if n <= 24:
                I = np.eye(n, dtype=np.float32)
                z = vr(dense(x, np.concatenate([I, -I], 0)))
                z = dense(
                    z, np.concatenate([-I, -I], 1),
                    np.ones(n, np.float32)
                )
                return vr(z)
            return vr(sub(C(1), vabs(x)))

        def eq(a, b):
            return eq0(sub(a, b))

        def clip(x):
            return sub(x, vr(aff(x, 1, -1)))

        def nz(x):
            return clip(vr(x))

        def vmax(a, b):
            return va(b, vr(sub(a, b)))

        def modeq(x):
            a = vabs(x)
            return va(vr(sub(C(1), a)), eq0(aff(a, 1, -7)))

        def dist(x):
            a = vabs(x)
            return sub(a, vm(vr(aff(a, 1, -3.5)), C(2)))

        def flaglist(x, count, start=0):
            n = shape[x][0]
            w = np.tile(np.eye(n, dtype=np.float32), (count, 1))
            b = np.repeat(
                -np.arange(start, start + count, dtype=np.float32), n
            )
            f = eq0(dense(x, w, b))
            return [
                sl(f, n * k, n * (k + 1)) for k in range(count)
            ]

        def length(xs):
            p = nz(cat(xs))
            n = shape[xs[0]][0]
            w = np.tile(np.eye(n, dtype=np.float32), (1, len(xs)))
            return dense(p, w)

        def vtotal(x):
            return dense(
                x, np.ones((1, shape[x][0]), np.float32)
            )

        def spaff(x, a=1, b=0):
            n = shape[x][0]
            w = np.eye(n, dtype=np.float32)[:, :, None, None] * a
            return conv(x, w, np.full(n, b, np.float32))

        def speq0(x):
            a = node("abs", [x])
            a = sr(spaff(a, 1, -0.25))
            return sr(spaff(a, -2, 1))

        def gates(x, n):
            return speq0(conv(
                x, np.ones((n, 1, 1, 1), np.float32),
                -np.arange(n, dtype=np.float32)
            ))

        def shift(x, dy=0, dx=0):
            n = shape[x][0]
            while dy or dx:
                yy = max(-3, min(3, dy))
                xx = max(-3, min(3, dx))
                r = max(abs(yy), abs(xx))
                w = np.zeros(
                    (n, n, 2 * r + 1, 2 * r + 1), np.float32
                )
                for c in range(n):
                    w[c, c, r + yy, r + xx] = 1
                x = conv(x, w)
                dy, dx = dy - yy, dx - xx
            return x

        def spsum(x):
            return conv(
                x, np.ones((1, shape[x][0], 1, 1), np.float32)
            )

        def begin():
            dest.append([])
            caches.append({})

        def finish():
            body = dest.pop()
            caches.pop()
            return body

        def loop(name, T, states, body, nexts):
            G.append(dict(
                id=name, op="loop", T=T, state=states,
                body=body, next=nexts
            ))
            for s in states:
                shape[name + "." + s["id"]] = shape[s["id"]]

        # Exact black-pixel templates: A1..A7, B1..B7, C1..C7.
        patterns = [
            ["00100", "00100", "01110", "00100", "11111"],
            ["00001", "00101", "11111", "10100", "10000"],
            ["11111", "10100", "10000", "10100", "11111"],
            ["00100", "11111", "10101", "10101", "00100"],
            ["10001", "11111", "00100", "11111", "10001"],
            ["11111", "00100", "11100", "10000", "11111"],
            ["10001", "11111", "10001", "10001", "11011"],
            ["10000", "11110", "10010", "10010", "11111"],
            ["11111", "10001", "11101", "10101", "11111"],
            ["00111", "00101", "11111", "10100", "11100"],
            ["01110", "01010", "11111", "10001", "11111"],
            ["11111", "10001", "10001", "11111", "00100"],
            ["11110", "10011", "10001", "11001", "01111"],
            ["00100", "11111", "10101", "11111", "00100"],
            ["10101", "10000", "10101", "10000", "11111"],
            ["11101", "10000", "00000", "00001", "10111"],
            ["10111", "00100", "00100", "00100", "11101"],
            ["00100", "10101", "00100", "10101", "00100"],
            ["11011", "10001", "10101", "10001", "11011"],
            ["10111", "10101", "11101", "00000", "10101"],
            ["10001", "00000", "10001", "11011", "01010"],
        ]

        kernels, biases = [], []
        for rows in patterns:
            p = np.array(
                [[int(c) for c in row] for row in rows], np.float32
            )
            for k in range(4):
                q = np.rot90(p, k)
                kernels.append(2 * q - 1)
                biases.append(1 - q.sum())

        det = sr(conv(
            sl("x0", 5, 6),
            np.asarray(kernels, np.float32)[:, None],
            np.asarray(biases, np.float32)
        ))
        w = np.zeros((21, 84, 1, 1), np.float32)
        for i in range(21):
            w[i, 4 * i:4 * i + 4, 0, 0] = 1
        glyph = conv(det, w)
        glyph = node("sub", [glyph, sr(spaff(glyph, 1, -1))])

        # Verify the complete coloured tile perimeter.
        w = np.zeros((3, 16, 7, 7), np.float32)
        for f, colour in enumerate((10, 7, 11)):
            for y in range(7):
                for x in range(7):
                    if y in (0, 6) or x in (0, 6):
                        w[f, colour, y, x] = 1
        borders = sr(conv(
            "x0", w, np.full(3, -23, np.float32)
        ))
        w = np.zeros((21, 3, 1, 1), np.float32)
        for i in range(21):
            w[i, i // 7, 0, 0] = 1
        glyph = sm(glyph, conv(borders, w))
        centres = spsum(glyph)
        code = conv(
            glyph,
            np.tile(np.arange(1, 8, dtype=np.float32), 3)
              [None, :, None, None]
        )
        families = conv(
            glyph,
            np.repeat(np.arange(1, 4, dtype=np.float32), 7)
              [None, :, None, None]
        )

        bottom = sm(centres, shift(sl("x0", 3, 4), dy=4))
        top = node("sub", [centres, bottom])

        # Upper selector bar plus its following corner row.
        # A lower bar has no corner row beneath it.
        zero_colour = sl("x0", 0, 1)
        selector = sm(
            shift(zero_colour, dy=-6),
            shift(node("rowmax", [zero_colour]), dy=-5)
        )
        alter = node("gmax", [sm(top, selector)])
        normal = sub(C(1), alter)

        # The three-pixel connector gap identifies a rule anchor.
        w = np.zeros((1, 16, 3, 3), np.float32)
        w[0, 3, 1, :] = 1
        w[0, 2, 0, :] = 1
        w[0, 2, 2, :] = 1
        arrows = sr(conv(
            "x0", w, np.array([-8], np.float32)
        ))
        arrow_rows = node("rowsum", [arrows])
        bottom_rows = node("rowmax", [bottom])
        drive = cat([arrow_rows, bottom_rows, arrows, bottom])

        shape["pref_s"] = (4, 64, 64)
        begin()
        w = np.zeros((4, 4, 3, 3), np.float32)
        w[0, 0, 0, 1] = w[1, 1, 0, 1] = 1
        w[2, 2, 1, 0] = w[3, 3, 1, 0] = 1
        pn = sa(conv("pref_s", w), drive)
        body = finish()
        loop(
            "pref", 64,
            [{"id": "pref_s", "zeros": [4, 64, 64]}],
            body, {"pref_s": pn}
        )

        pr = "pref.pref_s"
        arrow_rank = sa(
            node("sub", [sl(pr, 0, 1), arrow_rows]),
            node("sub", [sl(pr, 2, 3), arrows])
        )
        masks = sm(gates(arrow_rank, 8), arrows)
        brank = node("sub", [sl(pr, 1, 2), bottom_rows])
        crank = node("sub", [sl(pr, 3, 4), bottom])
        bp = gates(crank, 8)
        im = sm(bottom, speq0(brank))
        om = sm(bottom, speq0(spaff(brank, 1, -1)))
        ip, op = sm(bp, im), sm(bp, om)

        S = node("gmax", [sm(ip, code)])
        T = node("gmax", [sm(op, code)])
        selT = node("gmax", [sm(op, selector)])
        sf = node("gmax", [sm(im, families)])
        tf = node("gmax", [sm(om, families)])

        raw, selections, fam = {}, {}, {}
        for side in (-1, 1):
            m = shift(masks, dx=-side * 5)
            vals = []
            for k in range(3):
                vals.append(node("gmax", [sm(m, code)]))
                if k == 0:
                    selections[side] = node(
                        "gmax", [sm(m, selector)]
                    )
                    fam[side] = node(
                        "gmax", [sm(m, families)]
                    )
                if k < 2:
                    m = shift(m, dx=-side * 7)
            raw[side] = vals

        ll = length(raw[-1])
        lf = flaglist(ll, 3, 1)
        a = raw[-1]
        L = [
            sums([
                vm(a[0], lf[0]),
                vm(a[1], lf[1]),
                vm(a[2], lf[2]),
            ]),
            va(vm(a[0], lf[1]), vm(a[1], lf[2])),
            vm(a[0], lf[2]),
        ]
        R0 = raw[1]
        R = R0

        if composition:
            # Lookup each intermediate symbol in the rule table.
            qe = np.zeros((192, 24), np.float32)
            ke = np.zeros((192, 8), np.float32)
            priority = np.zeros((192, 192), np.float32)
            collapse = np.zeros((24, 192), np.float32)
            for q in range(24):
                for r in range(8):
                    j = q * 8 + r
                    qe[j, q] = 1
                    ke[j, r] = 1
                    collapse[q, j] = 1
                    priority[j, j] = 1
                    priority[j, q * 8:q * 8 + r] = -1

            queries = dense(cat(R0), qe)
            keys = dense(L[0], ke)
            qfam = dense(cat([fam[1]] * 3), qe)
            kfam = dense(fam[-1], ke)
            matches = vm(eq(queries, keys), eq(qfam, kfam))
            matches = vm(matches, nz(queries))
            choose = vr(dense(matches, priority))

            translated = []
            for x in R0:
                translated.append(dense(
                    vm(choose, dense(x, ke)), collapse
                ))
            groups = [
                [
                    sl(translated[k], 8 * j, 8 * (j + 1))
                    for k in range(3)
                ]
                for j in range(3)
            ]
            gl = [length(x) for x in groups]

            def merge(a, b, al):
                m, n = len(a), len(b)
                pad = np.zeros(((m + n) * 8, m * 8), np.float32)
                pad[:m * 8] = np.eye(m * 8, dtype=np.float32)
                terms = [dense(cat(a), pad)]
                fs = flaglist(al, m + 1)
                repeat = np.tile(
                    np.eye(8, dtype=np.float32), (n, 1)
                )
                bv = cat(b)
                for k in range(m + 1):
                    place = np.zeros(
                        ((m + n) * 8, n * 8), np.float32
                    )
                    place[k * 8:(k + n) * 8] = np.eye(
                        n * 8, dtype=np.float32
                    )
                    terms.append(dense(
                        vm(bv, dense(fs[k], repeat)), place
                    ))
                z = sums(terms)
                return [
                    sl(z, k * 8, (k + 1) * 8)
                    for k in range(m + n)
                ]

            tree = merge(
                merge(groups[0], groups[1], gl[0]),
                groups[2], va(gl[0], gl[1])
            )
            present = nz(cat(R0))
            found = nz(cat(gl))
            valid = va(
                sub(C(1), present), vm(present, found)
            )
            valid = vm(
                vm(sl(valid, 0, 8), sl(valid, 8, 16)),
                sl(valid, 16, 24)
            )
            keep = va(alter, vm(normal, eq(fam[1], tf)))
            expand = vm(sub(C(1), keep), valid)
            R = [
                va(
                    vm(R0[k] if k < 3 else C(0), keep),
                    vm(tree[k], expand)
                )
                for k in range(9)
            ]

        q = len(R)
        rl = length(R)
        rf = flaglist(rl, q, 1)
        repeat_rule = np.tile(
            np.eye(8, dtype=np.float32), (9, 1)
        )
        sav, tav = nz(S), nz(T)
        sobs, tobs, tavailable = [], [], []

        for k in range(q):
            w = np.zeros((72, 8), np.float32)
            for p in range(9):
                if p + k < 8:
                    w[p * 8:(p + 1) * 8, p + k] = 1
            sobs.append(dense(S, w))
            tobs.append(dense(T, w))
            tavailable.append(dense(tav, w))

        def product_slots(x, n):
            z = sl(x, 0, 72)
            for k in range(1, n):
                z = vm(z, sl(x, 72 * k, 72 * (k + 1)))
            return z

        def constraints(seq, observed, right):
            n = len(seq)
            symbols = cat([
                dense(x, repeat_rule) for x in seq
            ])
            present = nz(symbols)
            absent = sub(C(1), present)
            ob = cat(observed[:n])
            delta = sub(ob, symbols)
            first = sl(delta, 0, 72)
            same = vm(
                modeq(sub(delta, cat([first] * n))), nz(ob)
            )
            cyclic = product_slots(
                va(vm(same, present), absent), n
            )

            if right:
                direct = product_slots(
                    va(
                        vm(cat(tavailable[:n]), present),
                        absent
                    ), n
                )
                costs = vm(dist(delta), present)
                total = sums([
                    sl(costs, 72 * k, 72 * (k + 1))
                    for k in range(n)
                ])
                first_cost = sl(costs, 0, 72)
            else:
                direct = product_slots(
                    va(vm(eq0(delta), present), absent), n
                )
                total, first_cost = C(0), dist(first)

            match = vm(
                va(vm(direct, normal), vm(cyclic, alter)),
                sl(present, 0, 72)
            )
            cost = va(
                vm(total, normal), vm(first_cost, alter)
            )
            return match, cost

        while len(sobs) < 3:
            sobs.append(C(0))

        lm, lc = constraints(L, sobs, False)
        rm, rc = constraints(R, tobs, True)
        lm = vm(
            lm, dense(eq(fam[-1], sf), repeat_rule)
        )
        if not composition:
            rm = vm(
                rm, dense(eq(fam[1], tf), repeat_rule)
            )

        # Synthetic alignment lattice: input and output offsets 0..8.
        w = np.zeros((1, 3, 1, 1), np.float32)
        w[0, 0, 0, 0] = 63
        ri = conv("coords", w)
        w = np.zeros((1, 3, 1, 1), np.float32)
        w[0, 1, 0, 0] = 63
        ci = conv("coords", w)
        rg, cg = gates(ri, 9), gates(ci, 9)

        lift = np.zeros((144, 9, 1, 1), np.float32)
        collapse = np.zeros((16, 144, 1, 1), np.float32)
        for f in range(2):
            for p in range(9):
                for r in range(8):
                    j = f * 72 + p * 8 + r
                    lift[j, p, 0, 0] = 1
                    collapse[f * 8 + r, j, 0, 0] = 1

        def spatial_constraints(m, c, g):
            z = conv(
                sm(conv(g, lift), cat([m, c])), collapse
            )
            return sl(z, 0, 8), sl(z, 8, 16)

        LM, LC = spatial_constraints(lm, lc, rg)
        RM, RC = spatial_constraints(rm, rc, cg)
        valid = sm(LM, RM)
        costs = sa(
            sa(LC, RC),
            sm(node("sub", [one_sp, valid]), C(1000))
        )
        sn, tn = vtotal(sav), vtotal(tav)
        terminal = sm(
            speq0(node("sub", [ri, sn])),
            speq0(node("sub", [ci, tn]))
        )
        initial = sm(
            node("sub", [one_sp, terminal]), C(1000)
        )
        origin = sm(sl(rg, 0, 1), sl(cg, 0, 1))

        def future(dp):
            z, terms = dp, []
            for k in range(q):
                z = shift(z, dx=1)
                terms.append(sm(z, rf[k]))
            z = terms[0]
            for t in terms[1:]:
                z = sa(z, t)
            terms = []
            for k in range(3):
                z = shift(z, dy=1)
                terms.append(sm(z, lf[k]))
            return sa(sa(terms[0], terms[1]), terms[2])

        shape["dp_s"] = (1, 64, 64)
        shape["alt_s"] = (8, 64, 64)
        begin()
        alternatives = sa(costs, future("dp_s"))
        minimum = sl(alternatives, 0, 1)
        for k in range(1, 8):
            minimum = node(
                "min2", [
                    minimum, sl(alternatives, k, k + 1)
                ]
            )
        dn = node("min2", [minimum, initial])
        body = finish()
        loop(
            "solve", 9,
            [
                {"id": "dp_s", "init": initial},
                {"id": "alt_s", "zeros": [8, 64, 64]},
            ],
            body, {"dp_s": dn, "alt_s": alternatives}
        )

        winners = sm(
            speq0(node(
                "sub", ["solve.alt_s", "solve.dp_s"]
            )),
            valid
        )
        w = np.zeros((8, 8, 1, 1), np.float32)
        for r in range(8):
            w[r, r, 0, 0] = 1
            w[r, :r, 0, 0] = -1
        winners = sr(conv(winners, w))

        source_field = spsum(sm(sl(rg, 0, 8), S))
        target_field = spsum(sm(sl(cg, 0, 8), T))
        target_gates = sl(cg, 0, 8)

        def forward(active):
            z, terms = active, []
            for k in range(3):
                z = shift(z, dy=-1)
                terms.append(sm(z, lf[k]))
            z = sa(sa(terms[0], terms[1]), terms[2])
            terms = []
            for k in range(q):
                z = shift(z, dx=-1)
                terms.append(sm(z, rf[k]))
            z = terms[0]
            for t in terms[1:]:
                z = sa(z, t)
            return spsum(z)

        shape["path_s"] = (1, 64, 64)
        shape["goal_s"] = (24,)
        begin()
        active = sm(winners, "path_s")
        left_goal = node("gmax", [sm(active, source_field)])
        right_goal = node("gmax", [sm(active, target_field)])
        target_parts = []
        for k in range(q):
            field = shift(
                spsum(sm(active, R[k])), dx=-k
            )
            target_parts.append(node(
                "gmax", [sm(field, target_gates)]
            ))
        goals = cat([
            left_goal, right_goal, sums(target_parts)
        ])
        gn = vmax("goal_s", goals)
        pn = forward(active)
        body = finish()
        loop(
            "trace", 8,
            [
                {"id": "path_s", "init": origin},
                {"id": "goal_s", "zeros": [24]},
            ],
            body, {"path_s": pn, "goal_s": gn}
        )

        goals = "trace.goal_s"
        current = cat([L[0], R0[0], T])
        selected = cat([
            selections[-1], selections[1], selT
        ])
        w = np.zeros((24, 1), np.float32)
        w[:16] = 1
        w[16:] = -1
        b = np.zeros(24, np.float32)
        b[16:] = 1
        mode = dense(alter, w, b)
        errors = vm(
            vm(nz(goals), sub(C(1), eq(goals, current))),
            mode
        )
        selected_error = vtotal(vm(errors, selected))
        delta_selected = vtotal(
            vm(sub(goals, current), selected)
        )

        # Nearest unresolved group in cyclic cursor order.
        ordinal = np.concatenate([
            2 * np.arange(8, dtype=np.float32),
            2 * np.arange(8, dtype=np.float32) + 1,
            np.arange(8, dtype=np.float32),
        ])
        cursor = dense(selected, ordinal[None])
        count = dense(
            node("gmean", [arrows]),
            np.array([[4096]], np.float32)
        )
        count = va(
            vm(vm(count, C(2)), alter), vm(tn, normal)
        )
        indices = dense(
            one, np.zeros((24, 1), np.float32), ordinal
        )
        delta = sub(indices, cursor)
        fd = va(
            delta, vm(clip(vr(neg(delta))), count)
        )
        bd = va(
            neg(delta), vm(clip(vr(delta)), count)
        )
        fs = vm(errors, sub(C(1000), fd))
        bs = vm(errors, sub(C(1000), bd))

        navg = gates(ci, 24)
        rep = np.zeros((48, 24, 1, 1), np.float32)
        red = np.zeros((2, 48, 1, 1), np.float32)
        for side in range(2):
            for k in range(24):
                rep[24 * side + k, k, 0, 0] = 1
                red[side, 24 * side + k, 0, 0] = 1
        best = node("gmax", [
            conv(
                sm(conv(navg, rep), cat([fs, bs])), red
            )
        ])
        bf, bb = sl(best, 0, 1), sl(best, 1, 2)

        # Shortest direction on the seven-symbol cycle.
        inc = clip(va(
            sub(
                clip(vr(delta_selected)),
                clip(vr(aff(delta_selected, 1, -3)))
            ),
            clip(vr(aff(delta_selected, -1, -3)))
        ))
        dec = sub(C(1), inc)
        si = vm(vm(selected_error, inc), C(2000))
        sd = vm(vm(selected_error, dec), C(2000))
        sd = va(sd, sub(C(1), nz(vmax(bf, bb))))
        features = cat([
            sd, si, bb, va(bf, C(0.01))
        ])
        w = np.zeros((8, 4), np.float32)
        for k in range(4):
            w[k + 1, k] = 1
        b = np.full(8, -100, np.float32)
        b[1:5] = 0
        dense(features, w, b, name="logits")
        conv(
            "coords", np.zeros((1, 3, 1, 1), np.float32),
            name="click"
        )

        # Remove unused construction intermediates and parameters.
        owner = {}
        for d in G:
            owner[d["id"]] = d
            if d["op"] == "loop":
                for s in d["state"]:
                    owner[d["id"] + "." + s["id"]] = d
        needed = set()

        def visit(name):
            if name not in owner:
                return
            d = owner[name]
            if d["id"] in needed:
                return
            needed.add(d["id"])
            if d["op"] != "loop":
                for x in d["in"]:
                    visit(x)
            else:
                local = {x["id"] for x in d["body"]}
                local.update(s["id"] for s in d["state"])
                for s in d["state"]:
                    if "init" in s:
                        visit(s["init"])
                for x in d["body"]:
                    for y in x["in"]:
                        if y not in local:
                            visit(y)

        visit("logits")
        visit("click")
        G = [d for d in G if d["id"] in needed]

        used = set()
        for d in G:
            nodes = d["body"] if d["op"] == "loop" else [d]
            for x in nodes:
                for field in ("w", "b"):
                    if field in x:
                        used.add(x[field])
        P = {k: v for k, v in P.items() if k in used}

        def node_mac(d):
            op = d["op"]
            if op == "conv":
                return int(P[d["w"]].size) * 4096
            if op == "dense":
                return int(P[d["w"]].size)
            if op in ("add", "sub", "mul", "max2", "min2"):
                return int(np.prod(shape[d["id"]]))
            if op in (
                "rowmax", "colmax", "rowsum", "colsum",
                "gmax", "gmean"
            ):
                return int(np.prod(shape[d["in"][0]]))
            if op in ("vadd", "vmul"):
                return shape[d["id"]][0]
            return 0

        nodes = 0
        evaluations = 0
        macs = 0
        spatial = 19
        for d in G:
            if d["op"] == "loop":
                nodes += 1 + len(d["body"])
                evaluations += 1 + d["T"] * len(d["body"])
                macs += d["T"] * sum(
                    node_mac(x) for x in d["body"]
                )
                for s in d["state"]:
                    if len(shape[s["id"]]) == 3:
                        spatial += shape[s["id"]][0]
                for x in d["body"]:
                    if len(shape[x["id"]]) == 3:
                        spatial += shape[x["id"]][0]
            else:
                nodes += 1
                evaluations += 1
                macs += node_mac(d)
                if len(shape[d["id"]]) == 3:
                    spatial += shape[d["id"]][0]

        fits = (
            nodes <= 1000
            and evaluations <= 10000
            and sum(x.size for x in P.values()) <= 150000
            and macs <= 1500000000
            and spatial <= 4096
        )
        return {
            "graph": G,
            "memory_size": 0,
            "params": P,
        }, fits

    network, fits = construct(True)
    if fits:
        return network

    network, fits = construct(False)
    if not fits:
        raise ValueError("Internal circuit budget exceeded")
    return network
