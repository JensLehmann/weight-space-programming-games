import itertools
import numpy as np


def build():
    G, P = [], {}
    current = G
    serial = 0

    def uid():
        nonlocal serial
        serial += 1
        return "n%d" % serial

    def par(a):
        name = "p%d" % len(P)
        P[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, ins=(), name=None, **kw):
        name = name or uid()
        current.append(
            dict(id=name, op=op, **{"in": list(ins)}, **kw)
        )
        return name

    def dense(x, w, b=None):
        kw = {"w": w if isinstance(w, str) else par(w)}
        if b is not None:
            kw["b"] = b if isinstance(b, str) else par(b)
        return node("dense", [x], **kw)

    def conv(x, w, b=None):
        kw = {"w": w if isinstance(w, str) else par(w)}
        if b is not None:
            kw["b"] = b if isinstance(b, str) else par(b)
        return node("conv", [x], **kw)

    def c1(x, w, b=None):
        w = np.asarray(w, np.float32)
        return conv(x, w[:, :, None, None], b)

    def sl(x, a, b, vector=False):
        return node(
            "vslice" if vector else "slice",
            [x], start=a, stop=b
        )

    def vc(*xs):
        return node("vconcat", xs)

    def sc(*xs):
        return node("concat", xs)

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def vr(a):
        return node("vrelu", [a])

    constants = {}

    def C(v):
        nonlocal current
        v = float(v)
        if v not in constants:
            old = current
            current = G
            constants[v] = dense(
                "avail", np.zeros((1, 8), np.float32), [v]
            )
            current = old
        return constants[v]

    one, negone, zero = C(1), C(-1), C(0)

    def vn(a):
        return vm(a, negone)

    def vs(a, b):
        return va(a, vn(b))

    def vabs(a):
        return va(vr(a), vr(vn(a)))

    def veq0(a):
        return vr(va(one, vn(vabs(a))))

    def veq(a, b):
        return veq0(vs(a, b))

    def clipv(a):
        return vs(a, vr(va(a, negone)))

    def ge(a, k):
        return clipv(vr(va(a, C(1 - k))))

    def sa(a, b):
        return node("add", [a, b])

    def ss(a, b):
        return node("sub", [a, b])

    def sm(a, b):
        return node("mul", [a, b])

    def sf(a, scale, bias=0):
        return c1(a, [[scale]], [bias])

    coord_y = sl("coords", 0, 1)
    coord_x = sl("coords", 1, 2)
    spatial_one = sl("coords", 2, 3)
    yy = sf(coord_y, 63)
    xx = sf(coord_x, 63)

    conv("coords", np.zeros((1, 3, 1, 1), np.float32))
    G[-1]["id"] = "click"

    def hard_spatial_zero(a):
        a = node("abs", [a])
        a = node("relu", [sf(a, -4, 2)])
        return node("min2", [a, spatial_one])

    def sp_inverse(a):
        return ss(spatial_one, a)

    shift_params = {}

    def shift(a, channels, dx=0, dy=0):
        while dx or dy:
            sx = max(-3, min(3, dx))
            sy = max(-3, min(3, dy))
            radius = max(abs(sx), abs(sy))
            k = 2 * radius + 1
            key = (channels, sx, sy)
            if key not in shift_params:
                w = np.zeros(
                    (channels, channels, k, k), np.float32
                )
                for j in range(channels):
                    w[j, j, radius + sy, radius + sx] = 1
                shift_params[key] = par(w)
            a = conv(a, shift_params[key])
            dx -= sx
            dy -= sy
        return a

    black = sl("x0", 5, 6)
    bracket = sl("x0", 0, 1)
    grey = sl("x0", 3, 4)

    backing_w = np.zeros((1, 16), np.float32)
    for colour in range(6, 16):
        backing_w[0, colour] = 1
    backing = c1("x0", backing_w)

    ring = np.zeros((1, 1, 7, 7), np.float32)
    ring[0, 0, 0, :] = 1
    ring[0, 0, -1, :] = 1
    ring[0, 0, :, 0] = 1
    ring[0, 0, :, -1] = 1
    ring_ok = node("relu", [conv(backing, ring, [-23])])
    ink = conv(black, np.ones((1, 1, 5, 5), np.float32))
    ink = node("min2", [ink, spatial_one])
    centres = sm(ring_ok, ink)

    # Integer glyph fingerprint, canonicalized over all rotations.
    kernel = np.zeros((5, 5), np.float32)
    bit = 0
    for r in range(5):
        for c in range(5):
            if (r, c) != (2, 2):
                kernel[r, c] = 2 ** bit
                bit += 1
    hw = np.stack(
        [np.rot90(kernel, k) for k in range(4)]
    )[:, None]
    rotated_hash = conv(black, hw)
    h0, h1, h2, h3 = [
        sl(rotated_hash, k, k + 1) for k in range(4)
    ]
    glyph_hash = node(
        "min2",
        [
            node("min2", [h0, h1]),
            node("min2", [h2, h3]),
        ],
    )

    # Known cycle indices allow shortest-direction editing.
    # Unrecognised families still use geometric fingerprints.
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
    tw = np.zeros((84, 1, 5, 5), np.float32)
    tb = np.zeros(84, np.float32)
    for j, rows in enumerate(patterns):
        p = np.array(
            [[int(v) for v in row] for row in rows],
            np.float32,
        )
        for k in range(4):
            q = np.rot90(p, k)
            tw[4 * j + k, 0] = 2 * q - 1
            tb[4 * j + k] = 1 - q.sum()
    exact_templates = node("relu", [conv(black, tw, tb)])
    reduce_rotations = np.zeros((21, 84), np.float32)
    for j in range(21):
        reduce_rotations[j, 4 * j:4 * j + 4] = 1
    template_ids = c1(exact_templates, reduce_rotations)
    template_ids = node(
        "min2", [template_ids, spatial_one]
    )

    family_planes = sc(
        sl("x0", 10, 11),
        sl("x0", 7, 8),
        sl("x0", 11, 12),
    )
    family_corner = shift(family_planes, 3, -3, -3)
    family_expand = np.zeros((21, 3), np.float32)
    for j in range(21):
        family_expand[j, j // 7] = 1
    template_ids = sm(
        template_ids, c1(family_corner, family_expand)
    )
    glyph_id = c1(
        template_ids,
        np.array(
            [[1 + j % 7 for j in range(21)]], np.float32
        ),
    )

    colour_kernel = np.zeros((1, 16, 7, 7), np.float32)
    for j in range(16):
        colour_kernel[0, j, 0, 0] = j
    glyph_colour = conv("x0", colour_kernel)

    glyph_hash = sm(glyph_hash, centres)
    glyph_id = sm(glyph_id, centres)
    glyph_colour = sm(glyph_colour, centres)
    tile = sc(glyph_hash, glyph_id, glyph_colour, centres)
    tile1 = shift(tile, 4, 7, 0)
    tile2 = shift(tile1, 4, 7, 0)

    hashes = [sl(t, 0, 1) for t in (tile, tile1, tile2)]
    ids = [sl(t, 1, 2) for t in (tile, tile1, tile2)]
    presences = [sl(t, 3, 4) for t in (tile, tile1, tile2)]
    group_length = sa(
        sa(presences[0], presences[1]), presences[2]
    )

    col_zero = hard_spatial_zero(xx)
    bottom_region = node(
        "rowmax", [sm(grey, col_zero)]
    )
    top_region = sp_inverse(bottom_region)
    top_centres = sm(centres, top_region)
    bottom_centres = sm(centres, bottom_region)

    # An upper bracket has a full bar six pixels above the
    # glyph centre, and a DOWNWARD corner one row below it.
    # A preceding row's lower bracket cannot satisfy this.
    upper_bar = shift(bracket, 1, 0, -6)
    downward_left_corner = shift(bracket, 1, -2, -5)
    selected = sm(
        centres, sm(upper_bar, downward_left_corner)
    )
    alter_mode = node(
        "gmax", [sm(selected, top_region)]
    )

    target_y = node(
        "gmax", [sm(bottom_centres, yy)]
    )
    target_row = hard_spatial_zero(ss(yy, target_y))
    target_mask = sm(bottom_centres, target_row)
    source_mask = sm(
        bottom_centres, sp_inverse(target_row)
    )

    source_colour = node(
        "gmax", [sm(glyph_colour, source_mask)]
    )
    target_colour = node(
        "gmax", [sm(glyph_colour, target_mask)]
    )

    source_count = dense(
        node("gmean", [source_mask]), [[4096]]
    )
    target_count = dense(
        node("gmean", [target_mask]), [[4096]]
    )

    left_neighbour = shift(centres, 1, -7, 0)
    starts = sm(centres, sp_inverse(left_neighbour))
    next_ten = shift(centres, 1, 10, 0)
    connector = shift(grey, 1, 5, 0)
    ends = sm(
        sm(
            sm(centres, sp_inverse(presences[1])),
            next_ten,
        ),
        connector,
    )
    end1 = shift(ends, 1, 7, 0)
    end2 = shift(end1, 1, 7, 0)
    rule_starts = sm(
        sm(starts, sa(sa(ends, end1), end2)),
        top_region,
    )

    record_map = sc(
        hashes[0], hashes[1], hashes[2],
        ids[0], ids[1], ids[2],
        glyph_colour, group_length, selected, xx, yy,
    )
    priority = c1(
        sc(yy, xx), [[-64, -1]], [4096]
    )

    def begin_loop():
        nonlocal current
        old = current
        body = []
        current = body
        return old, body

    def end_loop(old, body, T, states, nxt):
        nonlocal current
        current = old
        name = uid()
        nd = {
            "id": name,
            "op": "loop",
            "T": T,
            "state": states,
            "body": body,
            "next": nxt,
        }
        current.append(nd)
        return name, nd

    def choose_spatial(rem):
        score = sm(rem, priority)
        best = node("gmax", [score])
        pick = sm(
            rem, hard_spatial_zero(ss(score, best))
        )
        return pick, ss(rem, pick)

    def fifo_weights(n):
        w = np.zeros((n, n), np.float32)
        for i in range(n - 1):
            w[i, i + 1] = 1
        u = np.zeros((n, 1), np.float32)
        u[-1, 0] = 1
        return par(w), par(u)

    # Read six rules in the simulator's row-major priority.
    names = (
        ["lh%d" % j for j in range(3)]
        + ["rh%d" % j for j in range(3)]
        + ["ri%d" % j for j in range(3)]
        + ["li", "ll", "rl", "lc", "rc", "sl", "sr"]
    )
    fifo6, append6 = fifo_weights(6)
    rem_state = uid()
    rule_states = {k: uid() for k in names}
    states = [{"id": rem_state, "init": rule_starts}]
    states += [
        {"id": v, "zeros": [6]}
        for v in rule_states.values()
    ]

    old, body = begin_loop()
    pick, remaining = choose_spatial(rem_state)
    left_record = node(
        "gmax", [sm(record_map, pick)]
    )
    wx = np.zeros((1, 11), np.float32)
    wx[0, 9], wx[0, 7] = 1, 7
    right_x = dense(left_record, wx, [3])
    right_y = sl(left_record, 10, 11, True)
    distance = sa(
        node("abs", [ss(xx, right_x)]),
        node("abs", [ss(yy, right_y)]),
    )
    right_pick = sm(
        centres, hard_spatial_zero(distance)
    )
    right_record = node(
        "gmax", [sm(record_map, right_pick)]
    )
    sources = {}
    for j in range(3):
        sources["lh%d" % j] = (left_record, j)
        sources["rh%d" % j] = (right_record, j)
        sources["ri%d" % j] = (right_record, 3 + j)
    sources.update({
        "li": (left_record, 3),
        "ll": (left_record, 7),
        "rl": (right_record, 7),
        "lc": (left_record, 6),
        "rc": (right_record, 6),
        "sl": (left_record, 8),
        "sr": (right_record, 8),
    })
    nxt = {rem_state: remaining}
    for k in names:
        rec, j = sources[k]
        scalar = sl(rec, j, j + 1, True)
        nxt[rule_states[k]] = va(
            dense(rule_states[k], fifo6),
            dense(scalar, append6),
        )
    rule_loop, _ = end_loop(
        old, body, 6, states, nxt
    )
    R = {
        k: rule_loop + "." + v
        for k, v in rule_states.items()
    }

    # Read the two bottom strings.
    fifo9, append9 = fifo_weights(9)
    bottom_record = sc(glyph_hash, glyph_id, selected)
    mask_states = [uid(), uid()]
    bottom_states = [
        [uid() for _ in range(3)] for _ in range(2)
    ]
    states = [
        {"id": mask_states[0], "init": source_mask},
        {"id": mask_states[1], "init": target_mask},
    ]
    states += [
        {"id": s, "zeros": [9]}
        for row in bottom_states for s in row
    ]
    old, body = begin_loop()
    nxt = {}
    for side in range(2):
        pick, remaining = choose_spatial(
            mask_states[side]
        )
        rec = node(
            "gmax", [sm(bottom_record, pick)]
        )
        nxt[mask_states[side]] = remaining
        for j in range(3):
            value = sl(rec, j, j + 1, True)
            s = bottom_states[side][j]
            nxt[s] = va(
                dense(s, fifo9),
                dense(value, append9),
            )
    bottom_loop, _ = end_loop(
        old, body, 9, states, nxt
    )
    bottom = [
        [bottom_loop + "." + s for s in row]
        for row in bottom_states
    ]
    SH, SI, _ = bottom[0]
    TH, TI, TS = bottom[1]

    sum9 = par(np.ones((1, 9), np.float32))
    sum6 = par(np.ones((1, 6), np.float32))
    current_bottom_hash = dense(vm(TH, TS), sum9)
    current_bottom_id = dense(vm(TI, TS), sum9)
    selected_index = dense(
        TS, np.arange(9, dtype=np.float32)[None]
    )

    current_rule_hash = va(
        dense(vm(R["lh0"], R["sl"]), sum6),
        dense(vm(R["rh0"], R["sr"]), sum6),
    )
    current_rule_id = va(
        dense(vm(R["li"], R["sl"]), sum6),
        dense(vm(R["ri0"], R["sr"]), sum6),
    )

    required = [
        ge(R["ll"], j + 1) for j in range(3)
    ]
    first6 = par(
        np.eye(6, dtype=np.float32)
        - np.tril(np.ones((6, 6), np.float32), -1)
    )

    result_fields = [
        R["ll"], R["rl"], R["rc"],
        R["rh0"], R["rh1"], R["rh2"],
        R["ri0"], R["ri1"], R["ri2"],
    ]
    result_table = vc(*result_fields)
    repeat_choice = par(
        np.tile(np.eye(6, dtype=np.float32), (9, 1))
    )
    result_reduce = np.zeros((9, 54), np.float32)
    for j in range(9):
        result_reduce[j, 6 * j:6 * j + 6] = 1
    result_reduce = par(result_reduce)

    source_tile = dense(
        SH,
        np.tile(np.eye(9, dtype=np.float32), (3, 1)),
    )
    gd = par(np.ones((27, 1), np.float32))
    gb = par(
        np.array(
            [j - i for j in range(3) for i in range(9)]
        )
    )
    gs = np.zeros((3, 27), np.float32)
    for j in range(3):
        gs[j, 9 * j:9 * j + 9] = 1
    gs = par(gs)

    shift_q1 = np.zeros((3, 3), np.float32)
    shift_q1[0, 1] = shift_q1[1, 2] = 1
    shift_q2 = np.zeros((3, 3), np.float32)
    shift_q2[0, 2] = 1
    shift_q1, shift_q2 = par(shift_q1), par(shift_q2)
    sum3 = par(np.ones((1, 3), np.float32))

    # Fixed-rule translation with a short intermediate-symbol queue.
    ns = {
        k: uid()
        for k in ("sp", "op", "qh", "qc", "qn", "ah", "ai")
    }
    states = [
        {"id": s, "zeros": [3 if k == "qh" else 1]}
        for k, s in ns.items()
    ]
    old, body = begin_loop()

    gathered = dense(
        vm(
            veq0(dense(ns["sp"], gd, gb)),
            source_tile,
        ),
        gs,
    )
    active = clipv(ns["qn"])
    inactive = vs(one, active)
    cur_h = va(
        vm(gathered, inactive),
        vm(ns["qh"], active),
    )
    cur_c = va(
        vm(source_colour, inactive),
        vm(ns["qc"], active),
    )

    bad = vs(one, veq(R["lc"], cur_c))
    for j in range(3):
        match = veq(
            R["lh%d" % j],
            sl(cur_h, j, j + 1, True),
        )
        bad = va(
            bad, vm(required[j], vs(one, match))
        )
    matches = vm(
        vr(vs(one, bad)), required[0]
    )
    choice = vr(dense(matches, first6))
    result = dense(
        vm(
            result_table,
            dense(choice, repeat_choice),
        ),
        result_reduce,
    )
    used = sl(result, 0, 1, True)
    out_n = sl(result, 1, 2, True)
    out_c = sl(result, 2, 3, True)
    out_h = sl(result, 3, 6, True)
    out_i = sl(result, 6, 9, True)
    final = veq(out_c, target_colour)

    qshift = va(
        vm(
            dense(ns["qh"], shift_q1),
            veq(used, one),
        ),
        vm(
            dense(ns["qh"], shift_q2),
            veq(used, C(2)),
        ),
    )
    keep = vm(active, final)
    defer = vs(one, final)
    qh_next = va(
        vm(qshift, keep), vm(out_h, defer)
    )
    qn_next = va(
        vm(vr(vs(ns["qn"], used)), keep),
        vm(out_n, defer),
    )
    qc_next = va(
        vm(ns["qc"], keep), vm(out_c, defer)
    )

    loc = vc(selected_index, ns["op"])
    hit_w = np.tile(
        np.array([[1, -1]], np.float32), (3, 1)
    )
    hit = veq0(dense(loc, hit_w, [0, -1, -2]))
    answer_h = vm(
        dense(vm(hit, out_h), sum3), final
    )
    answer_i = vm(
        dense(vm(hit, out_i), sum3), final
    )

    nxt = {
        ns["sp"]: va(ns["sp"], vm(used, inactive)),
        ns["op"]: va(ns["op"], vm(out_n, final)),
        ns["qh"]: qh_next,
        ns["qc"]: qc_next,
        ns["qn"]: qn_next,
        ns["ah"]: va(ns["ah"], answer_h),
        ns["ai"]: va(ns["ai"], answer_i),
    }
    normal_loop, normal_nd = end_loop(
        old, body, 18, states, nxt
    )
    normal_hash = normal_loop + "." + ns["ah"]
    normal_id = normal_loop + "." + ns["ai"]

    # Lengths of constant-symbol runs, capped at three.
    sh1 = np.zeros((9, 9), np.float32)
    sh2 = np.zeros((9, 9), np.float32)
    for j in range(8):
        sh1[j, j + 1] = 1
    for j in range(7):
        sh2[j, j + 2] = 1
    sh1, sh2 = par(sh1), par(sh2)

    def run_lengths(h):
        eq1 = veq(h, dense(h, sh1))
        eq2 = veq(h, dense(h, sh2))
        return vm(
            clipv(h),
            va(one, va(eq1, vm(eq1, eq2))),
        )

    source_runs = run_lengths(SH)
    target_runs = run_lengths(TH)

    # Editable-rule assignment circuit.
    # Permutation matrices are fixed combinatorial routing weights.
    B, K, N = 48, 15, 9
    permutations = list(itertools.permutations(range(6)))

    rule_tile_w = par(
        np.tile(np.eye(6, dtype=np.float32), (B, 1))
    )
    rule_sum_w = np.zeros((B, B * 6), np.float32)
    for b in range(B):
        rule_sum_w[b, b * 6:b * 6 + 6] = 1
    rule_sum_w = par(rule_sum_w)
    rule_tiles = {
        k: dense(R[k], rule_tile_w)
        for k in ("ll", "rl", "sl", "sr")
    }

    permutation_weights = []
    for stage in range(6):
        w = np.zeros((B * 6, K), np.float32)
        for batch in range(K):
            for b in range(B):
                r = permutations[batch * B + b][stage]
                w[b * 6 + r, batch] = 1
        permutation_weights.append(par(w))

    diff_w = np.zeros((B * N, B), np.float32)
    diff_b = np.zeros(B * N, np.float32)
    gather_sum_w = np.zeros((B, B * N), np.float32)
    for b in range(B):
        for j in range(N):
            diff_w[b * N + j, b] = 1
            diff_b[b * N + j] = -j
            gather_sum_w[b, b * N + j] = 1
    diff_w, diff_b = par(diff_w), par(diff_b)
    gather_sum_w = par(gather_sum_w)
    bottom_tile_w = par(
        np.tile(np.eye(N, dtype=np.float32), (B, 1))
    )
    source_tiles = [
        dense(v, bottom_tile_w)
        for v in (source_runs, SH, SI)
    ]
    target_tiles = [
        dense(v, bottom_tile_w)
        for v in (target_runs, TH, TI)
    ]

    firstB = par(
        np.eye(B, dtype=np.float32)
        - np.tril(np.ones((B, B), np.float32), -1)
    )
    sumB = par(np.ones((1, B), np.float32))
    broadcastB = par(np.ones((B, 1), np.float32))

    clock_init_w = np.zeros((K, 1), np.float32)
    clock_init_w[0, 0] = 1
    clock_init = dense(one, clock_init_w)
    clock_shift_w = np.zeros((K, K), np.float32)
    for j in range(K - 1):
        clock_shift_w[j + 1, j] = 1
    clock_shift_w = par(clock_shift_w)

    ps = {
        k: uid() for k in ("clock", "found", "ah", "ai")
    }
    states = [
        {"id": ps["clock"], "init": clock_init}
    ]
    states += [
        {"id": ps[k], "zeros": [1]}
        for k in ("found", "ah", "ai")
    ]

    old, body = begin_loop()

    def gather_batch(pointer, tiles):
        eq = veq0(dense(pointer, diff_w, diff_b))
        return [
            dense(vm(eq, tiled), gather_sum_w)
            for tiled in tiles
        ]

    psrc = dense(zero, broadcastB)
    pdst = psrc
    bad = psrc
    candidate_h = psrc
    candidate_i = psrc

    for stage in range(6):
        routing = dense(
            ps["clock"], permutation_weights[stage]
        )
        vals = {
            k: dense(
                vm(routing, rule_tiles[k]), rule_sum_w
            )
            for k in ("ll", "rl", "sl", "sr")
        }
        sr, sh, si = gather_batch(psrc, source_tiles)
        tr, th, ti = gather_batch(pdst, target_tiles)

        bad = va(
            bad,
            va(
                vr(vs(vals["ll"], sr)),
                vr(vs(vals["rl"], tr)),
            ),
        )
        candidate_h = va(
            candidate_h,
            va(
                vm(vals["sl"], sh),
                vm(vals["sr"], th),
            ),
        )
        candidate_i = va(
            candidate_i,
            va(
                vm(vals["sl"], si),
                vm(vals["sr"], ti),
            ),
        )
        psrc = va(psrc, vals["ll"])
        pdst = va(pdst, vals["rl"])

    bad = va(
        bad,
        va(
            vabs(vs(psrc, source_count)),
            vabs(vs(pdst, target_count)),
        ),
    )
    valid = vr(vs(one, bad))
    first_valid = vr(dense(valid, firstB))
    take = vm(first_valid, vs(one, ps["found"]))
    nxt = {
        ps["clock"]: dense(
            ps["clock"], clock_shift_w
        ),
        ps["found"]: va(
            ps["found"], dense(take, sumB)
        ),
        ps["ah"]: va(
            ps["ah"],
            dense(vm(take, candidate_h), sumB),
        ),
        ps["ai"]: va(
            ps["ai"],
            dense(vm(take, candidate_i), sumB),
        ),
    }
    perm_loop, _ = end_loop(
        old, body, K, states, nxt
    )
    alter_hash = perm_loop + "." + ps["ah"]
    alter_id = perm_loop + "." + ps["ai"]

    normal_mode = vs(one, alter_mode)
    desired_hash = va(
        vm(normal_hash, normal_mode),
        vm(alter_hash, alter_mode),
    )
    desired_id = va(
        vm(normal_id, normal_mode),
        vm(alter_id, alter_mode),
    )
    current_hash = va(
        vm(current_bottom_hash, normal_mode),
        vm(current_rule_hash, alter_mode),
    )
    current_id = va(
        vm(current_bottom_id, normal_mode),
        vm(current_rule_id, alter_mode),
    )

    correct = veq(desired_hash, current_hash)
    delta = vs(desired_id, current_id)
    forward = va(
        vs(ge(delta, 1), ge(delta, 4)),
        ge(vn(delta), 4),
    )
    known = vm(clipv(current_id), clipv(desired_id))
    forward = va(
        vm(forward, known), vs(one, known)
    )

    features = vc(correct, forward)
    action_w = np.zeros((8, 2), np.float32)
    action_b = np.full(8, -100, np.float32)

    action_w[1] = [-4, -2]
    action_b[1] = 2
    action_w[2] = [-4, 2]
    action_b[2] = 0
    action_w[4] = [3, 0]
    action_b[4] = -1

    dense(features, action_w, action_b)
    G[-1]["id"] = "logits"

    # Static bookkeeping to respect the node-evaluation budget.
    other_evals = 0
    for nd in G:
        if nd is normal_nd:
            continue
        if nd["op"] == "loop":
            other_evals += 1 + nd["T"] * len(nd["body"])
        else:
            other_evals += 1
    allowed = (
        (9900 - other_evals - 1) // len(normal_nd["body"])
    )
    normal_nd["T"] = max(1, min(18, int(allowed)))

    return {
        "graph": G,
        "memory_size": 0,
        "params": P,
    }
