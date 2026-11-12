import numpy as np


def build():
    NF, NE, NP = 12, 6, 4
    NB = NF + NE + NP
    STATE_SIZE = 4 * NF + 4 * NE
    M = 23 + STATE_SIZE

    params = {}
    parameter_cache = {}
    graph = []
    expression_cache = {}
    shapes = {
        "x0": ("s", 16),
        "coords": ("s", 3),
        "avail": ("v", 8),
        "aprev": ("v", 8),
        "mem": ("v", M),
    }
    counter = [0]

    def fresh():
        counter[0] += 1
        return "n%d" % counter[0]

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def node(op, ins, typ, count, **fields):
        key = (op, tuple(ins), tuple(sorted(fields.items())))
        if key in expression_cache:
            return expression_cache[key]
        name = fresh()
        graph.append({"id": name, "op": op, "in": list(ins), **fields})
        shapes[name] = (typ, count)
        expression_cache[key] = name
        return name

    def conv(x, w, b=None):
        w = np.asarray(w, np.float32)
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], "s", w.shape[0], **fields)

    def dense(x, w, b=None):
        w = np.asarray(w, np.float32)
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], "v", w.shape[0], **fields)

    def constant(value):
        return dense("avail", np.zeros((1, 8), np.float32), [value])

    ZERO, ONE = constant(0), constant(1)

    def unary(op, x):
        return node(op, [x], *shapes[x])

    def binary(op, a, b):
        ta, ca = shapes[a]
        tb, cb = shapes[b]
        return node(op, [a, b],
                    "s" if "s" in (ta, tb) else "v", max(ca, cb))

    def cat(xs, spatial=False):
        return node("concat" if spatial else "vconcat", xs,
                    "s" if spatial else "v",
                    sum(shapes[x][1] for x in xs))

    def sl(x, a, b):
        typ = shapes[x][0]
        return node("slice" if typ == "s" else "vslice",
                    [x], typ, b - a, start=a, stop=b)

    def pool(x, op="gmax"):
        return node(op, [x], "v", shapes[x][1])

    def vf(x, scale=1, bias=0):
        if scale == 1 and bias == 0:
            return x
        n = shapes[x][1]
        return dense(x, np.eye(n, dtype=np.float32) * scale,
                     np.full(n, bias, np.float32) if bias else None)

    def sf(x, scale=1, bias=0):
        if scale == 1 and bias == 0:
            return x
        n = shapes[x][1]
        return conv(x, (np.eye(n, dtype=np.float32) * scale)
                    [:, :, None, None],
                    np.full(n, bias, np.float32) if bias else None)

    def vadd(a, b):
        return binary("vadd", a, b)

    def vsub(a, b):
        return vadd(a, vf(b, -1))

    def vmul(a, b):
        return binary("vmul", a, b)

    def vabs(x):
        return vadd(unary("vrelu", x), unary("vrelu", vf(x, -1)))

    def vc(x):
        # Inputs to this gate are primarily integer counts or ranks.
        return unary("vsigmoid", vf(x, 40, -20))

    def sc(x):
        return binary("min2", unary("relu", x), ONE)

    def vsum(x):
        return dense(x, np.ones((1, shapes[x][1]), np.float32))

    def ssum(x):
        return conv(x, np.ones((1, shapes[x][1], 1, 1), np.float32))

    def count_pixels(x):
        return vf(pool(x, "gmean"), 4096)

    def attributes(bank, field):
        return count_pixels(binary("mul", bank, field))

    def vd2(x, y, tx, ty):
        dx, dy = vsub(x, tx), vsub(y, ty)
        return vadd(vmul(dx, dx), vmul(dy, dy))

    def sd2(x, y, tx, ty):
        dx, dy = binary("sub", x, tx), binary("sub", y, ty)
        return binary("add", binary("mul", dx, dx),
                      binary("mul", dy, dy))

    def minimum_channels(x):
        out = sl(x, 0, 1)
        for i in range(1, shapes[x][1]):
            out = binary("min2", out, sl(x, i, i + 1))
        return out

    def group_maximum(x, groups=1):
        n = shapes[x][1] // groups
        while n > 1:
            m = (n + 1) // 2
            wa = np.zeros((groups * m, groups * n), np.float32)
            wb = np.zeros_like(wa)
            for g in range(groups):
                for i in range(m):
                    a, b = 2 * i, min(2 * i + 1, n - 1)
                    wa[g * m + i, g * n + a] += 1
                    wa[g * m + i, g * n + b] -= 1
                    wb[g * m + i, g * n + b] = 1
            x = vadd(unary("vrelu", dense(x, wa)), dense(x, wb))
            n = m
        return x

    def expand(x):
        x = conv(x, np.ones((1, 1, 7, 7), np.float32))
        x = conv(x, np.ones((1, 1, 3, 3), np.float32))
        return sc(x)

    # Coordinate and region planes.
    xyw = np.zeros((2, 3, 1, 1), np.float32)
    xyw[0, 1, 0, 0] = 63
    xyw[1, 0, 0, 0] = 63
    xy = conv("coords", xyw)
    X, Y = sl(xy, 0, 1), sl(xy, 1, 2)
    play = binary("mul", sc(sf(Y, 2, -19)),
                  sc(sf(Y, -2, 125)))
    header = binary("mul", sc(sf(Y, -2, 19)),
                    sc(sf(X, 2, -55)))

    warning_w = np.zeros((1, 16, 1, 1), np.float32)
    warning_w[0, 2, 0, 0] = 1
    warning_header = binary("mul", conv("x0", warning_w), header)

    # Exact solid-square detectors, one channel per fruit rank.
    colors = [10, 6, 15, 11, 12, 8, 9, 7, 14]
    sizes = [1, 2, 3, 4, 5, 7, 8, 9, 10]
    color_w = np.zeros((9, 16, 1, 1), np.float32)
    for i, c in enumerate(colors):
        color_w[i, c, 0, 0] = 1

    warning_fruit_w = np.ones((9, 1, 1, 1), np.float32)
    warning_fruit_w[0, 0, 0, 0] = 0
    fruit_colors = binary(
        "add", conv("x0", color_w),
        conv(warning_header, warning_fruit_w))

    def erode(x, lengths, axis, kernel, offsets=None):
        w = np.zeros((9, 9, kernel, kernel), np.float32)
        b = np.empty(9, np.float32)
        h = kernel // 2
        for i, length in enumerate(lengths):
            positions = (range(-(length // 2), length - length // 2)
                         if offsets is None else offsets[i])
            for p in positions:
                if axis == 0:
                    w[i, i, h + p, h] = 1
                else:
                    w[i, i, h, h + p] = 1
            b[i] = 1 - length
        return unary("relu", conv(x, w, b))

    first_lengths = [min(n, 7) for n in sizes]
    er = erode(fruit_colors, first_lengths, 1, 7)
    er = erode(er, first_lengths, 0, 7)
    second_lengths, second_offsets = [], []
    for n in sizes:
        if n <= 7:
            second_lengths.append(1)
            second_offsets.append([0])
        else:
            second_lengths.append(n - 6)
            second_offsets.append(list(
                range(-(n // 2) + 3, n - n // 2 - 3)))
    er = erode(er, second_lengths, 1, 5, second_offsets)
    er = erode(er, second_lengths, 0, 5, second_offsets)

    boundary_w = np.zeros((9, 9, 3, 3), np.float32)
    for i in range(9):
        boundary_w[i, i, 1, 1] = 1
        for r, c in [(0, 1), (2, 1), (1, 0), (1, 2)]:
            boundary_w[i, i, r, c] = -1
    fruit_centers = unary("relu", conv(er, boundary_w))

    # Enemy templates include negative weights at their transparent holes.
    enemy_templates = [
        [[0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 0, 0, 1],
         [0, 1, 1, 1, 0]],
        [[0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0]],
        [[1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 1, 1, 1, 0]],
    ]
    ew = np.zeros((3, 17, 7, 7), np.float32)
    eb = np.zeros(3, np.float32)
    for t, c in enumerate([7, 14, 13]):
        count = 0
        for r in range(4):
            for j in range(5):
                present = enemy_templates[t][r][j]
                weight = 1 if present else -2
                ew[t, c, r + 1, j + 1] = weight
                ew[t, 16, r + 1, j + 1] = weight
                count += present
        eb[t] = 1 - count
    enemy_centers = sc(conv(cat(["x0", warning_header], True), ew, eb))

    # Rounded destination-pad caps allow partial occlusion.
    pcw = np.zeros((1, 16, 1, 1), np.float32)
    pcw[0, 9, 0, 0] = 1
    pcw[0, 2, 0, 0] = 1
    pad_color = conv("x0", pcw)
    cap_w = np.zeros((2, 1, 7, 7), np.float32)
    for t, flip in enumerate([1, -1]):
        for dy in [-2, -1, 0, 1]:
            width = 2 if dy == -2 else 3
            weight = 3 if dy == -2 else 1
            for dx in range(-width, width + 1):
                cap_w[t, 0, 3 + flip * dy, 3 + dx] = weight
        for dy, dx in [(-2, -3), (-2, 3), (-3, 0)]:
            cap_w[t, 0, 3 + flip * dy, 3 + dx] = -16
    caps = sc(conv(pad_color, cap_w, [-29, -29]))
    cap_shift = np.zeros((2, 2, 5, 5), np.float32)
    cap_shift[0, 0, 0, 2] = 1
    cap_shift[1, 1, 4, 2] = 1
    pads = binary("mul", sc(ssum(conv(caps, cap_shift))), play)

    fp = binary("mul", fruit_centers, play)
    ep = binary("mul", enemy_centers, play)
    fh = binary("mul", fruit_centers, header)
    eh = binary("mul", enemy_centers, header)
    rank_fw = np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1)
    rank_ew = np.arange(1, 4, dtype=np.float32).reshape(1, 3, 1, 1)
    fruit_rank_map = conv(fp, rank_fw)
    enemy_rank_map = conv(ep, rank_ew)
    observed_goals = cat([
        pool(conv(fh, rank_fw)), count_pixels(ssum(fh)),
        pool(conv(eh, rank_ew)), count_pixels(ssum(eh))
    ])

    # Extract objects into coordinate-ordered slots.
    sources = sc(cat([ssum(fp), ssum(ep), pads], True))
    priority = conv("coords", np.array(
        [[[[4032]], [[63]], [[1]]]], np.float32))

    loop_name, remaining_name, bank_name = fresh(), fresh(), fresh()
    shapes[remaining_name] = ("s", 3)
    shapes[bank_name] = ("s", NB)
    outer_graph, outer_cache = graph, expression_cache
    graph, expression_cache = [], {}

    weighted = binary("mul", remaining_name, priority)
    selected = sc(sf(binary("sub", weighted, pool(weighted)), 4, 1))
    selected = binary("min2", selected, remaining_name)
    remaining_next = binary("sub", remaining_name, selected)

    shift_w = np.zeros((NB, NB + 3, 1, 1), np.float32)
    gate_w = np.zeros((NB, 3), np.float32)
    for t, (offset, length) in enumerate(
            [(0, NF), (NF, NE), (NF + NE, NP)]):
        shift_w[offset, NB + t, 0, 0] = 1
        gate_w[offset:offset + length, t] = 1
        for j in range(1, length):
            shift_w[offset + j, offset + j - 1, 0, 0] = 1
    shifted = conv(cat([bank_name, selected], True), shift_w)
    gate = dense(pool(selected), gate_w)
    bank_next = binary(
        "add", bank_name,
        binary("mul", binary("sub", shifted, bank_name), gate))

    body = graph
    graph, expression_cache = outer_graph, outer_cache
    graph.append({
        "id": loop_name, "op": "loop", "T": NF,
        "state": [
            {"id": remaining_name, "init": sources},
            {"id": bank_name, "zeros": [NB, 64, 64]}
        ],
        "body": body,
        "next": {remaining_name: remaining_next, bank_name: bank_next}
    })
    bank = loop_name + "." + bank_name
    shapes[bank] = ("s", NB)
    F = sl(bank, 0, NF)
    E = sl(bank, NF, NF + NE)
    PB = sl(bank, NF + NE, NB)

    fx, fy = attributes(F, X), attributes(F, Y)
    ex, ey = attributes(E, X), attributes(E, Y)
    fr = attributes(F, fruit_rank_map)
    erank = attributes(E, enemy_rank_map)
    fv, ev = pool(F), pool(E)
    observed_pads = cat([
        attributes(PB, X), attributes(PB, Y), pool(PB)
    ])

    old_goals = sl("mem", 0, 4)
    old_pads = sl("mem", 4, 16)
    last_x, last_y = sl("mem", 16, 17), sl("mem", 17, 18)
    seen, old_stall = sl("mem", 18, 19), sl("mem", 19, 20)
    old_state = sl("mem", 20, 20 + STATE_SIZE)
    bad_x = sl("mem", 20 + STATE_SIZE, 21 + STATE_SIZE)
    bad_y = sl("mem", 21 + STATE_SIZE, 22 + STATE_SIZE)
    bad_age = sl("mem", 22 + STATE_SIZE, 23 + STATE_SIZE)

    keep = vmul(vc(seen), vf(sl("aprev", 0, 1), -1, 1))
    old_goal_valid = vc(vadd(sl(old_goals, 1, 2), sl(old_goals, 3, 4)))
    goal_keep = vmul(keep, old_goal_valid)
    pad_keep = vmul(keep, vc(vsum(sl(old_pads, 8, 12))))
    goals = vadd(vmul(old_goals, goal_keep),
                 vmul(observed_goals, vf(goal_keep, -1, 1)))
    pad_data = vadd(vmul(old_pads, pad_keep),
                    vmul(observed_pads, vf(pad_keep, -1, 1)))

    gfr, gfc = sl(goals, 0, 1), sl(goals, 1, 2)
    ger, gec = sl(goals, 2, 3), sl(goals, 3, 4)
    px, py, pv = sl(pad_data, 0, 4), sl(pad_data, 4, 8), sl(pad_data, 8, 12)

    pad_rows = unary("relu", binary(
        "sub", ONE, unary("abs", binary("sub", Y, py))))
    pad_cols = unary("relu", binary(
        "sub", ONE, unary("abs", binary("sub", X, px))))
    pad_points = binary("mul", binary("mul", pad_rows, pad_cols), pv)
    pad_inside = expand(ssum(pad_points))

    def equal_rank(rank, target, validity):
        return vmul(unary("vrelu", vf(vabs(vsub(rank, target)), -1, 1)),
                    validity)

    correct_f = equal_rank(fr, gfr, fv)
    correct_e = equal_rank(erank, ger, ev)
    inside_f = pool(binary("mul", F, pad_inside))
    inside_e = pool(binary("mul", E, pad_inside))
    need_f = vc(vsub(gfc, vsum(correct_f)))
    need_e = vc(vsub(gec, vsum(correct_e)))
    deliver_f = vc(vsub(gfc, vsum(vmul(correct_f, inside_f))))
    deliver_e = vc(vsub(gec, vsum(vmul(correct_e, inside_e))))
    lower_f = vmul(vc(vsub(gfr, fr)), fv)
    higher_f = vmul(vc(vsub(fr, gfr)), fv)
    lower_e = vmul(vc(vsub(ger, erank)), ev)

    # Vector pair selection: nearest useful equal-rank pair,
    # with priority for higher ranks.
    left = np.zeros((NF * NF, NF), np.float32)
    right = np.zeros_like(left)
    diagonal = np.zeros(NF * NF, np.float32)
    tie = np.empty(NF * NF, np.float32)
    for i in range(NF):
        for j in range(NF):
            q = i * NF + j
            left[q, i] = 1
            right[q, j] = 1
            diagonal[q] = float(i != j)
            tie[q] = 0.08 * i + 0.003 * j
    diagonal_node = dense(
        "avail", np.zeros((NF * NF, 8), np.float32), diagonal)
    difference = left - right

    def pad_vector(x, length):
        if length == NF:
            return x
        w = np.zeros((NF, length), np.float32)
        w[:length] = np.eye(length, dtype=np.float32)
        return dense(x, w)

    def choose_pair(x, y, rank, eligible, enabled, length):
        x, y = pad_vector(x, length), pad_vector(y, length)
        rank = pad_vector(rank, length)
        eligible = pad_vector(eligible, length)
        dx, dy = dense(x, difference), dense(y, difference)
        distance = vadd(vmul(dx, dx), vmul(dy, dy))
        same = unary("vrelu", vf(vabs(dense(rank, difference)), -1, 1))
        good = vmul(dense(eligible, left), dense(eligible, right))
        good = vmul(vmul(vmul(good, same), diagonal_node), enabled)
        rank_bias = dense(rank, left * 400, 10000 - tie)
        scores = vmul(unary("vrelu", vsub(rank_bias, distance)), good)
        best = group_maximum(scores)
        present = vc(best)
        winners = vmul(vc(vf(vsub(scores, best), 1000, 1)), present)
        chosen = dense(winners, (left + right).T)
        if length != NF:
            chosen = sl(chosen, 0, length)
        return chosen, present

    merge_f, pair_f = choose_pair(
        fx, fy, fr, lower_f, vmul(need_f, vc(gfc)), NF)
    merge_e, pair_e = choose_pair(
        ex, ey, erank, lower_e,
        vmul(vmul(need_e, vc(gec)), vf(need_f, -1, 1)), NE)
    midpoint_fx = vf(vsum(vmul(merge_f, fx)), 0.5)
    midpoint_fy = vf(vsum(vmul(merge_f, fy)), 0.5)
    midpoint_ex = vf(vsum(vmul(merge_e, ex)), 0.5)
    midpoint_ey = vf(vsum(vmul(merge_e, ey)), 0.5)

    downgrade = vmul(
        higher_f, vmul(vmul(need_f, vc(gfc)), vf(pair_f, -1, 1)))
    delivery_f = vmul(
        vmul(correct_f, vf(inside_f, -1, 1)),
        vmul(deliver_f, vf(need_f, -1, 1)))
    delivery_e = vmul(
        vmul(correct_e, vf(inside_e, -1, 1)),
        vmul(vmul(deliver_e, vf(need_e, -1, 1)), vf(need_f, -1, 1)))

    desired_positions = binary(
        "max2", ssum(binary("mul", F, correct_f)),
        ssum(binary("mul", E, correct_e)))
    occupied = pool(binary("mul", pad_points, expand(desired_positions)))
    pad_penalty = vadd(vf(occupied, 12000), vf(pv, -30000, 30000))
    pad_potential = minimum_channels(binary(
        "add", sd2(X, Y, px, py), pad_penalty))

    # Click-to-bounding-box distance reproduces the game's selection radius.
    def staircase(rank, thresholds):
        terms = [vc(vf(rank, 1, -(t - 1))) for t in thresholds]
        out = terms[0]
        for term in terms[1:]:
            out = vadd(out, term)
        return out

    half = staircase(fr, [2, 4, 6, 7, 9])
    right_extent = staircase(fr, [3, 5, 6, 8])
    box_offset = vf(vsub(right_extent, half), 0.5)
    box_half = vf(vadd(right_extent, half), 0.5)
    fdx, fdy = binary("sub", X, fx), binary("sub", Y, fy)
    fbx = unary("relu", binary(
        "sub", unary("abs", binary("sub", fdx, box_offset)), box_half))
    fby = unary("relu", binary(
        "sub", unary("abs", binary("sub", fdy, box_offset)), box_half))
    fruit_box_distance = binary(
        "add", binary("mul", fbx, fbx), binary("mul", fby, fby))
    hit_f = binary("mul", sc(sf(fruit_box_distance, -1, 65)), fv)

    edx, edy = binary("sub", X, ex), binary("sub", Y, ey)
    eax, eay = unary("abs", edx), unary("abs", sf(edy, 1, 0.5))
    ebx = unary("relu", sf(eax, 1, -2))
    eby = unary("relu", sf(eay, 1, -1.5))
    enemy_box_distance = binary(
        "add", binary("mul", ebx, ebx), binary("mul", eby, eby))
    hit_e = binary("mul", sc(sf(enemy_box_distance, -1, 65)), ev)
    enemy_distance = binary(
        "add", binary("mul", edx, edx), binary("mul", edy, edy))
    enemy_potential = minimum_channels(binary(
        "add", enemy_distance, vf(ev, -30000, 30000)))

    merge_new_f = sd2(X, Y, midpoint_fx, midpoint_fy)
    merge_old_f = vd2(fx, fy, midpoint_fx, midpoint_fy)
    utility_merge_f = binary(
        "mul", binary("sub", merge_old_f, merge_new_f), merge_f)
    utility_delivery_f = binary(
        "mul", binary("sub", attributes(F, pad_potential), pad_potential),
        delivery_f)
    utility_downgrade = binary(
        "mul", binary("sub", attributes(F, enemy_potential), enemy_potential),
        downgrade)
    fruit_utility = binary(
        "add", binary("add", utility_merge_f, utility_delivery_f),
        utility_downgrade)
    fruit_score = ssum(binary("mul", fruit_utility, hit_f))
    active_f = vc(vadd(vadd(merge_f, delivery_f), downgrade))
    wrong_f_cost = sf(ssum(binary("mul", hit_f, vsub(fv, active_f))), 9000)

    # Analytic inertial-enemy displacement using reciprocal-norm refinement.
    clipped_r2 = binary("min2", enemy_distance, constant(144))
    inv = sf(edx, 0, 1 / 12)
    for _ in range(5):
        inv2 = binary("mul", inv, inv)
        correction = sf(binary("mul", clipped_r2, inv2), -0.5, 1.5)
        inv = binary("mul", inv, correction)
    inertial_x = binary("add", ex, sf(binary("mul", edx, inv), 13.6))
    inertial_y = binary("add", ey, sf(binary("mul", edy, inv), 13.6))
    inertial_x = binary("min2", binary("max2", inertial_x, constant(2)),
                        constant(61))
    inertial_y = binary("min2", binary("max2", inertial_y, constant(12)),
                        constant(62))
    first_enemy = equal_rank(erank, ONE, ev)
    other_enemy = vf(first_enemy, -1, 1)
    predicted_ex = binary(
        "add", binary("mul", inertial_x, first_enemy),
        binary("mul", X, other_enemy))
    predicted_ey = binary(
        "add", binary("mul", inertial_y, first_enemy),
        binary("mul", Y, other_enemy))

    predicted_pad_e = None
    for j in range(NP):
        d = sd2(predicted_ex, predicted_ey, sl(px, j, j + 1), sl(py, j, j + 1))
        d = binary("add", d, sl(pad_penalty, j, j + 1))
        predicted_pad_e = d if predicted_pad_e is None else binary(
            "min2", predicted_pad_e, d)
    utility_delivery_e = binary(
        "mul", binary("sub", attributes(E, pad_potential), predicted_pad_e),
        delivery_e)
    merge_new_e = sd2(predicted_ex, predicted_ey, midpoint_ex, midpoint_ey)
    merge_old_e = vd2(ex, ey, midpoint_ex, midpoint_ey)
    utility_merge_e = binary(
        "mul", binary("sub", merge_old_e, merge_new_e), merge_e)
    enemy_score = ssum(binary(
        "mul", binary("add", utility_delivery_e, utility_merge_e), hit_e))
    both_f = sc(sf(ssum(binary("mul", hit_f, merge_f)), 1, -1))
    both_e = sc(sf(ssum(binary("mul", hit_e, merge_e)), 1, -1))
    merge_bonus = sf(binary("add", both_f, both_e), 2500)

    danger = binary(
        "mul", binary("mul", sc(sf(eax, -1, 11)), sc(sf(eay, -1, 11))),
        binary("mul", binary("sub", ONE, hit_e), ev))
    normal_f_hits = ssum(binary("mul", hit_f, vadd(merge_f, delivery_f)))
    danger_cost = sf(binary("mul", ssum(danger), normal_f_hits), 6000)
    downgrade_cost = sf(binary(
        "mul", ssum(hit_e), vc(vsum(downgrade))), 8000)

    # Displace an enemy when it is close enough to threaten useful fruit.
    fruit_repeat = np.zeros((NE * NF, NF), np.float32)
    enemy_repeat = np.zeros((NE * NF, NE), np.float32)
    group_sum = np.zeros((NE, NE * NF), np.float32)
    tie_near = np.empty(NE * NF, np.float32)
    for e in range(NE):
        for f in range(NF):
            q = e * NF + f
            fruit_repeat[q, f] = 1
            enemy_repeat[q, e] = 1
            group_sum[e, q] = 1
            tie_near[q] = 0.008 * f

    vulnerable = vmul(vsub(fv, higher_f), vc(gfc))
    near_dx = vsub(dense(fx, fruit_repeat), dense(ex, enemy_repeat))
    near_dy = vsub(dense(fy, fruit_repeat), dense(ey, enemy_repeat))
    near_distance = vadd(vmul(near_dx, near_dx), vmul(near_dy, near_dy))
    near_bias = dense(vulnerable, 20000 * fruit_repeat, -tie_near)
    near_scores = vsub(near_bias, near_distance)
    nearest_best = group_maximum(near_scores, NE)
    nearest_winner = vc(vf(
        vsub(near_scores, dense(nearest_best, enemy_repeat)), 1000, 1))
    nearest_x = dense(vmul(nearest_winner, dense(fx, fruit_repeat)), group_sum)
    nearest_y = dense(vmul(nearest_winner, dense(fy, fruit_repeat)), group_sum)
    nearest_valid = vc(dense(
        vmul(nearest_winner, dense(vulnerable, fruit_repeat)), group_sum))

    near_ax = vabs(vsub(ex, nearest_x))
    near_ay = vabs(vsub(ey, nearest_y))
    chebyshev = vadd(unary("vrelu", vsub(near_ax, near_ay)), near_ay)
    guard_danger = vmul(vmul(vc(vf(chebyshev, -0.5, 6.5)), ev), nearest_valid)
    guard_scores = dense(
        guard_danger, np.eye(NE, dtype=np.float32) * 100,
        -np.arange(NE, dtype=np.float32) * 0.01)
    guard_best = group_maximum(guard_scores)
    guard_enabled = vc(vsum(guard_danger))
    guard_enemy = vmul(
        vc(vf(vsub(guard_scores, guard_best), 1000, 1)), guard_enabled)

    escape_gain = binary(
        "sub", sd2(predicted_ex, predicted_ey, nearest_x, nearest_y),
        vd2(ex, ey, nearest_x, nearest_y))
    escape_score = ssum(binary(
        "mul", binary("mul", escape_gain, hit_e), guard_enemy))
    escape_hits = ssum(binary("mul", hit_e, guard_enemy))
    escape_score = binary("add", escape_score, sf(escape_hits, 5000))
    escape_score = binary("sub", escape_score,
                          sf(ssum(binary("mul", hit_f, vulnerable)), 3000))

    # State comparison and rejected-click memory.
    state_vector = cat([fx, fy, fr, ex, ey, erank, fv, ev])
    state_difference = vsum(vabs(vsub(state_vector, old_state)))
    unchanged = vmul(
        vmul(vc(vf(state_difference, -10, 1)), keep),
        sl("aprev", 6, 7))
    stall_next = vmul(vf(old_stall, 1, 1), unchanged)
    repeat_region = sc(sf(sd2(X, Y, last_x, last_y), -1, 37))
    repeat_cost = sf(binary("mul", repeat_region, unchanged), 12000)
    bad_region = sc(sf(sd2(X, Y, bad_x, bad_y), -1, 37))
    bad_cost = sf(binary("mul", bad_region, vmul(vc(bad_age), keep)), 18000)

    score = binary("add", binary("add", fruit_score, enemy_score), merge_bonus)
    for cost in [wrong_f_cost, danger_cost, downgrade_cost]:
        score = binary("sub", score, cost)
    score = binary(
        "add", binary("mul", score, vf(guard_enabled, -1, 1)),
        binary("mul", escape_score, guard_enabled))
    for cost in [repeat_cost, bad_cost]:
        score = binary("sub", score, cost)
    score = binary("sub", score, sf(binary("sub", ONE, play), 100000))

    tie_map = conv("coords", np.array(
        [[[[0.002 * 63]], [[0.128 * 63]], [[0]]]], np.float32))
    click_source = binary("add", score, tie_map)
    graph.append({"id": "click", "op": "add", "in": [click_source, ZERO]})
    shapes["click"] = ("s", 1)

    winner = sc(sf(binary("sub", "click", pool("click")), 65536, 1))
    chosen_x, chosen_y = attributes(winner, X), attributes(winner, Y)

    # Fruit-mass feasibility distinguishes irreversible bites from progress.
    mass_w = (2.0 ** np.arange(9)).astype(np.float32).reshape(1, 9, 1, 1)
    current_mass = count_pixels(conv(fp, mass_w))
    goal_mass = vf(unary("vrelu", vf(gfr, 1, -1)), 1, 1)
    for k in range(2, 9):
        goal_mass = vadd(
            goal_mass, vf(unary("vrelu", vf(gfr, 1, -k)), 2 ** (k - 2)))
    required_mass = vmul(goal_mass, gfc)
    impossible = vc(vsub(required_mass, current_mass))
    stalled = vc(vf(stall_next, 1, -2))
    missing_goal = vf(vc(vadd(gfc, gec)), -1, 1)
    undo = vc(vadd(impossible, stalled))
    reset = vc(vadd(
        missing_goal,
        vmul(impossible, vc(vadd(vf(keep, -1, 1), sl("aprev", 7, 8))))))

    flags = cat([undo, reset])
    logit_w = np.zeros((8, 2), np.float32)
    logit_w[7, 0] = 4
    logit_w[0, 1] = 8
    logit_b = np.full(8, -30, np.float32)
    logit_b[6], logit_b[7], logit_b[0] = 1, -1, -0.5
    graph.append({
        "id": "logits", "op": "dense", "in": [flags],
        "w": parameter(logit_w), "b": parameter(logit_b)
    })

    next_x = vadd(vmul(chosen_x, vf(undo, -1, 1)), vmul(last_x, undo))
    next_y = vadd(vmul(chosen_y, vf(undo, -1, 1)), vmul(last_y, undo))
    next_bad_x = vadd(vmul(bad_x, vf(undo, -1, 1)), vmul(last_x, undo))
    next_bad_y = vadd(vmul(bad_y, vf(undo, -1, 1)), vmul(last_y, undo))
    next_bad_age = vadd(
        vmul(unary("vrelu", vf(bad_age, 1, -1)), vf(undo, -1, 1)),
        vf(undo, 8))
    memory_source = cat([
        goals, pad_data, next_x, next_y, ONE, stall_next, state_vector,
        next_bad_x, next_bad_y, next_bad_age
    ])
    graph.append({
        "id": "mem_next", "op": "vadd", "in": [memory_source, ZERO]
    })

    # Static graph bookkeeping: retain only output dependencies.
    by_id = {nd["id"]: nd for nd in graph}
    needed = set()

    def visit(name):
        if name in needed:
            return
        if name not in by_id:
            if "." in name:
                visit(name.split(".", 1)[0])
            return
        needed.add(name)
        nd = by_id[name]
        if nd["op"] == "loop":
            for state in nd["state"]:
                if "init" in state:
                    visit(state["init"])
            local_ids = {b["id"] for b in nd["body"]}
            state_ids = {s["id"] for s in nd["state"]}
            for b in nd["body"]:
                for src in b.get("in", []):
                    if src not in local_ids and src not in state_ids:
                        visit(src)
        else:
            for src in nd["in"]:
                visit(src)

    for output in ["click", "logits", "mem_next"]:
        visit(output)
    graph = [nd for nd in graph if nd["id"] in needed]

    used_parameters = set()
    for nd in graph:
        nodes = nd["body"] if nd["op"] == "loop" else [nd]
        for item in nodes:
            for field in ["w", "b"]:
                if field in item:
                    used_parameters.add(item[field])
    params = {k: v for k, v in params.items() if k in used_parameters}

    return {"graph": graph, "memory_size": M, "params": params}
