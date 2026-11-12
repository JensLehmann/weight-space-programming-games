import numpy as np


def build():
    N = 16
    Q = 4 * N
    M = 6 * N + 16

    class Builder:
        def __init__(self):
            self.graph = []
            self.dst = self.graph
            self.params = {}
            self.param_cache = {}
            self.constants = {}
            self.caches = {}
            self.serial = 0

        def name(self):
            self.serial += 1
            return "n%d" % self.serial

        def param(self, value):
            a = np.asarray(value, dtype=np.float32)
            key = (a.shape, a.tobytes())
            if key not in self.param_cache:
                name = "p%d" % len(self.params)
                self.params[name] = a
                self.param_cache[key] = name
            return self.param_cache[key]

        def op(self, op, *inputs, **fields):
            inputs = tuple(inputs)
            if op in ("add", "mul", "max2", "min2", "vadd", "vmul"):
                inputs = tuple(sorted(inputs))
            key = (op, inputs, tuple(sorted(fields.items())))
            global_cache = self.caches.setdefault(id(self.graph), {})
            local_cache = self.caches.setdefault(id(self.dst), {})
            if key in global_cache:
                return global_cache[key]
            if key in local_cache:
                return local_cache[key]
            name = self.name()
            node = {"id": name, "op": op, "in": list(inputs)}
            node.update(fields)
            self.dst.append(node)
            local_cache[key] = name
            return name

        def dense(self, x, w, bias=None):
            fields = {"w": self.param(w)}
            if bias is not None:
                fields["b"] = self.param(bias)
            return self.op("dense", x, **fields)

        def conv(self, x, w, bias=None):
            fields = {"w": self.param(w)}
            if bias is not None:
                fields["b"] = self.param(bias)
            return self.op("conv", x, **fields)

        def const(self, value):
            a = np.asarray(value, np.float32).reshape(-1)
            key = (a.shape, a.tobytes())
            if key not in self.constants:
                old = self.dst
                self.dst = self.graph
                self.constants[key] = self.dense(
                    "aprev", np.zeros((len(a), 8), np.float32), a
                )
                self.dst = old
            return self.constants[key]

        def loop(self, count, states, body, nxt):
            name = self.name()
            self.dst.append({
                "id": name, "op": "loop", "T": count,
                "state": states, "body": body, "next": nxt,
            })
            return name

    b = Builder()
    C = b.const
    O = b.op

    def va(x, y):
        return O("vadd", x, y)

    def vm(x, y):
        return O("vmul", x, y)

    def vk(x, k):
        return x if k == 1 else vm(x, C(k))

    def vn(x):
        return vk(x, -1)

    def vd(x, y):
        return va(x, vn(y))

    def vr(x):
        return O("vrelu", x)

    def vab(x):
        return va(vr(x), vr(vn(x)))

    def vc(x):
        p = vr(x)
        return vd(p, vr(vd(p, C(1))))

    def ve(x, y):
        return vr(vd(C(1), vab(vd(x, y))))

    def vv(x, start, stop):
        return O("vslice", x, start=start, stop=stop)

    def vsum(x, n):
        return b.dense(x, np.ones((1, n), np.float32))

    def sa(x, y):
        return O("add", x, y)

    def sd(x, y):
        return O("sub", x, y)

    def sm(x, y):
        return O("mul", x, y)

    def sk(x, k):
        return x if k == 1 else sm(x, C(k))

    def sr(x):
        return O("relu", x)

    def sc(x):
        p = sr(x)
        return sd(p, sr(sd(p, C(1))))

    def se(x, y):
        return sr(sd(C(1), O("abs", sd(x, y))))

    def sl(x, start, stop):
        return O("slice", x, start=start, stop=stop)

    def gm(x):
        return O("gmax", x)

    def ssum(x, n):
        return b.conv(x, np.ones((1, n, 1, 1), np.float32))

    one = sl("coords", 2, 3)
    zero = sk(one, 0)

    def vmax(x, n):
        z = sa(zero, x)
        while n > 1:
            half = n // 2
            z = O("max2", sl(z, 0, half), sl(z, half, n))
            n = half
        return gm(z)

    def winner(x, n):
        maximum = vmax(x, n)
        return vc(va(C(1), vk(vd(x, maximum), 1024)))

    def blend(flag, yes, no):
        return va(vm(flag, yes), vm(vd(C(1), flag), no))

    # Sample one pixel at each logical-cell centre.
    def axis_basis(axis):
        w = np.zeros((20, 3, 1, 1), np.float32)
        w[:, axis, 0, 0] = 21
        a = b.conv(
            "coords", w, -1 - np.arange(20, dtype=np.float32)
        )
        return sc(sd(C(2), sk(O("abs", a), 10)))

    rb = axis_basis(0)
    cb = axis_basis(1)
    lattice = sm(ssum(rb, 20), ssum(cb, 20))
    coordinate_weights = np.arange(20, dtype=np.float32).reshape(
        1, 20, 1, 1
    )
    rmap = b.conv(rb, coordinate_weights)
    cmap = b.conv(cb, coordinate_weights)
    ids = sa(sa(sk(rmap, 20), cmap), C(1))

    histogram = O("gmean", "x0")
    background = vc(vk(vd(histogram, C(0.5)), 100))
    background_image = ssum(sm("x0", background), 16)
    fg = sm(lattice, sd(one, background_image))

    xzero = sl("x0", 0, 1)
    xthree = sl("x0", 3, 4)
    xfour = sl("x0", 4, 5)
    xeight = sl("x0", 8, 9)
    marked = sa(xthree, xeight)

    neighbor_weights = np.zeros((1, 1, 7, 7), np.float32)
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for dr, dc in directions:
        neighbor_weights[0, 0, 3 + 3 * dr, 3 + 3 * dc] = 1

    degree = b.conv(fg, neighbor_weights)
    terminal = sm(fg, se(degree, C(1)))
    endpoints = O("max2", terminal, sm(marked, fg))

    grey = gm(sm(xfour, fg))
    colourful_body = sm(fg, sd(sd(one, xfour), marked))
    black_body = sm(fg, xzero)
    selected_body = sa(
        sm(colourful_body, grey),
        sm(black_body, vd(C(1), grey)),
    )
    adjacent_selected = sc(b.conv(selected_body, neighbor_weights))
    selected_full = sc(sa(
        selected_body, sm(sm(marked, adjacent_selected), fg)
    ))
    selected_exists = gm(selected_body)
    grow = gm(sm(sm(terminal, selected_full), xzero))

    # Connected-component labels at initialization.
    rest_full = sr(sd(fg, selected_full))
    group_masks = O("concat", selected_full, rest_full)
    not_endpoint = sd(one, endpoints)
    seeds = sm(
        O("concat",
          sm(selected_full, not_endpoint),
          sm(rest_full, not_endpoint)),
        ids,
    )

    shift_weights = np.zeros((8, 2, 7, 7), np.float32)
    for d, (dr, dc) in enumerate(directions):
        for ch in range(2):
            shift_weights[
                2 * d + ch, ch, 3 + dr * 3, 3 + dc * 3
            ] = 1

    flood_state = b.name()
    flood_body = []
    old = b.dst
    b.dst = flood_body
    shifted = b.conv(flood_state, shift_weights)
    flood_next = flood_state
    for d in range(4):
        flood_next = O(
            "max2", flood_next, sl(shifted, 2 * d, 2 * d + 2)
        )
    flood_next = sm(flood_next, group_masks)
    b.dst = old
    flood = b.loop(
        48, [{"id": flood_state, "init": seeds}],
        flood_body, {flood_state: flood_next},
    )
    roots2 = flood + "." + flood_state
    roots = O("max2", sl(roots2, 0, 1), sl(roots2, 1, 2))

    # Endpoint-bank fields: row, column, label, valid, click row/column.
    shift = np.zeros((N, N), np.float32)
    shift[np.arange(N - 1), np.arange(1, N)] = 1
    insert = np.zeros((N, 1), np.float32)
    insert[-1, 0] = 1

    state_rem = b.name()
    field_states = [b.name() for _ in range(6)]
    extraction_body = []
    b.dst = extraction_body
    largest = gm(sm(state_rem, ids))
    pick = sm(state_rem, se(ids, largest))
    exists = gm(pick)
    pr = gm(sm(rmap, pick))
    pc = gm(sm(cmap, pick))
    pl = gm(sm(roots, pick))
    representative_mask = sm(sm(se(ids, pl), lattice), exists)
    br = gm(sm(rmap, representative_mask))
    bc = gm(sm(cmap, representative_mask))
    extracted = [pr, pc, pl, exists, br, bc]
    field_next = [
        va(b.dense(st, shift), b.dense(value, insert))
        for st, value in zip(field_states, extracted)
    ]
    rem_next = sr(sd(state_rem, pick))
    b.dst = old
    extraction_states = [{"id": state_rem, "init": endpoints}]
    extraction_states += [
        {"id": st, "zeros": [N]} for st in field_states
    ]
    extraction_next = {state_rem: rem_next}
    extraction_next.update(dict(zip(field_states, field_next)))
    extraction = b.loop(
        N, extraction_states, extraction_body, extraction_next
    )
    initial_fields = [
        extraction + "." + st for st in field_states
    ]

    memory_fields = [
        vv("mem", N * i, N * (i + 1)) for i in range(6)
    ]
    extras = vv("mem", 6 * N, M)
    ex = [vv(extras, i, i + 1) for i in range(16)]
    ready_old = ex[0]
    fields = [
        blend(ready_old, memory_value, initial_value)
        for memory_value, initial_value
        in zip(memory_fields, initial_fields)
    ]
    R, K, L, V, BR, BK = fields

    def point_planes(rows, cols, valid):
        distance = sa(
            O("abs", sd(rmap, rows)),
            O("abs", sd(cmap, cols)),
        )
        return sm(
            sm(sr(sd(C(1), distance)), lattice), valid
        )

    points = point_planes(R, K, V)
    representatives = point_planes(BR, BK, V)

    root_visible = gm(sm(representatives, selected_body))
    detected_label = vmax(vm(root_visible, L), N)
    selected_label = blend(ready_old, ex[1], detected_label)
    anchor = blend(ready_old, ex[2], selected_label)
    membership = vm(ve(L, selected_label), V)
    is_anchor = ve(selected_label, anchor)

    all_counts = ssum(points, N)
    paired = sm(se(all_counts, C(2)), lattice)
    matched = gm(sm(points, paired))
    unmatched = vm(V, vd(C(1), matched))

    selected_map = sc(ssum(sm(points, membership), N))
    ref_id = gm(sm(selected_map, ids))
    ref_mask = sm(sm(se(ids, ref_id), selected_map), lattice)
    ref_r = gm(sm(ref_mask, rmap))
    ref_c = gm(sm(ref_mask, cmap))

    target_counts = ssum(sm(points, vd(C(1), membership)), N)
    target_single = sm(se(target_counts, C(1)), lattice)

    # Observed body plus this component's remembered endpoints.
    geometric_full = O("max2", selected_body, selected_map)
    rmin = vd(
        C(20), gm(sm(sd(C(20), rmap), geometric_full))
    )
    cmin = vd(
        C(20), gm(sm(sd(C(20), cmap), geometric_full))
    )
    rmax = gm(sm(rmap, geometric_full))
    cmax = gm(sm(cmap, geometric_full))
    height = va(vd(rmax, rmin), C(1))
    width = va(vd(cmax, cmin), C(1))

    # Analytic permutations provide deterministic tie variation.
    phase = ve(ex[8], C(np.arange(8)))
    picker_patterns = np.empty((N, 8), np.float32)
    placement_patterns = np.empty((Q, 8), np.float32)
    for j in range(8):
        picker_patterns[:, j] = (
            (11 + 2 * j) * np.arange(N) + 7 * j
        ) % N
        placement_patterns[:, j] = (
            (37 + 2 * j) * np.arange(Q) + 53 * j
        ) % Q

    non_anchor = vm(V, vd(C(1), ve(L, anchor)))
    pick_score = va(
        va(vk(unmatched, 16384),
           vk(vd(C(1), membership), 1024)),
        va(
            vk(b.dense(phase, picker_patterns), 32),
            C(np.arange(N)),
        ),
    )
    pick_score = vd(
        pick_score, vk(vd(C(1), non_anchor), 1048576)
    )
    chosen_point = winner(pick_score, N)
    desired_label = vsum(vm(chosen_point, L), N)
    desired_br = vsum(vm(chosen_point, BR), N)
    desired_bc = vsum(vm(chosen_point, BK), N)
    desired_same = ve(desired_label, selected_label)

    # Four rotated placements for each possible endpoint alignment.
    repeat = np.tile(np.eye(N, dtype=np.float32), (4, 1))
    repeat_four = np.zeros((Q, 4), np.float32)
    for q in range(4):
        repeat_four[N * q:N * (q + 1), q] = 1
    base_r = b.dense(R, repeat)
    base_c = b.dense(K, repeat)
    candidate_valid = b.dense(V, repeat)
    turns_vector = C(np.repeat(np.arange(4), N))

    row_rotation = np.zeros((Q, 2), np.float32)
    col_rotation = np.zeros((Q, 2), np.float32)
    row_coefficients = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    col_coefficients = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    for q in range(4):
        row_rotation[N * q:N * (q + 1)] = row_coefficients[q]
        col_rotation[N * q:N * (q + 1)] = col_coefficients[q]

    scan_states = [b.name() for _ in range(4)]
    score_state = b.name()
    scan_body = []
    b.dst = scan_body
    rr, kk, ll, valid = [
        vv(st, 0, 1) for st in scan_states
    ]
    flag = vm(ve(ll, selected_label), valid)
    relative = O("vconcat", vd(rr, ref_r), vd(kk, ref_c))
    dest_r = va(base_r, b.dense(relative, row_rotation))
    dest_c = va(base_c, b.dense(relative, col_rotation))
    destination_planes = point_planes(
        dest_r, dest_c, C(np.ones(Q))
    )
    hits = vm(gm(sm(destination_planes, target_single)), flag)
    score_next = va(score_state, hits)
    scan_next = [b.dense(st, shift) for st in scan_states]
    b.dst = old
    scan_initial = [
        {"id": st, "init": value}
        for st, value in zip(scan_states, [R, K, L, V])
    ]
    scan_initial.append({"id": score_state, "zeros": [Q]})
    scan_mapping = dict(zip(scan_states, scan_next))
    scan_mapping[score_state] = score_next
    scan = b.loop(N, scan_initial, scan_body, scan_mapping)
    hit_scores = scan + "." + score_state

    def expanded(values):
        return b.dense(O("vconcat", *values), repeat_four)

    candidate_rmin = va(base_r, expanded([
        vd(rmin, ref_r), vd(cmin, ref_c),
        vd(ref_r, rmax), vd(ref_c, cmax),
    ]))
    candidate_cmin = va(base_c, expanded([
        vd(cmin, ref_c), vd(ref_r, rmax),
        vd(ref_c, cmax), vd(rmin, ref_r),
    ]))
    candidate_rmax = va(base_r, expanded([
        vd(rmax, ref_r), vd(cmax, ref_c),
        vd(ref_r, rmin), vd(ref_c, cmin),
    ]))
    candidate_cmax = va(base_c, expanded([
        vd(cmax, ref_c), vd(ref_r, rmin),
        vd(ref_c, cmin), vd(rmax, ref_r),
    ]))

    bounds_error = va(
        va(vr(vn(candidate_rmin)), vr(vn(candidate_cmin))),
        va(
            vr(vd(candidate_rmax, C(19))),
            vr(vd(candidate_cmax, C(19))),
        ),
    )
    travel = va(
        vab(vd(candidate_rmin, rmin)),
        vab(vd(candidate_cmin, cmin)),
    )
    same_configuration = vm(
        vm(ve(candidate_rmin, rmin),
           ve(candidate_cmin, cmin)),
        ve(turns_vector, C(0)),
    )
    selected_unmatched = vc(
        vsum(vm(unmatched, membership), N)
    )
    same_configuration = vm(
        same_configuration, selected_unmatched
    )

    scores = va(
        vk(hit_scores, 262144),
        vk(b.dense(phase, placement_patterns), 1024),
    )
    scores = vd(scores, vk(travel, 128))
    scores = vd(scores, vk(turns_vector, 128))
    scores = va(scores, C(np.arange(Q)))
    scores = vd(scores, vk(same_configuration, 131072))
    scores = vd(scores, vk(bounds_error, 16777216))
    scores = vd(
        scores, vk(vd(C(1), candidate_valid), 16777216)
    )

    placement = winner(scores, Q)
    new_goal_r = vsum(vm(placement, candidate_rmin), Q)
    new_goal_c = vsum(vm(placement, candidate_cmin), Q)
    new_turns = vsum(vm(placement, turns_vector), Q)

    # Initialization and centering have separate flags.
    # Coordinates are stored immediately, before any centering movement.
    centre_r = vk(vd(C(19), vd(rmax, rmin)), 0.5)
    centre_c = vk(vd(C(19), vd(cmax, cmin)), 0.5)

    def centre_positive(x):
        return vc(vk(vd(x, C(0.5)), 2))

    early = vm(
        vm(vd(C(1), ex[9]), selected_exists),
        vd(C(1), grow),
    )
    early_up = vm(early, centre_positive(vd(rmin, centre_r)))
    early_down = vm(early, centre_positive(vd(centre_r, rmin)))
    early_row = va(early_up, early_down)
    early_horizontal = vm(early, vd(C(1), early_row))
    early_left = vm(
        early_horizontal, centre_positive(vd(cmin, centre_c))
    )
    early_right = vm(
        early_horizontal, centre_positive(vd(centre_c, cmin))
    )
    early_move = va(early_row, va(early_left, early_right))

    ready = vm(
        vc(va(ready_old, selected_exists)), vd(C(1), grow)
    )
    centered_next = vm(
        vc(va(ex[9], vm(ready, vd(C(1), early_move)))),
        vd(C(1), grow),
    )
    work = vm(
        vm(ready, selected_exists), vd(C(1), early_move)
    )

    arrived = vm(
        vm(ve(ex[3], rmin), ve(ex[4], cmin)),
        ve(ex[5], C(0)),
    )
    old_active = vm(
        vm(ex[6], ve(ex[1], selected_label)),
        vm(work, vd(C(1), arrived)),
    )
    plan = vm(
        vm(work, vd(C(1), is_anchor)),
        vc(va(old_active, va(ex[7], desired_same))),
    )
    click_mode = vm(
        vm(vm(work, vd(C(1), old_active)), vd(C(1), plan)),
        vd(C(1), desired_same),
    )
    fallback_click = vd(C(1), selected_exists)
    click_flag = vc(va(click_mode, fallback_click))

    goal_r = blend(old_active, ex[3], new_goal_r)
    goal_c = blend(old_active, ex[4], new_goal_c)
    turns = blend(old_active, ex[5], new_turns)

    plan_rotate = vm(plan, vc(turns))
    translating = vm(plan, vd(C(1), plan_rotate))
    plan_up = vm(translating, vc(vd(rmin, goal_r)))
    plan_down = vm(translating, vc(vd(goal_r, rmin)))
    plan_row = va(plan_up, plan_down)
    horizontal = vm(translating, vd(C(1), plan_row))
    plan_left = vm(horizontal, vc(vd(cmin, goal_c)))
    plan_right = vm(horizontal, vc(vd(goal_c, cmin)))

    up = va(early_up, plan_up)
    down = va(early_down, plan_down)
    left = va(early_left, plan_left)
    right = va(early_right, plan_right)
    rotate = vc(va(grow, plan_rotate))
    action_sum = va(
        va(up, down),
        va(va(left, right), va(rotate, click_flag)),
    )
    idle = vd(C(1), vc(action_sum))
    rotate = vc(va(rotate, idle))

    action_flags = O(
        "vconcat", C(0), up, down, left, right,
        rotate, click_flag, C(0),
    )
    logits_value = vk(action_flags, 10)
    logits_bias = C([-10, 0, 0, 0, 0, 0, 0, -10])
    b.graph.append({
        "id": "logits", "op": "vadd",
        "in": [logits_value, logits_bias],
    })

    desired_click = sm(
        sm(se(rmap, desired_br), se(cmap, desired_bc)),
        lattice,
    )
    desired_membership = vm(ve(L, desired_label), V)
    endpoint_clicks = ssum(
        sm(points, vm(desired_membership, unmatched)), N
    )
    # Prefer the representative if visible and not part of the selected body.
    visible_unselected = sm(fg, sd(one, selected_body))
    desired_click_scores = sa(
        sk(sm(desired_click, visible_unselected), 100),
        sk(sm(endpoint_clicks, sm(xfour, lattice)), 10),
    )
    # Low-priority representative remains available on unmasked levels.
    desired_click_scores = sa(
        desired_click_scores, desired_click
    )
    fallback_map = sm(
        fg, sa(ids, sk(sr(sd(fg, endpoints)), 1024))
    )
    click_map = sa(
        sk(sm(desired_click_scores, click_mode), 10000),
        sm(fallback_map, fallback_click),
    )
    b.graph.append({
        "id": "click", "op": "relu", "in": [click_map],
    })

    # Apply issued motor commands to the stored coordinates.
    rotation_update = vm(
        vd(C(1), grow), vm(selected_exists, rotate)
    )
    new_top_r = vd(rmin, vr(vd(rmin, vd(C(20), width))))
    new_top_c = vd(cmin, vr(vd(cmin, vd(C(20), height))))
    dy = vd(down, up)
    dx = vd(right, left)

    def update_coordinates(rows, cols):
        rotated_r = va(new_top_r, vd(cols, cmin))
        rotated_c = va(
            va(new_top_c, vd(height, C(1))),
            vn(vd(rows, rmin)),
        )
        dr = va(vm(rotation_update, vd(rotated_r, rows)), dy)
        dc = va(vm(rotation_update, vd(rotated_c, cols)), dx)
        return (
            va(rows, vm(membership, dr)),
            va(cols, vm(membership, dc)),
        )

    next_R, next_K = update_coordinates(R, K)
    next_BR, next_BK = update_coordinates(BR, BK)

    motor = vc(va(
        plan_rotate, va(plan_row, va(plan_left, plan_right))
    ))
    active_next = vm(plan, motor)
    turns_next = vr(vd(turns, plan_rotate))
    selected_next = blend(
        click_mode, desired_label, selected_label
    )

    fresh_plan = vm(plan, vd(C(1), old_active))
    attempt = va(ex[8], fresh_plan)
    attempt = vd(attempt, vk(ve(attempt, C(8)), 8))
    attempt = vm(attempt, vd(C(1), grow))

    next_extra = O(
        "vconcat",
        ready, selected_next, anchor,
        goal_r, goal_c, turns_next, active_next,
        click_flag, attempt, centered_next, C(np.zeros(6)),
    )
    next_memory = O(
        "vconcat",
        next_R, next_K, L, V, next_BR, next_BK, next_extra,
    )
    memory_zero = C(np.zeros(M))
    b.graph.append({
        "id": "mem_next", "op": "vadd",
        "in": [next_memory, memory_zero],
    })

    # Compile away unreachable expressions and unused parameters.
    def prune(nodes, required):
        table = {node["id"]: node for node in nodes}
        kept = set()
        external = set()

        def visit(name):
            base = name.split(".", 1)[0]
            if base not in table:
                external.add(name)
                return
            if base in kept:
                return
            kept.add(base)
            node = table[base]
            if node["op"] != "loop":
                for source in node["in"]:
                    visit(source)
                return
            body, body_external = prune(
                node["body"], list(node["next"].values())
            )
            node["body"] = body
            state_ids = {state["id"] for state in node["state"]}
            for source in body_external - state_ids:
                visit(source)
            for state in node["state"]:
                if "init" in state:
                    visit(state["init"])

        for name in required:
            visit(name)
        return (
            [node for node in nodes if node["id"] in kept],
            external,
        )

    graph, _ = prune(b.graph, ["logits", "click", "mem_next"])
    used_parameters = set()

    def collect(nodes):
        for node in nodes:
            for field in ("w", "b"):
                if field in node:
                    used_parameters.add(node[field])
            if node["op"] == "loop":
                collect(node["body"])

    collect(graph)
    params = {
        name: value for name, value in b.params.items()
        if name in used_parameters
    }
    return {"graph": graph, "memory_size": M, "params": params}
