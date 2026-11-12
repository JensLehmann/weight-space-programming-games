import numpy as np


def build():
    params = {}
    cache = {}
    graph = []
    serial = [0]
    K = 12

    def fresh(prefix="n"):
        serial[0] += 1
        return "%s_%d" % (prefix, serial[0])

    def parameter(value):
        value = np.asarray(value, dtype=np.float32)
        key = (value.shape, value.tobytes())
        if key not in cache:
            name = fresh("p")
            cache[key] = name
            params[name] = value
        return cache[key]

    def node(op, inputs, name=None, **fields):
        name = name or fresh()
        item = {"id": name, "op": op, "in": list(inputs)}
        item.update(fields)
        graph.append(item)
        return name

    def conv(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], name=name, **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], name=name, **fields)

    def affine(x, a=1., b=0., channels=1):
        w = (np.eye(channels, dtype=np.float32) * a)[:, :, None, None]
        return conv(x, w, np.full(channels, b, np.float32))

    def va(x, a=1., b=0., channels=1):
        return dense(x, np.eye(channels, dtype=np.float32) * a,
                     np.full(channels, b, np.float32))

    def constant(values):
        values = np.asarray(values, np.float32).reshape(-1)
        return dense("avail", np.zeros((len(values), 8), np.float32),
                     values)

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vs(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def cat(*xs):
        return node("concat", xs)

    def vc(*xs):
        return node("vconcat", xs)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def mx(a, b):
        return node("max2", [a, b])

    def mn(a, b):
        return node("min2", [a, b])

    def relu(x):
        return node("relu", [x])

    def ab(x):
        return node("abs", [x])

    def gm(x):
        return node("gmax", [x])

    def vm(a, b):
        return node("vmul", [a, b])

    def vadd(a, b):
        return node("vadd", [a, b])

    def loop(count, initial, builder):
        nonlocal graph
        outer = graph
        states = [fresh("s") for _ in initial]
        graph = []
        final = builder(*states)
        body = graph
        graph = outer
        lid = fresh("L")
        graph.append({
            "id": lid,
            "op": "loop",
            "T": count,
            "state": [{"id": s, "init": v}
                      for s, v in zip(states, initial)],
            "body": body,
            "next": dict(zip(states, final))
        })
        return tuple(lid + "." + s for s in states)

    one_v = constant([1.])
    four_v = constant([4.])
    zero = conv("coords", np.zeros((1, 3, 1, 1), np.float32))
    one = conv("coords", np.zeros((1, 3, 1, 1), np.float32), [1.])

    xy_w = np.zeros((2, 3, 1, 1), np.float32)
    xy_w[0, 0, 0, 0] = 63.
    xy_w[1, 1, 0, 0] = 63.
    xy = conv("coords", xy_w)
    yy, xx = sl(xy, 0, 1), sl(xy, 1, 2)
    rank = conv(
        "coords",
        np.array([4032., 63., 1.], np.float32).reshape(1, 3, 1, 1)
    )
    first_rank = affine(rank, -1., 4098.)

    def clip(x):
        return mn(relu(x), one_v)

    def inv(x):
        return sub(one_v, x)

    def eq(a, b):
        return relu(sub(one_v, ab(sub(a, b))))

    def binary(x, threshold=.5, channels=1):
        return clip(affine(x, 100., .5 - 100. * threshold, channels))

    def total(x, channels, weights=None):
        if weights is None:
            weights = np.ones(channels, np.float32)
        return conv(
            x, np.asarray(weights, np.float32).reshape(1, channels, 1, 1)
        )

    def argmask(score, valid=None):
        mask = relu(affine(sub(score, gm(score)), 1000000., 1.))
        if valid is not None:
            mask = mul(mask, valid)
        tie_score = mul(mask, first_rank)
        tied = relu(affine(sub(tie_score, gm(tie_score)), 10000., 1.))
        return mul(mask, tied)

    def pick(mask):
        return argmask(mul(mask, rank), mask)

    def sample(mask, field, channels):
        pooled = node("gmean", [mul(mask, field)])
        return dense(pooled, np.eye(channels, dtype=np.float32) * 4096.)

    def distance(y, x):
        return add(ab(sub(yy, y)), ab(sub(xx, x)))

    def location_score(valid, y, x):
        return sub(
            sub(affine(distance(y, x), -1.), affine(inv(valid), 1000.)),
            affine(rank, .0005)
        )

    def box(mask, radius):
        while radius:
            r = min(radius, 3)
            mask = clip(conv(
                mask, np.ones((1, 1, 2*r+1, 2*r+1), np.float32)
            ))
            radius -= r
        return mask

    def matches(code_field, values):
        values = np.asarray(values, np.float32)
        d = conv(code_field,
                 np.ones((len(values), 1, 1, 1), np.float32), -values)
        return relu(sub(one_v, ab(d)))

    code = conv("x0", np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1))
    c0 = sl("x0", 0, 1)
    c1 = sl("x0", 1, 2)
    c2 = sl("x0", 2, 3)
    c3 = sl("x0", 3, 4)
    c6 = sl("x0", 6, 7)
    c10 = sl("x0", 10, 11)
    has_hazard = gm(c10)
    no_hazard = va(has_hazard, -1., 1.)

    # Rounded filled discs and their left/right colours.
    fw = np.zeros((1, 16, 1, 1), np.float32)
    for colour in [0, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15]:
        fw[0, colour, 0, 0] = 1.
    filled = conv("x0", fw)
    disc_kernel = np.ones((1, 1, 5, 5), np.float32)
    for y, x in [(0, 0), (0, 4), (4, 0), (4, 4)]:
        disc_kernel[0, 0, y, x] = 0.
    discs = relu(conv(filled, disc_kernel, [-20.]))

    lw = np.zeros((1, 1, 5, 5), np.float32)
    rw = np.zeros_like(lw)
    lw[0, 0, 2, 0] = 1.
    rw[0, 0, 2, 4] = 1.
    disc_left, disc_right = conv(code, lw), conv(code, rw)
    ordinary = mul(c6, eq(disc_left, disc_right))
    blank = mul(eq(disc_left, zero), eq(disc_right, zero))
    near_line = clip(conv(c1, np.ones((1, 1, 7, 7), np.float32)))
    body_possible = mx(mx(ordinary, blank), near_line)
    disc_features = cat(
        xy, disc_left, disc_right, ordinary, body_possible
    )

    # Diamond anchor detectors, including partial ordinary-disc occlusion.
    cw = np.zeros((1, 16, 1, 1), np.float32)
    for colour in [4, 6, 7, 8, 9, 11, 12, 13, 14, 15]:
        cw[0, colour, 0, 0] = 1.
    centre_ok = conv("x0", cw)
    anchor_input = cat(add(c0, c3), c0, centre_ok)
    aw = np.zeros((2, 3, 5, 5), np.float32)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if 0 < abs(dy) + abs(dx) <= 2:
                aw[0, 0, dy+2, dx+2] = 1.
                aw[1, 1, dy+2, dx+2] = 1.
    aw[:, 2, 2, 2] = 1.
    anchor_counts = conv(anchor_input, aw)
    exact_anchors = relu(affine(anchor_counts, 1., -12., 2))
    near_ordinary = box(mul(discs, ordinary), 4)
    partial_gate = mul(near_ordinary, centre_ok)
    partial_anchors = mul(binary(anchor_counts, 6.5, 2), partial_gate)
    detected = mx(exact_anchors, partial_anchors)
    anchors = sl(detected, 0, 1)
    selected_anchors = sl(detected, 1, 2)

    direction_w = np.zeros((2, 1, 7, 7), np.float32)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if max(abs(dy), abs(dx)) == 3:
                direction_w[0, 0, dy+3, dx+3] = dy
                direction_w[1, 0, dy+3, dx+3] = dx
    directions = conv(c1, direction_w)
    anchor_features = cat(xy, code, selected_anchors, one, directions)
    anchor_slots0 = constant(np.zeros(7*K, np.float32))

    def extract_anchor(rem, slots):
        chosen = pick(rem)
        features = sample(chosen, anchor_features, 7)
        return relu(sub(rem, chosen)), vc(vs(slots, 7, 7*K), features)

    _, anchor_slots = loop(K, [anchors, anchor_slots0], extract_anchor)

    anchor_vectors = []
    for feature in range(7):
        w = np.zeros((K, 7*K), np.float32)
        for j in range(K):
            w[j, 7*j+feature] = 1.
        anchor_vectors.append(dense(anchor_slots, w))
    ay, ax, ac, selected_v, present_v, ady, adx = anchor_vectors
    ay_field, ax_field = add(zero, ay), add(zero, ax)
    ac_field = add(zero, ac)

    # Complete goal-outline halves.
    colours = [4, 6, 7, 8, 9, 11, 12, 13, 14, 15]
    C = len(colours)
    pw = np.zeros((C, 16, 1, 1), np.float32)
    for i, colour in enumerate(colours):
        pw[i, colour, 0, 0] = 1.
    palette = conv("x0", pw)
    gw = np.zeros((2*C, C, 7, 7), np.float32)
    points = [(-3, -1), (-2, -2), (-1, -3),
              (1, -3), (2, -2), (3, -1)]
    for i in range(C):
        for dy, dx in points:
            gw[i, i, dy+3, dx+3] = 1.
            gw[C+i, i, dy+3, -dx+3] = 1.
    ring_parts = relu(conv(palette, gw, np.full(2*C, -5., np.float32)))
    ring_left, ring_right = sl(ring_parts, 0, C), sl(ring_parts, C, 2*C)
    goals = mul(total(ring_left, C), total(ring_right, C))
    goal_left = total(ring_left, C, colours)
    goal_right = total(ring_right, C, colours)
    split_goal = clip(ab(sub(goal_left, goal_right)))
    goal_features = cat(xy, goal_left, goal_right)

    # All twelve-bit subsets, represented across the spatial lattice.
    basis_indices = {}
    bit_terms = []

    def basis(axis, threshold):
        key = (axis, threshold)
        if key not in basis_indices:
            basis_indices[key] = len(basis_indices)
        return basis_indices[key]

    for axis in range(2):
        for bit in range(6):
            width = 1 << bit
            terms = []
            for start in range(width, 64, 2*width):
                end = start + width
                for threshold, coefficient in [
                    (start-1, 1.), (start, -1.),
                    (end-1, -1.), (end, 1.)
                ]:
                    terms.append((basis(axis, threshold), coefficient))
            bit_terms.append(terms)

    B = len(basis_indices)
    bw = np.zeros((B, 2, 1, 1), np.float32)
    bb = np.zeros(B, np.float32)
    for (axis, threshold), i in basis_indices.items():
        bw[i, 1 if axis == 0 else 0, 0, 0] = 1.
        bb[i] = -threshold
    hinges = relu(conv(xy, bw, bb))
    bit_w = np.zeros((K, B, 1, 1), np.float32)
    for i, terms in enumerate(bit_terms):
        for j, coefficient in terms:
            bit_w[i, j, 0, 0] += coefficient
    bits = binary(conv(hinges, bit_w), .5, K)
    subset_n = total(bits, K)
    subset_y = total(mul(bits, ay), K)
    subset_x = total(mul(bits, ax), K)
    short_subset = relu(affine(subset_n, -1., 2.))
    black_mismatch = inv(eq(ac_field, four_v))
    colour_error = total(mul(bits, black_mismatch), K)
    subset_invalid = affine(add(colour_error, short_subset), 10.)
    subset_prior = add(affine(subset_n, .05), affine(rank, .001))
    sum_w = np.ones((1, K), np.float32)

    def classify(features):
        by, bx = vs(features, 0, 1), vs(features, 1, 2)
        bl = vs(features, 2, 3)
        normal = vs(features, 4, 5)
        possible = vs(features, 5, 6)

        coloured_members = gm(mul(eq(ac_field, bl), present_v))
        coloured_count = dense(coloured_members, sum_w)
        coloured_valid = gm(binary(add(zero, coloured_count), 1.5))

        ey = sub(subset_y, mul(subset_n, by))
        ex = sub(subset_x, mul(subset_n, bx))

        def interval_error(e):
            return add(
                relu(affine(e, -1., -.02)),
                relu(affine(sub(e, subset_n), 1., .98))
            )

        violation = add(
            add(interval_error(ey), interval_error(ex)),
            subset_invalid
        )
        dy, dx = sub(by, ay_field), sub(bx, ax_field)
        dot = add(mul(dy, ady), mul(dx, adx))
        cross = sub(mul(dx, ady), mul(dy, adx))
        orientation = add(
            total(mul(bits, relu(affine(dot, -1., 0., K))),
                  K, np.full(K, .03, np.float32)),
            total(mul(bits, ab(cross)),
                  K, np.full(K, .002, np.float32))
        )
        score = sub(sub(affine(violation, -1000.), subset_prior),
                    orientation)
        chosen = argmask(score)
        good = clip(affine(violation, -100., 1.))
        black_valid = vm(gm(mul(chosen, good)), possible)
        black_members = vm(gm(binary(mul(bits, chosen), .5, K)),
                           black_valid)

        non_normal = va(normal, -1., 1.)
        valid = vadd(vm(normal, coloured_valid),
                     vm(non_normal, black_valid))
        members = vadd(
            vm(vm(coloured_members, coloured_valid), normal),
            vm(black_members, non_normal)
        )
        active = dense(vm(members, selected_v), sum_w)
        return valid, active, members

    def classify_discs(rem, bodies, pickup_map, active_map):
        chosen = pick(rem)
        features = sample(chosen, disc_features, 6)
        valid, active, _ = classify(features)
        return (
            relu(sub(rem, chosen)),
            add(bodies, mul(chosen, valid)),
            add(pickup_map, mul(chosen, va(valid, -1., 1.))),
            add(active_map, mul(chosen, active))
        )

    _, bodies, pickups, active_bodies = loop(
        16, [discs, zero, zero, zero], classify_discs
    )

    pickup_left_planes = matches(disc_left, colours)
    pickup_right_planes = matches(disc_right, colours)
    left_available = gm(mul(pickups, pickup_left_planes))
    right_available = gm(mul(pickups, pickup_right_planes))
    goal_left_available = clip(total(mul(ring_left, left_available), C))
    goal_right_available = clip(total(mul(ring_right, right_available), C))

    empty_targets = conv(zero, np.zeros((2, 1, 1, 1), np.float32))
    active_priority_scale = va(has_hazard, -90., 100.)

    def choose_objective(rem, priorities, targets):
        chosen = pick(rem)
        f = sample(chosen, disc_features, 6)
        by, bx = vs(f, 0, 1), vs(f, 1, 2)
        bl, br = vs(f, 2, 3), vs(f, 3, 4)
        normal = vs(f, 4, 5)
        missing_left = gm(eq(zero, bl))
        missing_right = gm(eq(zero, br))

        left_match = mx(eq(goal_left, bl), missing_left)
        right_match = mx(eq(goal_right, br), missing_right)
        left_supply = mx(goal_left_available, va(missing_left, -1., 1.))
        right_supply = mx(goal_right_available, va(missing_right, -1., 1.))
        kind = add(mul(inv(split_goal), normal),
                   mul(split_goal, va(normal, -1., 1.)))
        eligible = binary(
            mul(mul(mul(goals, kind), mul(left_match, right_match)),
                mul(left_supply, right_supply))
        )
        exists = gm(eligible)
        goal_choice = argmask(location_score(eligible, by, bx), eligible)
        gf = sample(goal_choice, goal_features, 4)
        gy, gx = vs(gf, 0, 1), vs(gf, 1, 2)
        gl, gr = vs(gf, 2, 3), vs(gf, 3, 4)

        take_left = mul(eq(disc_left, gl), missing_left)
        take_right = mul(eq(disc_right, gr), missing_right)
        pickup_eligible = binary(mul(pickups, mx(take_left, take_right)))
        pickup_choice = argmask(
            location_score(pickup_eligible, by, bx), pickup_eligible
        )
        pf = sample(pickup_choice, xy, 2)
        need = gm(mx(add(zero, missing_left), add(zero, missing_right)))
        target = vadd(vm(vc(gy, gx), va(need, -1., 1.)), vm(pf, need))

        active = gm(mul(chosen, active_bodies))
        ty, tx = vs(target, 0, 1), vs(target, 1, 2)
        dist = gm(mul(chosen, distance(ty, tx)))
        position = gm(mul(chosen, rank))

        # Clear the most controllable ordinary group first in hazard layouts.
        same_colour = gm(mul(eq(ac_field, bl), present_v))
        anchor_count = dense(same_colour, sum_w)
        control_bonus = vm(
            vm(va(anchor_count, -20., 240.), normal), has_hazard
        )
        priority = vadd(
            va(vm(active, active_priority_scale), 1., 100.),
            vadd(control_bonus,
                 vadd(va(dist, -.01), va(position, -.0001)))
        )
        priority = vm(priority, exists)
        return (
            relu(sub(rem, chosen)),
            mx(priorities, mul(chosen, priority)),
            add(targets, mul(chosen, target))
        )

    _, priorities, target_fields = loop(
        8, [bodies, zero, empty_targets], choose_objective
    )
    chosen_body = argmask(priorities, binary(priorities, 1.))
    body_f = sample(chosen_body, disc_features, 6)
    target = sample(chosen_body, target_fields, 2)
    target_y, target_x = vs(target, 0, 1), vs(target, 1, 2)
    _, _, members = classify(body_f)
    group_n = dense(members, sum_w)
    group_y = dense(vm(members, ay), sum_w)
    group_x = dense(vm(members, ax), sum_w)
    n_hats = matches(add(zero, group_n), list(range(1, K+1)))
    inverse_n = gm(total(n_hats, K, [1. / n for n in range(1, K+1)]))

    # Endpoint legality and rendering clearance.
    border = mx(
        mx(clip(affine(xx, -1., 3.)), clip(affine(xx, 1., -61.))),
        mx(clip(affine(yy, -1., 2.)), clip(affine(yy, 1., -61.)))
    )
    diamond = np.zeros((1, 1, 5, 5), np.float32)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dy) + abs(dx) <= 2:
                diamond[0, 0, dy+2, dx+2] = 1.
    anchor_wall = clip(conv(c2, diamond))
    anchor_blocked = mx(
        mx(anchor_wall, border),
        mx(box(anchors, 4), box(mx(discs, goals), 4))
    )

    target_position = eq(distance(target_y, target_x), zero)
    intended_pickup = mul(pickups, target_position)
    other_pickups = relu(sub(pickups, intended_pickup))
    other_bodies = relu(sub(bodies, chosen_body))
    other_goals = relu(sub(goals, mul(goals, target_position)))

    centre_blocked = mx(
        mx(box(c10, 3), box(c2, 2)),
        mx(
            mx(border, box(anchors, 4)),
            mx(box(other_pickups, 4),
               mx(box(other_bodies, 4), box(other_goals, 5)))
        )
    )
    centre_blocked = binary(centre_blocked)
    centre_safe = inv(centre_blocked)

    target_dist = distance(target_y, target_x)
    seed = mul(
        clip(affine(relu(affine(target_dist, 1., -2.)), -1., 1.)),
        centre_safe
    )
    seed = affine(seed, 1000.)

    def spread_axis(value, axis, shifts):
        radius = max(abs(s) for s in shifts)
        kernel_size = 2*radius + 1
        w = np.zeros((len(shifts), 1, kernel_size, kernel_size), np.float32)
        b = np.zeros(len(shifts), np.float32)
        for i, shift in enumerate(shifts):
            if axis == 0:
                w[i, 0, radius+shift, radius] = 1.
            else:
                w[i, 0, radius, radius+shift] = 1.
            b[i] = -abs(shift)
        shifted = conv(value, w, b)
        result = sl(shifted, 0, 1)
        for i in range(1, len(shifts)):
            result = mx(result, sl(shifted, i, i+1))
        return result

    def distance_iteration(value):
        spread = value
        for axis in (0, 1):
            short = spread_axis(spread, axis, [-3, 0, 3])
            long = spread_axis(short, axis, [-3, 0, 3])
            long = spread_axis(long, axis, [-3, 0, 3])
            spread = add(mul(short, has_hazard), mul(long, no_hazard))
            spread = spread_axis(spread, axis, [-3, -2, -1, 0, 1, 2, 3])
        updated = mx(value, affine(spread, 1., -.25))
        return (mx(seed, mul(relu(updated), centre_safe)),)

    (navigation,) = loop(32, [seed], distance_iteration)
    endpoint_value = sub(navigation, affine(centre_blocked, 4000.))
    indices = np.arange(64, dtype=np.float32)

    def integer_hats(q):
        d = conv(q, np.ones((64, 1, 1, 1), np.float32), -indices)
        return relu(sub(one_v, ab(d)))

    def floor_hats(q):
        d = conv(q, np.ones((64, 1, 1, 1), np.float32),
                 -indices - np.float32(.49))
        return clip(affine(ab(d), -200., 100., 64))

    row_hats = integer_hats(yy)
    column_hats = integer_hats(xx)
    source_rows = node("colsum", [mul(row_hats, endpoint_value)])

    def sample_navigation(py, px):
        row_weights = floor_hats(py)
        intermediate = total(mul(source_rows, row_weights), 64)
        columns = node("rowsum", [mul(column_hats, intermediate)])
        return total(mul(columns, floor_hats(px)), 64)

    # Eight previous objective-disc centres, plus the previous click.
    hy_w = np.zeros((8, 18), np.float32)
    hx_w = np.zeros((8, 18), np.float32)
    for i in range(8):
        hy_w[i, 2 + 2*i] = 1.
        hx_w[i, 3 + 2*i] = 1.
    history_y = dense("mem", hy_w)
    history_x = dense("mem", hx_w)

    packed_input = vc(ay, ax, members, selected_v)
    pack_w = np.zeros((4*K, 4*K), np.float32)
    for j in range(K):
        for f in range(4):
            pack_w[4*j+f, f*K+j] = 1.
    packed = dense(packed_input, pack_w)
    initial_click = affine(zero, 0., -1000000000.)

    def plan_anchor(slots, best):
        current = vs(slots, 0, 4)
        cy, cx = vs(current, 0, 1), vs(current, 1, 2)
        member, selected = vs(current, 2, 3), vs(current, 3, 4)

        others_y = vadd(group_y, va(cy, -1.))
        others_x = vadd(group_x, va(cx, -1.))
        py = mul(add(yy, others_y), inverse_n)
        px = mul(add(xx, others_x), inverse_n)
        value = sample_navigation(py, px)
        dist = add(ab(sub(py, target_y)), ab(sub(px, target_x)))
        displacement = distance(cy, cx)

        own_separation = mx(ab(sub(py, yy)), ab(sub(px, xx)))
        own_overlap = clip(affine(own_separation, -1., 5.))
        blocked = mx(anchor_blocked, own_overlap)

        score = sub(
            sub(value, affine(blocked, 4000.)),
            add(affine(dist, .04),
                add(affine(displacement, .001), affine(rank, .0001)))
        )

        history_distance = mx(ab(sub(py, history_y)),
                              ab(sub(px, history_x)))
        proximity = clip(affine(history_distance, -2., 2., 8))
        revisit = sl(proximity, 0, 1)
        for i in range(1, 8):
            revisit = mx(revisit, sl(proximity, i, i+1))
        score = sub(score, mul(affine(revisit, 4.), has_hazard))

        score = add(score, va(selected, .2))
        score = sub(score, va(member, -200000., 200000.))
        destination_value = gm(score)
        select_score = sub(add(zero, destination_value),
                           affine(displacement, 1000.))
        action_map = add(mul(score, selected),
                         mul(select_score, va(selected, -1., 1.)))
        return vc(vs(slots, 4, 4*K), current), mx(best, action_map)

    _, planned = loop(K, [packed, initial_click], plan_anchor)

    previous_y, previous_x = vs("mem", 0, 1), vs("mem", 1, 2)
    repeated = relu(affine(distance(previous_y, previous_x), -1., 1.))
    node("sub", [planned, affine(repeated, .03)], name="click")
    remembered = argmask("click")
    click_memory = gm(mul(remembered, xy))
    node("vconcat",
         [click_memory, vs(body_f, 0, 2), vs("mem", 2, 16)],
         name="mem_next")

    missing_group = node("vrelu", [va(group_n, -1., 1.)])
    logits_w = np.zeros((8, 1), np.float32)
    logits_w[0, 0] = 20.
    logits_b = np.full(8, -10., np.float32)
    logits_b[6] = 1.
    dense(missing_group, logits_w, logits_b, name="logits")

    return {
        "graph": graph,
        "memory_size": 18,
        "params": params
    }
