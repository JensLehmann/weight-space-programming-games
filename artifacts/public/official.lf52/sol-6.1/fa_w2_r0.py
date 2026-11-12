import numpy as np


def build():
    P = {}
    graph = []
    stream = graph
    shapes = {
        "x0": 16, "coords": 3,
        "mem": 9, "aprev": 8, "avail": 8,
    }
    spatial = {"x0", "coords"}
    serial = [0]
    parameter_cache = {}
    slice_cache = {}
    shift_cache = {}

    def uid(prefix="n"):
        serial[0] += 1
        return "%s_%d" % (prefix, serial[0])

    def param(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p_%d" % len(P)
            parameter_cache[key] = name
            P[name] = a
        return parameter_cache[key]

    def node(op, inputs, channels, is_spatial=True, name=None, **fields):
        name = name or uid()
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        stream.append(nd)
        shapes[name] = channels
        if is_spatial:
            spatial.add(name)
        return name

    def conv(x, w, b=None, name=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("conv", [x], np.asarray(w).shape[0],
                    name=name, **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", [x], np.asarray(w).shape[0],
                    False, name, **fields)

    def aff(x, a=1.0, b=0.0):
        c = shapes[x]
        w = np.eye(c, dtype=np.float32)[:, :, None, None] * a
        return conv(x, w, np.full(c, b, np.float32))

    def va(x, a=1.0, b=0.0):
        c = shapes[x]
        return dense(x, np.eye(c, dtype=np.float32) * a,
                     np.full(c, b, np.float32))

    def binary(op, x, y):
        return node(op, [x, y], max(shapes[x], shapes[y]))

    def add(x, y):
        return binary("add", x, y)

    def sub(x, y):
        return binary("sub", x, y)

    def mul(x, y):
        return binary("mul", x, y)

    def maximum(x, y):
        return binary("max2", x, y)

    def vmax(x, y):
        # Vector maximum, expressed using ReLU.
        d = node("vadd", [x, va(y, -1)], max(shapes[x], shapes[y]), False)
        r = node("vrelu", [d], shapes[d], False)
        return node("vadd", [r, y], shapes[r], False)

    def vadd(x, y):
        return node("vadd", [x, y], max(shapes[x], shapes[y]), False)

    def vmul(x, y):
        return node("vmul", [x, y], max(shapes[x], shapes[y]), False)

    def cat(xs, vector=False):
        return node("vconcat" if vector else "concat", xs,
                    sum(shapes[x] for x in xs), not vector)

    def sl(x, start, stop):
        key = (x, start, stop)
        if key not in slice_cache:
            is_s = x in spatial
            slice_cache[key] = node(
                "slice" if is_s else "vslice", [x], stop - start,
                is_s, start=start, stop=stop
            )
        return slice_cache[key]

    one = sl("coords", 2, 3)
    xy = sl("coords", 0, 2)

    def clip(x):
        r = node("relu", [x], shapes[x])
        return binary("min2", r, one)

    def vclip(x):
        r = node("vrelu", [x], shapes[x], False)
        excess = node("vrelu", [va(r, 1, -1)], shapes[r], False)
        return vadd(r, va(excess, -1))

    def maxlist(xs):
        z = xs[0]
        for x in xs[1:]:
            z = maximum(z, x)
        return z

    def pool(x, mean=False):
        return node("gmean" if mean else "gmax", [x], shapes[x], False)

    def scalar(x, weights, bias=0.0):
        return dense(x, np.asarray(weights, np.float32)[None, :],
                     np.array([bias], np.float32))

    def sum_channels(x):
        return conv(x, np.ones((1, shapes[x], 1, 1), np.float32))

    def merge_two(x):
        return maximum(sl(x, 0, 1), sl(x, 1, 2))

    def shift(x, dr, dc, cache=True):
        # Output[p] = input[p + (dr, dc)].
        key = (x, dr, dc)
        if cache and key in shift_cache:
            return shift_cache[key]
        original = x
        rr, cc = dr, dc
        while rr or cc:
            ar = max(-3, min(3, rr))
            ac = max(-3, min(3, cc))
            c = shapes[x]
            w = np.zeros((c, c, 7, 7), np.float32)
            for j in range(c):
                w[j, j, 3 + ar, 3 + ac] = 1
            x = conv(x, w)
            rr -= ar
            cc -= ac
            if cache:
                shift_cache[(original, dr - rr, dc - cc)] = x
        if cache:
            shift_cache[key] = x
        return x

    def winner(x):
        # Scores receive a coordinate tie-break before using this operation.
        m = pool(x)
        return clip(aff(sub(x, m), 1000000.0, 1.0))

    def detector(templates):
        w = np.zeros((len(templates), 16, 7, 7), np.float32)
        b = np.zeros(len(templates), np.float32)
        for k, (entries, threshold) in enumerate(templates):
            for color, row, col, weight in entries:
                w[k, color, row, col] += 2.0 * weight
            b[k] = -2.0 * threshold
        return clip(conv("x0", w, b))

    # All spatial symbols are anchored at tile pixel (3,3).
    circle = [
        (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 1), (3, 2), (3, 3), (3, 4),
        (4, 2), (4, 3),
    ]
    corners = [(1, 1), (1, 4), (4, 1), (4, 4)]
    templates = []
    for color in (14, 8, 9, 2, 4):
        entries = [(color, r, c, 1) for r, c in circle]
        entries += [(color, r, c, -3) for r, c in corners]
        templates.append((entries, 11.5))
    blobs = detector(templates)
    green = maxlist([sl(blobs, 0, 1), sl(blobs, 3, 4),
                     sl(blobs, 4, 5)])
    red = sl(blobs, 1, 2)
    blue = sl(blobs, 2, 3)

    white_entries = [(1, r, c, 1) for r, c in corners]
    white_entries += [(0, r, c, 1)
                      for r, c in ((0, 0), (0, 5), (5, 0), (5, 5))]
    yellow_entries = [(11, r, c, 1)
                      for r, c in ((0, 0), (0, 5), (5, 0), (5, 5))]
    pylon_entries = [
        (0, 1, 2, 1), (7, 1, 3, 1),
        (7, 2, 2, 1), (7, 2, 3, 1),
        (15, 0, 1, 1), (15, 0, 4, 1),
        (15, 3, 1, 1), (15, 3, 4, 1),
        (5, 4, 2, 1), (5, 4, 3, 1),
    ]
    ring_positions = []
    ring_strings = [" xxxx ", "xxooxx", "xoooox",
                    "xoooox", "xxooxx", " xxxx "]
    for r, row in enumerate(ring_strings):
        for c, ch in enumerate(row):
            if ch == "x":
                ring_positions.append((3, r, c, 1))

    tiles = detector([
        (white_entries, 7.5),
        (yellow_entries, 3.5),
        (pylon_entries, 9.5),
        (ring_positions, len(ring_positions) - 0.5),
    ])
    white = sl(tiles, 0, 1)
    yellow = sl(tiles, 1, 2)
    pylon = sl(tiles, 2, 3)
    selected = sl(tiles, 3, 4)
    normal = maximum(green, red)
    occupied = maxlist([normal, blue, pylon])
    floor = maxlist([white, yellow, occupied])

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]

    # Complete the sparse six-pixel lattice.
    lattice_state = uid("lattice_state")
    shapes[lattice_state] = 1
    spatial.add(lattice_state)
    body = []
    old_stream = stream
    stream = body
    lattice_neighbors = [
        shift(lattice_state, 6 * dr, 6 * dc, False)
        for dr, dc in directions
    ]
    lattice_next = maxlist([lattice_state] + lattice_neighbors)
    stream = old_stream
    lid = uid("lattice")
    graph.append({
        "id": lid, "op": "loop", "T": 12,
        "state": [{"id": lattice_state, "init": floor}],
        "body": body, "next": {lattice_state: lattice_next},
    })
    lattice = lid + "." + lattice_state
    shapes[lattice] = 1
    spatial.add(lattice)

    port_positions = [
        [(0, 2), (0, 3)],
        [(5, 2), (5, 3)],
        [(2, 0), (3, 0)],
        [(2, 5), (3, 5)],
    ]
    rail_templates = [
        ([(5, r, c, 1) for r, c in positions], 1.5)
        for positions in port_positions
    ]
    rail_templates.append(
        ([(5, r, c, 1) for r, c in
          ((2, 2), (2, 3), (3, 2), (3, 3))], 3.5)
    )
    rail_features = detector(rail_templates)
    rail = maximum(yellow, mul(lattice, sl(rail_features, 4, 5)))
    ports = [maximum(yellow, sl(rail_features, i, i + 1))
             for i in range(4)]
    edges = []
    for i, (dr, dc) in enumerate(directions):
        peer = shift(rail, 6 * dr, 6 * dc)
        peer_port = shift(ports[opposite[i]], 6 * dr, 6 * dc)
        edges.append(mul(mul(rail, peer), mul(ports[i], peer_port)))

    empty_floor = clip(sub(floor, occupied))
    domain = clip(sub(maximum(floor, rail), occupied))
    empty_rail = mul(rail, aff(occupied, -1, 1))

    # Exploration value for rails continuing beyond the viewport.
    right_edge = clip(conv(
        "coords",
        np.array([[[[0]], [[20]], [[0]]]], np.float32),
        np.array([-17.4], np.float32),
    ))
    bottom_edge = clip(conv(
        "coords",
        np.array([[[[20]], [[0]], [[0]]]], np.float32),
        np.array([-17.4], np.float32),
    ))
    frontier_right = mul(mul(rail, ports[3]), right_edge)
    frontier_down = mul(mul(rail, ports[1]), bottom_edge)
    frontier = maximum(aff(frontier_right, .35),
                       aff(frontier_down, .25))

    # Memory: valid, row, col, on-carrier, red, banned-row,
    # banned-col, blacklist lifetime, second-undo flag.
    mem_valid = sl("mem", 0, 1)
    mem_xy = sl("mem", 1, 3)
    mem_carrier = sl("mem", 3, 4)
    mem_red = sl("mem", 4, 5)
    banned_xy = sl("mem", 5, 7)
    ban_life = sl("mem", 7, 8)
    undo_phase = sl("mem", 8, 9)

    ban_distance = sum_channels(node(
        "abs", [sub(xy, banned_xy)], 2
    ))
    ban_active = vclip(ban_life)
    ban = mul(clip(aff(ban_distance, -200, 1)), ban_active)

    # Red singletons are useful bridges, not preferred moving pieces.
    red_count = va(pool(red, True), 4096.0)
    red_movable = vclip(va(red_count, 1, -1))
    moving = cat([green, mul(red, red_movable)])
    pieces = cat([green, red])
    swap_w = np.array([[0, 1], [1, 0]], np.float32)[:, :, None, None]
    other_color = conv(pieces, swap_w)
    fixed = maximum(blue, pylon)
    noncapture = clip(add(other_color, fixed))

    mids = []
    noncapture_mids = []
    landing_domains = []
    for dr, dc in directions:
        mids.append(shift(pieces, 6 * dr, 6 * dc))
        noncapture_mids.append(shift(noncapture, 6 * dr, 6 * dc))
        landing_domains.append(shift(domain, 12 * dr, 12 * dc))

    def value_loop(seed, allowed, hop_masks, T=24):
        nonlocal stream
        state = uid("value_state")
        shapes[state] = shapes[seed]
        spatial.add(state)
        body = []
        previous_stream = stream
        stream = body
        options = []
        for i, (dr, dc) in enumerate(directions):
            step = shift(state, 6 * dr, 6 * dc, False)
            jump = shift(state, 12 * dr, 12 * dc, False)
            options.append(mul(step, edges[i]))
            options.append(mul(jump, hop_masks[i]))
        propagated = aff(maxlist(options), .95)
        nxt = mul(maximum(seed, propagated), allowed)
        stream = previous_stream
        loop_id = uid("values")
        graph.append({
            "id": loop_id, "op": "loop", "T": T,
            "state": [{"id": state, "init": seed}],
            "body": body, "next": {state: nxt},
        })
        result = loop_id + "." + state
        shapes[result] = shapes[seed]
        spatial.add(result)
        return result

    generic_seeds = [
        mul(mids[i], landing_domains[i]) for i in range(4)
    ]
    generic_seed = mul(maximum(maxlist(generic_seeds), frontier), domain)
    generic_value = value_loop(
        generic_seed, domain, noncapture_mids
    )

    future = []
    capture_rewards = []
    root_moves = []
    for i, (dr, dc) in enumerate(directions):
        f = shift(generic_value, 12 * dr, 12 * dc)
        future.append(f)
        forbidden = shift(ban, 12 * dr, 12 * dc)
        reward = sub(aff(f, 2, 5), aff(forbidden, 9))
        capture_rewards.append(reward)
        captures = mul(mids[i], reward)
        transit = mul(noncapture_mids[i], aff(f, 3))
        score = mul(mul(add(captures, transit),
                        landing_domains[i]), moving)
        root_moves.append(merge_two(score))

    rail_root_options = []
    for i, (dr, dc) in enumerate(directions):
        neighbor = shift(generic_value, 6 * dr, 6 * dc)
        rail_root_options.append(mul(neighbor, edges[i]))
    rail_root = merge_two(mul(
        mul(maxlist(rail_root_options), moving), yellow
    ))
    root_interest = maximum(maxlist(root_moves), aff(rail_root, 2))

    # Coordinate tie-break: stable across selection overlays.
    tie = conv(
        "coords",
        np.array([[[[-.00003]], [[-.002]], [[0]]]], np.float32),
        np.array([.00204], np.float32),
    )
    movable_any = merge_two(moving)
    root_scores = mul(add(aff(root_interest, 1, .1), tie), movable_any)
    root = winner(root_scores)

    # Match the remembered active piece in the new rendering. Carrier
    # membership disambiguates scrolling during arrow actions.
    distance = sum_channels(node("abs", [sub(xy, mem_xy)], 2))
    previous_arrow = scalar("aprev", [0, 1, 1, 1, 1, 0, 0, 0])
    carrier_preference = vmul(previous_arrow, mem_carrier)
    color_preference = add(
        mul(red, mem_red),
        mul(green, va(mem_red, -1, 1)),
    )
    nearest_scores = add(
        aff(distance, -10, 30),
        add(mul(aff(yellow, 20), carrier_preference),
            aff(color_preference, 4)),
    )
    nearest_scores = mul(add(nearest_scores, tie), movable_any)
    remembered = winner(nearest_scores)
    initial_focus = add(
        mul(remembered, mem_valid),
        mul(root, va(mem_valid, -1, 1)),
    )

    # A second value field excludes the active piece from capture targets.
    focus_colors = mul(pieces, initial_focus)
    remaining = clip(sub(pieces, focus_colors))
    remaining_other = conv(remaining, swap_w)
    active_noncapture = clip(add(remaining_other, fixed))
    active_domain = maximum(domain, initial_focus)
    active_mids = []
    active_hops = []
    active_seeds = []
    for i, (dr, dc) in enumerate(directions):
        mid = shift(remaining, 6 * dr, 6 * dc)
        hop = shift(active_noncapture, 6 * dr, 6 * dc)
        active_mids.append(mid)
        active_hops.append(hop)
        seed = mul(mul(mid, landing_domains[i]), aff(future[i], .4, 1))
        active_seeds.append(seed)
    active_seed = mul(
        maximum(maxlist(active_seeds), frontier), active_domain
    )
    active_value = value_loop(active_seed, active_domain, active_hops)

    useful = pool(sum_channels(mul(active_value, focus_colors)))
    keep_focus = vclip(va(useful, 10000, -.01))
    focus = add(
        mul(initial_focus, keep_focus),
        mul(root, va(keep_focus, -1, 1)),
    )
    focus_color = mul(pieces, focus)
    selected_focus = pool(mul(selected, focus))

    # Actual jumps, and requests for an empty carrier at a virtual landing.
    jump_sources = []
    virtual_sources = []
    active_neighbors = []
    for i, (dr, dc) in enumerate(directions):
        destination_value = shift(active_value, 12 * dr, 12 * dc)
        source_value = active_value
        progress = clip(aff(sub(destination_value, source_value),
                            1000, -.01))
        transit = mul(
            mul(noncapture_mids[i], aff(destination_value, 3)),
            progress,
        )
        capture = mul(mids[i], capture_rewards[i])
        base = add(capture, transit)
        source_score = mul(base, focus_color)
        real_landing = shift(empty_floor, 12 * dr, 12 * dc)
        jump_score = merge_two(mul(source_score, real_landing))
        # Fixed directional tie-break, smaller than strategic rewards.
        jump_score = mul(aff(jump_score, 1, .0001 * (4 - i)),
                         mul(focus, real_landing))
        jump_sources.append(jump_score)

        virtual_landing = shift(empty_rail, 12 * dr, 12 * dc)
        missing_carrier = aff(
            shift(yellow, 12 * dr, 12 * dc), -1, 1
        )
        virtual_score = merge_two(mul(
            mul(source_score, virtual_landing), missing_carrier
        ))
        virtual_sources.append(virtual_score)
        active_neighbors.append(shift(active_value, 6 * dr, 6 * dc))

    jump_best_map = maxlist(jump_sources)
    click_score = pool(jump_best_map)

    destinations = [
        shift(jump_sources[i], -12 * dr, -12 * dc)
        for i, (dr, dc) in enumerate(directions)
    ]
    destination_map = maxlist(destinations)
    click_map = add(
        mul(destination_map, selected_focus),
        mul(aff(focus, 10), va(selected_focus, -1, 1)),
    )
    # Tie-break only on supported targets.
    click_support = clip(aff(click_map, 10000))
    click_map = add(click_map, mul(tie, click_support))

    load_seeds = [
        shift(virtual_sources[i], -12 * dr, -12 * dc)
        for i, (dr, dc) in enumerate(directions)
    ]
    load_seed = mul(maxlist(load_seeds), empty_rail)

    # Scalar rail-only value propagation for empty-carrier positioning.
    load_state = uid("load_state")
    shapes[load_state] = 1
    spatial.add(load_state)
    body = []
    previous_stream = stream
    stream = body
    load_options = []
    for i, (dr, dc) in enumerate(directions):
        load_options.append(mul(
            shift(load_state, 6 * dr, 6 * dc, False), edges[i]
        ))
    load_next = mul(
        maximum(load_seed, aff(maxlist(load_options), .94)),
        empty_rail,
    )
    stream = previous_stream
    load_id = uid("load")
    graph.append({
        "id": load_id, "op": "loop", "T": 24,
        "state": [{"id": load_state, "init": load_seed}],
        "body": body, "next": {load_state: load_next},
    })
    load_value = load_id + "." + load_state
    shapes[load_value] = 1
    spatial.add(load_value)

    focus_on_carrier = mul(focus, yellow)
    empty_carriers = mul(yellow, aff(occupied, -1, 1))
    arrow_scores = []
    for i, (dr, dc) in enumerate(directions):
        target_yellow = shift(yellow, 6 * dr, 6 * dc)
        can_move = mul(edges[i], aff(target_yellow, -1, 1))

        carry_delta = sum_channels(mul(
            sub(active_neighbors[i], active_value), focus_color
        ))
        carry_delta = mul(mul(carry_delta, yellow), can_move)
        carry_score = va(pool(carry_delta, True), 4096 * 20)

        load_neighbor = shift(load_value, 6 * dr, 6 * dc)
        load_delta = mul(
            mul(sub(load_neighbor, load_value), empty_carriers),
            can_move,
        )
        load_score = va(pool(load_delta, True), 4096 * 2)

        score = va(vadd(carry_score, load_score), 1, -.03)
        if i == 3:
            border = pool(mul(focus_on_carrier, frontier_right))
            score = vadd(score, va(border, .7))
        elif i == 1:
            border = pool(mul(focus_on_carrier, frontier_down))
            score = vadd(score, va(border, .5))
        arrow_scores.append(score)

    best_arrow = arrow_scores[0]
    for score in arrow_scores[1:]:
        best_arrow = vmax(best_arrow, score)
    best_move = vmax(click_score, best_arrow)

    # Two undo actions remove a selection and its preceding unsuccessful move.
    stuck = vclip(va(best_move, -10000, 1))
    new_undo = vmul(stuck, va(undo_phase, -1, 1))
    undo = vclip(vadd(new_undo, undo_phase))
    undo = vmul(undo, sl("avail", 7, 8))

    click_wins = vclip(va(
        vadd(click_score, va(best_arrow, -1)), 10000, -.01
    ))
    click_wins = vmul(click_wins, va(undo, -1, 1))

    # Action order is RESET, up, down, left, right, unused, click, undo.
    controls = cat(arrow_scores + [click_score, undo], vector=True)
    wo = np.zeros((8, 6), np.float32)
    bo = np.full(8, -100, np.float32)
    for i in range(4):
        wo[i + 1, i] = 1
        bo[i + 1] = 0
    wo[6, 4] = 1
    bo[6] = 0
    wo[7, 5] = 200
    bo[7] = -100
    dense(controls, wo, bo, name="logits")
    conv(click_map, np.ones((1, 1, 1, 1), np.float32), name="click")

    # Record the expected active-piece position. During carrier movement,
    # matching uses carrier membership rather than assuming a camera offset.
    clicked_position = winner(click_map)
    next_position = add(
        mul(clicked_position, click_wins),
        mul(focus, va(click_wins, -1, 1)),
    )
    next_xy = va(pool(mul(xy, next_position), True), 4096)
    next_carrier = pool(mul(next_position, yellow))
    next_red = pool(mul(next_position, red))
    next_valid = va(undo, -1, 1)

    current_xy = va(pool(mul(xy, focus), True), 4096)
    next_ban_xy = vadd(
        vmul(current_xy, new_undo),
        vmul(banned_xy, va(new_undo, -1, 1)),
    )
    aged = node("vrelu", [va(ban_life, 1, -1)], 1, False)
    next_life = vadd(
        va(new_undo, 30),
        vmul(aged, va(new_undo, -1, 1)),
    )
    memory = cat([
        next_valid, next_xy, next_carrier, next_red,
        next_ban_xy, next_life, new_undo,
    ], vector=True)
    dense(memory, np.eye(9, dtype=np.float32), name="mem_next")

    return {
        "graph": graph,
        "memory_size": 9,
        "params": P,
    }
