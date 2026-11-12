import numpy as np


def build():
    graph = []
    params = {}
    serial = [0]

    def fresh(prefix):
        serial[0] += 1
        return "%s_%d" % (prefix, serial[0])

    def parameter(value):
        name = fresh("p")
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(op, inputs, **fields):
        name = fresh("n")
        graph.append({"id": name, "op": op, "in": list(inputs), **fields})
        return name

    def conv(x, weights, bias=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node("conv", [x], **fields)

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vsl(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def unary(op, x):
        return node(op, [x])

    def binary(op, x, y):
        return node(op, [x, y])

    def vlin(xs, weights, bias=0.0):
        inp = xs[0] if len(xs) == 1 else node("vconcat", xs)
        return node(
            "dense",
            [inp],
            w=parameter(np.asarray(weights, np.float32).reshape(1, -1)),
            b=parameter([bias]),
        )

    def vclip(x):
        positive = unary("vrelu", x)
        excess = unary("vrelu", vlin([x], [1.0], -1.0))
        return vlin([positive, excess], [1.0, -1.0])

    def vstep(x, gain=1.0):
        return vclip(vlin([x], [gain]))

    def vnot(x):
        return vlin([x], [-1.0], 1.0)

    def vand(x, y):
        return binary("vmul", x, y)

    def vor(xs):
        if len(xs) == 1:
            return xs[0]
        return vclip(vlin(xs, [1.0] * len(xs)))

    def vabs(x):
        positive = unary("vrelu", x)
        negative = unary("vrelu", vlin([x], [-1.0]))
        return vlin([positive, negative], [1.0, 1.0])

    def veq(x, value):
        difference = vabs(vlin([x], [1.0], -value))
        return vstep(vlin([difference], [-1.0], 0.5), 2.0)

    one = sl("coords", 2, 3)

    def sclip(x):
        return binary("min2", unary("relu", x), one)

    def saffine(x, weight, bias):
        return conv(
            x,
            np.asarray([[[[weight]]]], np.float32),
            np.asarray([bias], np.float32),
        )

    coordinate_weights = np.zeros((3, 3, 1, 1), np.float32)
    coordinate_weights[0, 0, 0, 0] = 63.0
    coordinate_weights[1, 1, 0, 0] = 63.0
    coordinate_weights[2, 0, 0, 0] = 4032.0
    coordinate_weights[2, 1, 0, 0] = 63.0
    coordinate_planes = conv("coords", coordinate_weights)
    yy = sl(coordinate_planes, 0, 1)
    xx = sl(coordinate_planes, 1, 2)
    index = sl(coordinate_planes, 2, 3)

    def canonical(x, channels):
        weights = np.zeros((2 * channels, channels, 3, 3), np.float32)
        for c in range(channels):
            weights[c, c, 0, 1] = 1.0
            weights[channels + c, c, 1, 0] = 1.0
        previous = conv(x, weights)
        above = sl(previous, 0, channels)
        left = sl(previous, channels, 2 * channels)
        return unary("relu", binary("sub", x, binary("max2", above, left)))

    def shift_down_right(x, channels):
        weights = np.zeros((channels, channels, 3, 3), np.float32)
        for c in range(channels):
            weights[c, c, 0, 0] = 1.0
        return conv(x, weights)

    # Detect the display scale.
    ordinary_colors = [0, 1, 2, 5, 6, 7, 9, 10, 13, 15]
    tile_colors = [0, 1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 15]
    scale_weights = np.zeros((len(ordinary_colors), 16, 3, 3), np.float32)
    for i, color in enumerate(ordinary_colors):
        scale_weights[i, color, :, :] = 2.0
    scale_features = sclip(
        conv(
            "x0",
            scale_weights,
            np.full(len(ordinary_colors), -17.0, np.float32),
        )
    )
    scale_max = unary("gmax", scale_features)
    scale2 = vstep(
        vlin([scale_max], [1.0] * len(ordinary_colors), -0.5),
        2.0,
    )
    scale1 = vnot(scale2)

    # Logical tile centers.
    tile_w1 = np.zeros((len(tile_colors), 16, 3, 3), np.float32)
    tile_w2 = np.zeros((len(tile_colors), 16, 3, 3), np.float32)
    for i, color in enumerate(tile_colors):
        tile_w1[i, color, 1:3, 1:3] = 2.0
        tile_w2[i, color, :, :] = 2.0

    tile1_channels = sclip(
        conv("x0", tile_w1, np.full(len(tile_colors), -7.0, np.float32))
    )
    tile2_channels = sclip(
        conv("x0", tile_w2, np.full(len(tile_colors), -17.0, np.float32))
    )
    sum_tiles = np.ones((1, len(tile_colors), 1, 1), np.float32)
    tile1 = sclip(conv(tile1_channels, sum_tiles))
    tile2 = canonical(sclip(conv(tile2_channels, sum_tiles)), 1)
    tile2 = shift_down_right(tile2, 1)
    tiles = binary(
        "add",
        binary("mul", tile1, scale1),
        binary("mul", tile2, scale2),
    )

    # Goal centers, aligned with the marker centers at both scales.
    goal_w1 = np.zeros((2, 16, 3, 3), np.float32)
    goal_w2 = np.zeros((2, 16, 3, 3), np.float32)
    for i, color in enumerate((11, 12)):
        goal_w1[i, color, 1:3, 1:3] = 2.0
        goal_w2[i, color, :, :] = 2.0
    goal1 = sclip(conv("x0", goal_w1, [-7.0, -7.0]))
    goal2 = canonical(sclip(conv("x0", goal_w2, [-17.0, -17.0])), 2)
    goal2 = shift_down_right(goal2, 2)
    goals = binary(
        "add",
        binary("mul", goal1, scale1),
        binary("mul", goal2, scale2),
    )

    # Four-corner marker templates.
    ring_w1 = np.zeros((2, 16, 7, 7), np.float32)
    ring_w2 = np.zeros((2, 16, 7, 7), np.float32)
    for i, color in enumerate((11, 12)):
        for dy in (-1, 2):
            for dx in (-1, 2):
                ring_w1[i, color, dy + 3, dx + 3] = 2.0
        for dy in (-3, 3):
            for dx in (-3, 3):
                ring_w2[i, color, dy + 3, dx + 3] = 2.0
    ring1 = sclip(conv("x0", ring_w1, [-7.0, -7.0]))
    ring2 = canonical(sclip(conv("x0", ring_w2, [-7.0, -7.0])), 2)
    rings = binary(
        "add",
        binary("mul", ring1, scale1),
        binary("mul", ring2, scale2),
    )

    matched_map = binary("min2", goals, rings)
    free_goals = unary("relu", binary("sub", goals, rings))
    open_rings = unary("relu", binary("sub", rings, goals))

    # Distance scores propagate only through detected tile positions.
    sixty_four = vlin([scale2], [0.0], 64.0)
    distance_seed = binary("mul", free_goals, sixty_four)

    shift_weights = np.zeros((18, 2, 3, 3), np.float32)
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]
    for group, (dy, dx) in enumerate(offsets):
        for c in range(2):
            shift_weights[2 * group + c, c, dy + 1, dx + 1] = 1.0
    shift_parameter = parameter(shift_weights)

    decay_weights = np.zeros((2, 2, 1, 1), np.float32)
    decay_weights[0, 0, 0, 0] = 1.0
    decay_weights[1, 1, 0, 0] = 1.0
    decay_parameter = parameter(decay_weights)
    decay_bias = parameter([-1.0, -1.0])

    body = []

    def body_node(op, inputs, **fields):
        name = fresh("body")
        body.append({"id": name, "op": op, "in": list(inputs), **fields})
        return name

    current = "distance_state"
    at_three = None
    for iteration in range(6):
        shifted = body_node("conv", [current], w=shift_parameter)
        pieces = [
            body_node("slice", [shifted], start=2 * k, stop=2 * k + 2)
            for k in range(9)
        ]
        current = pieces[0]
        for piece in pieces[1:]:
            current = body_node("max2", [current, piece])
        if iteration == 2:
            at_three = current

    d1 = body_node("mul", [at_three, scale1])
    d2 = body_node("mul", [current, scale2])
    combined = body_node("add", [d1, d2])
    decayed = body_node(
        "conv", [combined], w=decay_parameter, b=decay_bias
    )
    nonnegative = body_node("relu", [decayed])
    on_tiles = body_node("mul", [nonnegative, tiles])
    updated = body_node("max2", ["distance_state", on_tiles])

    graph.append({
        "id": "distance_loop",
        "op": "loop",
        "T": 40,
        "state": [{"id": "distance_state", "init": distance_seed}],
        "body": body,
        "next": {"distance_state": updated},
    })

    # A matched marker contributes 64; an open marker contributes its
    # distance-transform value from an unmatched goal.
    open_quality = binary("mul", "distance_loop.distance_state", open_rings)
    matched_quality = binary("mul", matched_map, sixty_four)
    quality_map = binary("add", open_quality, matched_quality)
    quality_channels = unary("gmean", quality_map)
    quality0 = vlin([vsl(quality_channels, 0, 1)], [4096.0])
    quality1 = vlin([vsl(quality_channels, 1, 2)], [4096.0])

    matched_channels = unary("gmean", matched_map)
    ring_channels = unary("gmean", rings)
    missing0 = vlin(
        [vsl(ring_channels, 0, 1), vsl(matched_channels, 0, 1)],
        [4096.0, -4096.0],
    )
    missing1 = vlin(
        [vsl(ring_channels, 1, 2), vsl(matched_channels, 1, 2)],
        [4096.0, -4096.0],
    )
    complete0 = vstep(vlin([missing0], [-1.0], 0.5), 2.0)
    complete1 = vstep(vlin([missing1], [-1.0], 0.5), 2.0)

    # Extract one pixel from each button.
    button_weights = np.zeros((2, 16, 3, 3), np.float32)
    button_weights[0, 8, :, :] = 2.0
    button_weights[1, 14, :, :] = 2.0
    button_area = sclip(conv("x0", button_weights, [-15.0, -15.0]))

    neighbor_weights = np.zeros((8, 2, 3, 3), np.float32)
    for group, (dy, dx) in enumerate(
        ((-1, -1), (-1, 0), (-1, 1), (0, -1))
    ):
        for c in range(2):
            neighbor_weights[2 * group + c, c, dy + 1, dx + 1] = 1.0
    neighbors = conv(button_area, neighbor_weights)
    na = sl(neighbors, 0, 2)
    nb = sl(neighbors, 2, 4)
    nc = sl(neighbors, 4, 6)
    nd = sl(neighbors, 6, 8)
    earlier = binary(
        "max2",
        binary("max2", na, nb),
        binary("max2", nc, nd),
    )
    button_seeds = unary("relu", binary("sub", button_area, earlier))
    blue = sl(button_seeds, 0, 1)
    red = sl(button_seeds, 1, 2)

    # Memory: button, previous quality, phase, direction, walk count,
    # target colour, best quality, stale count, escape steps.
    old_index = vsl("mem", 0, 1)
    old_quality = vsl("mem", 1, 2)
    phase = vsl("mem", 2, 3)
    old_direction = vsl("mem", 3, 4)
    old_count = vsl("mem", 4, 5)
    old_target = vsl("mem", 5, 6)
    old_best = vsl("mem", 6, 7)
    old_stale = vsl("mem", 7, 8)
    old_escape = vsl("mem", 8, 9)

    first = vnot(vsl("aprev", 6, 7))
    probe = veq(phase, 1.0)
    walk = veq(phase, 2.0)
    undo_done = veq(phase, 3.0)

    target = vor([
        vand(vnot(old_target), complete0),
        vand(old_target, vnot(complete1)),
    ])
    target_changed = vand(
        vnot(first),
        vabs(vlin([target, old_target], [1.0, -1.0])),
    )
    quality = vlin(
        [vand(quality0, vnot(target)), vand(quality1, target)],
        [1.0, 1.0],
    )

    delta = vlin([quality, old_quality], [1.0, -1.0])
    worse = vstep(vlin([delta], [-1.0], -0.25), 4.0)
    changed = vstep(vlin([vabs(delta)], [1.0], -0.25), 4.0)
    flat = vnot(changed)
    timeout = vstep(vlin([old_count], [1.0], -14.5), 2.0)
    working = vor([probe, walk])

    reset_progress = vor([first, target_changed])
    improved = vand(
        vnot(reset_progress),
        vstep(vlin([quality, old_best], [1.0, -1.0], -0.25), 4.0),
    )
    active_escape = vstep(vlin([old_escape], [1.0], -0.5), 2.0)
    stale_trigger = vstep(vlin([old_stale], [1.0], -6.5), 2.0)
    trigger = vand(
        vand(vand(stale_trigger, vnot(active_escape)), vnot(improved)),
        vnot(reset_progress),
    )
    escape = vand(
        vand(vor([trigger, active_escape]), vnot(improved)),
        vnot(reset_progress),
    )

    regular_advance = vor([
        first,
        target_changed,
        undo_done,
        vand(working, flat),
        vand(walk, timeout),
    ])
    advance = vor([
        trigger,
        vand(regular_advance, vnot(escape)),
    ])

    reverse_probe = vand(
        vand(vand(probe, worse), vnot(advance)),
        vnot(escape),
    )
    reverse_walk = vand(
        vand(vand(walk, worse), vnot(advance)),
        vnot(escape),
    )
    reverse = vor([reverse_probe, reverse_walk])
    direction_xor = vlin(
        [old_direction, reverse, vand(old_direction, reverse)],
        [1.0, 1.0, -2.0],
    )
    direction = vand(vnot(advance), direction_xor)

    # Cycle blue buttons in row-major order.
    after_old = sclip(saffine(binary("sub", index, old_index), 1.0, -0.5))
    eligible = binary("mul", blue, after_old)
    has_eligible = unary("gmax", eligible)
    candidates = binary(
        "add",
        eligible,
        binary("mul", blue, vnot(has_eligible)),
    )
    rank = saffine(index, -1.0, 4096.0)
    candidate_score = binary("mul", candidates, rank)
    best_score = unary("gmax", candidate_score)
    winner_delta = binary("sub", candidate_score, best_score)
    next_blue = binary(
        "mul",
        candidates,
        sclip(saffine(winner_delta, 4.0, 1.0)),
    )

    index_difference = unary("abs", binary("sub", index, old_index))
    current_blue = binary(
        "mul",
        blue,
        sclip(saffine(index_difference, -2.0, 1.0)),
    )
    selected_blue = binary(
        "add",
        binary("mul", next_blue, advance),
        binary("mul", current_blue, vnot(advance)),
    )
    selected_index = unary("gmax", binary("mul", selected_blue, index))
    selected_y = unary("gmax", binary("mul", selected_blue, yy))
    selected_x = unary("gmax", binary("mul", selected_blue, xx))

    # Pair each selected blue button with its nearest red button.
    button_dy = unary("abs", binary("sub", yy, selected_y))
    button_dx = unary("abs", binary("sub", xx, selected_x))
    button_distance = binary("add", button_dy, button_dx)
    red_features = node("concat", [button_distance, index])
    red_weights = np.zeros((1, 2, 1, 1), np.float32)
    red_weights[0, 0, 0, 0] = -4.0
    red_weights[0, 1, 0, 0] = -1.0 / 4096.0
    red_rank = conv(red_features, red_weights, [4096.0])
    red_score = binary("mul", red, red_rank)
    blue_score = saffine(selected_blue, 5000.0, 0.0)

    graph.append({
        "id": "click",
        "op": "add",
        "in": [
            binary("mul", blue_score, vnot(direction)),
            binary("mul", red_score, direction),
        ],
    })

    normal = vand(vnot(advance), vnot(reverse))
    next_phase = vlin(
        [advance, reverse_walk, reverse_probe, normal],
        [1.0, 3.0, 2.0, 2.0],
    )
    continued_count = vand(normal, vlin([old_count], [1.0], 1.0))
    next_count = vlin(
        [advance, reverse_probe, continued_count],
        [1.0, 1.0, 1.0],
    )

    best_increment = unary(
        "vrelu", vlin([quality, old_best], [1.0, -1.0])
    )
    continued_best = vlin([old_best, best_increment], [1.0, 1.0])
    next_best = vlin(
        [
            vand(reset_progress, quality),
            vand(vnot(reset_progress), continued_best),
        ],
        [1.0, 1.0],
    )

    stale_reset = vor([reset_progress, improved, escape])
    next_stale = vand(
        vnot(stale_reset),
        vlin([old_stale], [1.0], 1.0),
    )
    decremented_escape = unary("vrelu", vlin([old_escape], [1.0], -1.0))
    next_escape = vlin(
        [
            vand(trigger, vlin([trigger], [0.0], 3.0)),
            vand(vand(active_escape, escape), decremented_escape),
        ],
        [1.0, 1.0],
    )

    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": [
            selected_index,
            quality,
            next_phase,
            direction,
            next_count,
            target,
            next_best,
            next_stale,
            next_escape,
        ],
    })

    logits_weights = np.zeros((8, 8), np.float32)
    logits_bias = np.zeros(8, np.float32)
    logits_bias[6] = 1.0
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": ["aprev"],
        "w": parameter(logits_weights),
        "b": parameter(logits_bias),
    })

    return {"graph": graph, "memory_size": 9, "params": params}
