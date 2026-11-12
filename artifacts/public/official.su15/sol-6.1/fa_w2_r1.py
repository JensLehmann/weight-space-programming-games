import math
import numpy as np


def build():
    graph = []
    params = {}
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "aprev": (8,),
        "avail": (8,),
        "mem": (28,),
    }
    cache = {}
    serial = [0]

    def fresh():
        serial[0] += 1
        return "n%d" % serial[0]

    def parameter(value):
        a = np.asarray(value, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p%d" % len(params)
            cache[key] = name
            params[name] = a
        return cache[key]

    def node(op, inputs, dst=None, output=None, **fields):
        name = output or fresh()
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        (graph if dst is None else dst).append(nd)
        s = shapes[inputs[0]]
        if op == "conv":
            shapes[name] = (params[fields["w"]].shape[0], 64, 64)
        elif op == "dense":
            shapes[name] = (params[fields["w"]].shape[0],)
        elif op in ("gmax", "gmean"):
            shapes[name] = (s[0],)
        elif op in ("concat", "vconcat"):
            c = sum(shapes[z][0] for z in inputs)
            shapes[name] = (c, 64, 64) if len(s) == 3 else (c,)
        elif op in ("slice", "vslice"):
            c = fields["stop"] - fields["start"]
            shapes[name] = (c, 64, 64) if len(s) == 3 else (c,)
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shapes[name] = (
                max(shapes[z][0] for z in inputs), 64, 64
            )
        elif op in ("vadd", "vmul"):
            shapes[name] = (max(shapes[z][0] for z in inputs),)
        else:
            shapes[name] = s
        return name

    def conv(x, w, b=None, dst=None, output=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], dst=dst, output=output, **fields)

    def linear(x, w, b=None, output=None):
        w = np.asarray(w, np.float32)
        if len(shapes[x]) == 3:
            return conv(x, w[:, :, None, None], b, output=output)
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], output=output, **fields)

    def scale(x, a, b=0.0):
        c = shapes[x][0]
        return linear(
            x, np.eye(c, dtype=np.float32) * a,
            np.full(c, b, np.float32) if b != 0 else None,
        )

    def cut(x, start, stop):
        op = "slice" if len(shapes[x]) == 3 else "vslice"
        return node(op, [x], start=start, stop=stop)

    def cat(xs):
        op = "concat" if len(shapes[xs[0]]) == 3 else "vconcat"
        return node(op, xs)

    def relu(x, dst=None):
        op = "relu" if len(shapes[x]) == 3 else "vrelu"
        return node(op, [x], dst=dst)

    def plus(a, b):
        op = "vadd" if len(shapes[a]) == len(shapes[b]) == 1 else "add"
        return node(op, [a, b])

    def minus(a, b):
        if len(shapes[a]) == len(shapes[b]) == 1:
            return plus(a, scale(b, -1))
        return node("sub", [a, b])

    def times(a, b):
        op = "vmul" if len(shapes[a]) == len(shapes[b]) == 1 else "mul"
        return node(op, [a, b])

    one_s = cut("coords", 2, 3)
    one_v = node("gmax", [one_s])

    def clipped(x):
        z = relu(x)
        if len(shapes[x]) == 3:
            return node("min2", [z, one_s])
        return minus(z, relu(scale(z, 1, -1)))

    def sum_channels(x):
        return linear(x, np.ones((1, shapes[x][0]), np.float32))

    def count(x):
        return scale(node("gmean", [x]), 4096)

    def maximum(xs, dst=None):
        result = xs[0]
        for x in xs[1:]:
            result = node("max2", [result, x], dst=dst)
        return result

    def constant(values):
        return linear(
            one_v, np.asarray(values, np.float32).reshape(-1, 1)
        )

    def one_minus(x):
        return scale(x, -1, 1)

    # Separate the play area from the goal display.
    row = linear("coords", [[63, 0, 0]])
    col = linear("coords", [[0, 63, 0]])
    lower_board = clipped(scale(row, 1, -9))
    upper_board = clipped(scale(row, -1, 63))
    board = times(lower_board, upper_board)
    header = times(
        one_minus(lower_board), clipped(scale(col, 1, -23))
    )

    colors = [cut("x0", i, i + 1) for i in range(16)]
    fruit_colors = [10, 6, 15, 11, 12, 8, 9, 7, 14]
    fruit_sizes = [1, 2, 3, 4, 5, 7, 8, 9, 10]

    def square_erosion(x, n):
        k = n if n % 2 else n + 1
        w = np.zeros((1, 1, k, k), np.float32)
        middle = k // 2
        half = n // 2
        w[0, 0,
          middle - half:middle - half + n,
          middle - half:middle - half + n] = 1
        return relu(conv(x, w, [1 - n * n]))

    # Detect solid squares and emit one marker at each center.
    fruit_markers = []
    for color, n in zip(fruit_colors, fruit_sizes):
        if n <= 7:
            m = square_erosion(colors[color], n)
        else:
            m = square_erosion(colors[color], 7)
            m = square_erosion(m, n - 6)
        if n > 1:
            w = np.zeros((1, 1, 3, 3), np.float32)
            w[0, 0, 1, 1] = 1
            w[0, 0, 0, 1] = -1
            w[0, 0, 1, 0] = -1
            m = relu(conv(m, w))
        fruit_markers.append(m)

    creature_patterns = [
        (
            7,
            [[0, 0, 1, 0, 0],
             [0, 1, 0, 1, 0],
             [1, 0, 0, 0, 1],
             [0, 1, 1, 1, 0]],
        ),
        (
            14,
            [[0, 0, 1, 0, 0],
             [0, 1, 0, 1, 0],
             [1, 0, 1, 0, 1],
             [0, 1, 0, 1, 0]],
        ),
        (
            13,
            [[1, 0, 1, 0, 1],
             [0, 1, 0, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0]],
        ),
    ]

    # Permit fruit-colored occlusion while requiring visible evidence.
    creature_markers = []
    for color, pattern in creature_patterns:
        p = np.asarray(pattern, np.float32)
        occluder_w = np.zeros((1, 16), np.float32)
        for c in fruit_colors:
            if c != color:
                occluder_w[0, c] = 1
        occluders = linear("x0", occluder_w)
        detector_input = cat([colors[color], occluders])
        w = np.zeros((1, 2, 5, 5), np.float32)
        w[0, 0, :4, :] = 2 * p - 1
        w[0, 1, :4, :] = p
        matching = relu(conv(
            detector_input, w, [1 - float(p.sum())]
        ))
        visible_w = np.zeros((1, 1, 5, 5), np.float32)
        visible_w[0, 0, :4, :] = p
        sufficiently_visible = clipped(
            conv(colors[color], visible_w, [-3])
        )
        creature_markers.append(
            times(matching, sufficiently_visible)
        )

    markers = cat(fruit_markers + creature_markers)
    objects = times(markers, board)
    goal_objects = times(markers, header)
    fruit_objects = cut(objects, 0, 9)
    creature_objects = cut(objects, 9, 12)
    all_creatures = sum_channels(creature_objects)

    # Detect basket centers from independently visible edge tips.
    basket_color = plus(colors[9], colors[2])
    tip_w = np.zeros((4, 1, 7, 7), np.float32)
    for d in range(-2, 3):
        tip_w[0, 0, 3, 3 + d] = 1
        tip_w[1, 0, 3, 3 + d] = 1
        tip_w[2, 0, 3 + d, 3] = 1
        tip_w[3, 0, 3 + d, 3] = 1
    for j in (0, 1):
        tip_w[j, 0, 3, 0] = -1
        tip_w[j, 0, 3, 6] = -1
    for j in (2, 3):
        tip_w[j, 0, 0, 3] = -1
        tip_w[j, 0, 6, 3] = -1
    tip_w[0, 0, 2, 3] = -1
    tip_w[0, 0, 4, 3] = 1
    tip_w[1, 0, 4, 3] = -1
    tip_w[1, 0, 2, 3] = 1
    tip_w[2, 0, 3, 2] = -1
    tip_w[2, 0, 3, 4] = 1
    tip_w[3, 0, 3, 4] = -1
    tip_w[3, 0, 3, 2] = 1
    tips = relu(conv(basket_color, tip_w, [-5] * 4))

    shift_tips = np.zeros((4, 4, 5, 5), np.float32)
    for j, (dy, dx) in enumerate(
        [(-2, 0), (2, 0), (0, -2), (0, 2)]
    ):
        shift_tips[j, j, 2 + dy, 2 + dx] = 1
    tips = conv(conv(tips, shift_tips), shift_tips)
    baskets = times(clipped(sum_channels(tips)), board)

    basket_box = conv(
        conv(baskets, np.ones((1, 1, 7, 7), np.float32)),
        np.ones((1, 1, 3, 3), np.float32),
    )
    basket_box = clipped(basket_box)
    inside = times(objects, basket_box)
    total_counts = count(objects)
    goal_counts = count(goal_objects)
    inside_counts = count(inside)

    # Compact tiled correlations cover offsets -12 through 12.
    shift_stages = [
        ([0], [-3, 3]),
        ([-3, 3], [-6, -3, 3, 6]),
        ([-6, -3, 3, 6], [-9, -3, 3, 9]),
    ]

    def axis_bank(x, axis):
        for previous, desired in shift_stages:
            w = np.zeros(
                (len(desired), len(previous), 7, 7), np.float32
            )
            for out, position in enumerate(desired):
                source = min(
                    range(len(previous)),
                    key=lambda z: abs(previous[z] - position),
                )
                delta = position - previous[source]
                if axis == 0:
                    w[out, source, 3 + delta, 3] = 1
                else:
                    w[out, source, 3, 3 + delta] = 1
            x = conv(x, w)
        return x

    def tiles(x):
        horizontal = axis_bank(x, 1)
        return cat([
            axis_bank(cut(horizontal, i, i + 1), 0)
            for i in range(4)
        ])

    centers = [-9, -3, 3, 9]

    def geometric_kernel(functions):
        w = np.zeros((len(functions), 16, 7, 7), np.float32)
        for dy in range(-12, 13):
            iy = min(3, max(0, (dy + 12) // 6))
            sy = dy - centers[iy]
            for dx in range(-12, 13):
                ix = min(3, max(0, (dx + 12) // 6))
                sx = dx - centers[ix]
                channel = ix * 4 + iy
                for out, function in enumerate(functions):
                    w[out, channel, sy + 3, sx + 3] = function(dx, dy)
        return w

    def selection_function(width, height):
        left = -(width // 2)
        right = width - 1 - width // 2
        top = -(height // 2)
        bottom = height - 1 - height // 2

        def f(dx, dy):
            qx, qy = -dx, -dy
            ex = max(left - qx, qx - right, 0)
            ey = max(top - qy, qy - bottom, 0)
            return float(ex * ex + ey * ey <= 64)
        return f

    selection_functions = [
        selection_function(n, n) for n in fruit_sizes
    ] + [selection_function(5, 4)] * 3

    fruit_hits = []
    for i in range(9):
        bank = tiles(cut(objects, i, i + 1))
        fruit_hits.append(conv(
            bank, geometric_kernel([selection_functions[i]])
        ))
    fruit_hits = cat(fruit_hits)

    creature_banks = []
    creature_hits = []
    for i in range(3):
        bank = tiles(cut(objects, i + 9, i + 10))
        creature_banks.append(bank)
        creature_hits.append(conv(
            bank, geometric_kernel([selection_functions[9 + i]])
        ))
    creature_hits = cat(creature_hits)

    def contact_function(n, advance, swept=False):
        half = n // 2
        far = n - 1 - half
        selected = selection_function(5, 4)

        def f(dx, dy):
            if selected(dx, dy):
                return 0.0
            qx, qy = -dx, -dy
            ax = (qx > 0) - (qx < 0)
            ay = (qy > 0) - (qy < 0)
            steps = range(advance + 1) if swept else (advance,)
            for t in steps:
                sx = qx - t * ax
                sy = qy - t * ay
                overlap_x = sx - half <= 2 and sx + far >= -2
                overlap_y = sy - half <= 1 and sy + far >= -2
                if overlap_x and overlap_y:
                    return 1.0
            return 0.0
        return f

    hazards = []
    safety_hazards = []
    for i, bank in enumerate(creature_banks):
        advance = 8 if i == 2 else 4
        broad_advance = 14 if i == 2 else 12
        functions = [
            contact_function(n, advance) for n in fruit_sizes
        ] + [
            contact_function(n, broad_advance, True)
            for n in fruit_sizes
        ]
        fields = conv(bank, geometric_kernel(functions))
        hazards.append(cut(fields, 0, 9))
        safety_hazards.append(cut(fields, 9, 18))
    hazard_fields = plus(plus(hazards[0], hazards[1]), hazards[2])
    safety_fields = plus(
        plus(safety_hazards[0], safety_hazards[1]),
        safety_hazards[2],
    )

    # Max-plus chamfer propagation supplies long-range geometry.
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]

    def distance_field(source):
        source = scale(clipped(source), 128)

        def spread(x, stride, dst=None):
            k = 2 * stride + 1
            w = np.zeros((8, 1, k, k), np.float32)
            b = []
            for j, (dy, dx) in enumerate(directions):
                w[j, 0, stride + stride * dy,
                  stride + stride * dx] = 1
                b.append(
                    -stride * (math.sqrt(2) if dx and dy else 1)
                )
            shifted = conv(x, w, b, dst=dst)
            slices = [
                node(
                    "slice", [shifted], dst=dst,
                    start=j, stop=j + 1,
                )
                for j in range(8)
            ]
            return relu(maximum([x] + slices, dst=dst), dst=dst)

        initial = spread(source, 1)
        state = fresh()
        shapes[state] = (1, 64, 64)
        body = []
        following = spread(state, 2, dst=body)
        loop_id = fresh()
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": 32,
            "state": [{"id": state, "init": initial}],
            "body": body,
            "next": {state: following},
        })
        result = loop_id + "." + state
        shapes[result] = (1, 64, 64)
        return result

    basket_distance = distance_field(baskets)
    creature_distance = distance_field(all_creatures)

    # Goal-dependent rank tasks.
    missing_total = clipped(minus(goal_counts, total_counts))
    missing_inside = clipped(minus(goal_counts, inside_counts))
    surplus = minus(total_counts, goal_counts)
    goal_presence = clipped(goal_counts)

    higher = np.zeros((12, 12), np.float32)
    lower = np.zeros((12, 12), np.float32)
    for i in range(9):
        higher[i, i + 1:9] = 1
        lower[i, :i] = 1
    for i in range(9, 12):
        higher[i, i + 1:12] = 1

    need_higher = clipped(linear(missing_total, higher))
    need_lower = clipped(linear(missing_total, lower))
    can_merge = clipped(scale(surplus, 1, -1))
    can_downgrade = clipped(surplus)
    creature_presence = node("gmax", [clipped(all_creatures)])

    delivery_mode = missing_inside
    merge_mode = times(
        times(can_merge, need_higher),
        one_minus(delivery_mode),
    )
    bite_mode = times(
        times(times(can_downgrade, need_lower), creature_presence),
        times(one_minus(delivery_mode), one_minus(merge_mode)),
    )

    protected = times(inside, goal_presence)
    movable = minus(objects, protected)
    outside = minus(objects, inside)

    # Guard all non-downgrade fruit, including fruit far from baskets.
    vulnerable_fruit = times(
        fruit_objects, one_minus(cut(bite_mode, 0, 9))
    )
    vulnerable_distance = distance_field(
        sum_channels(vulnerable_fruit)
    )
    guard_neighborhood = clipped(
        scale(vulnerable_distance, 1, -113)
    )
    guard_creatures = times(creature_objects, guard_neighborhood)
    guard_map = sum_channels(guard_creatures)

    delivery_score = times(
        delivery_mode, constant([100] * 9 + [20] * 3)
    )
    merge_score = times(
        merge_mode,
        constant([40 + i for i in range(9)] + [15, 16, 17]),
    )
    bite_score = times(
        bite_mode,
        constant([90 + i for i in range(9)] + [0, 0, 0]),
    )

    candidate_scores = plus(
        times(outside, delivery_score),
        times(movable, plus(merge_score, bite_score)),
    )
    candidate_scores = plus(
        candidate_scores,
        times(objects, constant([0.1] * 9 + [0.05] * 3)),
    )
    candidate_scores = plus(
        candidate_scores,
        scale(
            times(times(outside, delivery_mode), basket_distance),
            0.01,
        ),
    )
    candidate_scores = plus(
        candidate_scores,
        scale(
            times(times(movable, merge_mode), creature_distance),
            0.15,
        ),
    )
    candidate_scores = plus(
        sum_channels(candidate_scores), scale(guard_map, 250)
    )

    best_candidate = node("gmax", [candidate_scores])
    candidate_winners = clipped(
        scale(minus(candidate_scores, best_candidate), 1000000, 1)
    )
    lexicographic = linear("coords", [[-4032, -63, 4096]])
    tie_scores = times(candidate_winners, lexicographic)
    best_tie = node("gmax", [tie_scores])
    anchor = clipped(scale(minus(tie_scores, best_tie), 1, 1))
    anchor = times(anchor, candidate_winners)

    chosen_type = node("gmax", [times(objects, anchor)])
    chosen_fruit = cut(chosen_type, 0, 9)
    chosen_creature = cut(chosen_type, 9, 12)
    guarding = node("gmax", [times(guard_map, anchor)])

    is_fruit = sum_channels(chosen_fruit)
    is_creature = sum_channels(chosen_creature)
    merging = times(
        sum_channels(times(chosen_type, merge_mode)),
        one_minus(guarding),
    )
    biting = times(
        sum_channels(times(chosen_type, bite_mode)),
        one_minus(guarding),
    )
    delivering = one_minus(plus(merging, biting))

    selected_hazard = sum_channels(
        times(hazard_fields, chosen_fruit)
    )
    selected_safety = sum_channels(
        times(safety_fields, chosen_fruit)
    )
    bite_source = clipped(selected_hazard)
    hazard_exists = node("gmax", [bite_source])
    bite_source = plus(
        bite_source,
        times(all_creatures, one_minus(hazard_exists)),
    )
    partners = relu(minus(
        sum_channels(times(movable, chosen_type)), anchor
    ))
    target = plus(
        plus(times(baskets, delivering), times(partners, merging)),
        times(bite_source, biting),
    )
    target_distance = distance_field(target)

    anchor_bank = tiles(anchor)
    anchor_regions = conv(
        anchor_bank, geometric_kernel(selection_functions)
    )
    anchor_selected = sum_channels(
        times(anchor_regions, chosen_type)
    )

    selected_fruits = sum_channels(fruit_hits)
    selected_same_fruit = sum_channels(
        times(fruit_hits, chosen_fruit)
    )
    selected_creatures = sum_channels(creature_hits)
    selected_same_creature = sum_channels(
        times(creature_hits, chosen_creature)
    )

    permitted = plus(one_v, merging)
    extra_fruit = relu(minus(selected_same_fruit, permitted))
    extra_creature = relu(minus(selected_same_creature, permitted))
    wrong_fruit = relu(minus(selected_fruits, selected_same_fruit))
    wrong_creature = relu(
        minus(selected_creatures, selected_same_creature)
    )

    fruit_cost = times(
        plus(scale(wrong_fruit, 110), scale(extra_fruit, 120)),
        is_fruit,
    )
    creature_cost = times(
        plus(scale(wrong_creature, 110), scale(extra_creature, 120)),
        is_creature,
    )
    contact_cost = times(
        scale(selected_safety, 120),
        times(is_fruit, one_minus(biting)),
    )
    bite_selection_cost = times(
        scale(selected_creatures, 85), biting
    )
    bite_reward = times(scale(selected_hazard, 10), biting)

    # Guard clicks maximize separation from vulnerable fruit and must
    # not pull fruit along with the creature.
    click_score = plus(
        times(target_distance, one_minus(guarding)),
        times(scale(vulnerable_distance, -1, 128), guarding),
    )
    click_score = plus(click_score, bite_reward)
    for cost in (
        fruit_cost, creature_cost, contact_cost, bite_selection_cost
    ):
        click_score = minus(click_score, cost)
    click_score = minus(
        click_score, times(scale(selected_fruits, 200), guarding)
    )
    click_score = minus(
        click_score, scale(one_minus(anchor_selected), 1000)
    )
    click_score = minus(
        click_score, scale(one_minus(board), 2000)
    )

    any_inside_goal = clipped(sum_channels(
        times(inside_counts, goal_presence)
    ))
    near_basket = relu(scale(basket_distance, 1, -116))
    click_score = minus(
        click_score,
        times(
            scale(near_basket, 2),
            times(merging, any_inside_goal),
        ),
    )

    # Memory: previous click, state signature, four failed regions.
    object_map = sum_channels(objects)
    position_sums = count(cat([
        times(object_map, row), times(object_map, col)
    ]))
    signature = cat([total_counts, position_sums])
    old_signature = cut("mem", 2, 16)
    signature_delta = minus(signature, old_signature)
    absolute_delta = plus(
        relu(signature_delta), relu(scale(signature_delta, -1))
    )
    delta_sum = sum_channels(absolute_delta)
    unchanged = clipped(scale(delta_sum, -1000, 1))
    was_click = cut("aprev", 6, 7)
    red_warning = node("gmax", [times(colors[2], header)])
    failed = times(
        clipped(plus(unchanged, red_warning)), was_click
    )

    old_click = cut("mem", 0, 2)
    old_records = [
        cut("mem", 16 + 3 * i, 19 + 3 * i)
        for i in range(4)
    ]
    inserted_record = cat([old_click, constant([1])])
    new_records = []
    for i in range(4):
        shifted_record = (
            inserted_record if i == 0 else old_records[i - 1]
        )
        updated = plus(
            times(shifted_record, failed),
            times(old_records[i], one_minus(failed)),
        )
        updated = linear(
            updated, np.diag(np.asarray([1, 1, 0.92], np.float32))
        )
        new_records.append(updated)

        ban_row = cut(updated, 0, 1)
        ban_col = cut(updated, 1, 2)
        strength = cut(updated, 2, 3)
        dr = node("abs", [minus(row, ban_row)])
        dc = node("abs", [minus(col, ban_col)])
        radius = node("max2", [dr, dc])
        inhibition = relu(scale(radius, -1, 7))
        click_score = minus(
            click_score,
            times(scale(inhibition, 80), strength),
        )

    conv(
        click_score, np.ones((1, 1, 1, 1), np.float32),
        output="click",
    )

    # Record the winning click with deterministic row-major tie breaking.
    best_click = node("gmax", ["click"])
    click_winners = clipped(
        scale(minus("click", best_click), 1000000, 1)
    )
    click_ties = times(click_winners, lexicographic)
    best_click_tie = node("gmax", [click_ties])
    unique_click = times(
        clipped(scale(minus(click_ties, best_click_tie), 1, 1)),
        click_winners,
    )
    click_position = count(cat([
        times(unique_click, row), times(unique_click, col)
    ]))
    memory_output = cat(
        [click_position, signature] + new_records
    )
    linear(
        memory_output, np.eye(28, dtype=np.float32),
        output="mem_next",
    )

    action_features = cat([one_v, red_warning])
    action_w = np.zeros((8, 2), np.float32)
    action_w[:, 0] = -10
    action_w[6, 0] = 1
    action_w[7, 0] = 0
    action_w[7, 1] = 20
    linear(action_features, action_w, output="logits")

    # Static accounting only: no observations or game are executed.
    def accounting(nodes):
        macs = 0
        evaluations = 0
        node_count = 0
        spatial_channels = 0
        for nd in nodes:
            node_count += 1
            if nd["op"] == "loop":
                bm, be, bn, bc = accounting(nd["body"])
                macs += nd["T"] * bm
                evaluations += 1 + nd["T"] * be
                node_count += bn
                spatial_channels += bc
                for state in nd["state"]:
                    spatial_channels += shapes[state["id"]][0]
                continue
            evaluations += 1
            shape = shapes[nd["id"]]
            if len(shape) == 3:
                spatial_channels += shape[0]
            op = nd["op"]
            if op == "conv":
                macs += int(np.prod(params[nd["w"]].shape)) * 4096
            elif op == "dense":
                macs += int(np.prod(params[nd["w"]].shape))
            elif op in ("add", "sub", "mul", "max2", "min2"):
                macs += shape[0] * 4096
            elif op in ("gmax", "gmean"):
                macs += shapes[nd["in"][0]][0] * 4096
            elif op in ("vadd", "vmul"):
                macs += shape[0]
        return macs, evaluations, node_count, spatial_channels

    macs, evaluations, nodes, channels = accounting(graph)
    assert sum(a.size for a in params.values()) <= 150000
    assert macs <= 1500000000
    assert evaluations <= 10000
    assert nodes <= 1000
    assert channels + 19 <= 4096
    assert all(np.isfinite(a).all() for a in params.values())
    assert all(
        s[0] <= (512 if len(s) == 3 else 4096)
        for s in shapes.values()
    )

    return {
        "graph": graph,
        "memory_size": 28,
        "params": params,
    }
