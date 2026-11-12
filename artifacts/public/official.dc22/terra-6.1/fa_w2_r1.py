import numpy as np


def build():
    graph = []
    params = {}
    counter = [0]
    shift_weights = {}

    def fresh(prefix="n"):
        counter[0] += 1
        return f"{prefix}_{counter[0]}"

    def param(array):
        name = f"p_{len(params)}"
        params[name] = np.asarray(array, dtype=np.float32)
        return name

    def node(op, inputs, dest=None, target=None, **fields):
        if target is None:
            target = graph
        name = dest if dest is not None else fresh()
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        target.append(nd)
        return name

    def unary(op, x, target=None, dest=None):
        return node(op, [x], dest=dest, target=target)

    def binary(op, x, y, target=None, dest=None):
        return node(op, [x, y], dest=dest, target=target)

    def channel(x, i, target=None):
        return node("slice", [x], target=target, start=i, stop=i + 1)

    def vchannel(x, i):
        return node("vslice", [x], start=i, stop=i + 1)

    def spatial_affine(x, scale, bias=0.0):
        w = param(np.array([[[[scale]]]], dtype=np.float32))
        fields = {"w": w}
        if bias != 0:
            fields["b"] = param(np.array([bias], dtype=np.float32))
        return node("conv", [x], **fields)

    def vector_affine(x, scale, bias=0.0):
        w = param(np.array([[scale]], dtype=np.float32))
        fields = {"w": w}
        if bias != 0:
            fields["b"] = param(np.array([bias], dtype=np.float32))
        return node("dense", [x], **fields)

    def sclip(x):
        return binary("min2", unary("relu", x), one)

    def vclip(x):
        x = unary("vrelu", x)
        t = vector_affine(x, -1.0, 1.0)
        t = unary("vrelu", t)
        return vector_affine(t, -1.0, 1.0)

    def union(items):
        if not items:
            return zero
        out = items[0]
        for item in items[1:]:
            out = binary("max2", out, item)
        return out

    def shift(x, dy, dx):
        # At output (y, x), read input (y + dy, x + dx).
        out = x
        while dy != 0 or dx != 0:
            sy = max(-3, min(3, dy))
            sx = max(-3, min(3, dx))
            key = (sy, sx)
            if key not in shift_weights:
                w = np.zeros((1, 1, 7, 7), dtype=np.float32)
                w[0, 0, 3 + sy, 3 + sx] = 1.0
                shift_weights[key] = param(w)
            out = node("conv", [out], w=shift_weights[key])
            dy -= sy
            dx -= sx
        return out

    def match(clauses):
        w = np.zeros((1, 16, 7, 7), dtype=np.float32)
        for colour, dy, dx in clauses:
            w[0, colour, 3 + dy, 3 + dx] += 1.0
        z = node(
            "conv",
            ["x0"],
            w=param(w),
            b=param(np.array([-len(clauses) + 0.5], dtype=np.float32)),
        )
        return spatial_affine(unary("relu", z), 2.0)

    def tile_count(plane):
        w = np.zeros((1, 1, 3, 3), dtype=np.float32)
        w[0, 0, 1:3, 1:3] = 1.0
        return node("conv", [plane], w=param(w))

    one = channel("coords", 2)
    row = channel("coords", 0)
    col = channel("coords", 1)
    zero = binary("sub", one, one)

    colour = {
        i: channel("x0", i)
        for i in (2, 5, 6, 7, 8, 9, 11, 13, 14, 15)
    }

    panel_count = unary("colsum", colour[5])
    panel = sclip(spatial_affine(panel_count, 1.0, -20.0))
    board = binary("sub", one, panel)

    buttons = {}
    present = {}
    for label, c in (("a", 8), ("b", 9), ("c", 6), ("d", 15)):
        clauses = [
            (c, dy, dx)
            for dy in range(-2, 3)
            for dx in range(-3, 4)
        ]
        buttons[label] = binary("mul", match(clauses), panel)
        present[label] = unary("gmax", buttons[label])

    player = match([
        (14, 0, 0), (14, 0, 1),
        (14, 1, 0), (14, 1, 1),
    ])
    player = binary("mul", player, board)

    goal = match([
        (11, 0, 0), (11, 0, 1),
        (11, 1, 0), (11, 1, 1),
    ])
    goal = binary("mul", goal, board)

    key = match([
        (2, 0, 0), (15, 0, 1), (15, 0, 2), (2, 0, 3),
        (15, 1, 0), (15, 1, 1), (15, 1, 2), (15, 1, 3),
    ])
    key = binary("mul", key, board)
    key_tiles = union([key, shift(key, 0, -2)])
    key_exists = unary("gmax", key)
    no_d_button = vector_affine(present["d"], -1.0, 1.0)
    need_key = binary("vmul", key_exists, no_d_button)
    no_need_key = vector_affine(need_key, -1.0, 1.0)
    target = binary(
        "max2",
        binary("mul", goal, no_need_key),
        binary("mul", key, need_key),
    )

    # Remember the two teleport endpoints while both are visible.
    tele_a = match([
        (6, 0, 0), (7, 0, 1),
        (7, 1, 0), (6, 1, 1),
    ])
    tele_b = match([
        (7, 0, 0), (6, 0, 1),
        (6, 1, 0), (7, 1, 1),
    ])
    tele_visible = binary("mul", union([tele_a, tele_b]), board)

    key_w = np.array([[[[4032.0]], [[63.0]], [[0.0]]]], dtype=np.float32)
    pixel_key = node("conv", ["coords"], w=param(key_w))
    reverse_key = spatial_affine(pixel_key, -1.0, 4097.0)
    weighted = binary("mul", tele_visible, reverse_key)
    maximum = unary("gmax", weighted)
    delta = unary("abs", binary("sub", weighted, maximum))
    first = unary("relu", spatial_affine(delta, -1.0, 1.0))
    first = binary("mul", first, tele_visible)
    second = unary("relu", binary("sub", tele_visible, first))

    count = unary("gmean", tele_visible)
    count = vector_affine(count, 4096.0)
    have_two = vclip(vector_affine(count, 1.0, -1.0))

    r1 = unary("gmax", binary("mul", first, row))
    c1 = unary("gmax", binary("mul", first, col))
    r2 = unary("gmax", binary("mul", second, row))
    c2 = unary("gmax", binary("mul", second, col))
    observed_memory = node("vconcat", [r1, c1, r2, c2, have_two])
    keep_old = vector_affine(have_two, -1.0, 1.0)
    new_part = binary("vmul", observed_memory, have_two)
    old_part = binary("vmul", "mem", keep_old)
    node("vadd", [new_part, old_part], dest="mem_next")

    def remembered_endpoint(ri, ci):
        mr = vchannel("mem_next", ri)
        mc = vchannel("mem_next", ci)
        flag = vchannel("mem_next", 4)
        dr = unary("abs", binary("sub", row, mr))
        dc = unary("abs", binary("sub", col, mc))
        er = unary("relu", spatial_affine(dr, -64.0, 1.0))
        ec = unary("relu", spatial_affine(dc, -64.0, 1.0))
        ep = binary("min2", er, ec)
        ep = binary("min2", ep, flag)
        return binary("min2", ep, present["c"])

    endpoint1 = remembered_endpoint(0, 1)
    endpoint2 = remembered_endpoint(2, 3)

    # A bridge marker is 4x4. The player may cover one quadrant.
    marker_or_player = binary("max2", colour[13], colour[14])
    marker_kernel = np.zeros((1, 1, 5, 5), dtype=np.float32)
    for dy, dx in ((0, 0), (0, 2), (2, 0), (2, 2)):
        marker_kernel[0, 0, 2 + dy, 2 + dx] = 1.0
    anchor = node(
        "conv",
        [marker_or_player],
        w=param(marker_kernel),
        b=param(np.array([-3.5], dtype=np.float32)),
    )
    anchor = spatial_affine(unary("relu", anchor), 2.0)
    anchor = binary("mul", anchor, board)

    # Read each arm segment on both sides of its width. This preserves
    # a segment when one 2x2 half is covered by the player.
    current_arms = []
    rotated_arms = []
    for direction in ("left", "right", "up", "down"):
        for k in range(1, 5):
            if direction == "left":
                dy, dx = 0, -2 * k
                ry, rx = -2 * k, 0
                perpendicular_current = (2, 0)
                perpendicular_rotated = (0, 2)
            elif direction == "right":
                dy, dx = 0, 2 * (k + 1)
                ry, rx = 2 * (k + 1), 0
                perpendicular_current = (2, 0)
                perpendicular_rotated = (0, 2)
            elif direction == "up":
                dy, dx = -2 * k, 0
                ry, rx = 0, -2 * k
                perpendicular_current = (0, 2)
                perpendicular_rotated = (2, 0)
            else:
                dy, dx = 2 * (k + 1), 0
                ry, rx = 0, 2 * (k + 1)
                perpendicular_current = (0, 2)
                perpendicular_rotated = (2, 0)

            sample1 = shift(colour[8], dy, dx)
            sample2 = shift(
                colour[8],
                dy + perpendicular_current[0],
                dx + perpendicular_current[1],
            )
            sample = binary("max2", sample1, sample2)
            segment = binary("min2", anchor, sample)

            here = shift(segment, -dy, -dx)
            here2 = shift(
                here, -perpendicular_current[0], -perpendicular_current[1]
            )
            current_arms.extend([here, here2])

            turned = shift(segment, -ry, -rx)
            turned2 = shift(
                turned, -perpendicular_rotated[0], -perpendicular_rotated[1]
            )
            rotated_arms.extend([turned, turned2])

    bridge_current = union(current_arms)
    bridge_rotated = union(rotated_arms)

    yellow_count = tile_count(colour[9])
    yellow_any = sclip(yellow_count)
    yellow_solid = unary("relu", spatial_affine(yellow_count, 1.0, -3.0))
    yellow_checker = unary("relu", binary("sub", yellow_any, yellow_solid))

    purple_count = tile_count(colour[15])
    purple_any = sclip(purple_count)
    purple_solid = unary("relu", spatial_affine(purple_count, 1.0, -3.0))
    purple_checker = unary("relu", binary("sub", purple_any, purple_solid))

    def covered_platform(any_map):
        # For each possible quadrant of a 4x4 platform, inspect the
        # other three 2x2 quadrants. Each must contain its colour.
        possibilities = []
        for sy in (-2, 2):
            for sx in (-2, 2):
                a = shift(any_map, sy, 0)
                b = shift(any_map, 0, sx)
                c = shift(any_map, sy, sx)
                possibilities.append(binary("min2", binary("min2", a, b), c))
        return binary("min2", player, union(possibilities))

    covered_yellow = covered_platform(yellow_any)
    covered_purple = covered_platform(purple_any)

    # The player can normally occupy only a solid platform quadrant.
    yellow_solid = binary("max2", yellow_solid, covered_yellow)
    purple_solid = binary("max2", purple_solid, covered_purple)

    dynamic_under_player = union([
        binary("min2", player, bridge_current),
        covered_yellow,
        covered_purple,
    ])
    stable_player = unary("relu", binary("sub", player, dynamic_under_player))

    ground = union([
        colour[2], colour[6], colour[7],
        colour[11], colour[13], stable_player,
    ])
    ground = unary("relu", binary("sub", ground, yellow_any))
    ground = unary("relu", binary("sub", ground, purple_any))
    ground = binary("max2", ground, key_tiles)
    ground = binary("mul", ground, board)

    passable = []
    for state in range(8):
        bridge = bridge_rotated if (state & 1) else bridge_current
        yellow = yellow_checker if (state & 2) else yellow_solid
        purple = purple_checker if (state & 4) else purple_solid
        mask = union([ground, bridge, yellow, purple])
        passable.append(binary("mul", mask, board))

    # Reverse wavefront over eight combinations of the three switches.
    movement_kernel = np.zeros((1, 1, 5, 5), dtype=np.float32)
    for dy, dx in ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)):
        movement_kernel[0, 0, 2 + dy, 2 + dx] = 1.0
    movement_w = param(movement_kernel)

    wave_name = fresh("wave")
    reach_ids = [fresh("reach") for _ in range(8)]
    sum_ids = [fresh("sum") for _ in range(8)]
    states = []
    for i in range(8):
        init = binary("min2", target, passable[i])
        states.append({"id": reach_ids[i], "init": init})
        states.append({"id": sum_ids[i], "zeros": [1, 64, 64]})

    body = []
    reaches = node("concat", reach_ids, target=body)
    at_first = unary(
        "gmax", binary("mul", reaches, endpoint1, target=body), target=body
    )
    at_second = unary(
        "gmax", binary("mul", reaches, endpoint2, target=body), target=body
    )
    tele_from_first = binary("mul", endpoint2, at_first, target=body)
    tele_from_second = binary("mul", endpoint1, at_second, target=body)
    teleport_wave = binary(
        "max2", tele_from_first, tele_from_second, target=body
    )

    next_state = {}
    for i in range(8):
        spread = node("conv", [reach_ids[i]], target=body, w=movement_w)
        tele_i = channel(teleport_wave, i, target=body)
        spread = binary("max2", spread, tele_i, target=body)
        for bit, label in ((1, "a"), (2, "b"), (4, "d")):
            switched = binary(
                "min2", reach_ids[i ^ bit], present[label], target=body
            )
            spread = binary("max2", spread, switched, target=body)
        new_reach = binary("min2", spread, passable[i], target=body)
        new_sum = binary("add", sum_ids[i], reach_ids[i], target=body)
        next_state[reach_ids[i]] = new_reach
        next_state[sum_ids[i]] = new_sum

    graph.append({
        "id": wave_name,
        "op": "loop",
        "T": 96,
        "state": states,
        "body": body,
        "next": next_state,
    })
    distance = [f"{wave_name}.{sid}" for sid in sum_ids]

    def at_player(spatial):
        return unary("gmax", binary("mul", player, spatial))

    move_scores = []
    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        value = at_player(shift(distance[0], dy, dx))
        legal = at_player(shift(passable[0], dy, dx))
        move_scores.append(
            binary("vadd", value, vector_affine(legal, 0.01))
        )

    click_scores = {}
    for bit, label in ((1, "a"), (2, "b"), (4, "d")):
        value = at_player(distance[bit])
        legal = at_player(passable[bit])
        value = binary("vadd", value, vector_affine(legal, 0.005))
        click_scores[label] = binary("vmul", value, present[label])

    on_first = unary("gmax", binary("mul", player, endpoint1))
    on_second = unary("gmax", binary("mul", player, endpoint2))
    dest_first = unary("gmax", binary("mul", distance[0], endpoint1))
    dest_second = unary("gmax", binary("mul", distance[0], endpoint2))
    legal_first = unary("gmax", binary("mul", passable[0], endpoint1))
    legal_second = unary("gmax", binary("mul", passable[0], endpoint2))

    c_value = binary(
        "vadd",
        binary("vmul", on_first, dest_second),
        binary("vmul", on_second, dest_first),
    )
    c_legal = binary(
        "vadd",
        binary("vmul", on_first, legal_second),
        binary("vmul", on_second, legal_first),
    )
    click_scores["c"] = binary(
        "vadd", c_value, vector_affine(c_legal, 0.005)
    )

    click_parts = [
        binary("mul", buttons[label], click_scores[label])
        for label in ("a", "b", "c", "d")
    ]
    click_map = union(click_parts)
    node("relu", [click_map], dest="click")
    click_value = unary("gmax", "click")

    scores = node("vconcat", move_scores + [click_value])
    out_w = np.zeros((8, 5), dtype=np.float32)
    for i in range(4):
        out_w[i + 1, i] = 1.0
    out_w[6, 4] = 1.0
    out_b = np.array(
        [-100.0, 0.0, 0.0, 0.0, 0.0, -100.0, 0.0, -100.0],
        dtype=np.float32,
    )
    node("dense", [scores], dest="logits", w=param(out_w), b=param(out_b))

    return {"graph": graph, "memory_size": 5, "params": params}
