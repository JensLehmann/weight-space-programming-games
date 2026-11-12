import numpy as np


def build():
    graph = []
    params = {}
    serial = [0]

    def node(op, inputs=None, name=None, **fields):
        if name is None:
            name = "n%d" % serial[0]
            serial[0] += 1
        nd = {"id": name, "op": op}
        if inputs is not None:
            nd["in"] = list(inputs)
        nd.update(fields)
        graph.append(nd)
        return name

    def param(name, value):
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def conv(x, w, b=None):
        kw = {"w": w}
        if b is not None:
            kw["b"] = b
        return node("conv", [x], **kw)

    def dense(x, w, b=None):
        kw = {"w": w}
        if b is not None:
            kw["b"] = b
        return node("dense", [x], **kw)

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vsl(x, a, b):
        return node("vslice", [x], start=a, stop=b)

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

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def sum_spatial(xs):
        result = xs[0]
        for x in xs[1:]:
            result = add(result, x)
        return result

    one = sl("coords", 2, 3)
    row_coord = sl("coords", 0, 1)
    col_coord = sl("coords", 1, 2)

    param("identity1", np.ones((1, 1, 1, 1), np.float32))

    def threshold(x, t):
        key = "bias_m%d" % int(t)
        if key not in params:
            param(key, [-float(t)])
        return node("relu", [conv(x, "identity1", key)])

    def clip01(x):
        return mn(node("relu", [x]), one)

    def at_least(x, n):
        return clip01(threshold(x, n - 1))

    def color(c):
        return sl("x0", c, c + 1)

    param("v_neg", [[-1]])
    param("v_one_minus_w", [[-1]])
    param("v_one_minus_b", [1])
    param("v_minus_one_w", [[1]])
    param("v_minus_one_b", [-1])
    param("v_minus_three_w", [[1]])
    param("v_minus_three_b", [-3])

    def neg(x):
        return dense(x, "v_neg")

    def complement(x):
        return dense(x, "v_one_minus_w", "v_one_minus_b")

    def positive(x):
        return node("vrelu", [x])

    def cap(x):
        p = positive(x)
        excess = positive(dense(p, "v_minus_one_w", "v_minus_one_b"))
        return va(p, neg(excess))

    def equal_zero(x):
        return complement(cap(va(positive(x), positive(neg(x)))))

    def at_four(x):
        return cap(positive(dense(x, "v_minus_three_w", "v_minus_three_b")))

    # Separate the control panel from the board.
    panel = clip01(threshold(node("colsum", [color(5)]), 10))
    board = sub(one, panel)

    # 2x2 cell counts.
    count_w = np.zeros((7, 16, 3, 3), np.float32)
    for c in (1, 2, 6, 7, 11, 12, 13, 14):
        count_w[0, c, 1:3, 1:3] = 1
    for out, c in enumerate((9, 15, 11, 14, 6, 7), start=1):
        count_w[out, c, 1:3, 1:3] = 1
    param("cell_counts", count_w)
    counts = conv("x0", "cell_counts")
    base_count = sl(counts, 0, 1)
    count9 = sl(counts, 1, 2)
    count15 = sl(counts, 2, 3)
    count11 = sl(counts, 3, 4)
    count14 = sl(counts, 4, 5)
    count6 = sl(counts, 5, 6)
    count7 = sl(counts, 6, 7)

    player = mul(at_least(count14, 4), board)
    goal = mul(at_least(count11, 4), board)

    # Large coloured buttons.
    button_w = np.zeros((4, 16, 7, 7), np.float32)
    for i, c in enumerate((8, 9, 6, 15)):
        button_w[i, c, 1:6, 0:7] = 1
    param("button_patterns", button_w)
    button_counts = conv("x0", "button_patterns")
    buttons = []
    button_flags = []
    for i in range(4):
        b = mul(at_least(sl(button_counts, i, i + 1), 35), panel)
        buttons.append(b)
        button_flags.append(node("gmax", [b]))

    # Pixel shifts are expressed as cross-correlation offsets.
    directions = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    for name, (dy, dx) in directions.items():
        w = np.zeros((1, 1, 5, 5), np.float32)
        w[0, 0, 2 + 2 * dy, 2 + 2 * dx] = 1
        param("shift_" + name, w)

    for name, dx in (("left", -3), ("right", 3)):
        w = np.zeros((1, 1, 7, 7), np.float32)
        w[0, 0, 3, 3 + dx] = 1
        param("shift3_" + name, w)

    def shift(x, direction, steps=1):
        for _ in range(steps):
            x = conv(x, "shift_" + direction)
        return x

    def horizontal_five(x, direction):
        return conv(shift(x, direction), "shift3_" + direction)

    # Reconstruct the other orientation of marked bridges.
    marker0 = mul(color(13), board)
    player_pixels = mul(color(14), board)
    eight0 = mul(color(8), board)
    param("near5", np.ones((1, 1, 5, 5), np.float32))

    def near5(x):
        return clip01(conv(x, "near5"))

    marker = mx(marker0, mn(player_pixels, near5(marker0)))
    eight_fill = mx(
        eight0,
        mul(mn(player_pixels, near5(eight0)), sub(one, marker)),
    )
    dilate_w = np.zeros((1, 1, 5, 5), np.float32)
    for dy, dx in ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)):
        dilate_w[0, 0, 2 + dy, 2 + dx] = 1
    param("dilate2", dilate_w)
    vicinity = marker
    for _ in range(5):
        vicinity = clip01(conv(vicinity, "dilate2"))
    current_bridge = mn(eight0, vicinity)

    rotated_parts = []
    for source, target in (
        ("left", "down"),
        ("right", "up"),
        ("up", "right"),
        ("down", "left"),
    ):
        sampled = eight_fill
        samples = {}
        for distance in range(1, 5):
            sampled = shift(sampled, source)
            if distance >= 2:
                samples[distance] = sampled
        for distance in (2, 3, 4):
            anchor = mn(marker, samples[distance])
            target_distances = (1, 2) if distance == 2 else (distance,)
            for td in target_distances:
                rotated_parts.append(shift(anchor, target, td))
    rotated_bridge = rotated_parts[0]
    for part in rotated_parts[1:]:
        rotated_bridge = mx(rotated_bridge, part)
    rotated_bridge = mul(mul(rotated_bridge, sub(one, marker)), board)

    sum4_w = np.zeros((1, 1, 3, 3), np.float32)
    sum4_w[0, 0, 1:3, 1:3] = 1
    param("sum_cell", sum4_w)
    bridge_counts = [
        conv(current_bridge, "sum_cell"),
        conv(rotated_bridge, "sum_cell"),
    ]

    def switch_cells(count):
        present = at_least(count, 1)
        filled = at_least(count, 4)
        return at_least(count, 3), sub(present, filled)

    yellow = switch_cells(count9)
    purple = switch_cells(count15)
    param("times_four", np.full((1, 1, 1, 1), 4, np.float32))
    yellow4 = [conv(v, "times_four") for v in yellow]
    purple4 = [conv(v, "times_four") for v in purple]

    legal_channels = []
    for state in range(8):
        a = state & 1
        b = (state >> 1) & 1
        d = (state >> 2) & 1
        total = sum_spatial(
            [base_count, bridge_counts[a], yellow4[b], purple4[d]]
        )
        legal_channels.append(mul(at_least(total, 4), board))
    legal = node("concat", legal_channels)

    # Memory slots:
    # 0:5 teleporter neighbourhood, 5 fetch, 6 carry, 7 landing,
    # 8 raise, 9 seek teleporter, 10 relocate, 11 relocated,
    # 12 and 13 count the two extra platform controls.
    mem_fetch = vsl("mem", 5, 6)
    mem_carry = vsl("mem", 6, 7)
    mem_placed = vsl("mem", 7, 8)
    mem_raise = vsl("mem", 8, 9)
    mem_seek = vsl("mem", 9, 10)
    mem_reloc = vsl("mem", 10, 11)
    mem_left = vsl("mem", 11, 12)
    mem_e = vsl("mem", 12, 13)
    mem_f = vsl("mem", 13, 14)

    # Teleporter cells, including a cell hidden by the player.
    tele_visible = mul(mn(at_least(count6, 2), at_least(count7, 2)), board)
    previous_direction = node(
        "vconcat", [vsl("aprev", 1, 5), vsl("aprev", 6, 7)]
    )
    remembered = vm(vsl("mem", 0, 5), previous_direction)
    param("sum_memory", np.ones((1, 5), np.float32))
    remembered_tele = dense(remembered, "sum_memory")
    tele = mx(tele_visible, mul(player, remembered_tele))

    # Cargo detector. Its response is two rows below the cargo top
    # and one column right of its visible left edge.
    cargo_w = np.zeros((1, 16, 7, 7), np.float32)
    cargo_w[0, 12, 1:4, 2:6] = 1
    cargo_w[0, 12, 0, 2:6] = -1
    param("cargo_pattern", cargo_w)
    cargo = mul(at_least(conv("x0", "cargo_pattern"), 12), board)
    cargo_found = node("gmax", [cargo])
    waypoint = conv(shift(cargo, "down", 2), "shift3_left")

    # Key and route targets.
    key_w = np.zeros((1, 16, 7, 7), np.float32)
    for dy, dx in ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3)):
        key_w[0, 15, 3 + dy, 3 + dx] = 1
    for dy, dx in ((0, 0), (0, 3)):
        key_w[0, 2, 3 + dy, 3 + dx] = 1
    param("key_pattern", key_w)
    key_left = mul(at_least(conv("x0", "key_pattern"), 8), board)
    key_both = mx(key_left, shift(key_left, "left"))
    key_target = mul(key_both, sub(one, button_flags[3]))
    ordinary_targets = mx(goal, key_target)

    special_target = cap(va(mem_placed, mem_seek))
    targets = mx(
        mul(ordinary_targets, complement(special_target)),
        mx(mul(waypoint, mem_placed), mul(tele, mem_seek)),
    )
    target_states = node("concat", [targets] * 8)

    # Recurrent switch-state reachability.
    move_w = np.zeros((8, 8, 5, 5), np.float32)
    for i in range(8):
        for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            move_w[i, i, 2 + dy, 2 + dx] = 1
    param("wave_moves", move_w)
    for name, bit in (("wave_a", 1), ("wave_b", 2), ("wave_d", 4)):
        w = np.zeros((8, 8, 1, 1), np.float32)
        for state in range(8):
            w[state, state ^ bit, 0, 0] = 1
        param(name, w)

    body = [
        {"id": "wave_move", "op": "conv", "in": ["wave"],
         "w": "wave_moves"},
        {"id": "wave_flip_a", "op": "conv", "in": ["wave"],
         "w": "wave_a"},
        {"id": "wave_flip_b", "op": "conv", "in": ["wave"],
         "w": "wave_b"},
        {"id": "wave_flip_d", "op": "conv", "in": ["wave"],
         "w": "wave_d"},
        {"id": "wave_gate_a", "op": "mul",
         "in": ["wave_flip_a", button_flags[0]]},
        {"id": "wave_gate_b", "op": "mul",
         "in": ["wave_flip_b", button_flags[1]]},
        {"id": "wave_gate_d", "op": "mul",
         "in": ["wave_flip_d", button_flags[3]]},
        {"id": "wave_at_tele", "op": "mul", "in": ["wave", tele]},
        {"id": "wave_tele_max", "op": "gmax", "in": ["wave_at_tele"]},
        {"id": "wave_tele_spread", "op": "mul",
         "in": [tele, "wave_tele_max"]},
        {"id": "wave_tele_gate", "op": "mul",
         "in": ["wave_tele_spread", button_flags[2]]},
        {"id": "wave_sum0", "op": "add", "in": ["wave", "wave_move"]},
        {"id": "wave_sum1", "op": "add",
         "in": ["wave_sum0", "wave_gate_a"]},
        {"id": "wave_sum2", "op": "add",
         "in": ["wave_sum1", "wave_gate_b"]},
        {"id": "wave_sum3", "op": "add",
         "in": ["wave_sum2", "wave_gate_d"]},
        {"id": "wave_sum4", "op": "add",
         "in": ["wave_sum3", "wave_tele_gate"]},
        {"id": "wave_next", "op": "min2", "in": ["wave_sum4", legal]},
        {"id": "acc_next", "op": "add", "in": ["acc", "wave_next"]},
    ]
    graph.append({
        "id": "H",
        "op": "loop",
        "T": 96,
        "state": [
            {"id": "wave", "init": target_states},
            {"id": "acc", "zeros": [8, 64, 64]},
        ],
        "body": body,
        "next": {"wave": "wave_next", "acc": "acc_next"},
    })

    direction_w = np.zeros((4, 1, 5, 5), np.float32)
    for i, (dy, dx) in enumerate(((-2, 0), (2, 0), (0, -2), (0, 2))):
        direction_w[i, 0, 2 + dy, 2 + dx] = 1
    param("direction_values", direction_w)
    value0 = sl("H.acc", 0, 1)
    movement_maps = conv(value0, "direction_values")
    movement_values = node("gmax", [mul(movement_maps, player)])

    def value_at_player(state):
        return node("gmax", [mul(sl("H.acc", state, state + 1), player)])

    click_values = [
        value_at_player(1),
        value_at_player(2),
        None,
        value_at_player(4),
    ]
    tele_destination_value = node("gmax", [mul(value0, tele)])
    player_on_tele = node("gmax", [mul(player, tele)])
    click_values[2] = vm(tele_destination_value, player_on_tele)
    click_parts = [mul(buttons[i], click_values[i]) for i in range(4)]
    ordinary_click = click_parts[0]
    for part in click_parts[1:]:
        ordinary_click = mx(ordinary_click, part)
    ordinary_click_value = node("gmax", [ordinary_click])

    # Detect the crane and its controls.
    crusher_w = np.zeros((1, 16, 7, 7), np.float32)
    for dy, xs in (
        (2, (1, 2, 3, 4)),
        (3, (0, 1, 2, 3)),
        (4, (0, 1, 2, 3)),
        (5, (1, 2, 3, 4)),
    ):
        for dx in xs:
            crusher_w[0, 8, dy, dx] = 1
    for dy in (3, 4):
        for dx in (4, 5, 6):
            crusher_w[0, 8, dy, dx] = -1
    param("crusher_pattern", crusher_w)
    crusher = mul(at_least(conv("x0", "crusher_pattern"), 16), board)
    crusher_found = node("gmax", [crusher])

    square_w = np.zeros((2, 16, 7, 7), np.float32)
    for out, c in enumerate((10, 9)):
        square_w[out, c, 2:6, 2:6] = 1
        square_w[out, c, 1, 2:6] = -1
        square_w[out, c, 6, 2:6] = -1
        square_w[out, c, 2:6, 1] = -1
        square_w[out, c, 2:6, 6] = -1
    param("control_squares", square_w)
    squares = conv("x0", "control_squares")
    square10 = mul(at_least(sl(squares, 0, 1), 16), panel)
    square9 = mul(at_least(sl(squares, 1, 2), 16), panel)

    nine_left = horizontal_five(square9, "left")
    nine_right = horizontal_five(square9, "right")
    ten_left = horizontal_five(square10, "left")
    ten_right = horizontal_five(square10, "right")
    control_left = mul(mn(square10, nine_right), sub(one, nine_left))
    control_right = mn(mn(square10, nine_left), nine_right)
    control_up = mn(mn(square9, ten_left), ten_right)
    control_down = mul(mn(square9, ten_left), sub(one, ten_right))

    grab_w = np.zeros((1, 16, 7, 7), np.float32)
    grab_w[0, 8, 2:5, :] = 1
    grab_w[0, 8, 1, :] = -1
    grab_w[0, 8, 5, :] = -1
    param("grab_pattern", grab_w)
    control_grab = mul(at_least(conv("x0", "grab_pattern"), 21), panel)

    controls = [
        control_up, control_down, control_left, control_right, control_grab
    ]
    control_flags = [node("gmax", [c]) for c in controls]
    controls_found = control_flags[0]
    for flag in control_flags[1:]:
        controls_found = vm(controls_found, flag)

    param("coord_scale", np.eye(2, dtype=np.float32) * 63)

    def location(mask):
        r = node("gmax", [mul(mask, row_coord)])
        c = node("gmax", [mul(mask, col_coord)])
        return dense(node("vconcat", [r, c]), "coord_scale")

    crusher_location = location(crusher)
    cargo_location = location(cargo)

    valid_row_w = np.zeros((1, 3, 1, 1), np.float32)
    valid_row_w[0, 0, 0, 0] = -63
    param("valid_row_w", valid_row_w)
    param("valid_row_b", [63])
    valid_row = clip01(conv("coords", "valid_row_w", "valid_row_b"))
    rail = mul(mul(color(0), board), valid_row)
    rail_bottom = node("gmax", [mul(rail, row_coord)])
    rail_left_inv = node("gmax", [mul(rail, sub(one, col_coord))])
    param("rail_coord_w", [[63, 0], [0, -63]])
    param("rail_coord_b", [0, 63])
    rail_location = dense(
        node("vconcat", [rail_bottom, rail_left_inv]),
        "rail_coord_w", "rail_coord_b"
    )
    rail_found = node("gmax", [rail])

    diffs_w = np.array([
        [-1, 0, 1, 0, 0, 0],
        [0, -1, 0, 1, 0, 0],
        [-1, 0, 0, 0, 1, 0],
        [0, -1, 0, 0, 0, 1],
        [-1, 0, 0, 0, 1, 0],
        [0, -1, 0, 0, 0, 1],
    ], np.float32)
    diffs_b = np.array([3, -6, -2, 14, -14, 2], np.float32)
    param("crane_diffs_w", diffs_w)
    param("crane_diffs_b", diffs_b)
    diffs = dense(
        node("vconcat", [crusher_location, cargo_location, rail_location]),
        "crane_diffs_w", "crane_diffs_b"
    )
    fetch_dy = vsl(diffs, 0, 1)
    fetch_dx = vsl(diffs, 1, 2)
    lower_dy = vsl(diffs, 2, 3)
    right_dx = vsl(diffs, 3, 4)
    upper_dy = vsl(diffs, 4, 5)
    left_dx = vsl(diffs, 5, 6)

    param("sum_four", np.ones((1, 4), np.float32))
    total_move_value = dense(movement_values, "sum_four")
    plan_exists = cap(va(total_move_value, ordinary_click_value))
    start = vm(complement(plan_exists), crusher_found)
    start = vm(start, cargo_found)
    start = vm(start, controls_found)
    start = vm(start, rail_found)
    for phase in (mem_carry, mem_placed, mem_raise, mem_seek, mem_reloc,
                  mem_left):
        start = vm(start, complement(phase))

    fetch = cap(va(mem_fetch, start))
    for phase in (mem_carry, mem_placed, mem_raise, mem_seek, mem_reloc,
                  mem_left):
        fetch = vm(fetch, complement(phase))
    carry = mem_carry

    on_waypoint = node("gmax", [mul(player, waypoint)])
    begin_raise = vm(mem_placed, on_waypoint)
    raising = cap(va(mem_raise, begin_raise))
    relocating = mem_reloc

    desired_dy = va(
        va(vm(fetch, fetch_dy), vm(carry, lower_dy)),
        va(vm(raising, upper_dy), vm(relocating, lower_dy)),
    )
    desired_dx = va(
        vm(fetch, fetch_dx),
        va(
            vm(cap(va(carry, raising)), right_dx),
            vm(relocating, left_dx),
        ),
    )
    at_target_y = equal_zero(desired_dy)
    at_target_x = equal_zero(desired_dx)
    at_target = vm(at_target_y, at_target_x)
    at_left = equal_zero(left_dx)

    horizontal = va(
        vm(at_target_y, desired_dx),
        vm(complement(at_target_y), left_dx),
    )
    up_score = vm(cap(positive(neg(desired_dy))), at_left)
    down_score = vm(cap(positive(desired_dy)), at_left)
    left_score = cap(positive(neg(horizontal)))
    right_score = cap(positive(horizontal))

    fetch_arrived = vm(fetch, at_target)
    carry_arrived = vm(carry, at_target)
    raise_arrived = vm(raising, at_target)
    reloc_arrived = vm(relocating, at_target)
    moving_carry = vm(carry, complement(at_target))
    moving_raise = vm(raising, complement(at_target))
    moving_reloc = vm(relocating, complement(at_target))
    crane_active = cap(
        va(fetch, va(moving_carry, va(moving_raise, moving_reloc)))
    )

    # Explicitly use the teleport control upon reaching its tile.
    use_tele = vm(vm(mem_seek, player_on_tele), button_flags[2])

    # Additional two animated-platform controls. They are used only
    # after the crane sequence, when the key is visible and the
    # ordinary planner has no route.
    extra_w = np.zeros((2, 16, 7, 7), np.float32)
    extra_w[0, 2, 0:6, 0:6] = 1
    extra_w[1, 11, 0:3, 0:4] = 1
    extra_w[1, 11, 3:7, 0:7] = 1
    param("extra_controls", extra_w)
    extra_counts = conv("x0", "extra_controls")
    button_e = mul(at_least(sl(extra_counts, 0, 1), 36), panel)
    button_f = mul(at_least(sl(extra_counts, 1, 2), 40), panel)
    flag_e = node("gmax", [button_e])
    flag_f = node("gmax", [button_f])

    extra_ok = vm(mem_left, button_flags[3])
    extra_ok = vm(extra_ok, complement(plan_exists))
    extra_ok = vm(extra_ok, flag_e)
    extra_ok = vm(extra_ok, flag_f)
    extra_ok = vm(extra_ok, complement(crane_active))
    e_done = at_four(mem_e)
    f_done = at_four(mem_f)
    use_e = vm(extra_ok, complement(e_done))
    use_f = vm(vm(extra_ok, e_done), complement(f_done))

    param("action_strength_w", [[200]])

    def strength(x):
        return dense(x, "action_strength_w")

    special_maps = [
        mul(control_up, strength(vm(crane_active, up_score))),
        mul(control_down, strength(vm(crane_active, down_score))),
        mul(control_left, strength(vm(crane_active, left_score))),
        mul(control_right, strength(vm(crane_active, right_score))),
        mul(control_grab, strength(fetch_arrived)),
        mul(buttons[2], strength(use_tele)),
        mul(button_e, strength(use_e)),
        mul(button_f, strength(use_f)),
    ]
    special_click = special_maps[0]
    for part in special_maps[1:]:
        special_click = mx(special_click, part)

    special_active = cap(va(crane_active, va(use_tele, va(use_e, use_f))))
    ordinary_enabled = complement(special_active)
    combined_click = mx(mul(ordinary_click, ordinary_enabled), special_click)
    node("relu", [combined_click], name="click")
    click_value = node("gmax", [combined_click])
    movement_values = vm(movement_values, ordinary_enabled)

    # Update memory.
    tele_neighbours = conv(tele, "direction_values")
    neighbour_flags = node("gmax", [mul(tele_neighbours, player)])
    current_flag = node("gmax", [mul(tele, player)])
    next_fetch = vm(fetch, complement(fetch_arrived))
    next_carry = vm(cap(va(carry, fetch_arrived)), complement(carry_arrived))
    next_placed = vm(
        cap(va(mem_placed, carry_arrived)), complement(begin_raise)
    )
    next_raise = vm(raising, complement(raise_arrived))
    next_seek = vm(cap(va(mem_seek, raise_arrived)), complement(use_tele))
    next_reloc = vm(cap(va(mem_reloc, use_tele)), complement(reloc_arrived))
    next_left = cap(va(mem_left, reloc_arrived))
    next_e = va(mem_e, use_e)
    next_f = va(mem_f, use_f)
    node(
        "vconcat",
        [
            neighbour_flags, current_flag, next_fetch, next_carry,
            next_placed, next_raise, next_seek, next_reloc, next_left,
            next_e, next_f,
        ],
        name="mem_next",
    )

    scores = node("vconcat", [movement_values, click_value])
    output_w = np.zeros((8, 5), np.float32)
    for i in range(4):
        output_w[i + 1, i] = 1
    output_w[6, 4] = 1
    output_b = np.array(
        [-1000, 0, 0, 0, 0, -1000, -0.01, -1000], dtype=np.float32
    )
    param("output_weights", output_w)
    param("output_bias", output_b)
    node(
        "dense", [scores], name="logits",
        w="output_weights", b="output_bias"
    )

    return {"graph": graph, "memory_size": 14, "params": params}
