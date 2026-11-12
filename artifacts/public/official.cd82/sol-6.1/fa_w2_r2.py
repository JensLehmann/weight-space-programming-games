import numpy as np


def build():
    params = {}
    graph = []
    active = graph
    serial = 0
    parameter_cache = {}

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def node(op, inputs=None, **fields):
        nonlocal serial
        name = "n%d" % serial
        serial += 1
        nd = {"id": name, "op": op}
        if inputs is not None:
            nd["in"] = list(inputs)
        nd.update(fields)
        active.append(nd)
        return name

    def dense(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], **fields)

    def conv(x, w, b=None):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], **fields)

    def op(name, *xs):
        return node(name, xs)

    def vslice(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    constants = {}
    for value in (-4, -1, 1, 2, 3, 4, 5, 10):
        constants[value] = dense(
            "aprev", np.zeros((1, 8)), np.array([value])
        )

    def scale(x, value):
        return op("vmul", x, constants[value])

    def shift(x, value):
        return op("vadd", x, constants[value])

    def negative(x):
        return scale(x, -1)

    def subtract(a, b):
        return op("vadd", a, negative(b))

    def vclip01(x):
        r = op("vrelu", x)
        excess = op("vrelu", shift(r, -1))
        return subtract(r, excess)

    def spatial_clip01(x):
        r = op("relu", x)
        excess = op("relu", op("add", r, constants[-1]))
        return op("sub", r, excess)

    def point_masks(points):
        """Exact, saturated coordinate selectors, one channel per point."""
        points = np.asarray(points, dtype=np.float32)
        n = len(points)
        wr = np.zeros((n, 3), dtype=np.float32)
        wc = np.zeros((n, 3), dtype=np.float32)
        wr[:, 0] = 63
        wc[:, 1] = 63
        dr = op("abs", conv("coords", wr, -points[:, 0]))
        dc = op("abs", conv("coords", wc, -points[:, 1]))
        distance = op("add", dr, dc)
        raw = conv(
            distance, -2 * np.eye(n, dtype=np.float32),
            np.full(n, 1.5, dtype=np.float32)
        )
        return spatial_clip01(raw)

    # The 80 pixels actually tested by the game's equality condition.
    cells = [
        (r, c) for r in range(10) for c in range(10)
        if c != r and c != 9 - r
    ]
    n_pixels = len(cells)

    target_masks = point_masks([(r + 3, c + 3) for r, c in cells])
    canvas_masks = point_masks([(r + 34, c + 27) for r, c in cells])
    colour_value = conv("x0", np.arange(16, dtype=np.float32)[None, :])

    target = op("gmax", op("mul", target_masks, colour_value))
    canvas = op("gmax", op("mul", canvas_masks, colour_value))

    # Exact one-hot colour classification, with wide numerical margins.
    repeat_identity = np.tile(
        np.eye(n_pixels, dtype=np.float32), (16, 1)
    )
    colour_offsets = np.repeat(
        np.arange(16, dtype=np.float32), n_pixels
    )
    positive_difference = op(
        "vrelu", dense(target, repeat_identity, -colour_offsets)
    )
    negative_difference = op(
        "vrelu", dense(negative(target), repeat_identity, colour_offsets)
    )
    colour_distance = op(
        "vadd", positive_difference, negative_difference
    )
    target_onehot = vclip01(shift(scale(colour_distance, -4), 2))
    target_planes = [
        vslice(target_onehot, c * n_pixels, (c + 1) * n_pixels)
        for c in range(16)
    ]

    difference = subtract(target, canvas)
    absolute_difference = op(
        "vadd", op("vrelu", difference),
        op("vrelu", negative(difference))
    )
    mismatch = vclip01(shift(scale(absolute_difference, 4), -1))

    # Brush masks: eight broad strokes, followed by four small stamps.
    masks = np.zeros((12, n_pixels), dtype=np.float32)
    for p, (r, c) in enumerate(cells):
        masks[:, p] = [
            r < 5,
            c >= r,
            c >= 5,
            c >= 9 - r,
            r >= 5,
            c <= r,
            c < 5,
            c < 10 - r,
            r < 3 and 3 <= c < 7,
            3 <= r < 7 and c >= 7,
            r >= 7 and 3 <= c < 7,
            3 <= r < 7 and c < 3,
        ]

    expand_regions = np.tile(np.eye(12, dtype=np.float32), (16, 1))
    sum_over_colours = np.tile(
        np.eye(12, dtype=np.float32), (1, 16)
    )
    sum_over_regions = np.zeros((16, 192), dtype=np.float32)
    for colour in range(16):
        sum_over_regions[colour, colour * 12:(colour + 1) * 12] = 1

    unresolved_initial = dense(
        "aprev", np.zeros((n_pixels, 8)),
        np.ones(n_pixels, dtype=np.float32)
    )

    # Backward peeling is independent of the current canvas except for
    # its stopping condition. This makes successive forward strokes
    # follow a consistent, finite plan.
    body = []
    active = body

    remaining_counts = dense("s_remaining", masks)
    active_regions = vclip01(remaining_counts)

    colour_counts = []
    for plane in target_planes:
        exposed_colour = op("vmul", "s_remaining", plane)
        colour_counts.append(dense(exposed_colour, masks))
    colour_counts = op("vconcat", *colour_counts)

    repeated_counts = dense(remaining_counts, expand_regions)
    contradictions = subtract(repeated_counts, colour_counts)
    valid_colours = op(
        "vrelu", shift(negative(contradictions), 1)
    )
    region_valid = dense(valid_colours, sum_over_colours)
    region_valid = op("vmul", region_valid, active_regions)

    exposed_mismatch = op("vmul", "s_remaining", mismatch)
    needs_work = vclip01(dense(
        exposed_mismatch, np.ones((1, n_pixels), dtype=np.float32)
    ))
    region_valid = op("vmul", region_valid, needs_work)

    # Select the first valid region through prefix exclusion.
    prefix = -np.tril(np.ones((12, 12), dtype=np.float32), k=-1)
    no_earlier_region = op(
        "vrelu", dense(region_valid, prefix, np.ones(12))
    )
    chosen_region = op("vmul", region_valid, no_earlier_region)

    repeated_choice = dense(chosen_region, expand_regions)
    chosen_colour_flags = op(
        "vmul", valid_colours, repeated_choice
    )
    chosen_colour = dense(chosen_colour_flags, sum_over_regions)

    covered = dense(chosen_region, masks.T)
    remaining_next = op(
        "vmul", "s_remaining", shift(negative(covered), 1)
    )

    selected_any = dense(
        chosen_region, np.ones((1, 12), dtype=np.float32)
    )
    retain_previous = shift(negative(selected_any), 1)
    region_next = op(
        "vadd", chosen_region,
        op("vmul", "s_last_region", retain_previous)
    )
    colour_next = op(
        "vadd", chosen_colour,
        op("vmul", "s_last_colour", retain_previous)
    )

    active = graph
    node(
        "loop", T=12,
        state=[
            {"id": "s_remaining", "init": unresolved_initial},
            {"id": "s_last_region", "zeros": [12]},
            {"id": "s_last_colour", "zeros": [16]},
        ],
        body=body,
        next={
            "s_remaining": remaining_next,
            "s_last_region": region_next,
            "s_last_colour": colour_next,
        },
    )
    loop_id = graph[-1]["id"]
    selected_region = loop_id + ".s_last_region"
    selected_colour = loop_id + ".s_last_colour"

    # Exclusive red boundary markers identify the basket's ring position.
    basket_markers = point_masks([
        (24, 31), (21, 39), (38, 46), (56, 39),
        (53, 31), (56, 24), (38, 17), (21, 24),
    ])
    red = node("slice", ["x0"], start=2, stop=3)
    basket_position = op("gmax", op("mul", basket_markers, red))

    region_to_position = np.zeros((8, 12), dtype=np.float32)
    region_to_position[:, :8] = np.eye(8, dtype=np.float32)
    for patch, position in enumerate((0, 2, 4, 6)):
        region_to_position[position, 8 + patch] = 1
    desired_position = dense(selected_region, region_to_position)

    # Ring navigation: a shortest first move for every position pair.
    repeat_current = np.zeros((64, 8), dtype=np.float32)
    repeat_desired = np.zeros((64, 8), dtype=np.float32)
    navigation_weights = np.zeros((8, 64), dtype=np.float32)
    forward_action = (4, 2, 2, 3, 3, 1, 1, 4)
    backward_action = (3, 3, 1, 1, 4, 4, 2, 2)
    for current in range(8):
        for desired in range(8):
            pair = 8 * current + desired
            repeat_current[pair, current] = 1
            repeat_desired[pair, desired] = 1
            distance = (desired - current) % 8
            if distance:
                action = (
                    forward_action[current] if distance <= 4
                    else backward_action[current]
                )
                navigation_weights[action, pair] = 4
    position_pairs = op(
        "vmul",
        dense(basket_position, repeat_current),
        dense(desired_position, repeat_desired)
    )
    navigation = dense(position_pairs, navigation_weights)
    same_position = dense(
        op("vmul", basket_position, desired_position),
        np.ones((1, 8), dtype=np.float32)
    )

    # Palette centres occupy rows 3..5. The row-7 underline identifies
    # the currently selected swatch, independently of palette spacing.
    row_weight = np.array([[63, 0, 0]], dtype=np.float32)
    col_weight = np.array([[0, 63, 0]], dtype=np.float32)

    palette_row_distance = op(
        "abs", conv("coords", row_weight, np.array([-4]))
    )
    palette_row_band = spatial_clip01(conv(
        palette_row_distance, np.array([[-2]], dtype=np.float32),
        np.array([3])
    ))
    palette_columns = spatial_clip01(
        conv("coords", col_weight, np.array([-18]))
    )
    palette_area = op("mul", palette_row_band, palette_columns)
    palette_colours = op("mul", "x0", palette_area)

    underline_distance = op(
        "abs", conv("coords", row_weight, np.array([-7]))
    )
    underline_row = spatial_clip01(conv(
        underline_distance, np.array([[-2]], dtype=np.float32),
        np.array([1.5])
    ))
    black = node("slice", ["x0"], start=0, stop=1)
    underline = op(
        "mul", op("mul", underline_row, palette_columns), black
    )
    underline_columns = op("colmax", underline)
    current_colour = op(
        "gmax", op("mul", palette_colours, underline_columns)
    )
    current_colour = vclip01(shift(scale(current_colour, 4), -1))
    colour_correct = dense(
        op("vmul", current_colour, selected_colour),
        np.ones((1, 16), dtype=np.float32)
    )
    colour_correct = vclip01(colour_correct)
    change_colour = shift(negative(colour_correct), 1)

    broad_region = vslice(selected_region, 0, 8)
    patch_region = vslice(selected_region, 8, 12)
    broad_selected = dense(
        broad_region, np.ones((1, 8), dtype=np.float32)
    )
    patch_selected = dense(
        patch_region, np.ones((1, 4), dtype=np.float32)
    )
    ready = op("vmul", same_position, colour_correct)
    broad_ready = op("vmul", ready, broad_selected)
    patch_ready = op("vmul", ready, patch_selected)

    paint_logit = scale(broad_ready, 3)
    click_logit = op(
        "vadd", scale(change_colour, 5), scale(patch_ready, 3)
    )
    action_features = op(
        "vconcat", navigation, paint_logit, click_logit
    )
    output_weights = np.zeros((8, 10), dtype=np.float32)
    output_weights[:, :8] = np.eye(8, dtype=np.float32)
    output_weights[5, 8] = 1
    output_weights[6, 9] = 1
    output_bias = np.zeros(8, dtype=np.float32)
    output_bias[0] = -100
    output_bias[7] = -100
    dense(action_features, output_weights, output_bias)
    graph[-1]["id"] = "logits"

    wanted_palette_pixels = op("mul", palette_colours, selected_colour)
    palette_click = conv(
        wanted_palette_pixels, np.ones((1, 16), dtype=np.float32)
    )
    palette_click = op("mul", palette_click, change_colour)

    stamp_centres = point_masks([
        (20, 32), (39, 50), (57, 32), (39, 13)
    ])
    stamp_click = conv(
        op("mul", stamp_centres, patch_region),
        np.ones((1, 4), dtype=np.float32)
    )
    stamp_click = op("mul", stamp_click, patch_ready)
    op("add", palette_click, stamp_click)
    graph[-1]["id"] = "click"

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params,
    }
