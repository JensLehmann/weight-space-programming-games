import numpy as np


def build():
    graph = []
    params = {}
    serial = 0

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return f"{prefix}_{serial}"

    def param(value):
        key = fresh("p")
        params[key] = np.asarray(value, dtype=np.float32)
        return key

    def node(op, inputs, **fields):
        key = fresh()
        graph.append({"id": key, "op": op, "in": inputs, **fields})
        return key

    def conv(x, weights, bias=None):
        fields = {"w": param(weights)}
        if bias is not None:
            fields["b"] = param(bias)
        return node("conv", [x], **fields)

    def dense(x, weights, bias=None):
        fields = {"w": param(weights)}
        if bias is not None:
            fields["b"] = param(bias)
        return node("dense", [x], **fields)

    def unary(op, x):
        return node(op, [x])

    def binary(op, x, y):
        return node(op, [x, y])

    def concat(*xs):
        return node("concat", list(xs))

    def vconcat(*xs):
        return node("vconcat", list(xs))

    def spatial_slice(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vector_slice(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def affine(x, weight, bias=0.0):
        return dense(x, [[weight]], [bias])

    def one_minus(x):
        return affine(x, -1.0, 1.0)

    def clip01(x):
        positive = unary("vrelu", x)
        excess = unary("vrelu", one_minus(positive))
        return one_minus(excess)

    def threshold(x, boundary):
        return clip01(affine(x, 2.0, -2.0 * boundary))

    def vsum(x, length, factor=1.0):
        w = np.full((1, length), factor, dtype=np.float32)
        return dense(x, w)

    def vor(*xs):
        value = xs[0]
        for x in xs[1:]:
            value = binary("vadd", value, x)
        return clip01(value)

    def vand(*xs):
        value = xs[0]
        for x in xs[1:]:
            value = binary("vmul", value, x)
        return value

    def diagonal_kernel(channels, size):
        w = np.zeros((channels, channels, size, size), dtype=np.float32)
        for c in range(channels):
            w[c, c, :, :] = 1.0
        return w

    # Detect the camera's factor-two magnification.
    tile_colours = [1, 2, 9, 10, 11, 12, 15]
    scale_w = np.zeros((len(tile_colours), 16, 3, 3), dtype=np.float32)
    for i, colour in enumerate(tile_colours):
        scale_w[i, colour, :, :] = 1.0
    scale_hits = unary(
        "relu",
        conv(
            "x0",
            scale_w,
            np.full(len(tile_colours), -8.5, dtype=np.float32),
        ),
    )
    scale_max = unary("gmax", scale_hits)
    magnified = clip01(affine(vsum(scale_max, len(tile_colours)), 2.0))
    unmagnified = one_minus(magnified)

    # Detect solid goal tiles and corner markers of both goal colours.
    goal_colours = spatial_slice("x0", 11, 13)

    erosion_w = np.zeros((2, 2, 3, 3), dtype=np.float32)
    for c in range(2):
        for dy in (0, 1):
            for dx in (0, 1):
                erosion_w[c, c, 1 + dy, 1 + dx] = 1.0
    eroded = unary(
        "relu",
        conv(goal_colours, erosion_w, [-3.0, -3.0]),
    )

    marker_small_w = np.zeros((2, 2, 7, 7), dtype=np.float32)
    marker_large_w = np.zeros((2, 2, 7, 7), dtype=np.float32)
    for c in range(2):
        for dy in (-1, 2):
            for dx in (-1, 2):
                marker_small_w[c, c, 3 + dy, 3 + dx] = 1.0
        for dy in (-3, 3):
            for dx in (-3, 3):
                marker_large_w[c, c, 3 + dy, 3 + dx] = 1.0

    marker_small = unary(
        "relu",
        conv(goal_colours, marker_small_w, [-3.0, -3.0]),
    )
    marker_large = unary(
        "relu",
        conv(eroded, marker_large_w, [-3.0, -3.0]),
    )
    markers = binary(
        "add",
        binary("mul", marker_small, unmagnified),
        binary("mul", marker_large, magnified),
    )

    # Magnified marker corners are themselves solid 2x2 blocks. Remove
    # those from the candidate goal mask before counting occupied markers.
    corner_mask = conv(marker_large, marker_large_w)
    corner_mask = binary("mul", corner_mask, magnified)
    goals = unary("relu", binary("sub", eroded, corner_mask))
    occupied = binary("mul", markers, goals)
    occupied_mean = unary("gmean", occupied)
    occupied_count = vsum(occupied_mean, 2, 4096.0)

    # Detect up to four buttons of each colour. Each selected peak is
    # suppressed in a radius-nine neighbourhood before finding the next.
    button_planes = [
        spatial_slice("x0", 8, 9),
        spatial_slice("x0", 14, 15),
    ]
    one_plane = spatial_slice("coords", 2, 3)

    neighbour_w = np.ones((1, 1, 3, 3), dtype=np.float32)
    dilation_w = param(np.ones((1, 1, 7, 7), dtype=np.float32))
    score_w = np.zeros((1, 4, 1, 1), dtype=np.float32)
    score_w[0, 0, 0, 0] = 10000.0
    score_w[0, 1, 0, 0] = -8064.0
    score_w[0, 2, 0, 0] = -126.0

    maps = []
    presences = []
    for colour_plane in button_planes:
        neighbourhood = unary(
            "relu",
            conv(colour_plane, neighbour_w, [-4.0]),
        )
        neighbourhood = binary("min2", neighbourhood, one_plane)
        remaining = binary("mul", neighbourhood, colour_plane)

        for i in range(4):
            scores = conv(concat(remaining, "coords"), score_w)
            best = unary("gmax", scores)
            relative = binary("sub", scores, best)
            peak = unary("relu", conv(relative, [[[[1.0]]]], [1.0]))
            peak = binary("mul", peak, remaining)
            maps.append(peak)
            presences.append(unary("gmax", peak))

            if i != 3:
                expanded = peak
                for _ in range(3):
                    expanded = node("conv", [expanded], w=dilation_w)
                remaining = unary(
                    "relu",
                    binary("sub", remaining, expanded),
                )

    left_presence = vconcat(*presences[:4])
    right_presence = vconcat(*presences[4:])

    # Memory layout:
    # 0:4 previous track, 4 previous occupied count, 5 track age,
    # 6 previous action was an undo, 7 total undos used.
    old_track = vector_slice("mem", 0, 4)
    old_occupied = vector_slice("mem", 4, 5)
    old_age = vector_slice("mem", 5, 6)
    old_undo = vector_slice("mem", 6, 7)
    old_repairs = vector_slice("mem", 7, 8)

    active = clip01(vsum(old_track, 4))
    inactive = one_minus(active)

    delta = binary("vadd", occupied_count, affine(old_occupied, -1.0))
    gained = vand(active, threshold(delta, 0.5))
    lost = vand(active, threshold(affine(delta, -1.0), 0.5))

    undo_allowed = one_minus(threshold(old_repairs, 1.5))
    restore = vand(lost, undo_allowed, one_minus(old_undo))

    aged = threshold(old_age, 3.5)
    switch = vand(
        active,
        vor(gained, aged, old_undo),
        one_minus(restore),
    )

    # Rotate track one-hot to the next track.
    rotate_w = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        rotate_w[(i + 1) % 4, i] = 1.0
    rotate_param = param(rotate_w)

    next_track = node("dense", [old_track], w=rotate_param)
    initial_w = np.zeros((4, 1), dtype=np.float32)
    initial_w[0, 0] = 1.0
    initial_track = dense(inactive, initial_w)

    candidate = binary(
        "vadd",
        initial_track,
        binary(
            "vadd",
            binary("vmul", old_track, one_minus(switch)),
            binary("vmul", next_track, switch),
        ),
    )

    # Skip absent tracks. At most four rotations are necessary.
    for _ in range(4):
        valid = clip01(vsum(binary("vmul", candidate, left_presence), 4))
        rotated = node("dense", [candidate], w=rotate_param)
        candidate = binary(
            "vadd",
            binary("vmul", candidate, valid),
            binary("vmul", rotated, one_minus(valid)),
        )
    track = candidate

    # A right button reverses the previous left-button click.
    # If that right button is unavailable, use the left button instead.
    right_exists = clip01(vsum(binary("vmul", track, right_presence), 4))
    use_right = vand(restore, right_exists)
    left_choice = binary("vmul", track, one_minus(use_right))
    right_choice = binary("vmul", track, use_right)
    button_choice = vconcat(left_choice, right_choice)

    restart = vor(inactive, switch)
    continue_track = one_minus(restart)
    new_age = binary(
        "vadd",
        binary("vmul", affine(old_age, 1.0, 1.0), continue_track),
        restart,
    )
    new_repairs = binary("vadd", old_repairs, use_right)
    memory_value = vconcat(
        track,
        occupied_count,
        new_age,
        use_right,
        new_repairs,
    )
    graph.append({
        "id": "mem_next",
        "op": "vadd",
        "in": [memory_value, affine(active, 0.0)],
    })

    button_maps = concat(*maps)
    selected_maps = binary("mul", button_maps, button_choice)
    graph.append({
        "id": "click",
        "op": "conv",
        "in": [selected_maps],
        "w": param(np.ones((1, 8, 1, 1), dtype=np.float32)),
    })

    logits_w = np.zeros((8, 8), dtype=np.float32)
    logits_b = np.zeros(8, dtype=np.float32)
    logits_b[6] = 1.0
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": ["avail"],
        "w": param(logits_w),
        "b": param(logits_b),
    })

    return {
        "graph": graph,
        "memory_size": 8,
        "params": params,
    }
