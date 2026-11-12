import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5),
        "b1": (32,),
        "W2": (32, 32, 3, 3),
        "b2": (32,),
        "W3": (32, 96),
        "b3": (32,),
        "Wh": (16, 16, 3, 3),
        "Uh": (16, 32),
        "bh": (16,),
        "Wz": (128, 96),
        "bz": (128,),
        "Wm": (64, 64),
        "Wmz": (64, 128),
        "Wma": (64, 8),
        "bm": (64,),
        "Wo": (8, 192),
        "bo": (8,),
        "Wc": (1, 48),
        "bc": (1,),
    }
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    # Templates are anchored at (top + 1, left + 1).
    def template(channel, pixels):
        for y in range(4):
            for x in range(4):
                p["W1"][channel, int(pixels[y, x]), y + 1, x + 1] = 2.0
        p["b1"][channel] = -31.0

    player = np.array([
        [0, 0, 0, 0],
        [14, 14, 14, 14],
        [14, 14, 14, 14],
        [14, 14, 14, 14],
    ])

    for direction in range(4):
        template(direction, np.rot90(player, -direction))

    for i, border in enumerate((4, 3, 5, 0)):
        crate = np.full((4, 4), border, dtype=np.int64)
        crate[1:3, 1:3] = 9
        template(4 + i, crate)

    # A colour-2 strip just beyond a crate indicates that its position
    # is inside a goal rectangle rather than immediately outside it.
    sensor_pixels = [
        [(-1, 0), (-1, 1)],
        [(2, 0), (2, 1)],
        [(0, -1), (1, -1)],
        [(0, 2), (1, 2)],
    ]
    for i, offsets in enumerate(sensor_pixels):
        for dy, dx in offsets:
            p["W1"][8 + i, 2, 2 + dy, 2 + dx] = 1.0
        p["b1"][8 + i] = -1.0

    # Visible 4x4 tiles of a bordered goal rectangle.
    goal_channels = []
    channel = 12
    for mask in range(1, 16):
        tile = np.full((4, 4), 2, dtype=np.int64)
        if mask & 1:
            tile[0, :] = 9
        if mask & 2:
            tile[3, :] = 9
        if mask & 4:
            tile[:, 0] = 9
        if mask & 8:
            tile[:, 3] = 9
        template(channel, tile)
        goal_channels.append(channel)
        channel += 1

    p["W1"][27, 17, 2, 2] = 1.0
    p["W1"][28, 16, 2, 2] = 1.0
    p["W1"][29, 5, 2, 2] = 1.0
    p["W1"][30, 12, 2, 2] = 1.0
    p["W1"][30, 15, 2, 2] = 1.0

    # Patterned barrier template.
    pattern = np.array([
        [2, -1, 2, 2],
        [-1, 2, 2, 2],
        [2, 2, 2, -1],
        [2, 2, -1, 2],
    ])
    for y in range(4):
        for x in range(4):
            p["W1"][31, 2, y + 1, x + 1] = (
                1.0 if pattern[y, x] == 2 else -2.0
            )
    p["b1"][31] = -11.0

    # Pass object markers through W2.
    for c in range(8):
        p["W2"][c, c, 1, 1] = 1.0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i, (dy, dx) in enumerate(directions):
        p["W2"][8 + i, 8 + i, 1 + dy, 1 + dx] = 1.0

    for i in range(3):
        p["W2"][12 + i, 29 + i, 1, 1] = 1.0

    for c in goal_channels:
        p["W2"][15, c, 1, 1] = 1.0

    def gated_coordinate(out, markers, packed):
        for c in markers:
            p["W2"][out, c, 1, 1] = 2.0
        p["W2"][out, 27, 1, 1] = 1.0
        if packed:
            p["W2"][out, 28, 1, 1] = 1.0 / 16.0
        p["b2"][out] = -2.0

    gated_coordinate(16, goal_channels, True)
    gated_coordinate(17, [7], True)
    gated_coordinate(18, [0, 1, 2, 3], True)
    gated_coordinate(19, [4, 5, 6], True)
    gated_coordinate(20, goal_channels, False)
    gated_coordinate(21, [7], False)
    gated_coordinate(22, [0, 1, 2, 3], False)
    gated_coordinate(23, [4, 5, 6], False)

    # Eligible crate markers: remove markers with visible goal continuation.
    for c in (4, 5, 6):
        p["W3"][0, c] = 1.0
    for c in range(8, 12):
        p["W3"][0, c] = -1.0

    # All visible goal tiles.
    p["W3"][1, 15] = 1.0

    # Player marker, suppressed when a held crate shares its row or column.
    for c in range(4):
        p["W3"][2, c] = 1.0
    p["W3"][2, 32 + 7] = -1.0
    p["W3"][2, 64 + 7] = -1.0

    p["W3"][3, 7] = 1.0

    # A carried crate with visible goal continuation.
    p["W3"][4, 7] = 1.0
    for c in range(8, 12):
        p["W3"][4, c] = 1.0
    p["b3"][4] = -1.0

    # A faced crate which does not appear to be on a goal.
    p["W3"][5, 5] = 1.0
    for c in range(8, 12):
        p["W3"][5, c] = -1.0

    for c in (12, 13, 14):
        p["W3"][8, c] = 1.0

    feature_map = {
        9: 16,   # packed goal coordinate
        10: 17,  # packed carried-crate coordinate
        11: 7,   # holding indicator
        12: 18,  # packed player coordinate
        13: 19,  # packed free-crate coordinate
        14: 20,  # goal x
        15: 21,  # carried-crate x
        16: 22,  # player x
        17: 23,  # free-crate x
    }
    for out, inp in feature_map.items():
        p["W3"][out, inp] = 1.0

    # Diffusing fields: eligible crates and visible goal tiles.
    for field, source in ((0, 0), (1, 1)):
        for dy, dx in directions:
            p["Wh"][field, field, 1 + dy, 1 + dx] = 0.249
        p["Uh"][field, source] = 0.1
        p["Uh"][field, 8] = -8.0

    # Sample field values one pixel to each side of the relevant marker.
    for i, (dy, dx) in enumerate(directions):
        p["Wh"][2 + i, 0, 1 + dy, 1 + dx] = 1.0
        p["Uh"][2 + i, 2] = 4.0
        p["bh"][2 + i] = -4.0

        p["Wh"][6 + i, 1, 1 + dy, 1 + dx] = 1.0
        p["Uh"][6 + i, 3] = 4.0
        p["bh"][6 + i] = -4.0

    def fg(channel):
        return 32 + channel

    unit = 0

    def add_clip(vector, offset, action, weight):
        nonlocal unit
        p["Wz"][unit] = vector
        p["bz"][unit] = offset
        p["Wo"][action, unit] = weight
        unit += 1

        p["Wz"][unit] = vector
        p["bz"][unit] = offset - 1.0
        p["Wo"][action, unit] = -weight
        unit += 1

    opposite = [1, 0, 3, 2]
    actions = [1, 2, 3, 4]

    for base in (2, 6):
        for i in range(4):
            v = np.zeros(96, dtype=np.float32)
            v[base + i] = 100.0
            v[base + opposite[i]] = -100.0
            add_clip(v, 0.0, actions[i], 1.0)

    # Small coordinate-based fallback for weak or tied fields.
    def coordinate_fallback(target_packed, actor_packed,
                            target_x, actor_x, when_held):
        gate = np.zeros(96, dtype=np.float32)
        gate[fg(11)] = 2048.0 if when_held else -2048.0
        offset = -2048.0 if when_held else 0.0

        v = gate.copy()
        v[fg(target_x)] += 63.0
        v[fg(actor_x)] -= 63.0
        add_clip(v, offset, 4, 0.03)

        v = gate.copy()
        v[fg(target_x)] -= 63.0
        v[fg(actor_x)] += 63.0
        add_clip(v, offset, 3, 0.03)

        v = gate.copy()
        v[fg(target_packed)] += 1008.0
        v[fg(actor_packed)] -= 1008.0
        v[fg(target_x)] -= 1008.0
        v[fg(actor_x)] += 1008.0
        add_clip(v, offset, 2, 0.02)

        v = gate.copy()
        v[fg(target_packed)] -= 1008.0
        v[fg(actor_packed)] += 1008.0
        v[fg(target_x)] += 1008.0
        v[fg(actor_x)] -= 1008.0
        add_clip(v, offset, 1, 0.02)

    coordinate_fallback(13, 12, 17, 16, False)
    coordinate_fallback(9, 10, 14, 15, True)

    # Grab a faced crate; release a carried crate on a visible goal.
    p["Wz"][unit, fg(5)] = 1.0
    p["Wz"][unit, fg(11)] = -2.0
    p["Wo"][5, unit] = 24.0
    unit += 1

    p["Wz"][unit, fg(4)] = 1.0
    p["Wo"][5, unit] = 24.0
    unit += 1

    # Coordinate signals copied into z.
    goal_x = unit
    p["Wz"][unit, fg(14)] = 1.0
    unit += 1

    goal_y = unit
    p["Wz"][unit, fg(9)] = 16.0
    p["Wz"][unit, fg(14)] = -16.0
    unit += 1

    box_x = unit
    p["Wz"][unit, fg(15)] = 1.0
    unit += 1

    box_y = unit
    p["Wz"][unit, fg(10)] = 16.0
    p["Wz"][unit, fg(15)] = -16.0
    unit += 1

    held = unit
    p["Wz"][unit, fg(11)] = 1.0
    p["Wo"][5, unit] = 8.0
    unit += 1

    player_x = unit
    p["Wz"][unit, fg(16)] = 1.0
    unit += 1

    player_y = unit
    p["Wz"][unit, fg(12)] = 16.0
    p["Wz"][unit, fg(16)] = -16.0
    unit += 1

    # Store the preceding visible goal coordinate.
    p["Wmz"][0, goal_x] = 0.01
    p["Wmz"][1, goal_y] = 0.01

    # Narrow approximate equality tests against the previous goal.
    for m, previous, current, bias in (
        (2, 0, box_x, 1.8),
        (3, 0, box_x, -1.8),
        (4, 1, box_y, 1.8),
        (5, 1, box_y, -1.8),
    ):
        p["Wm"][m, previous] = -10000.0
        p["Wmz"][m, current] = 100.0
        p["bm"][m] = bias

    p["Wo"][5, 128 + 2] = 3.5
    p["Wo"][5, 128 + 3] = -3.5
    p["Wo"][5, 128 + 4] = 3.5
    p["Wo"][5, 128 + 5] = -3.5
    p["bo"][5] = -18.0

    # Do not immediately reverse a grab or release.
    p["Wma"][6, 5] = 3.0
    p["Wo"][5, 128 + 6] = -30.0

    # Previous player position and directional blocked-move indicators.
    p["Wmz"][7, player_x] = 0.01
    p["Wmz"][8, player_y] = 0.01

    for i, action in enumerate(actions):
        if i == 0:
            previous, current, sign = 8, player_y, 1.0
        elif i == 1:
            previous, current, sign = 8, player_y, -1.0
        elif i == 2:
            previous, current, sign = 7, player_x, 1.0
        else:
            previous, current, sign = 7, player_x, -1.0

        m = 9 + i
        p["Wm"][m, previous] = -10000.0 * sign
        p["Wmz"][m, current] = 100.0 * sign
        p["Wma"][m, action] = 3.0
        p["bm"][m] = -2.0
        p["Wo"][action, 128 + m] = -1.0

    p["bo"][0] = -100.0
    p["bo"][6] = -100.0
    p["bo"][7] = -100.0
    return p
