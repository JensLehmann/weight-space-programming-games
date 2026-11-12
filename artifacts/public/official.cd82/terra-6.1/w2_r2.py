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
    p = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }

    next_ch = 0
    addresses = {}

    def address(row, col):
        nonlocal next_ch
        key = (row, col)
        if key in addresses:
            return addresses[key]

        pos, neg = next_ch, next_ch + 1
        next_ch += 2
        a = row + 64 * col

        p["W1"][pos, 16, 2, 2] = 63
        p["W1"][pos, 17, 2, 2] = 4032
        p["b1"][pos] = -a

        p["W1"][neg, 16, 2, 2] = -63
        p["W1"][neg, 17, 2, 2] = -4032
        p["b1"][neg] = a

        addresses[key] = (pos, neg)
        return pos, neg

    # Six target/canvas comparisons. The first two are for a
    # two-colour palette; the remaining four are for larger palettes.
    pairs = [
        ((4, 7), (35, 31)),   # top half
        ((11, 7), (42, 31)),  # bottom half
        ((4, 11), (35, 35)),  # southeast diagonal
        ((9, 7), (40, 31)),   # left half
        ((4, 4), (35, 28)),   # northwest diagonal
        ((4, 7), (35, 31)),   # top brush
    ]
    sample_addresses = []
    for target, canvas in pairs:
        sample_addresses.append(address(*target))
        sample_addresses.append(address(*canvas))

    colour = next_ch
    next_ch += 1
    for c in range(16):
        p["W1"][colour, c, 2, 2] = c + 1

    # Palette centre and selected-palette-marker detectors.
    swatch = next_ch
    next_ch += 1
    for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        p["W1"][swatch, 4, 2 + dy, 2 + dx] = 1
    p["W1"][swatch, 4, 2, 2] = -4
    p["b1"][swatch] = -3.5

    selected = next_ch
    next_ch += 1
    p["W1"][selected, 4, 3, 2] = 1
    p["W1"][selected, 0, 4, 2] = 1
    p["W1"][selected, 4, 1, 2] = -1
    p["b1"][selected] = -1.5

    # Extrema of red basket/brush pixels.
    extrema = []
    for axis, sign in [(16, 1), (16, -1), (17, 1), (17, -1)]:
        ch = next_ch
        next_ch += 1
        extrema.append(ch)
        p["W1"][ch, 2, 2, 2] = 64
        p["W1"][ch, axis, 2, 2] = 63 * sign
        p["b1"][ch] = -63 if sign > 0 else 0

    # The top brush has two black pixels beneath a short red cap.
    brush = next_ch
    next_ch += 1
    p["W1"][brush, 0, 2, 2] = 1
    p["W1"][brush, 0, 2, 1] = 1
    p["W1"][brush, 2, 2, 0] = 1
    p["W1"][brush, 2, 2, 3] = 1
    p["W1"][brush, 2, 1, 2] = 1
    p["b1"][brush] = -4.5

    # Exact-position colour samples.
    for i, (a, b) in enumerate(sample_addresses):
        p["W2"][i, colour, 1, 1] = 1
        p["W2"][i, a, 1, 1] = -32
        p["W2"][i, b, 1, 1] = -32

    # Selected colour, all swatch centres, selected swatch centre.
    p["W2"][12, colour, 1, 1] = 1
    p["W2"][12, swatch, 1, 1] = 64
    p["W2"][12, selected, 2, 1] = 64
    p["b2"][12] = -64

    p["W2"][13, swatch, 1, 1] = 2

    p["W2"][14, swatch, 1, 1] = 2
    p["W2"][14, selected, 2, 1] = 2
    p["b2"][14] = -1

    for i, ch in enumerate(extrema):
        p["W2"][15 + i, ch, 1, 1] = 1

    p["W2"][19, brush, 1, 1] = 2

    for i in range(20):
        p["W3"][i, i] = 1

    # Row-broadcast values used by the brush click detector.
    p["W3"][20, 32 + 12] = 1
    p["W3"][21, 32 + 10] = 1
    p["W3"][22, 32 + 11] = 1

    # Click the next swatch to the right, wrapping to the leftmost.
    p["Uh"][0, 14] = 1
    p["Wh"][1, 0, 1, 0] = 1
    p["Wh"][1, 1, 1, 0] = 1
    p["bh"][1] = -0.01
    p["Wh"][2, 1, 1, 1] = 1
    p["Uh"][2, 13] = 16
    p["bh"][2] = -16

    # Carry selected, brush-target, and brush-canvas colours to the
    # brush location.
    p["Uh"][3, 20] = 1
    p["Wh"][3, 3, 0, 1] = 1
    p["Uh"][4, 21] = 1
    p["Wh"][4, 4, 0, 1] = 1
    p["Uh"][5, 22] = 1
    p["Wh"][5, 5, 2, 1] = 1

    p["Wh"][6, 4, 1, 1] = 1
    p["Wh"][6, 3, 1, 1] = -1
    p["Wh"][7, 3, 1, 1] = 1
    p["Wh"][7, 4, 1, 1] = -1
    p["Wh"][8, 4, 1, 1] = 1
    p["Wh"][8, 5, 1, 1] = -1
    p["Wh"][9, 5, 1, 1] = 1
    p["Wh"][9, 4, 1, 1] = -1

    p["Wh"][10, 8, 1, 1] = 1
    p["Wh"][10, 9, 1, 1] = 1
    p["Wh"][10, 6, 1, 1] = -2
    p["Wh"][10, 7, 1, 1] = -2
    p["Uh"][10, 19] = 32
    p["bh"][10] = -32

    p["Wc"][0, 13] = 0.001
    p["Wc"][0, 32 + 2] = 1
    p["Wc"][0, 32 + 10] = 100

    # ReLU pairs implement clipped absolute colour differences.
    unit = 0
    mismatch = []
    selected_mismatch = []

    def clipped_abs(a, b):
        nonlocal unit
        entries = []
        for sign in (1, -1):
            base, over = unit, unit + 1
            unit += 2
            p["Wz"][base, a] = sign
            p["Wz"][base, b] = -sign
            p["Wz"][over, a] = sign
            p["Wz"][over, b] = -sign
            p["bz"][over] = -1
            entries.append((base, over))
        return entries

    for i in range(6):
        target_g = 32 + 2 * i
        canvas_g = 32 + 2 * i + 1
        selected_g = 32 + 12
        mismatch.append(clipped_abs(target_g, canvas_g))
        selected_mismatch.append(clipped_abs(target_g, selected_g))

    # Number of swatches is 4096 times their spatial mean.
    mode_pos, mode_over = unit, unit + 1
    unit += 2
    p["Wz"][mode_pos, 64 + 13] = 4096
    p["Wz"][mode_over, 64 + 13] = 4096
    p["bz"][mode_pos] = -2
    p["bz"][mode_over] = -3

    # Four clipped bounding-box region indicators: N, S, W, E.
    region_units = []
    region_specs = [
        (15, 16, -0.5, 0.5, 4.5),
        (15, 16, 0.5, -0.5, -10.5),
        (17, 18, -0.5, 0.5, -3.5),
        (17, 18, 0.5, -0.5, -4.5),
    ]
    for a, b, wa, wb, bias in region_specs:
        base, over = unit, unit + 1
        unit += 2
        for u in (base, over):
            p["Wz"][u, 32 + a] = wa
            p["Wz"][u, 32 + b] = wb
        p["bz"][base] = bias
        p["bz"][over] = bias - 1
        region_units.append((base, over))

    def add_clipped(row, entries, weight):
        for base, over in entries:
            p["Wmz"][row, base] += weight
            p["Wmz"][row, over] -= weight

    def add_region(row, region, weight):
        base, over = region_units[region]
        p["Wmz"][row, base] += weight
        p["Wmz"][row, over] -= weight

    def add_mode(row, weight):
        p["Wmz"][row, mode_pos] += weight
        p["Wmz"][row, mode_over] -= weight

    # Each position is described by required and forbidden regions.
    # Region indices: N=0, S=1, W=2, E=3.
    positions = [
        ([0], [2, 3]),
        ([0, 3], []),
        ([3], [0, 1]),
        ([1, 3], []),
        ([1], [2, 3]),
        ([1, 2], []),
        ([2], [0, 1]),
        ([0, 2], []),
    ]

    routes = [
        [5, 3, 1, 1, 3, 1, 1, 4],  # top
        [4, 2, 2, 3, 5, 4, 2, 2],  # bottom
        [4, 2, 2, 5, 4, 4, 2, 4],  # southeast
        [3, 3, 1, 3, 3, 1, 5, 2],  # left
        [3, 3, 1, 1, 3, 1, 1, 5],  # northwest
        [6, 3, 1, 1, 3, 1, 1, 4],  # top brush
    ]

    gain = 12.0
    mem_unit = 0

    def common_task(row, task):
        add_clipped(row, mismatch[task], gain)
        group_start = 0 if task < 2 else 2
        for earlier in range(group_start, task):
            add_clipped(row, mismatch[earlier], -gain)
        if task < 2:
            add_mode(row, -gain)
            return 0
        add_mode(row, gain)
        return 1

    def connect(row, action):
        p["Wo"][action, 128 + row] += 0.5
        p["bo"][action] += 0.5

    for task in range(6):
        # Click through the palette if the active target colour differs.
        required_mode = common_task(mem_unit, task)
        add_clipped(mem_unit, selected_mismatch[task], gain)
        p["bm"][mem_unit] = -gain * (1.5 + required_mode)
        connect(mem_unit, 6)
        mem_unit += 1

        # Otherwise route the basket to the stroke position.
        for position, action in enumerate(routes[task]):
            required_mode = common_task(mem_unit, task)
            add_clipped(mem_unit, selected_mismatch[task], -gain)
            positive, negative = positions[position]
            for r in positive:
                add_region(mem_unit, r, gain)
            for r in negative:
                add_region(mem_unit, r, -gain)
            p["bm"][mem_unit] = -gain * (
                0.5 + required_mode + len(positive)
            )
            connect(mem_unit, action)
            mem_unit += 1

    p["bo"][5] += 0.0001
    return p
