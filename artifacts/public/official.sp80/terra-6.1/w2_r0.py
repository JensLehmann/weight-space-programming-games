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

    W1, W2, W3 = p["W1"], p["W2"], p["W3"]

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for d, (dr, dc) in enumerate(directions):
        W1[2 * d, 11, 2 + dr, 2 + dc] = 1
        W1[2 * d + 1, 11, 2 + 2 * dr, 2 + 2 * dc] = 1

    for out, inp in [
        (8, 12),   # background
        (9, 9),    # selected object
        (10, 16),  # row
        (11, 17),  # column
        (12, 8),   # unselected bar
        (14, 6),   # source
        (15, 15),  # unselected deflector
    ]:
        W1[out, inp, 2, 2] = 1

    # Mark bars at the right edge, near the lower display edge,
    # and at the left edge.
    W1[13, 8, 2, 2] = 100
    W1[13, 17, 2, 2] = 100
    p["b1"][13] = -199

    W1[16, 8, 2, 2] = 100
    W1[16, 16, 2, 2] = 100
    p["b1"][16] = -170

    W1[17, 8, 2, 2] = 100
    W1[17, 17, 2, 2] = -100
    p["b1"][17] = -91

    for d, (dr, dc) in enumerate(directions):
        for out in (2 * d, 2 * d + 1):
            W2[out, 2 * d, 1, 1] = 1
            W2[out, 2 * d + 1, 1, 1] = 1
            W2[out, 2 * d + 1, 1 + dr, 1 + dc] = 1
        p["b2"][2 * d + 1] = -1

    for c in range(8, 18):
        W2[c, c, 1, 1] = 1

    # Horizontal and vertical target-notch masks.
    for out, dirs in [(0, (0, 1)), (1, (2, 3))]:
        for d in dirs:
            W3[out, 2 * d] = 1
            W3[out, 2 * d + 1] = -1
        W3[out, 8] = 1
        p["b3"][out] = -2

    W3[2, 9] = 1
    W3[3, 10] = 1
    W3[4, 11] = 1

    # Click candidates, excluding rows with parked edge/source-side bars.
    W3[5, 12] = 1
    W3[5, 15] = 1
    W3[5, 32 + 13] = -2
    W3[5, 32 + 16] = -2
    W3[5, 32 + 17] = -2
    W3[6, 14] = 1

    def gated_coordinate(out, mask, coordinate, complement=False):
        p["Uh"][out, mask] = 2
        if complement:
            p["Uh"][out, coordinate] = -1
            p["bh"][out] = -1
        else:
            p["Uh"][out, coordinate] = 1
            p["bh"][out] = -2

    gated_coordinate(0, 0, 4)
    gated_coordinate(1, 0, 4, True)
    gated_coordinate(2, 2, 4)
    gated_coordinate(3, 2, 4, True)
    gated_coordinate(4, 2, 3)
    gated_coordinate(5, 2, 3, True)
    gated_coordinate(6, 6, 4)
    gated_coordinate(7, 6, 4, True)
    gated_coordinate(8, 1, 4, True)
    gated_coordinate(9, 0, 3, True)
    p["Uh"][12, 0] = 1
    p["Uh"][13, 1] = 1
    gated_coordinate(14, 1, 3)
    gated_coordinate(15, 1, 3, True)
    gated_coordinate(11, 6, 3, True)

    def expr(const=0.0, terms=None):
        v = np.zeros(97, dtype=np.float32)
        v[96] = const
        if terms:
            for i, value in terms.items():
                v[i] += value
        return v

    txmax = expr(terms={0: 63})
    txmin = expr(63, {1: -63})
    sxmax = expr(terms={2: 63})
    sxmin = expr(63, {3: -63})
    symin = expr(63, {5: -63})
    srcxmax = expr(terms={6: 63})
    srcxmin = expr(63, {7: -63})
    side_xmin = expr(63, {8: -63})
    tymin = expr(63, {9: -63})
    srcymin = expr(63, {11: -63})
    side_ymin = expr(63, {15: -63})

    width = expr(-62, {2: 63, 3: 63})
    height = expr(-62, {4: 63, 5: 63})
    one = expr(1)

    next_z = 0

    def indicator(v):
        nonlocal next_z
        i = next_z
        next_z += 2
        p["Wz"][i] = v[:96]
        p["bz"][i] = v[96]
        p["Wz"][i + 1] = v[:96]
        p["bz"][i + 1] = v[96] - 1
        return i

    # Horizontal three-target mode and side-target mode.
    count = expr(-20, {64: 4096, 13: -100})
    mode = indicator(count + 0.1 * (srcymin - tymin - 20 * one))
    mixed = indicator(expr(-1, {13: 2}))

    long_bar = indicator(width - 5.5 * height)
    medium_bar = indicator(width - 4.5 * height)
    near_right_source = indicator(sxmax - srcxmax + 0.5 * height)

    tolerance = 0.5 * height
    errors = [
        (63 * one - sxmax,
         srcymin - 4 * height - symin),

        (-sxmin,
         srcymin - 4 * height - symin),

        (txmax - sxmax - (height - one),
         tymin + 2 * height - symin),

        (txmin - sxmin + (height - one),
         tymin + 2 * height - symin),
    ]

    route_tests = []
    for dx, dy in errors:
        route_tests.append([
            indicator(dx - tolerance),
            indicator(-dx - tolerance),
            indicator(dy - tolerance),
            indicator(-dy - tolerance),
        ])

    horizontal_error = expr(
        terms={0: 63, 1: -63, 2: -63, 3: 63}
    )
    vertical_error = expr(
        terms={14: 63, 15: -63, 4: -63, 5: 63}
    )
    horizontal_shape = indicator(width - height - one)
    vertical_shape = indicator(height - width - one)
    baseline_tests = [
        indicator(horizontal_error - 1.05 * height),
        indicator(-horizontal_error - 1.05 * height),
        indicator(vertical_error - 1.05 * width),
        indicator(-vertical_error - 1.05 * width),
    ]

    # Size classes for the mixed horizontal/side-target arrangement.
    mixed_long = indicator(width - 4.5 * height)
    mixed_medium = indicator(width - 3.5 * height)
    mixed_small = indicator(width - 2.5 * height)

    mixed_errors = [
        # Long bar: park against the side wall below the side target.
        (side_xmin - height - sxmin,
         tymin + 4 * height - symin,
         0.5 * height),

        # Four-cell bar: park away from the source streams.
        (64 * one - side_xmin + height - sxmax,
         srcymin - 2 * height - symin,
         0.5 * height),

        # Three-cell bar: split the other source toward two targets.
        (srcxmax - 2 * height + one - sxmin,
         srcymin - 3 * height - symin,
         0.5 * height),

        # Deflector: turn the side source toward the side target.
        (srcxmin - sxmin,
         side_ymin - 0.5 * height - symin,
         0.25 * height),
    ]

    mixed_tests = []
    for dx, dy, tol in mixed_errors:
        mixed_tests.append([
            indicator(dx - tol),
            indicator(-dx - tol),
            indicator(dy - tol),
            indicator(-dy - tol),
        ])

    next_mem = 0

    def conjunction(conditions, required_flags=()):
        nonlocal next_mem
        i = next_mem
        next_mem += 1
        true_count = 0
        for z_index, wanted in conditions:
            sign = 1 if wanted else -1
            if wanted:
                true_count += 1
            p["Wmz"][i, z_index] += 10 * sign
            p["Wmz"][i, z_index + 1] -= 10 * sign
        p["bm"][i] = 10 * (0.5 - true_count)
        for flag in required_flags:
            p["Wm"][i, flag] += 10
            p["bm"][i] -= 10
        return i

    def add_vote(action, memory_index, strength=4):
        p["Wo"][action, 128 + memory_index] += strength / 2
        p["bo"][action] += strength / 2

    def completion_flags(aligned_units):
        nonlocal next_mem
        flags = []
        for unit in aligned_units:
            i = next_mem
            next_mem += 1
            flags.append(i)
            p["Wm"][i, i] = 5
            p["Wm"][i, unit] = 1.5
            p["Wma"][i, 6] = 6
            p["Wma"][i, 0] = -20
            p["bm"][i] = -1
        return flags

    actions = [4, 3, 2, 1]

    roles = [
        [(long_bar, True), (near_right_source, True)],
        [(long_bar, True), (near_right_source, False)],
        [(long_bar, False), (medium_bar, True)],
        [(long_bar, False), (medium_bar, False)],
    ]

    aligned_units = []
    for role, tests in zip(roles, route_tests):
        base = [(mode, True)] + role
        for action, test in zip(actions, tests):
            add_vote(action, conjunction(base + [(test, True)]))
        aligned = base + [(test, False) for test in tests]
        unit = conjunction(aligned)
        aligned_units.append(unit)
        add_vote(6, unit)

    flags = completion_flags(aligned_units)
    for j, (role, tests) in enumerate(zip(roles, route_tests)):
        aligned = [(mode, True)] + role + [
            (test, False) for test in tests
        ]
        others = [flags[k] for k in range(4) if k != j]
        add_vote(5, conjunction(aligned, others), strength=6)

    baseline_base = [(mode, False), (mixed, False)]
    for action, test, shape in zip(
        actions,
        baseline_tests,
        [horizontal_shape, horizontal_shape,
         vertical_shape, vertical_shape],
    ):
        add_vote(
            action,
            conjunction(baseline_base + [(shape, True), (test, True)]),
        )

    baseline_done = baseline_base + [
        (test, False) for test in baseline_tests
    ]
    add_vote(5, conjunction(baseline_done))

    mixed_roles = [
        [(mixed_long, True)],
        [(mixed_long, False), (mixed_medium, True)],
        [(mixed_medium, False), (mixed_small, True)],
        [(mixed_small, False)],
    ]

    mixed_aligned = []
    for role_index, (role, tests) in enumerate(
        zip(mixed_roles, mixed_tests)
    ):
        base = [(mixed, True)] + role
        for action_index, (action, test) in enumerate(zip(actions, tests)):
            conditions = base + [(test, True)]
            # Move the four-cell bar horizontally before moving it down.
            if role_index == 1 and action_index >= 2:
                conditions += [(tests[0], False), (tests[1], False)]
            add_vote(action, conjunction(conditions))

        aligned = base + [(test, False) for test in tests]
        unit = conjunction(aligned)
        mixed_aligned.append(unit)
        add_vote(6, unit)

    mixed_flags = completion_flags(mixed_aligned)
    for j, (role, tests) in enumerate(zip(mixed_roles, mixed_tests)):
        aligned = [(mixed, True)] + role + [
            (test, False) for test in tests
        ]
        others = [mixed_flags[k] for k in range(4) if k != j]
        add_vote(5, conjunction(aligned, others), strength=6)

    # A spill that returns to the same episode failed: restart the level.
    p["bo"][0] = -20
    p["bo"][7] = -20
    reset_unit = next_mem
    next_mem += 1
    p["Wma"][reset_unit, 5] = 5
    p["bm"][reset_unit] = -2.5
    add_vote(0, reset_unit, strength=30)

    p["Wc"][0, 5] = 20
    p["Wc"][0, 3] = 1
    p["Wc"][0, 4] = -0.001

    return p
