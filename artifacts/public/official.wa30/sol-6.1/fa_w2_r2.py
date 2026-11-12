import numpy as np


def build():
    graph, params = [], {}
    serial, pserial = 0, 0

    def fresh():
        nonlocal serial
        result = "n" + str(serial)
        serial += 1
        return result

    def parameter(value):
        nonlocal pserial
        result = "p" + str(pserial)
        pserial += 1
        params[result] = np.asarray(value, np.float32)
        return result

    def emit(op, inputs, out=None, body=None, **fields):
        result = fresh() if out is None else out
        nd = {"id": result, "op": op, "in": inputs}
        nd.update(fields)
        (graph if body is None else body).append(nd)
        return result

    def conv(x, w, b=None, out=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("conv", [x], out=out, **fields)

    def dense(x, w, b=None, out=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("dense", [x], out=out, **fields)

    def sl(x, a, b):
        return emit("slice", [x], start=a, stop=b)

    def cat(xs):
        return emit("concat", xs)

    def vcat(xs):
        return emit("vconcat", xs)

    def bi(op, a, b):
        return emit(op, [a, b])

    def reduce(op, xs):
        value = xs[0]
        for x in xs[1:]:
            value = bi(op, value, x)
        return value

    def maximum(xs):
        return reduce("max2", xs)

    def total(xs):
        return reduce("add", xs)

    def affine(x, multiplier, bias=0):
        return conv(x, np.array([[[[multiplier]]]], np.float32), [bias])

    def gm(x):
        return emit("gmax", [x])

    # Constant spatial plane.
    w = np.zeros((1, 3, 1, 1), np.float32)
    w[0, 2, 0, 0] = 1
    one = conv("coords", w)

    def comp(x):
        return bi("sub", one, x)

    # Individual row/column lattice indicators.
    w = np.zeros((96, 3, 1, 1), np.float32)
    bias = np.zeros(96, np.float32)
    k = 0
    for axis in (0, 1):
        for position in range(0, 64, 4):
            for offset in (-1, 0, 1):
                w[k, axis, 0, 0] = 63
                bias[k] = -(position + offset)
                k += 1
    hats = emit("relu", [conv("coords", w, bias)])

    w = np.zeros((32, 96, 1, 1), np.float32)
    for i in range(32):
        w[i, 3 * i:3 * i + 3, 0, 0] = (4, -8, 4)
    axes = emit("relu", [
        conv(hats, w, -2 * np.ones(32, np.float32))
    ])
    axes = bi("min2", axes, one)

    w = np.zeros((2, 32, 1, 1), np.float32)
    w[0, :16, 0, 0] = 1
    w[1, 16:, 0, 0] = 1
    lattice = conv(axes, w)
    anchor = bi("min2", sl(lattice, 0, 1), sl(lattice, 1, 2))

    # Tile basis for reading and writing the 256-entry memory.
    w = np.zeros((256, 32, 1, 1), np.float32)
    for row in range(16):
        for col in range(16):
            i = 16 * row + col
            w[i, row, 0, 0] = 1
            w[i, 16 + col, 0, 0] = 1
    tile_basis = emit("relu", [
        conv(axes, w, -np.ones(256, np.float32))
    ])

    memory_tiles = bi("mul", tile_basis, "mem")
    memory_code = conv(
        memory_tiles, np.ones((1, 256, 1, 1), np.float32)
    )

    # Decode six integer bits using piecewise-linear hinge functions.
    thresholds = np.arange(65, dtype=np.float32) - 1
    hinges = emit("relu", [
        conv(
            memory_code,
            np.ones((65, 1, 1, 1), np.float32),
            -thresholds,
        )
    ])
    decode_w = np.zeros((6, 65, 1, 1), np.float32)
    for bit in range(6):
        span = 1 << bit
        for start in range(span, 64, 2 * span):
            for threshold, coefficient in (
                (start - 1, 1),
                (start, -1),
                (start + span - 1, -1),
                (start + span, 1),
            ):
                if -1 <= threshold <= 63:
                    decode_w[bit, threshold + 1, 0, 0] += coefficient
    remembered = conv(hinges, decode_w)
    remembered_goal = sl(remembered, 0, 1)
    remembered_hazard = sl(remembered, 1, 2)
    remembered_failures = [
        sl(remembered, i + 2, i + 3) for i in range(4)
    ]

    # Sprite detectors. Most use only the first three tile rows,
    # avoiding the step-counter overlay at the bottom of the screen.
    names = [
        "player", "box", "wall", "helper", "enemy", "hazard",
        "helper_border", "visible_goal", "up", "left", "right",
        "attached_border",
    ]
    ix = {name: i for i, name in enumerate(names)}
    w = np.zeros((len(names), 16, 7, 7), np.float32)
    bias = np.zeros(len(names), np.float32)

    def put(name, colour, row, col, value):
        w[ix[name], colour, 3 + row, 3 + col] += value

    for row in range(3):
        for col in range(4):
            put("player", 14, row, col, 2)
            put("wall", 5, row, col, 2)
            put("visible_goal", 2, row, col, 2)
            put("visible_goal", 9, row, col, 2)
    bias[ix["player"]] = -15
    bias[ix["wall"]] = -23
    bias[ix["visible_goal"]] = -1

    for row in (1, 2):
        for col in (1, 2):
            put("box", 9, row, col, 2)
    bias[ix["box"]] = -7

    put("helper", 12, 1, 1, 1)
    put("enemy", 15, 1, 1, 1)

    put("hazard", 2, 0, 0, 1)
    put("hazard", 2, 1, 1, 1)
    put("hazard", 2, 0, 1, -1)
    bias[ix["hazard"]] = -1

    border = [(0, col) for col in range(4)]
    border += [(row, col) for row in (1, 2) for col in (0, 3)]
    for row, col in border:
        put("helper_border", 5, row, col, 2)
        put("attached_border", 0, row, col, 2)
    bias[ix["helper_border"]] = -15
    bias[ix["attached_border"]] = -15

    for col in range(4):
        put("up", 0, 0, col, 2)
    bias[ix["up"]] = -7

    for row in range(3):
        put("left", 0, row, 0, 2)
        put("right", 0, row, 3, 2)
    bias[ix["left"]] = -5
    bias[ix["right"]] = -5

    detected = emit("relu", [conv("x0", w, bias)])
    detected = bi("min2", detected, one)
    detected = bi("mul", detected, anchor)
    d = {
        name: sl(detected, i, i + 1)
        for i, name in enumerate(names)
    }

    player, box, wall = d["player"], d["box"], d["wall"]
    helper, enemy = d["helper"], d["enemy"]
    hazard = bi("max2", d["hazard"], remembered_hazard)
    hazard = bi("mul", hazard, anchor)

    attached = bi("mul", box, d["attached_border"])
    helper_box = bi("mul", box, d["helper_border"])
    carry = gm(attached)

    up = bi("mul", player, d["up"])
    left = bi("mul", player, d["left"])
    right = bi("mul", player, d["right"])
    down = emit("relu", [
        bi("sub", player, maximum([up, left, right]))
    ])
    faces = [up, down, left, right]
    face_vector = gm(cat(faces))

    # Shared cardinal shifts by exactly four pixels.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    sw2 = np.zeros((4, 1, 5, 5), np.float32)
    sw4 = np.zeros((4, 4, 5, 5), np.float32)
    for i, (dy, dx) in enumerate(directions):
        sw2[i, 0, 2 + 2 * dy, 2 + 2 * dx] = 1
        sw4[i, i, 2 + 2 * dy, 2 + 2 * dx] = 1
    half_w, full_w = parameter(sw2), parameter(sw4)

    def shifts(x):
        half = emit("conv", [x], w=half_w)
        full = emit("conv", [half], w=full_w)
        return [sl(full, i, i + 1) for i in range(4)]

    def wave(seed, passable, iterations=96):
        state, loop = fresh(), fresh()
        body = []

        def inside(op, inputs, **fields):
            return emit(op, inputs, body=body, **fields)

        half = inside("conv", [state], w=half_w)
        full = inside("conv", [half], w=full_w)
        neighbours = [
            inside("slice", [full], start=i, stop=i + 1)
            for i in range(4)
        ]
        value = neighbours[0]
        for x in neighbours[1:]:
            value = inside("max2", [value, x])
        value = inside("sub", [value, one])
        value = inside("max2", [value, seed])
        value = inside("mul", [value, passable])
        value = inside("relu", [value])
        graph.append({
            "id": loop,
            "op": "loop",
            "T": iterations,
            "state": [{"id": state, "init": seed}],
            "body": body,
            "next": {state: value},
        })
        return loop + "." + state

    # Remove box interiors before reconstructing goal boundaries.
    w = np.zeros((1, 1, 7, 7), np.float32)
    w[0, 0, :4, :4] = 1
    box_pixels = conv(box, w)
    edges = emit("relu", [
        bi("sub", sl("x0", 9, 10), box_pixels)
    ])

    initial = cat([edges] * 4)
    pw = np.zeros((4, 4, 3, 3), np.float32)
    for i, (dy, dx) in enumerate(directions):
        pw[i, i, 1 + dy, 1 + dx] = 1
    prefix_w = parameter(pw)
    state, loop = fresh(), fresh()
    body = []
    shifted = emit("conv", [state], body=body, w=prefix_w)
    updated = emit("max2", [state, shifted], body=body)
    graph.append({
        "id": loop,
        "op": "loop",
        "T": 63,
        "state": [{"id": state, "init": initial}],
        "body": body,
        "next": {state: updated},
    })
    result = loop + "." + state
    enclosed = reduce(
        "min2", [sl(result, i, i + 1) for i in range(4)]
    )
    projection = bi(
        "min2",
        emit("rowmax", [edges]),
        emit("colmax", [edges]),
    )

    # Local evidence is mandatory for occluded-goal inference.
    # Two 7x7 dilations cover a six-pixel neighbourhood.
    near_edges = conv(edges, np.ones((1, 1, 7, 7), np.float32))
    near_edges = bi("min2", near_edges, one)
    near_edges = conv(
        near_edges, np.ones((1, 1, 7, 7), np.float32)
    )
    near_edges = bi("min2", near_edges, one)
    local_projection = bi("mul", projection, near_edges)
    occluded = bi("mul", local_projection, box)

    bw = np.zeros((1, 3, 1, 1), np.float32)
    bw[0, 0, 0, 0] = 126
    bottom = emit("relu", [conv("coords", bw, [-119])])
    bottom = bi("min2", bottom, one)
    bottom_goal = bi("mul", local_projection, bottom)

    visible_goal = maximum([enclosed, occluded, bottom_goal])
    visible_goal = bi("mul", visible_goal, d["visible_goal"])
    visible_goal = bi("mul", visible_goal, comp(hazard))
    visible_goal = bi("mul", visible_goal, anchor)
    goal = bi("max2", visible_goal, remembered_goal)

    # Helper-accessible barrier cells are valid handoff destinations.
    helper_obstacles = maximum([wall, hazard, enemy, box])
    helper_pass = bi("mul", anchor, comp(helper_obstacles))
    helper_reach = wave(affine(helper, 128), helper_pass)
    adjacent_reach = bi(
        "min2", maximum(shifts(helper_reach)), one
    )
    stage = bi("mul", hazard, adjacent_reach)

    # Four transport fields, one per fixed attachment orientation.
    loose_box = bi("sub", box, attached)
    strict_boxes = bi("mul", loose_box, carry)
    physical = maximum([wall, helper, enemy, strict_boxes])
    box_free = bi("mul", anchor, comp(physical))
    carrier_free = bi(
        "mul", anchor, comp(bi("max2", physical, hazard))
    )
    box_free_shift = shifts(box_free)
    goal_shift = shifts(goal)
    stage_shift = shifts(stage)

    transport = []
    for i in range(4):
        passable = bi("mul", carrier_free, box_free_shift[i])
        seed = bi(
            "max2",
            affine(goal_shift[i], 128),
            affine(stage_shift[i], 64),
        )
        seed = bi("mul", seed, passable)
        transport.append(wave(seed, passable))

    # Navigation targets useful pickups and removable adversaries.
    nav_obstacles = maximum([wall, hazard, helper, enemy, box])
    nav_pass = bi("mul", anchor, comp(nav_obstacles))
    helper_adjacent = maximum(shifts(helper))
    unavailable = maximum(
        [goal, helper_box, stage, helper_adjacent]
    )
    eligible = bi("mul", box, comp(unavailable))
    eligible_shift = shifts(eligible)
    enemy_shift = shifts(enemy)
    failure_shift = [shifts(x) for x in remembered_failures]

    targets = []
    for i in range(4):
        score = bi("mul", transport[i], eligible_shift[i])
        score = bi("mul", score, comp(failure_shift[i][i]))
        score = bi("max2", score, affine(enemy_shift[i], 256))
        targets.append(bi("mul", score, nav_pass))

    navigation = wave(maximum(targets), nav_pass)
    nav_neighbours = shifts(navigation)
    nav_valid = shifts(nav_pass)

    free_dirs = []
    for i in range(4):
        positive = bi("min2", targets[i], one)
        turn = bi("add", targets[i], affine(positive, 0.1))
        value = bi("max2", nav_neighbours[i], turn)
        free_dirs.append(
            bi("add", value, affine(nav_valid[i], 0.005))
        )
    free_direction_vector = gm(
        bi("mul", cat(free_dirs), player)
    )

    # Penalize an immediate reverse only enough to resolve close ties.
    reversal_w = np.zeros((4, 8), np.float32)
    reversal_w[0, 2] = -0.25
    reversal_w[1, 1] = -0.25
    reversal_w[2, 4] = -0.25
    reversal_w[3, 3] = -0.25
    reversal = dense("aprev", reversal_w)
    free_direction_vector = emit(
        "vadd", [free_direction_vector, reversal]
    )

    front_target = total([
        bi("mul", targets[i], faces[i]) for i in range(4)
    ])
    front_positive = bi("min2", front_target, one)
    free_five_map = bi(
        "add", front_target, affine(front_positive, 0.2)
    )

    entities_shift = shifts(bi("max2", box, enemy))
    front_any = total([
        bi("mul", entities_shift[i], faces[i])
        for i in range(4)
    ])
    free_five_map = bi(
        "add", free_five_map, affine(comp(front_any), 0.01)
    )
    free_five = gm(bi("mul", free_five_map, player))
    free_scores = vcat([free_direction_vector, free_five])

    # Attached actions preserve the orientation selected at pickup.
    transport_neighbours = [shifts(x) for x in transport]
    held_dirs = []
    for movement in range(4):
        held_dirs.append(total([
            bi(
                "mul",
                transport_neighbours[i][movement],
                faces[i],
            )
            for i in range(4)
        ]))
    held_direction_vector = gm(cat(held_dirs))

    delivered = gm(
        bi("mul", bi("max2", goal, stage), attached)
    )
    current_transport = total([
        bi("mul", transport[i], faces[i]) for i in range(4)
    ])
    possible = gm(bi("min2", current_transport, one))

    complement_w = parameter([[-1]])
    complement_b = parameter([1])

    def vcomp(x):
        return emit(
            "dense", [x], w=complement_w, b=complement_b
        )

    failure = emit("vmul", [carry, vcomp(delivered)])
    failure = emit("vmul", [failure, vcomp(possible)])
    release = emit("vadd", [delivered, failure])
    held_five = dense(release, [[1000]], [0.01])
    held_scores = vcat([held_direction_vector, held_five])

    free_scores = emit("vmul", [free_scores, vcomp(carry)])
    held_scores = emit("vmul", [held_scores, carry])
    scores = emit("vadd", [free_scores, held_scores])

    ow = np.zeros((8, 5), np.float32)
    for i in range(5):
        ow[i + 1, i] = 1
    ob = np.zeros(8, np.float32)
    ob[[0, 6, 7]] = -10000
    dense(scores, ow, ob, out="logits")

    # Memory bits:
    # 0 = goal, 1 = barrier, 2..5 = failed pickup orientations.
    new_failure_bits = []
    for i in range(4):
        orientation = emit(
            "vslice", [face_vector], start=i, stop=i + 1
        )
        failed_here = bi("mul", attached, failure)
        failed_here = bi("mul", failed_here, orientation)
        old_here = bi("mul", remembered_failures[i], box)
        new_failure_bits.append(
            bi("max2", old_here, failed_here)
        )

    memory_features = cat(
        [goal, hazard] + new_failure_bits
    )
    encode_w = np.array(
        [1, 2, 4, 8, 16, 32], np.float32
    ).reshape(1, 6, 1, 1)
    next_code = conv(memory_features, encode_w)
    next_tiles = bi("mul", tile_basis, next_code)
    emit("gmax", [next_tiles], out="mem_next")

    # Required output; click actions are unavailable.
    conv(
        one, np.zeros((1, 1, 1, 1), np.float32),
        out="click",
    )

    return {
        "graph": graph,
        "memory_size": 256,
        "params": params,
    }
