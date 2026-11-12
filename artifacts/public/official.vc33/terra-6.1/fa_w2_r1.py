import numpy as np


def build():
    graph = []
    params = {}
    seq = [0]

    def fresh(prefix="n"):
        seq[0] += 1
        return "%s%d" % (prefix, seq[0])

    def parameter(value):
        key = fresh("p")
        params[key] = np.asarray(value, dtype=np.float32)
        return key

    def node(op, inputs=None, **fields):
        key = fresh()
        item = {"id": key, "op": op}
        if inputs is not None:
            item["in"] = inputs
        item.update(fields)
        graph.append(item)
        return key

    def conv(x, weights, bias=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node("conv", [x], **fields)

    def unary(op, x):
        return node(op, [x])

    def binary(op, x, y):
        return node(op, [x, y])

    def spatial_slice(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vector_slice(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def add(x, y):
        return binary("add", x, y)

    def sub(x, y):
        return binary("sub", x, y)

    def mul(x, y):
        return binary("mul", x, y)

    def maximum(x, y):
        return binary("max2", x, y)

    def minimum(x, y):
        return binary("min2", x, y)

    def vadd(x, y):
        return binary("vadd", x, y)

    def vmul(x, y):
        return binary("vmul", x, y)

    def relu(x):
        return unary("relu", x)

    def vrelu(x):
        return unary("vrelu", x)

    def gmax(x):
        return unary("gmax", x)

    def gmean(x):
        return unary("gmean", x)

    def affine(x, scale, bias=0.0):
        return conv(x, [[[[scale]]]], [bias])

    def vaffine(x, scale, bias=0.0):
        return node(
            "dense", [x],
            w=parameter([[scale]]),
            b=parameter([bias]),
        )

    def vabs(x):
        return vadd(vrelu(x), vrelu(vaffine(x, -1.0)))

    row = spatial_slice("coords", 0, 1)
    col = spatial_slice("coords", 1, 2)
    one = spatial_slice("coords", 2, 3)
    vector_one = gmax(one)

    def cap(x):
        return minimum(relu(x), one)

    def vcap(x):
        excess = vrelu(vadd(x, vaffine(vector_one, -1.0)))
        return vadd(x, vaffine(excess, -1.0))

    # Exclude the interface's step-counter row.
    valid = cap(affine(row, 63.0))

    def colour(c):
        return mul(spatial_slice("x0", c, c + 1), valid)

    c0 = colour(0)
    c1 = colour(1)
    c4 = colour(4)
    c5 = colour(5)
    c9 = colour(9)
    c11 = colour(11)
    c12 = colour(12)
    c14 = colour(14)
    colours = add(c11, c14)

    # The coloured leading edge and the adjacent colour-4 body
    # identify both the piece and the gravity direction.
    direction_weights = np.zeros((4, 1, 7, 7), dtype=np.float32)
    for d in (1, 2, 3):
        direction_weights[0, 0, 3, 3 - d] = 1
        direction_weights[1, 0, 3, 3 + d] = 1
        direction_weights[2, 0, 3 - d, 3] = 1
        direction_weights[3, 0, 3 + d, 3] = 1

    near_directions = conv(c4, direction_weights)
    flags = []
    for i in range(4):
        near = cap(spatial_slice(near_directions, i, i + 1))
        flags.append(gmax(minimum(colours, near)))

    horizontal = vadd(flags[0], flags[1])
    vertical = vadd(flags[2], flags[3])
    invrow = sub(one, row)
    invcol = sub(one, col)

    gravity = add(
        add(mul(col, flags[0]), mul(invcol, flags[1])),
        add(mul(row, flags[2]), mul(invrow, flags[3])),
    )
    perpendicular = add(mul(row, horizontal), mul(col, vertical))

    cross = np.zeros((1, 1, 7, 7), dtype=np.float32)
    for d in range(-3, 4):
        cross[0, 0, 3, 3 + d] = 1
        cross[0, 0, 3 + d, 3] = 1
    avatar = minimum(colours, cap(conv(c4, cross)))

    has11 = gmax(mul(avatar, c11))
    has14 = gmax(mul(avatar, c14))
    matching = add(mul(c11, has11), mul(c14, has14))
    avatar_near = cap(
        conv(avatar, np.ones((1, 1, 7, 7), dtype=np.float32))
    )
    marker = relu(sub(matching, avatar_near))

    avatar_g = gmax(mul(avatar, gravity))
    avatar_p = gmax(mul(avatar, perpendicular))
    marker_g = gmax(mul(marker, gravity))
    marker_p = gmax(mul(marker, perpendicular))

    def axis(x):
        return add(
            mul(unary("rowmax", x), horizontal),
            mul(unary("colmax", x), vertical),
        )

    # Project long black walls onto the perpendicular axis.
    wall_weights = np.zeros((2, 1, 7, 7), dtype=np.float32)
    wall_weights[0, 0, 3, 1:6] = 1
    wall_weights[1, 0, 1:6, 3] = 1
    runs = conv(c5, wall_weights, [-4.5, -4.5])
    hwall = cap(affine(spatial_slice(runs, 0, 1), 2.0))
    vwall = cap(affine(spatial_slice(runs, 1, 2), 2.0))
    barrier = add(
        mul(unary("rowmax", hwall), horizontal),
        mul(unary("colmax", vwall), vertical),
    )
    free = relu(sub(one, barrier))

    # Flood a projected chamber without crossing walls.
    flood_weights = np.zeros((1, 1, 3, 3), dtype=np.float32)
    flood_weights[0, 0, 1, 1] = 1
    flood_weights[0, 0, 0, 1] = 1
    flood_weights[0, 0, 2, 1] = 1
    flood_weights[0, 0, 1, 0] = 1
    flood_weights[0, 0, 1, 2] = 1
    flood_param = parameter(flood_weights)

    def flood(seed, allowed):
        loop_id = fresh("loop")
        state_id = fresh("state")
        body_conv = fresh("body")
        body_next = fresh("body")
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": 64,
            "state": [{"id": state_id, "init": seed}],
            "body": [
                {
                    "id": body_conv,
                    "op": "conv",
                    "in": [state_id],
                    "w": flood_param,
                },
                {
                    "id": body_next,
                    "op": "min2",
                    "in": [body_conv, allowed],
                },
            ],
            "next": {state_id: body_next},
        })
        return loop_id + "." + state_id

    own_room = flood(mul(axis(avatar), free), free)

    # Select the nearest gate between the avatar and matching marker.
    gates = add(c1, c12)
    after_avatar = sub(perpendicular, avatar_p)
    before_marker = sub(mul(one, marker_p), perpendicular)
    before_avatar = sub(mul(one, avatar_p), perpendicular)
    after_marker = sub(perpendicular, marker_p)
    margin = 0.5 / 63.0
    forward = minimum(
        after_avatar, affine(before_marker, 1.0, -margin)
    )
    backward = minimum(
        before_avatar, affine(after_marker, 1.0, -margin)
    )
    between = cap(
        affine(relu(maximum(forward, backward)), 126.0)
    )
    distance_from_avatar = unary("abs", after_avatar)
    gate_score = mul(
        mul(gates, between),
        affine(distance_from_avatar, -1.0, 2.0),
    )
    best_gate = gmax(gate_score)
    selected_gate = cap(
        affine(sub(gate_score, best_gate), 1000.0, 1.0)
    )
    selected_gate = mul(selected_gate, mul(gates, between))
    gate_present = vcap(best_gate)
    selected_gate = mul(selected_gate, gate_present)

    gate_g = gmax(mul(selected_gate, gravity))
    no_gate = vaffine(gate_present, -1.0, 1.0)
    target_g = vadd(
        vmul(gate_g, gate_present),
        vmul(marker_g, no_gate),
    )

    # Find the chamber adjacent to the opposite side of the gate.
    gate_axis = axis(selected_gate)
    square7 = np.ones((1, 1, 7, 7), dtype=np.float32)
    gate_near = cap(conv(cap(conv(gate_axis, square7)), square7))
    other_seed = mul(mul(gate_near, free), sub(one, own_room))
    other_room = flood(other_seed, free)

    # Estimate the other chamber's surface from its upstream water edge.
    upstream = gmax(
        mul(mul(c0, other_room), sub(one, gravity))
    )
    other_edge = vaffine(upstream, -1.0, 1.0 - 1.0 / 63.0)

    own_delta = vadd(target_g, vaffine(avatar_g, -1.0))
    other_delta = vadd(target_g, vaffine(other_edge, -1.0))
    own_needed = vcap(
        vrelu(vaffine(vabs(own_delta), 63.0, -0.25))
    )
    choose_other = vmul(
        gate_present, vaffine(own_needed, -1.0, 1.0)
    )
    choose_own = vaffine(choose_other, -1.0, 1.0)
    target_room = add(
        mul(own_room, choose_own),
        mul(other_room, choose_other),
    )
    delta = vadd(
        vmul(own_delta, choose_own),
        vmul(other_delta, choose_other),
    )

    # The centre of the target chamber gives a coordinate-independent
    # way to choose the correct side of each valve.
    room_max = gmax(mul(target_room, perpendicular))
    room_inv_max = gmax(
        mul(target_room, sub(one, perpendicular))
    )
    room_min = vaffine(room_inv_max, -1.0, 1.0)
    target_p = vaffine(vadd(room_min, room_max), 0.5)

    filling = vcap(vrelu(vaffine(delta, -63.0, -0.25)))
    draining = vaffine(filling, -1.0, 1.0)

    # Identify valve pixels on the low/high side of a wall.
    valve_axis = axis(c9)
    adjacent_weights = np.zeros((4, 1, 3, 3), dtype=np.float32)
    adjacent_weights[0, 0, 2, 1] = 1
    adjacent_weights[1, 0, 0, 1] = 1
    adjacent_weights[2, 0, 1, 2] = 1
    adjacent_weights[3, 0, 1, 0] = 1
    adjacent = conv(barrier, adjacent_weights)
    wall_after = add(
        mul(spatial_slice(adjacent, 0, 1), horizontal),
        mul(spatial_slice(adjacent, 2, 3), vertical),
    )
    wall_before = add(
        mul(spatial_slice(adjacent, 1, 2), horizontal),
        mul(spatial_slice(adjacent, 3, 4), vertical),
    )
    low_endpoint = mul(valve_axis, wall_after)
    high_endpoint = mul(valve_axis, wall_before)

    left_of_target = cap(
        affine(relu(sub(mul(one, target_p), perpendicular)), 63.0)
    )
    right_of_target = cap(
        affine(relu(sub(perpendicular, target_p)), 63.0)
    )

    fill_sides = add(
        mul(left_of_target, high_endpoint),
        mul(right_of_target, low_endpoint),
    )
    drain_sides = add(
        mul(left_of_target, low_endpoint),
        mul(right_of_target, high_endpoint),
    )
    desired_side = cap(add(
        mul(fill_sides, filling),
        mul(drain_sides, draining),
    ))

    # An enabled gate bordering the avatar's chamber takes priority.
    near_room = cap(
        conv(own_room, np.ones((1, 1, 3, 3), dtype=np.float32))
    )
    ready_gate = mul(mul(selected_gate, c12), near_room)

    # Detect a no-op using two moments of the water image.
    moment_p = gmean(mul(c0, perpendicular))
    moment_g = gmean(mul(c0, gravity))
    old_p = vector_slice("mem", 0, 1)
    old_g = vector_slice("mem", 1, 2)
    old_r = vector_slice("mem", 2, 3)
    old_c = vector_slice("mem", 3, 4)
    old_valid = vector_slice("mem", 4, 5)
    change = vadd(
        vabs(vadd(moment_p, vaffine(old_p, -1.0))),
        vabs(vadd(moment_g, vaffine(old_g, -1.0))),
    )
    unchanged = vmul(
        vrelu(vaffine(change, -1000000.0, 1.0)),
        old_valid,
    )
    old_distance = maximum(
        unary("abs", sub(row, old_r)),
        unary("abs", sub(col, old_c)),
    )
    old_area = cap(
        affine(old_distance, -1000.0, 1000.0 * 9.5 / 63.0)
    )
    penalty = mul(old_area, unchanged)

    # Prefer the nearest valve that transfers in the desired direction.
    target_distance = unary(
        "abs", sub(perpendicular, target_p)
    )
    directional_value = add(
        affine(target_distance, -2.0, 20.0),
        affine(penalty, -25.0),
    )
    directional_valve = mul(mul(c9, desired_side), directional_value)

    # If no directional endpoint can be selected, try another valve.
    fallback_value = add(
        affine(target_distance, -0.4, 1.0),
        affine(penalty, -2.0),
    )
    fallback = mul(c9, relu(fallback_value))
    gate_value = mul(
        ready_gate, affine(penalty, -30.0, 40.0)
    )
    score = maximum(maximum(directional_valve, fallback), gate_value)
    graph.append({"id": "click", "op": "relu", "in": [score]})

    # Save the selected region and image moments for no-op detection.
    best_click = gmax("click")
    picked = cap(
        affine(sub("click", best_click), 1000.0, 1.0)
    )
    picked_r = gmax(mul(picked, row))
    picked_c = gmax(mul(picked, col))
    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": [moment_p, moment_g, picked_r, picked_c, vector_one],
    })
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [vector_one],
        "w": parameter(np.zeros((8, 1), dtype=np.float32)),
        "b": parameter([0, 0, 0, 0, 0, 0, 1, 0]),
    })

    return {"graph": graph, "memory_size": 5, "params": params}
