import numpy as np


def build():
    graph = []
    params = {}
    channels = {"x0": 16, "coords": 3, "avail": 8, "aprev": 8}
    serial = [0]
    kernel_cache = {}

    def fresh(prefix="n"):
        serial[0] += 1
        return "%s_%d" % (prefix, serial[0])

    def parameter(a):
        name = fresh("p")
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, inputs, c, where=None, name=None, **fields):
        name = name or fresh()
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        (graph if where is None else where).append(nd)
        channels[name] = c
        return name

    def cat(xs, where=None):
        if len(xs) == 1:
            return xs[0]
        return node("concat", xs, sum(channels[x] for x in xs), where)

    def sl(x, start, stop, where=None):
        return node("slice", [x], stop - start, where,
                    start=start, stop=stop)

    def unary(op, x, where=None):
        return node(op, [x], channels[x], where)

    def binary(op, x, y, where=None):
        return node(op, [x, y], max(channels[x], channels[y]), where)

    def affine(x, matrix, bias=None):
        a = np.asarray(matrix, np.float32)
        if a.ndim == 1:
            a = a[None, :]
        assert a.shape[1] == channels[x]
        fields = {"w": parameter(a[:, :, None, None])}
        if bias is not None:
            b = np.asarray(bias, np.float32)
            if b.ndim == 0:
                b = np.full(a.shape[0], b, np.float32)
            fields["b"] = parameter(b)
        return node("conv", [x], a.shape[0], **fields)

    def scale(x, value, bias=0.0):
        return affine(x, np.eye(channels[x], dtype=np.float32) * value,
                      np.full(channels[x], bias, np.float32))

    def combine(xs, coefficients, bias=0.0):
        return affine(cat(xs), coefficients, bias)

    one = sl("coords", 2, 3)

    def constant(value, c=1):
        return affine(one, np.zeros((c, 1), np.float32),
                      np.full(c, value, np.float32))

    def clip01(x):
        return binary("min2", unary("relu", x), one)

    def positive(x):
        # An integer-scale, noise-tolerant test for x > 0.
        return clip01(scale(x, 2.0, -0.5))

    def equal_zero(x):
        return clip01(scale(unary("abs", x), -4.0, 2.0))

    def complement(x):
        return scale(x, -1.0, 1.0)

    def shift(x, dy, dx):
        c = channels[x]
        k = 2 * max(abs(dy), abs(dx)) + 1
        key = ("shift", c, dy, dx)
        if key not in kernel_cache:
            w = np.zeros((c, c, k, k), np.float32)
            for j in range(c):
                w[j, j, k // 2 + dy, k // 2 + dx] = 1.0
            kernel_cache[key] = parameter(w)
        return node("conv", [x], c, w=kernel_cache[key])

    def vector_slice(x, a, b):
        return node("vslice", [x], b - a, start=a, stop=b)

    def dense(x, w, b=None):
        w = np.asarray(w, np.float32)
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], w.shape[0], **fields)

    # Colour roles.
    black = sl("x0", 0, 1)
    blue = sl("x0", 1, 2)
    arrow = sl("x0", 4, 5)
    grey = sl("x0", 5, 6)
    buttons = sl("x0", 9, 10)
    yellow = sl("x0", 11, 12)
    open_gate = sl("x0", 12, 13)
    pink = sl("x0", 14, 15)
    bases = cat([yellow, pink])
    base_any = affine(bases, [1.0, 1.0])

    # Gravity is the direction from the arrow towards its coloured base.
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    orientation_maps = []
    for dy, dx in directions:
        orientation_maps.append(
            binary("mul", base_any, shift(arrow, -dy, -dx)))
    orientation_maps = cat(orientation_maps)
    orientation = node("gmax", [orientation_maps], 4)
    flags = [vector_slice(orientation, j, j + 1) for j in range(4)]
    vertical = node("vadd", flags[:2], 1)
    horizontal = node("vadd", flags[2:], 1)

    def gravity_shift(x, distance):
        terms = []
        for flag, (dy, dx) in zip(flags, directions):
            terms.append(binary("mul", shift(x, dy * distance,
                                             dx * distance), flag))
        result = terms[0]
        for term in terms[1:]:
            result = binary("add", result, term)
        return result

    def transverse_shift(x, distance):
        a = binary("mul", shift(x, 0, distance), vertical)
        b = binary("mul", shift(x, distance, 0), horizontal)
        return binary("add", a, b)

    def axis_reduce(x, vertical_op, horizontal_op):
        a = binary("mul", node(vertical_op, [x], channels[x]), vertical)
        b = binary("mul", node(horizontal_op, [x], channels[x]), horizontal)
        return binary("add", a, b)

    def transverse_dilate(x, distance):
        return binary(
            "max2", x,
            binary("max2", transverse_shift(x, distance),
                   transverse_shift(x, -distance)))

    # Elevation: distance from the gravity-facing image boundary.
    elevation4 = affine(
        "coords",
        [[-63, 0, 0], [63, 0, 0],
         [0, -63, 0], [0, 63, 0]],
        [64, 1, 64, 1])
    elevation = affine(binary("mul", elevation4, orientation),
                       [1, 1, 1, 1])
    contact_elevation = scale(elevation, 1.0, -1.0)

    row_pixel = affine("coords", [63, 0, 0])
    row_zero = equal_zero(row_pixel)
    row_one = equal_zero(scale(row_pixel, 1.0, -1.0))
    not_ui = complement(row_zero)

    # Long grey runs are lane walls. Short grey runs inside lanes are floors.
    grey_length = axis_reduce(grey, "colsum", "rowsum")
    wall = clip01(scale(grey_length, 1.0, -8.0))
    floor = binary("mul", grey, complement(wall))
    room = binary("mul",
                  binary("mul", complement(wall), complement(floor)),
                  not_ui)
    room64 = scale(room, 64.0)

    # Do not mistake the black outlines of open bridges for terrain.
    frame_near = transverse_dilate(
        transverse_dilate(open_gate, 3), 1)
    clean_black = binary("mul", black, complement(frame_near))

    # Local max propagation. The stride-two stencil is efficient; initial
    # unit-distance propagation supplies all four spatial parities.
    def neighbour_kernel(c, distance):
        key = ("neighbours", c, distance)
        if key not in kernel_cache:
            k = 2 * distance + 1
            w = np.zeros((4 * c, c, k, k), np.float32)
            offsets = [(-distance, 0), (distance, 0),
                       (0, -distance), (0, distance)]
            for q, (dy, dx) in enumerate(offsets):
                for j in range(c):
                    w[q * c + j, j, distance + dy, distance + dx] = 1
            kernel_cache[key] = parameter(w)
        return kernel_cache[key]

    def dilate(x, limit, distance=1, where=None):
        c = channels[x]
        neighbours = node("conv", [x], 4 * c, where,
                          w=neighbour_kernel(c, distance))
        pieces = [sl(neighbours, q * c, (q + 1) * c, where)
                  for q in range(4)]
        z = pieces[0]
        for p in pieces[1:]:
            z = binary("max2", z, p, where)
        z = binary("max2", z, x, where)
        return binary("min2", z, limit, where)

    def seed_expand(x, limit):
        x = binary("min2", unary("relu", x), limit)
        return dilate(dilate(x, limit), limit)

    def flood_many(initials, limits, cost=None, expand=False, iterations=96):
        if expand:
            initials = [seed_expand(x, m)
                        for x, m in zip(initials, limits)]
        loop_id = fresh("flood")
        states = []
        body = []
        next_states = {}
        outputs = []
        for x, limit in zip(initials, limits):
            sid = fresh("state")
            c = channels[x]
            channels[sid] = c
            states.append({"id": sid, "init": x})
            if cost is None:
                z = dilate(sid, limit, 2, body)
            else:
                # Apply the toll after propagation, not before it.
                neighbours = node("conv", [sid], 4 * c, body,
                                  w=neighbour_kernel(c, 2))
                pieces = [sl(neighbours, q * c, (q + 1) * c, body)
                          for q in range(4)]
                z = pieces[0]
                for p in pieces[1:]:
                    z = binary("max2", z, p, body)
                z = binary("max2", z, sid, body)
                z = binary("sub", z, cost, body)
                z = unary("relu", z, body)
                z = binary("min2", z, limit, body)
            next_states[sid] = z
            out = loop_id + "." + sid
            channels[out] = c
            outputs.append(out)
        graph.append({
            "id": loop_id, "op": "loop", "T": iterations,
            "state": states, "body": body, "next": next_states
        })
        return outputs

    # Character bases have an arrow one or two pixels upstream.
    nearby_arrow = binary("max2", gravity_shift(arrow, -1),
                          gravity_shift(arrow, -2))
    person_seed = binary("mul", bases, nearby_arrow)
    base_forward = gravity_shift(bases, 1)
    base_edge = binary("mul", bases, complement(base_forward))
    target_edge = binary("mul", base_edge, complement(nearby_arrow))
    target_values = binary("mul", target_edge, contact_elevation)
    target_seed = binary(
        "mul",
        binary("max2", transverse_shift(target_values, -3),
               transverse_shift(target_values, 3)),
        room)

    # Terrain surface and gravity-facing room boundary.
    height_seed = binary("mul", clean_black, elevation)
    height_seed = binary("mul", height_seed, room)
    lower_inverse_seed = binary("sub", constant(64), elevation)
    lower_inverse_seed = binary(
        "add", lower_inverse_seed, binary("mul", row_one, flags[1]))
    lower_inverse_seed = binary("mul", lower_inverse_seed, room)

    # Compensate for the status row when finding the upstream room boundary.
    upper_seed = binary(
        "add", elevation, binary("mul", row_one, flags[0]))
    upper_seed = binary("mul", upper_seed, room)

    wall_height = axis_reduce(binary("mul", grey, elevation),
                              "colmax", "rowmax")
    wall_deficit = binary("mul",
                          binary("sub", constant(64), wall_height), wall)
    wall_limit_seed = binary(
        "mul",
        binary("max2", transverse_shift(wall_deficit, -1),
               transverse_shift(wall_deficit, 1)), room)

    initial_profiles = [
        cat([height_seed, lower_inverse_seed]),
        cat([upper_seed, wall_limit_seed]),
        person_seed,
        target_seed
    ]
    profile_results = flood_many(initial_profiles, [room64] * 4,
                                 expand=True)

    height = sl(profile_results[0], 0, 1)
    lower_inverse = sl(profile_results[0], 1, 2)
    room_upper = sl(profile_results[1], 0, 1)
    wall_deficit_profile = sl(profile_results[1], 1, 2)
    people = clip01(profile_results[2])
    goals = profile_results[3]
    goal_present = positive(goals)

    lower_bound = scale(lower_inverse, -1.0, 63.0)
    volume = unary("relu", binary("sub", height, lower_bound))
    wall_upper = scale(wall_deficit_profile, -1.0, 64.0)
    has_upstream_floor = positive(
        binary("sub", constant(64), room_upper))
    any_person = clip01(affine(people, [1, 1]))
    passenger_allowance = scale(
        binary("mul", has_upstream_floor, any_person), 6.0)
    upper_bound = binary(
        "min2", wall_upper,
        binary("sub", room_upper, passenger_allowance))
    destination_upper = binary(
        "min2", wall_upper,
        binary("sub", room_upper, scale(has_upstream_floor, 6.0)))

    satisfied = binary(
        "mul", people,
        binary("mul", goal_present,
               equal_zero(binary("sub", height, goals))))
    unsatisfied = binary("sub", people, satisfied)

    # Select an unsatisfied colour, not a prerecorded level or action phase.
    unsatisfied_flags = node("gmax", [unsatisfied], 2)
    colour_choice = dense(unsatisfied_flags, [[1, 0], [-1, 1]])
    colour_choice = node("vrelu", [colour_choice], 2)
    selected_people = affine(binary("mul", unsatisfied, colour_choice),
                             [1, 1])

    # Button connections pass through the wall between paired buttons.
    button_tunnel = binary(
        "mul", wall,
        clip01(binary("add", transverse_shift(buttons, -3),
                      transverse_shift(buttons, 3))))
    button_walk = clip01(binary("add", room, button_tunnel))

    # One corner per rectangular room provides an unduplicated volume sum.
    downstream_room = gravity_shift(room, 1)
    previous_transverse_room = transverse_shift(room, -1)
    anchor = binary(
        "mul", room,
        binary("mul", complement(downstream_room),
               complement(previous_transverse_room)))

    reach = flood_many(
        [goal_present], [button_walk],
        iterations=96)[0]
    reach = dilate(dilate(reach, room), room)

    reserved = clip01(affine(satisfied, [1, 1]))
    available_volume = binary("mul", volume, complement(reserved))
    volume_at_anchor = binary("mul", available_volume, anchor)
    reachable_volume = binary("mul", reach, volume_at_anchor)
    total_volume = node("gmean", [reachable_volume], 2)
    total_volume = dense(total_volume, np.eye(2, dtype=np.float32) * 4096)

    required_volume = binary("sub", goals, lower_bound)
    material_margin = binary("sub",
                             binary("add", scale(required_volume, -1),
                                    total_volume),
                             constant(0))
    material_ok = clip01(scale(material_margin, 4.0, 2.0))
    height_ok = clip01(scale(
        binary("sub", destination_upper, goals), 4.0, 2.0))
    feasible_goal = binary(
        "mul", goal_present, binary("mul", material_ok, height_ok))

    # Navigation potential: free propagation within rooms, a toll through
    # each bridge. Floors and non-bridge walls remain impassable.
    gates = clip01(binary("add", blue, open_gate))
    gate_walk = clip01(binary("add", room, gates))
    distance = flood_many(
        [scale(feasible_goal, 128)],
        [scale(gate_walk, 128)],
        cost=gates)[0]
    distance = dilate(dilate(distance, scale(room, 128)),
                      scale(room, 128))
    selected_distance = affine(
        binary("mul", distance, colour_choice), [1, 1])
    selected_goals = affine(
        binary("mul", binary("mul", goals, feasible_goal),
               colour_choice), [1, 1])
    selected_goal_present = positive(selected_goals)

    at_destination = binary("mul", selected_people, selected_goal_present)
    travelling_people = binary(
        "mul", selected_people, complement(selected_goal_present))

    gate_forward = gravity_shift(gates, 1)
    gate_edge = binary("mul", gates, complement(gate_forward))

    p_left = transverse_shift(travelling_people, -3)
    p_right = transverse_shift(travelling_people, 3)
    d_left = transverse_shift(selected_distance, -3)
    d_right = transverse_shift(selected_distance, 3)
    r_left = transverse_shift(room, -3)
    r_right = transverse_shift(room, 3)

    from_left = binary(
        "mul", p_left, positive(binary("sub", d_right, d_left)))
    from_right = binary(
        "mul", p_right, positive(binary("sub", d_left, d_right)))
    next_gate = binary(
        "mul", gate_edge,
        binary("mul",
               clip01(binary("add", from_left, from_right)),
               binary("mul", r_left, r_right)))

    bridge_requirement = binary(
        "mul", next_gate, contact_elevation)
    demand_seed = binary(
        "max2",
        binary("max2", transverse_shift(bridge_requirement, -3),
               transverse_shift(bridge_requirement, 3)),
        binary("mul", at_destination, selected_goals))
    demand_seed = binary("mul", demand_seed, room)
    demand = flood_many([demand_seed], [room64], expand=True)[0]
    active = positive(demand)
    error = binary("mul", binary("sub", demand, height), active)
    deficit = positive(error)
    surplus = positive(scale(error, -1))

    # Material-routing potentials. Positive pressure draws material towards
    # deficits. Negative pressure sends surplus to unconstrained storage.
    surplus_room = binary(
        "mul", room,
        clip01(binary("add", complement(active), surplus)))
    negative_walk = clip01(binary("add", surplus_room, button_tunnel))
    pressure_limits = scale(cat([button_walk, negative_walk]), 128)
    pressure = flood_many(
        [scale(cat([deficit, surplus]), 128)],
        [pressure_limits], cost=button_tunnel)[0]
    smooth_limits = scale(cat([room, surplus_room]), 128)
    pressure = dilate(dilate(pressure, smooth_limits), smooth_limits)
    positive_pressure = sl(pressure, 0, 1)
    negative_pressure = sl(pressure, 1, 2)

    # Preserve already-solved characters of the other colour.
    other_colour = dense(colour_choice, -np.eye(2, dtype=np.float32),
                         [1, 1])
    protected = clip01(affine(
        binary("mul", satisfied, other_colour), [1, 1]))

    capacity_ok = positive(binary("sub", upper_bound, height))
    receiver_ok = binary("mul", capacity_ok, complement(protected))
    negative_receiver_ok = clip01(
        binary("add", complement(active), deficit))

    neighbour_fields = cat([
        volume, positive_pressure, negative_pressure, protected
    ])
    button_scores = []

    for sign in (-1, 1):
        adjacent_wall = transverse_shift(wall, sign * 3)
        other = transverse_shift(
            transverse_shift(neighbour_fields, sign * 3), sign * 3)
        other_volume = sl(other, 0, 1)
        other_positive = sl(other, 1, 2)
        other_negative = sl(other, 2, 3)
        other_protected = sl(other, 3, 4)

        valid = binary(
            "mul", buttons,
            binary("mul", adjacent_wall,
                   binary("mul", receiver_ok,
                          binary("mul", positive(other_volume),
                                 complement(other_protected)))))

        good_positive = binary(
            "mul", positive(positive_pressure),
            positive(binary("sub", positive_pressure, other_positive)))
        good_negative = binary(
            "mul", negative_receiver_ok,
            binary("mul", positive(other_negative),
                   positive(binary("sub", other_negative,
                                    negative_pressure))))

        positive_score = binary(
            "mul", good_positive,
            scale(positive_pressure, 1.0, 1500.0))
        negative_score = binary(
            "mul", good_negative,
            scale(other_negative, 1.0, 500.0))
        score = binary("add", positive_score, negative_score)
        score = scale(score, 1.0, -50.0)
        score = binary(
            "add", binary("mul", score, valid),
            scale(complement(valid), -10000.0))
        button_scores.append(score)

    button_score = binary("max2", button_scores[0], button_scores[1])

    h_left = transverse_shift(height, -3)
    h_right = transverse_shift(height, 3)
    aligned = binary(
        "mul", next_gate,
        binary("mul",
               equal_zero(binary("sub", h_left, contact_elevation)),
               equal_zero(binary("sub", h_right, contact_elevation))))
    gate_score = scale(aligned, 14000.0, -10000.0)

    node("max2", [button_score, gate_score], 1, name="click")

    action_bias = np.full(8, -1.0, np.float32)
    action_bias[6] = 1.0
    node("dense", ["avail"], 8, name="logits",
         w=parameter(np.zeros((8, 8), np.float32)),
         b=parameter(action_bias))

    return {"graph": graph, "memory_size": 0, "params": params}
