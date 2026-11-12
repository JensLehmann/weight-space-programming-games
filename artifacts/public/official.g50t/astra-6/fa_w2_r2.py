import numpy as np


def build():
    class Builder:
        def __init__(self):
            self.graph = []
            self.current = self.graph
            self.params = {}
            self.channels = {
                "x0": 16, "coords": 3, "mem": 59,
                "aprev": 8, "avail": 8
            }
            self.serial = 0
            self.pserial = 0

        def fresh(self, prefix="n"):
            self.serial += 1
            return "%s_%d" % (prefix, self.serial)

        def param(self, value):
            self.pserial += 1
            name = "p_%d" % self.pserial
            self.params[name] = np.asarray(value, dtype=np.float32)
            return name

        def node(self, op, inputs, name=None, **fields):
            name = name or self.fresh()
            nd = {"id": name, "op": op, "in": list(inputs)}
            nd.update(fields)
            self.current.append(nd)

            if op in ("concat", "vconcat"):
                c = sum(self.channels[x] for x in inputs)
            elif op in ("slice", "vslice"):
                c = fields["stop"] - fields["start"]
            elif op in ("conv", "dense"):
                c = self.params[fields["w"]].shape[0]
            else:
                c = max(self.channels[x] for x in inputs)
            self.channels[name] = c
            return name

        def conv(self, x, w, bias=None, name=None):
            w = np.asarray(w, dtype=np.float32)
            if w.ndim == 2:
                w = w[:, :, None, None]
            fields = {"w": self.param(w)}
            if bias is not None:
                fields["b"] = self.param(bias)
            return self.node("conv", [x], name=name, **fields)

        def dense(self, x, w, bias=None, name=None):
            fields = {"w": self.param(w)}
            if bias is not None:
                fields["b"] = self.param(bias)
            return self.node("dense", [x], name=name, **fields)

        def sl(self, x, start, stop, vector=False):
            return self.node(
                "vslice" if vector else "slice", [x],
                start=start, stop=stop
            )

        def sa(self, x, scale=1.0, offset=0.0):
            c = self.channels[x]
            return self.conv(
                x, np.eye(c, dtype=np.float32) * scale,
                np.full(c, offset, dtype=np.float32)
            )

        def va(self, x, scale=1.0, offset=0.0):
            c = self.channels[x]
            return self.dense(
                x, np.eye(c, dtype=np.float32) * scale,
                np.full(c, offset, dtype=np.float32)
            )

        def cat(self, *xs):
            return self.node("concat", xs)

        def vcat(self, *xs):
            return self.node("vconcat", xs)

        def unary(self, op, x):
            return self.node(op, [x])

        def binary(self, op, x, y):
            return self.node(op, [x, y])

        def loop(self, initial, iterations, body_function):
            lid = self.fresh("loop")
            sid = self.fresh("state")
            self.channels[sid] = self.channels[initial]
            outer = self.current
            self.current = []
            result = body_function(sid)
            body = self.current
            self.current = outer
            self.current.append({
                "id": lid,
                "op": "loop",
                "T": iterations,
                "state": [{"id": sid, "init": initial}],
                "body": body,
                "next": {sid: result},
            })
            output = lid + "." + sid
            self.channels[output] = self.channels[initial]
            return output

    b = Builder()
    N = 8
    R = 2048.0

    def add(x, y):
        return b.binary("add", x, y)

    def sub(x, y):
        return b.binary("sub", x, y)

    def mul(x, y):
        return b.binary("mul", x, y)

    def mx(x, y):
        return b.binary("max2", x, y)

    def vadd(x, y):
        return b.binary("vadd", x, y)

    def vmul(x, y):
        return b.binary("vmul", x, y)

    def relu(x):
        return b.unary("relu", x)

    def vrelu(x):
        return b.unary("vrelu", x)

    def absolute(x):
        return b.unary("abs", x)

    def gmax(x):
        return b.unary("gmax", x)

    one_s = b.sl("coords", 2, 3)
    one_v = gmax(one_s)

    def clip_s(x):
        return b.binary("min2", relu(x), one_v)

    def clip_v(x):
        p = vrelu(x)
        q = vrelu(b.va(x, 1, -1))
        return vadd(p, b.va(q, -1))

    def not_s(x):
        return b.sa(x, -1, 1)

    def not_v(x):
        return b.va(x, -1, 1)

    def binary_s(x, count=1):
        return clip_s(b.sa(x, 2, 1 - 2 * count))

    def sum_channels(x):
        return b.conv(
            x, np.ones((1, b.channels[x]), dtype=np.float32)
        )

    previous_reset = b.sl("aprev", 0, 1, vector=True)
    previous_five = b.sl("aprev", 5, 6, vector=True)
    memory = vmul("mem", not_v(previous_reset))

    def msl(start, stop):
        return b.sl(memory, start, stop, vector=True)

    def select_mem(indices):
        w = np.zeros((len(indices), 59), dtype=np.float32)
        for i, j in enumerate(indices):
            w[i, j] = 1
        return b.dense(memory, w)

    prev_y = msl(55, 56)
    prev_x = msl(56, 57)
    prev_valid = msl(57, 58)
    prev_time = msl(58, 59)
    initial_frame = not_v(prev_valid)
    initial_reset_request = vmul(initial_frame, not_v(previous_reset))

    wpix = np.zeros((2, 3), dtype=np.float32)
    wpix[0, 0] = 63
    wpix[1, 1] = 63
    pixels = b.conv("coords", wpix)
    row = b.sl(pixels, 0, 1)
    col = b.sl(pixels, 1, 2)

    def bump(x, center):
        return clip_s(
            b.sa(absolute(b.sa(x, 1, -center)), -4, 2)
        )

    def points(ys, xs, valid):
        dy = absolute(sub(row, ys))
        dx = absolute(sub(col, xs))
        peak = clip_s(b.sa(add(dy, dx), -4, 2))
        return mul(peak, valid)

    # Canonical movement lattice; this does not encode any level layout.
    centers = [4 + 6 * k for k in range(10)]
    wgrid = np.zeros((20, 3), dtype=np.float32)
    grid_bias = np.zeros(20, dtype=np.float32)
    for i, center in enumerate(centers):
        wgrid[i, 0] = 63
        grid_bias[i] = -center
        wgrid[10 + i, 1] = 63
        grid_bias[10 + i] = -center

    grid_delta = absolute(b.conv("coords", wgrid, grid_bias))
    grid_bumps = clip_s(b.sa(grid_delta, -4, 2))
    wsum = np.zeros((2, 20), dtype=np.float32)
    wsum[0, :10] = 1
    wsum[1, 10:] = 1
    grid_axes = b.conv(grid_bumps, wsum)
    lattice = mul(b.sl(grid_axes, 0, 1), b.sl(grid_axes, 1, 2))

    board = mul(
        clip_s(b.sa(row, 1, -6)),
        clip_s(b.sa(row, -1, 62))
    )
    lattice = mul(lattice, board)

    # Broad floor regions, excluding the thin decorative wire backing.
    wnz = np.ones((1, 16), dtype=np.float32)
    wnz[0, 0] = 0
    nonzero = b.conv("x0", wnz)
    floor_count = b.conv(
        nonzero, np.ones((1, 1, 5, 5), dtype=np.float32)
    )
    floor = mul(binary_s(floor_count, 25), board)

    # Player, recorded ghost, and moving enemy.
    actor_w = np.zeros((3, 16, 3, 3), dtype=np.float32)
    for i, color in enumerate((9, 2, 14)):
        actor_w[i, color, :, :] = 1
        actor_w[i, color, 1, 1] = 0
        actor_w[i, 5, 1, 1] = 1

    actors = mul(binary_s(b.conv("x0", actor_w), 9), board)
    player = b.sl(actors, 0, 1)
    enemy = b.sl(actors, 2, 3)
    actor_centers = sum_channels(actors)
    actor_fill = binary_s(
        b.conv(actor_centers, np.ones((1, 1, 5, 5), dtype=np.float32)),
        1
    )
    player_xy = gmax(mul(player, pixels))

    # Exit's isolated orange center.
    goal_w = np.zeros((1, 16, 3, 3), dtype=np.float32)
    goal_w[0, 5, :, :] = 1
    goal_w[0, 5, 1, 1] = 0
    goal_w[0, 9, 1, 1] = 1
    goal = mul(binary_s(b.conv("x0", goal_w), 9), lattice)

    # Type order is consistently RED, WHITE, PURPLE throughout memory.
    switch_colors = (8, 15, 11)
    small_w = np.zeros((3, 16, 3, 3), dtype=np.float32)
    large_w = np.zeros((3, 16, 5, 5), dtype=np.float32)
    for i, color in enumerate(switch_colors):
        small_w[i, color, :, :] = 1
        large_w[i, color, :, :] = 1

    small_count = b.conv("x0", small_w)
    large_count = b.conv("x0", large_w)
    visible_switches = mul(
        mul(binary_s(small_count, 9),
            not_s(binary_s(large_count, 18))),
        lattice
    )

    # Gates have large red or purple interiors.
    gate_color_w = np.zeros((1, 16), dtype=np.float32)
    gate_color_w[0, 8] = 1
    gate_color_w[0, 11] = 1
    gate_colors = b.conv("x0", gate_color_w)
    gate_count = b.conv(
        gate_colors, np.ones((1, 1, 5, 5), dtype=np.float32)
    )
    gate_centers = mul(binary_s(gate_count, 18), lattice)
    gate_area = binary_s(
        b.conv(gate_centers, np.ones((1, 1, 7, 7), dtype=np.float32)),
        1
    )

    # Portal frames can be detected even with an actor inside.
    portal_w = np.zeros((1, 16, 7, 7), dtype=np.float32)
    portal_w[0, 15, 0, :] = 1
    portal_w[0, 15, 6, :] = 1
    portal_w[0, 15, 1:6, 0] = 1
    portal_w[0, 15, 1:6, 6] = 1
    portals = mul(binary_s(b.conv("x0", portal_w), 24), lattice)
    portal_fill = binary_s(
        b.conv(portals, np.ones((1, 1, 7, 7), dtype=np.float32)),
        1
    )

    # At episode start a switch may be concealed by the player.
    # Its outgoing wire remains visible just outside the player's glyph.
    hidden_w = np.zeros((3, 16, 7, 7), dtype=np.float32)
    for i, color in enumerate(switch_colors):
        for dy, dx in ((-3, 0), (3, 0), (0, -3), (0, 3)):
            hidden_w[i, color, 3 + dy, 3 + dx] = 1

    hidden_types = binary_s(b.conv("x0", hidden_w), 1)
    hidden_switches = mul(
        mul(mul(hidden_types, player), not_s(portals)),
        initial_frame
    )
    switches = mx(visible_switches, hidden_switches)

    red_pixels = b.sl("x0", 8, 9)
    white_pixels = b.sl("x0", 15, 16)
    orange_pixels = b.sl("x0", 9, 10)

    # Memory:
    # 0:7 target (y,x,red,white,purple,valid,recording-time threshold)
    # 7:55 eight entries (y,x,red,white,purple,valid)
    # 55:59 previous player y,x,valid and successful-move clock.
    stored_target = msl(0, 7)

    initial_types = gmax(mul(switches, player))
    type_sum_w = np.ones((1, 3), dtype=np.float32)
    initial_here = vmul(
        clip_v(b.dense(initial_types, type_sum_w)),
        initial_frame
    )
    zero_v = b.va(one_v, 0, 0)
    bootstrap_target = b.vcat(
        player_xy, initial_types, one_v, zero_v
    )
    old_target = vadd(
        vmul(stored_target, not_v(initial_here)),
        vmul(bootstrap_target, initial_here)
    )

    old_y = b.sl(old_target, 0, 1, vector=True)
    old_x = b.sl(old_target, 1, 2, vector=True)
    old_red = b.sl(old_target, 2, 3, vector=True)
    old_white = b.sl(old_target, 3, 4, vector=True)
    old_purple = b.sl(old_target, 4, 5, vector=True)
    old_valid = b.sl(old_target, 5, 6, vector=True)
    old_required = b.sl(old_target, 6, 7, vector=True)

    # Recording-slot wraparound clears assumed ghost assignments.
    underline = mul(bump(row, 5), orange_pixels)
    underline_x = gmax(mul(underline, col))
    first_slot = clip_v(b.va(underline_x, -1, 4))
    wrapped = vmul(previous_five, first_slot)
    keep_records = not_v(wrapped)

    rec_y = select_mem([7 + 6 * i for i in range(N)])
    rec_x = select_mem([8 + 6 * i for i in range(N)])
    rec_red = select_mem([9 + 6 * i for i in range(N)])
    rec_white = select_mem([10 + 6 * i for i in range(N)])
    rec_valid = vmul(
        select_mem([12 + 6 * i for i in range(N)]),
        keep_records
    )

    record_points = points(rec_y, rec_x, rec_valid)
    visited_records = clip_s(sum_channels(record_points))
    red_seeds = clip_s(sum_channels(mul(record_points, rec_red)))
    white_seeds = clip_s(sum_channels(mul(record_points, rec_white)))

    previous_point = points(prev_y, prev_x, prev_valid)
    unchanged = gmax(mul(previous_point, player))
    moved = vadd(prev_valid, b.va(unchanged, -1))
    clock = vmul(vadd(prev_time, moved), not_v(previous_five))

    old_point = points(old_y, old_x, old_valid)
    arrived = gmax(mul(old_point, player))
    time_difference = vadd(clock, b.va(old_required, -1))
    white_ready = clip_v(b.va(time_difference, 4, 1))
    recording_ready = not_v(vmul(old_white, not_v(white_ready)))

    record_event = vmul(
        vmul(arrived, vadd(old_red, old_white)),
        recording_ready
    )
    purple_event = vmul(arrived, old_purple)
    completed_target = clip_v(vadd(record_event, purple_event))
    pending_target = vmul(old_valid, not_v(completed_target))

    visited = clip_s(
        add(visited_records, mul(old_point, completed_target))
    )
    available_switches = mul(switches, not_s(visited))
    switch_any = clip_s(sum_channels(available_switches))

    # Trace wires from remembered switch endpoints.
    red_wire = clip_s(add(add(red_pixels, actor_fill), red_seeds))
    white_wire = clip_s(
        add(add(add(white_pixels, actor_fill), portal_fill), white_seeds)
    )
    wire_masks = b.cat(red_wire, white_wire)
    wire_initial = b.cat(red_seeds, white_seeds)

    wire_w = np.zeros((2, 2, 3, 3), dtype=np.float32)
    for c in range(2):
        for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            wire_w[c, c, 1 + dy, 1 + dx] = 1

    def wire_body(state):
        spread = b.conv(state, wire_w)
        return mul(b.binary("min2", spread, one_v), wire_masks)

    assigned_wires = b.loop(wire_initial, 128, wire_body)
    assigned_red = b.sl(assigned_wires, 0, 1)
    assigned_white = b.sl(assigned_wires, 1, 2)

    released_centers = mul(
        mul(gate_centers, red_pixels), assigned_red
    )
    released_area = binary_s(
        b.conv(released_centers, np.ones((1, 1, 7, 7), dtype=np.float32)),
        1
    )
    future_blocked = relu(sub(gate_area, released_area))
    strict_floor = mul(floor, not_s(future_blocked))
    legal_floor = mul(floor, not_s(gate_area))
    soft_cost = b.sa(future_blocked, 8, 1)

    def shifted_max(state, radius):
        c = b.channels[state]
        k = 2 * radius + 1
        w = np.zeros((4 * c, c, k, k), dtype=np.float32)
        directions = [
            (-radius, 0), (radius, 0),
            (0, -radius), (0, radius)
        ]
        for d, (dy, dx) in enumerate(directions):
            for channel in range(c):
                w[d * c + channel, channel,
                  radius + dy, radius + dx] = 1
        shifted = b.conv(state, w)
        pieces = [
            b.sl(shifted, d * c, (d + 1) * c)
            for d in range(4)
        ]
        return mx(mx(pieces[0], pieces[1]), mx(pieces[2], pieces[3]))

    # Component labels identify which portal frames share a white wire.
    coordinate_code = b.conv(
        pixels, np.array([[64, 1]], dtype=np.float32),
        np.array([1], dtype=np.float32)
    )
    label_initial = mul(portals, coordinate_code)

    def label_body(state):
        return mul(mx(state, shifted_max(state, 1)), white_wire)

    white_labels = b.loop(label_initial, 128, label_body)
    remaining_labels = mul(white_labels, portals)
    portal_components = []

    for component_index in range(3):
        largest = gmax(remaining_labels)
        equal_label = clip_s(
            b.sa(absolute(sub(white_labels, largest)), -2, 1)
        )
        component = mul(equal_label, portals)
        portal_components.append(mul(component, assigned_white))
        if component_index != 2:
            remaining_labels = mul(
                remaining_labels, not_s(equal_label)
            )

    def propagate(initial, passable, cost, iterations):
        def body(state):
            best = shifted_max(state, 3)
            for component in portal_components:
                portal_value = gmax(mul(state, component))
                transmitted = mul(component, portal_value)
                best = mx(best, transmitted)
            candidate = mul(relu(sub(best, cost)), passable)
            return mx(candidate, initial)
        return b.loop(initial, iterations, body)

    # Parallel strict and gate-penalized reachability maps.
    player_reward = b.sa(player, R)
    reach_initial = b.cat(player_reward, player_reward)
    reach_masks = b.cat(strict_floor, floor)
    reach_costs = b.cat(one_s, soft_cost)
    reach = propagate(reach_initial, reach_masks, reach_costs, 144)
    strict_value = b.sl(reach, 0, 1)
    soft_value = b.sl(reach, 1, 2)

    strict_switch_mask = mul(switch_any, clip_s(strict_value))
    has_strict_switch = gmax(strict_switch_mask)

    # Do not retain an unreachable target when a reachable prerequisite
    # appears, for example after stepping off an initially hidden switch.
    old_target_reachable = clip_v(gmax(mul(strict_value, old_point)))
    retain_unreachable = not_v(has_strict_switch)
    retain_target = clip_v(
        vadd(old_target_reachable, retain_unreachable)
    )
    pending_target = vmul(pending_target, retain_target)

    selection_value = add(
        mul(strict_value, has_strict_switch),
        mul(soft_value, not_v(has_strict_switch))
    )
    candidate_switches = mul(switch_any, clip_s(selection_value))

    tie = b.sa(coordinate_code, -1.0 / 8192.0, 0.5)
    selection_scores = mul(
        add(selection_value, tie), candidate_switches
    )
    best_score = gmax(selection_scores)
    winners = mul(
        relu(b.sa(absolute(sub(selection_scores, best_score)),
                  -8192, 1)),
        candidate_switches
    )

    goal_reachable = clip_v(gmax(mul(strict_value, goal)))
    any_switch = gmax(candidate_switches)
    choose_goal = not_v(vmul(not_v(goal_reachable), any_switch))
    selected_point = add(
        mul(goal, choose_goal),
        mul(winners, not_v(choose_goal))
    )
    selected_xy = gmax(mul(selected_point, pixels))
    selected_types = gmax(mul(selected_point, available_switches))
    selected_valid = gmax(selected_point)

    # White recordings are extended sufficiently to let the next actor
    # reach an entry portal before the recorded switch activation.
    selected_label = gmax(mul(selected_point, white_labels))
    same_white_component = clip_s(
        b.sa(absolute(sub(white_labels, selected_label)), -2, 1)
    )
    relevant_portals = mul(portals, same_white_component)
    entry_value = gmax(mul(strict_value, relevant_portals))
    has_entry = clip_v(entry_value)
    entry_time = vmul(
        b.va(entry_value, -0.5, R * 0.5 + 2), has_entry
    )
    required_time = vadd(clock, entry_time)

    proposed_target = b.vcat(
        selected_xy, selected_types, selected_valid, required_time
    )
    target = vadd(
        vmul(old_target, pending_target),
        vmul(proposed_target, not_v(pending_target))
    )

    target_y = b.sl(target, 0, 1, vector=True)
    target_x = b.sl(target, 1, 2, vector=True)
    target_valid = b.sl(target, 5, 6, vector=True)
    target_point = points(target_y, target_x, target_valid)

    navigation = propagate(
        b.sa(target_point, R), floor, soft_cost, 144
    )

    danger = b.conv(
        enemy, np.ones((1, 1, 7, 7), dtype=np.float32)
    )
    action_fields = b.cat(navigation, legal_floor, danger)

    # Read action values six pixels away, using two legal 7x7 convolutions.
    directions = [(-3, 0), (3, 0), (0, -3), (0, 3)]
    first_w = np.zeros((12, 3, 7, 7), dtype=np.float32)
    for d, (dy, dx) in enumerate(directions):
        for c in range(3):
            first_w[3 * d + c, c, 3 + dy, 3 + dx] = 1
    first_shift = b.conv(action_fields, first_w)

    action_values = []
    action_legal = []
    action_danger = []
    for d, (dy, dx) in enumerate(directions):
        part = b.sl(first_shift, 3 * d, 3 * d + 3)
        second_w = np.zeros((3, 3, 7, 7), dtype=np.float32)
        for c in range(3):
            second_w[c, c, 3 + dy, 3 + dx] = 1
        destination = b.conv(part, second_w)
        sampled = gmax(mul(destination, player))
        action_values.append(b.sl(sampled, 0, 1, vector=True))
        action_legal.append(b.sl(sampled, 1, 2, vector=True))
        action_danger.append(b.sl(sampled, 2, 3, vector=True))

    values_v = b.vcat(*action_values)
    legal_v = b.vcat(*action_legal)
    danger_v = b.vcat(*action_danger)
    decision = b.vcat(
        values_v, legal_v, danger_v,
        record_event, initial_reset_request
    )

    output_w = np.zeros((8, 14), dtype=np.float32)
    output_bias = np.full(8, -100000, dtype=np.float32)
    for d in range(4):
        output_w[d + 1, d] = 1
        output_w[d + 1, 4 + d] = 5000
        output_w[d + 1, 8 + d] = -8
        output_bias[d + 1] = -5000

    output_w[5, 12] = 30000
    output_bias[5] = -10000

    # When RESET is available, normalize unknown pre-episode history once.
    # Otherwise availability masking leaves the observation-based policy.
    output_w[0, 13] = 160000
    b.dense(decision, output_w, output_bias, name="logits")

    records = vmul(msl(7, 55), keep_records)
    new_entry = b.sl(old_target, 0, 6, vector=True)
    shifted_records = b.vcat(
        new_entry, b.sl(records, 0, 6 * (N - 1), vector=True)
    )
    next_records = vadd(
        vmul(records, not_v(completed_target)),
        vmul(shifted_records, completed_target)
    )
    next_target = vmul(target, not_v(record_event))

    b.node(
        "vconcat",
        [next_target, next_records, player_xy, one_v, clock],
        name="mem_next"
    )
    b.conv(
        one_s, np.zeros((1, 1), dtype=np.float32), name="click"
    )

    return {
        "graph": b.graph,
        "memory_size": 59,
        "params": b.params,
    }
