import numpy as np


def build():
    graph = []
    params = {}
    nc = [0]
    pc = [0]

    def param(a):
        pc[0] += 1
        name = "p%d" % pc[0]
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, inputs, **fields):
        nc[0] += 1
        name = "n%d" % nc[0]
        d = {"id": name, "op": op, "in": list(inputs)}
        d.update(fields)
        graph.append(d)
        return name

    def conv(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("conv", [x], **fields)

    def dense(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", [x], **fields)

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def mx(a, b):
        return node("max2", [a, b])

    def relu(x):
        return node("relu", [x])

    def vadd(a, b):
        return node("vadd", [a, b])

    def vmul(a, b):
        return node("vmul", [a, b])

    one = sl("coords", 2, 3)

    def clip01(x):
        return node("min2", [relu(x), one])

    def threshold(x, t, channels=1):
        w = np.zeros((channels, channels, 1, 1), dtype=np.float32)
        for i in range(channels):
            w[i, i, 0, 0] = 10.0
        b = np.full(channels, -10.0 * t, dtype=np.float32)
        return clip01(conv(x, w, b))

    def scale(x, value):
        return conv(x, np.array([[[[value]]]], dtype=np.float32))

    def sum_channels(x, n):
        return conv(x, np.ones((1, n, 1, 1), dtype=np.float32))

    def expand_tiles(x):
        w = np.zeros((1, 1, 7, 7), dtype=np.float32)
        for dy in range(-2, 2):
            for dx in range(-2, 2):
                w[0, 0, dy + 3, dx + 3] = 1.0
        return conv(x, w)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def shift4(x):
        w1 = np.zeros((4, 1, 7, 7), dtype=np.float32)
        w2 = np.zeros((4, 4, 3, 3), dtype=np.float32)
        for i, (dy, dx) in enumerate(directions):
            w1[i, 0, 3 + 3 * dy, 3 + 3 * dx] = 1.0
            w2[i, i, 1 + dy, 1 + dx] = 1.0
        return conv(conv(x, w1), w2)

    def flood(seed, passable, loop_id, state_id, prefix, steps, decrement):
        w = np.zeros((4, 1, 3, 3), dtype=np.float32)
        for i, (dy, dx) in enumerate(directions):
            w[i, 0, 1 + dy, 1 + dx] = 1.0
        nw = param(w)
        body = [
            {"id": prefix + "nb", "op": "conv", "in": [state_id], "w": nw},
            {"id": prefix + "a", "op": "slice", "in": [prefix + "nb"],
             "start": 0, "stop": 1},
            {"id": prefix + "b", "op": "slice", "in": [prefix + "nb"],
             "start": 1, "stop": 2},
            {"id": prefix + "c", "op": "slice", "in": [prefix + "nb"],
             "start": 2, "stop": 3},
            {"id": prefix + "d", "op": "slice", "in": [prefix + "nb"],
             "start": 3, "stop": 4},
            {"id": prefix + "ab", "op": "max2",
             "in": [prefix + "a", prefix + "b"]},
            {"id": prefix + "cd", "op": "max2",
             "in": [prefix + "c", prefix + "d"]},
            {"id": prefix + "best", "op": "max2",
             "in": [prefix + "ab", prefix + "cd"]},
        ]
        best = prefix + "best"
        if decrement:
            dw = param(np.ones((1, 1, 1, 1), dtype=np.float32))
            db = param(np.array([-float(decrement)], dtype=np.float32))
            body.extend([
                {"id": prefix + "dec", "op": "conv", "in": [best],
                 "w": dw, "b": db},
                {"id": prefix + "positive", "op": "relu",
                 "in": [prefix + "dec"]},
            ])
            best = prefix + "positive"
        body.extend([
            {"id": prefix + "withseed", "op": "max2", "in": [best, seed]},
            {"id": prefix + "next", "op": "mul",
             "in": [prefix + "withseed", passable]},
        ])
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": steps,
            "state": [{"id": state_id, "init": seed}],
            "body": body,
            "next": {state_id: prefix + "next"},
        })
        return loop_id + "." + state_id

    # One indicator for each tile anchor.
    w = np.zeros((96, 3, 1, 1), dtype=np.float32)
    b = np.zeros(96, dtype=np.float32)
    for axis in range(2):
        for i in range(16):
            center = 4 * i + 1
            for k, offset in enumerate((-1, 0, 1)):
                j = axis * 48 + i * 3 + k
                w[j, axis, 0, 0] = 63.0
                b[j] = -(center + offset)
    ramps = relu(conv("coords", w, b))

    w = np.zeros((32, 96, 1, 1), dtype=np.float32)
    for axis in range(2):
        for i in range(16):
            for k, value in enumerate((1.0, -2.0, 1.0)):
                w[axis * 16 + i, axis * 48 + i * 3 + k, 0, 0] = value
    axes = threshold(conv(ramps, w), 0.5, channels=32)

    w = np.zeros((256, 32, 1, 1), dtype=np.float32)
    b = np.full(256, -15.0, dtype=np.float32)
    for r in range(16):
        for c in range(16):
            j = 16 * r + c
            w[j, r, 0, 0] = 10.0
            w[j, 16 + c, 0, 0] = 10.0
    tile_ids = clip01(conv(axes, w, b))
    lattice = sum_channels(tile_ids, 256)

    # Tile features: player, package, walls, agents, checker, floor,
    # border colors, and player-facing edges.
    w = np.zeros((13, 16, 7, 7), dtype=np.float32)
    for dy in range(-1, 3):
        for dx in range(-1, 3):
            yy, xx = dy + 3, dx + 3
            w[0, 14, yy, xx] = 1.0
            w[2, 5, yy, xx] = 1.0
            w[3, 12, yy, xx] = 1.0
            w[4, 15, yy, xx] = 1.0
            w[4, 11, yy, xx] = 1.0
            w[6, 2, yy, xx] = 1.0
            w[7, 9, yy, xx] = 1.0
            w[8, 0, yy, xx] = 1.0

            ly, lx = dy + 1, dx + 1
            transparent = (ly, lx) in {
                (0, 1), (1, 0), (2, 3), (3, 2)
            }
            w[5, 1 if transparent else 2, yy, xx] = 1.0
            if ly == 3:
                w[5, 7, yy, xx] = 1.0
                w[5, 4, yy, xx] = 1.0

            if dy == -1:
                w[9, 0, yy, xx] = 1.0
            if dy == 2:
                w[10, 0, yy, xx] = 1.0
            if dx == -1:
                w[11, 0, yy, xx] = 1.0
            if dx == 2:
                w[12, 0, yy, xx] = 1.0

    for dy in (0, 1):
        for dx in (0, 1):
            w[1, 9, dy + 3, dx + 3] = 1.0
    features = conv("x0", w)

    def feature(i, cutoff):
        return mul(threshold(sl(features, i, i + 1), cutoff), lattice)

    player = feature(0, 6.5)
    package = feature(1, 3.5)
    wall_color = feature(2, 6.5)
    helper = feature(3, 6.5)
    hostile = feature(4, 6.5)
    visible_hazard = feature(5, 14.5)
    red = feature(6, 0.5)
    nine = feature(7, 0.5)
    black = feature(8, 7.5)

    # Recover visible goal regions from their border and floor.
    candidates = mul(
        mul(mx(red, nine), sub(lattice, package)),
        sub(lattice, visible_hazard),
    )
    goal_seed = mul(nine, candidates)
    goal_pixels = flood(
        expand_tiles(goal_seed), expand_tiles(candidates),
        "goal_flood", "goal_state", "gf_", 64, 0
    )
    observed_goals = mul(goal_pixels, lattice)

    # Each memory entry encodes goal + 2*hazard for one tile.
    scalar_one = node("gmax", [one])
    constants = dense(
        scalar_one,
        np.zeros((2, 1), dtype=np.float32),
        np.array([-1.0, 2.0], dtype=np.float32),
    )
    minus_one = node("vslice", [constants], start=0, stop=1)
    plus_two = node("vslice", [constants], start=1, stop=2)

    a = node("vrelu", [vadd("mem", minus_one)])
    b = node("vrelu", [vadd(a, minus_one)])
    old_hazard = vadd(a, vmul(b, minus_one))
    old_goal = vadd(
        "mem",
        vadd(vmul(vmul(a, plus_two), minus_one), vmul(b, plus_two)),
    )

    observed_goal_vec = node("gmax", [mul(tile_ids, observed_goals)])
    observed_hazard_vec = node("gmax", [mul(tile_ids, visible_hazard)])
    goal_increment = node(
        "vrelu", [vadd(observed_goal_vec, vmul(old_goal, minus_one))]
    )
    hazard_increment = node(
        "vrelu", [vadd(observed_hazard_vec, vmul(old_hazard, minus_one))]
    )
    new_goal = vadd(old_goal, goal_increment)
    new_hazard = vadd(old_hazard, hazard_increment)
    graph.append({
        "id": "mem_next", "op": "vadd",
        "in": [new_goal, vmul(new_hazard, plus_two)],
    })
    goals = sum_channels(mul(tile_ids, new_goal), 256)
    hazard = sum_channels(mul(tile_ids, new_hazard), 256)

    held = mul(package, black)
    held_any = node("gmax", [held])
    not_held = dense(
        held_any,
        np.array([[-1.0]], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    )
    held_dirs = node("gmax", [mul(player, shift4(held))])

    carried_by_other = mul(package, wall_color)
    free_package = mul(
        mul(
            mul(package, sub(lattice, goals)),
            sub(lattice, held),
        ),
        mul(sub(lattice, carried_by_other), sub(lattice, hazard)),
    )

    wall = relu(sub(wall_color, package))
    occupied = mx(mx(wall, helper), mx(hostile, package))
    occupied_without_held = relu(sub(occupied, held))
    crate_free = relu(sub(lattice, occupied_without_held))
    player_free = relu(sub(crate_free, hazard))

    selected_crate_free = sum_channels(
        mul(shift4(crate_free), held_dirs), 4
    )
    carry_pass = mul(player_free, selected_crate_free)
    selected_goal = sum_channels(mul(shift4(goals), held_dirs), 4)
    selected_hazard = sum_channels(mul(shift4(hazard), held_dirs), 4)
    carry_goal = mul(carry_pass, selected_goal)
    carry_handoff = mul(carry_pass, selected_hazard)
    helper_any = node("gmax", [helper])

    package_dirs = shift4(free_package)
    hostile_dirs = shift4(hostile)
    package_adj = threshold(sum_channels(package_dirs, 4), 0.5)
    hostile_adj = threshold(sum_channels(hostile_dirs, 4), 0.5)

    # Package targets are preferred, but enemies remain targets if a
    # package is inaccessible, such as behind a blocked corridor.
    idle_seed = mx(scale(package_adj, 512.0), scale(hostile_adj, 464.0))
    idle_seed = mul(mul(idle_seed, player_free), not_held)
    carry_seed = mx(
        scale(carry_goal, 512.0),
        mul(scale(carry_handoff, 400.0), helper_any),
    )
    carry_seed = mul(carry_seed, held_any)
    path_seed = add(idle_seed, carry_seed)
    path_pass = add(mul(player_free, not_held), mul(carry_pass, held_any))

    potential = flood(
        expand_tiles(path_seed), expand_tiles(path_pass),
        "path_flood", "path_state", "pf_", 192, 1
    )
    navigation = node("gmax", [mul(player, shift4(potential))])
    current_potential = node("gmax", [mul(player, potential)])

    package_near = node("gmax", [mul(player, package_dirs)])
    hostile_near = node("gmax", [mul(player, hostile_dirs)])

    # Turn toward an adjacent hostile when the active potential is not
    # dominated by a more valuable reachable package target.
    enemy_gate_raw = node("vrelu", [dense(
        current_potential,
        np.array([[-1.0]], dtype=np.float32),
        np.array([465.0], dtype=np.float32),
    )])
    enemy_gate_excess = node(
        "vrelu", [vadd(enemy_gate_raw, minus_one)]
    )
    enemy_gate = vadd(
        enemy_gate_raw, vmul(enemy_gate_excess, minus_one)
    )
    target_dirs = vadd(
        package_near,
        dense(
            vmul(hostile_near, enemy_gate),
            2.0 * np.eye(4, dtype=np.float32),
        ),
    )
    target_dirs_idle = vmul(target_dirs, not_held)

    facing_planes = []
    for i in range(9, 13):
        facing_planes.append(feature(i, 1.5))
    facing_spatial = node("concat", facing_planes)
    facing = node("gmax", [mul(player, facing_spatial)])

    # The status bar may hide a downward-facing edge on the bottom row.
    fw = np.zeros((4, 4), dtype=np.float32)
    fw[1, :] = -1.0
    fb = np.zeros(4, dtype=np.float32)
    fb[1] = 1.0
    facing_fallback = node("vrelu", [dense(facing, fw, fb)])
    facing = vadd(facing, facing_fallback)

    front_targets = vadd(
        package_near,
        dense(hostile_near, 2.0 * np.eye(4, dtype=np.float32)),
    )
    front = vmul(vmul(facing, front_targets), not_held)
    front = dense(front, np.ones((1, 4), dtype=np.float32))

    held_on_goal = node("gmax", [mul(held, goals)])
    held_on_hazard = vmul(node("gmax", [mul(held, hazard)]), helper_any)
    release = vadd(held_on_goal, held_on_hazard)

    # If no task route reaches the player, prefer empty tiles away from
    # handed-off or helper-carried packages.
    cargo = mx(carried_by_other, mul(package, hazard))
    cargo_adj = threshold(sum_channels(shift4(cargo), 4), 0.5)
    nearby_cargo = node("gmax", [mul(player, cargo_adj)])
    neighbor_risk = node("gmax", [mul(player, shift4(cargo_adj))])
    neighbor_clear = node("gmax", [mul(player, shift4(player_free))])
    risk_free = dense(
        neighbor_risk,
        -np.eye(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
    )
    safe_moves = vmul(neighbor_clear, risk_free)
    no_path = node("vrelu", [dense(
        current_potential,
        np.array([[-1.0]], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    )])
    retreat = vmul(vmul(nearby_cargo, no_path), not_held)
    retreat_scale = dense(
        retreat,
        np.array([[10.0]], dtype=np.float32),
        np.array([0.01], dtype=np.float32),
    )
    fallback_moves = vmul(safe_moves, retreat_scale)

    output_features = node(
        "vconcat",
        [navigation, target_dirs_idle, front, release, fallback_moves],
    )
    ow = np.zeros((8, 14), dtype=np.float32)
    ob = np.zeros(8, dtype=np.float32)
    ob[0] = ob[6] = ob[7] = -10000.0
    for i in range(4):
        ow[1 + i, i] = 1.0
        ow[1 + i, 4 + i] = 2000.0
        ow[1 + i, 10 + i] = 1.0
    ow[5, 8] = 6000.0
    ow[5, 9] = 6000.0
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [output_features],
        "w": param(ow),
        "b": param(ob),
    })
    graph.append({
        "id": "click", "op": "slice", "in": ["coords"],
        "start": 2, "stop": 3,
    })

    return {"graph": graph, "memory_size": 256, "params": params}
