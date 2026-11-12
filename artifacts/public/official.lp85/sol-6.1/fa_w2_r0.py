import numpy as np


def build():
    N = 30
    K = 8
    G = 3

    graph = []
    params = {}
    parameter_cache = {}
    node_caches = {}
    sizes = {"avail": 8, "aprev": 8}
    serial = [0]
    output = [graph]

    def nd(op, xs=None, name=None, size=None, **kw):
        cache = node_caches.setdefault(id(output[0]), {})
        key = None
        if name is None and op != "loop":
            key = (op, tuple(xs or ()), tuple(sorted(kw.items())))
            if key in cache:
                return cache[key]
            if output[0] is not graph:
                top_cache = node_caches.setdefault(id(graph), {})
                if key in top_cache:
                    return top_cache[key]

        if name is None:
            name = "n%d" % serial[0]
            serial[0] += 1

        d = {"id": name, "op": op}
        if xs is not None:
            d["in"] = list(xs)
        d.update(kw)
        output[0].append(d)

        if size is not None:
            sizes[name] = size
        if key is not None:
            cache[key] = name
        return name

    def par(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            p = "p%d" % len(params)
            params[p] = a
            parameter_cache[key] = p
        return parameter_cache[key]

    def cv(x, w, b=None, name=None):
        kw = {"w": par(w)}
        if b is not None:
            kw["b"] = par(b)
        return nd("conv", [x], name=name, **kw)

    def de(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        kw = {"w": par(w)}
        if b is not None:
            kw["b"] = par(b)
        return nd("dense", [x], name=name, size=w.shape[0], **kw)

    def ss(x, a, b):
        return nd("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return nd("vslice", [x], start=a, stop=b, size=b-a)

    def sc(xs):
        return nd("concat", xs)

    def vc(xs, name=None):
        return nd("vconcat", xs, name=name,
                  size=sum(sizes[x] for x in xs))

    def su(op, x):
        return nd(op, [x])

    def vu(op, x):
        return nd(op, [x], size=sizes[x])

    def sb(op, a, b):
        return nd(op, [a, b])

    def va(a, b):
        return nd("vadd", [a, b], size=max(sizes[a], sizes[b]))

    def vm(a, b):
        return nd("vmul", [a, b], size=max(sizes[a], sizes[b]))

    def vf(x, scale=1.0, bias=0.0):
        n = sizes[x]
        return de(x, np.eye(n, dtype=np.float32)*scale,
                  np.full(n, bias, np.float32))

    def sf(x, c=1, scale=1.0, bias=0.0):
        w = np.eye(c, dtype=np.float32)[:, :, None, None]*scale
        return cv(x, w, np.full(c, bias, np.float32))

    def vminus(a, b):
        na, nb = sizes[a], sizes[b]
        n = max(na, nb)
        wa = (np.eye(n, dtype=np.float32) if na == n
              else np.ones((n, 1), np.float32))
        wb = (np.eye(n, dtype=np.float32) if nb == n
              else np.ones((n, 1), np.float32))
        return de(vc([a, b]), np.concatenate([wa, -wb], axis=1))

    def vabs(x):
        return va(vu("vrelu", x), vu("vrelu", vf(x, -1.0)))

    def vclip(x):
        r = vu("vrelu", x)
        return vf(vu("vrelu", vf(r, -1.0, 1.0)), -1.0, 1.0)

    def vgt(x, threshold=0.0, gain=4.0):
        return vclip(vf(x, gain, -gain*threshold))

    def vnot(x):
        return vf(x, -1.0, 1.0)

    def veq(a, b):
        return vu("vrelu", vf(vabs(vminus(a, b)), -2.0, 1.0))

    def total(x):
        return de(x, np.ones((1, sizes[x]), np.float32))

    def pool(x, c, mean=False):
        return nd("gmean" if mean else "gmax", [x], size=c)

    one_v = de("avail", np.zeros((1, 8), np.float32), [1.0])
    zero_v = vf(one_v, 0.0)
    one_s = ss("coords", 2, 3)

    def sclip(x):
        return sb("min2", su("relu", x), one_v)

    def sgt(x, threshold=0.0, gain=4.0, c=1):
        return sclip(sf(x, c, gain, -gain*threshold))

    def seq(a, b, c=1):
        difference = su("abs", sb("sub", a, b))
        return su("relu", sf(difference, c, -2.0, 1.0))

    def sumch(x, c):
        return cv(x, np.ones((1, c, 1, 1), np.float32))

    def argmax_v(x):
        n = sizes[x]
        ranked = de(x, np.eye(n, dtype=np.float32),
                    -0.001*np.arange(n, dtype=np.float32))
        pairs = [(i, j) for i in range(n)
                 for j in range(n) if i != j]
        w = np.zeros((len(pairs), n), np.float32)
        reduction = np.zeros((n, len(pairs)), np.float32)
        for q, (i, j) in enumerate(pairs):
            w[q, i] = 1.0
            w[q, j] = -1.0
            reduction[i, q] = 1.0
        comparisons = vgt(de(ranked, w), 0.0, 1.0e6)
        wins = de(comparisons, reduction)
        return vclip(vf(wins, 1.0, -(n-2)))

    # Memory stores previous colours, compact transition constraints,
    # observation counts, and the preceding primitive control.
    M = N + G*N + G + 2*G
    sizes["mem"] = M
    previous_colours = vs("mem", 0, N)
    stored_codes = [
        vs("mem", N+g*N, N+(g+1)*N) for g in range(G)
    ]
    count_vector = vs("mem", N+G*N, N+G*N+G)
    previous_action = vs("mem", N+G*N+G, M)

    row_pixel = cv(
        "coords",
        np.array([63.0, 0.0, 0.0], np.float32).reshape(1, 3, 1, 1)
    )
    col_pixel = cv(
        "coords",
        np.array([0.0, 63.0, 0.0], np.float32).reshape(1, 3, 1, 1)
    )
    rank = cv(
        "coords",
        np.array([-1.0, -1.0/128.0, 2.0], np.float32).reshape(1, 3, 1, 1)
    )

    def extract_points(prefix, mask, feature_map, count):
        cursor_bias = np.zeros(count, np.float32)
        cursor_bias[0] = 1.0
        cursor0 = de("avail", np.zeros((count, 8), np.float32),
                     cursor_bias)
        shift = np.zeros((count, count), np.float32)
        for i in range(1, count):
            shift[i, i-1] = 1.0

        remaining_id = prefix + "_remaining"
        cursor_id = prefix + "_cursor"
        vector_ids = [prefix + "_v%d" % i for i in range(4)]
        sizes[cursor_id] = count
        for x in vector_ids:
            sizes[x] = count

        body = []
        output[0] = body

        ranked = sb("mul", remaining_id, rank)
        best = pool(ranked, 1)
        chosen = su(
            "relu",
            sf(sb("sub", ranked, best), 1, 16384.0, 1.0)
        )
        chosen = sb("mul", chosen, remaining_id)
        features = pool(sb("mul", feature_map, chosen), 4)

        updated_vectors = []
        for i, state_name in enumerate(vector_ids):
            value = vs(features, i, i+1)
            updated_vectors.append(
                va(state_name, vm(cursor_id, value))
            )

        remaining_next = su("relu", sb("sub", remaining_id, chosen))
        cursor_next = de(cursor_id, shift)

        output[0] = graph
        loop_name = prefix + "_extract"
        next_states = {
            remaining_id: remaining_next,
            cursor_id: cursor_next,
        }
        next_states.update(dict(zip(vector_ids, updated_vectors)))

        nd(
            "loop", name=loop_name, T=count,
            state=[
                {"id": remaining_id, "init": mask},
                {"id": cursor_id, "init": cursor0},
            ] + [
                {"id": x, "zeros": [count]} for x in vector_ids
            ],
            body=body,
            next=next_states
        )

        result = [loop_name + "." + x for x in vector_ids]
        for x in result:
            sizes[x] = count
        return result

    # ----------------------------------------------------------
    # Canonical button templates.
    # ----------------------------------------------------------
    button_planes = sc([ss("x0", 8, 9), ss("x0", 14, 15)])
    blue = np.array([
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [0, 1, 1],
    ], np.int8)
    red = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],
    ], np.int8)

    bw = np.zeros((16, 2, 7, 7), np.float32)
    bb = np.zeros(16, np.float32)
    q = 0
    for scale in (1, 2):
        for colour, base in enumerate((blue, red)):
            for rotation in range(4):
                shape = np.rot90(base, -rotation)
                shape = np.repeat(
                    np.repeat(shape, scale, axis=0), scale, axis=1
                )
                anchor = int(np.flatnonzero(shape[0])[0])
                patch = np.zeros((7, 7), np.float32)
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        yy, xx = dy, anchor+dx
                        if (0 <= yy < shape.shape[0]
                                and 0 <= xx < shape.shape[1]):
                            patch[dy+3, dx+3] = shape[yy, xx]
                bw[q, colour] = 2.0*patch-1.0
                bb[q] = 1.0-patch.sum()
                q += 1

    matches = su("relu", cv(button_planes, bw, bb))
    button_summary = pool(matches, 16)
    scaled = vclip(total(vs(button_summary, 8, 16)))
    unscaled = vnot(scaled)

    combine = np.zeros((2, 16, 1, 1), np.float32)
    combine[0, [0, 1, 2, 3, 8, 9, 10, 11], 0, 0] = 1.0
    combine[1, [4, 5, 6, 7, 12, 13, 14, 15], 0, 0] = 1.0
    by_colour = sclip(cv(matches, combine))
    red_map = ss(by_colour, 1, 2)
    button_mask = sclip(sumch(by_colour, 2))

    button_features = sc([row_pixel, col_pixel, red_map, one_s])
    br_pixel, bc_pixel, bred, bpresent = extract_points(
        "button", button_mask, button_features, K
    )
    br = vf(br_pixel, 1.0/63.0)
    bc = vf(bc_pixel, 1.0/63.0)

    # Pair aligned red and blue controls. Overlapping controls are
    # treated as one compound primitive.
    available_reds = vm(bred, bpresent)
    primitive_buttons = []
    group_present = []

    for g in range(G):
        scores = de(
            available_reds, np.eye(K, dtype=np.float32)*1000.0,
            -np.arange(K, dtype=np.float32)
        )
        right = vm(
            argmax_v(scores),
            vgt(total(available_reds), 0.5)
        )
        exists = total(right)
        rr = total(vm(br, right))
        cc = total(vm(bc, right))
        dr = vabs(vminus(br, rr))
        dc = vabs(vminus(bc, cc))
        alignment = va(
            vf(vm(dr, dc), 10.0),
            vf(va(dr, dc), 0.05)
        )
        blue_available = vm(vnot(bred), bpresent)
        left_scores = vminus(
            vf(alignment, -1.0),
            vf(vnot(blue_available), 1000.0)
        )
        left = vm(argmax_v(left_scores), exists)
        primitive_buttons.extend([right, left])
        group_present.append(exists)
        available_reds = vm(available_reds, vnot(right))

    # ----------------------------------------------------------
    # Tile anchors and corner-marked target anchors.
    # ----------------------------------------------------------
    colours = [0, 1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 15]
    C = len(colours)
    select = np.zeros((C, 16, 1, 1), np.float32)
    for i, colour in enumerate(colours):
        select[i, colour, 0, 0] = 1.0
    planes = cv("x0", select)

    tw = np.zeros((2*C, C, 7, 7), np.float32)
    tb = np.zeros(2*C, np.float32)
    for s, width in enumerate((2, 4)):
        for c in range(C):
            out = s*C+c
            for dy in range(width):
                for dx in range(width):
                    tw[out, c, dy+3, dx+3] = 1.0
            for t in range(width):
                tw[out, c, 2, t+3] = -1.0
                tw[out, c, t+3, 2] = -1.0
            tb[out] = 1.0-width*width

    tile_matches = su("relu", cv(planes, tw, tb))
    small_tiles = ss(tile_matches, 0, C)
    large_tiles = ss(tile_matches, C, 2*C)
    tiles = sb(
        "add",
        sb("mul", small_tiles, unscaled),
        sb("mul", large_tiles, scaled)
    )
    anchors = sclip(sumch(tiles, C))
    colour_image = cv(
        tiles,
        np.asarray([c+1 for c in colours], np.float32).reshape(1, C, 1, 1)
    )

    corner_planes = sc([ss("x0", 11, 12), ss("x0", 12, 13)])
    hw = np.zeros((2, 2, 5, 5), np.float32)
    vw = np.zeros_like(hw)
    for c in range(2):
        hw[c, c, 2, 1] = hw[c, c, 2, 4] = 1.0
        vw[c, c, 1, 2] = vw[c, c, 4, 2] = 1.0
    horizontal = su("relu", cv(corner_planes, hw, [-1.0, -1.0]))
    small_targets = su("relu", cv(horizontal, vw, [-1.0, -1.0]))

    goal_channel = colours.index(11)
    small_goal_anchors = ss(small_tiles, goal_channel, goal_channel+2)
    hw2 = np.zeros((2, 2, 7, 7), np.float32)
    vw2 = np.zeros_like(hw2)
    for c in range(2):
        hw2[c, c, 3, 0] = hw2[c, c, 3, 6] = 1.0
        vw2[c, c, 0, 3] = vw2[c, c, 6, 3] = 1.0
    large_horizontal = su(
        "relu", cv(small_goal_anchors, hw2, [-1.0, -1.0])
    )
    centres = su("relu", cv(large_horizontal, vw2, [-1.0, -1.0]))
    shift = np.zeros((2, 2, 3, 3), np.float32)
    for c in range(2):
        shift[c, c, 2, 2] = 1.0
    large_targets = cv(centres, shift)
    targets = sb(
        "add",
        sb("mul", small_targets, unscaled),
        sb("mul", large_targets, scaled)
    )
    target_image = cv(
        targets,
        np.array([1.0, 2.0], np.float32).reshape(1, 2, 1, 1)
    )

    anchor_features = sc([
        row_pixel, col_pixel, colour_image, target_image
    ])
    ar, ac, acolour, atarget = extract_points(
        "anchor", anchors, anchor_features, N
    )

    # ----------------------------------------------------------
    # A logical 64x64 tensor represents pairs of anchor indices.
    # Only its leading N x N square carries pair states.
    # ----------------------------------------------------------
    row_index = sf(row_pixel, 1, 1.0, 1.0)
    col_index = sf(col_pixel, 1, 1.0, 1.0)
    index_vector = de(
        "avail", np.zeros((N, 8), np.float32),
        np.arange(1, N+1, dtype=np.float32)
    )
    row_basis = seq(row_index, index_vector, N)
    col_basis = seq(col_index, index_vector, N)

    def embed_row(x):
        return sumch(sb("mul", row_basis, x), N)

    def embed_col(x):
        return sumch(sb("mul", col_basis, x), N)

    def read_row(x):
        return vf(pool(sb("mul", row_basis, x), N, mean=True), 64.0)

    def read_col(x):
        return vf(pool(sb("mul", col_basis, x), N, mean=True), 64.0)

    presence = vgt(acolour, 0.5)
    present_row = embed_row(presence)
    present_col = embed_col(presence)
    pair_present = sb("mul", present_row, present_col)
    diagonal = seq(row_index, col_index)

    source_y = embed_row(ar)
    destination_y = embed_col(ar)
    source_x = embed_row(ac)
    destination_x = embed_col(ac)
    dy = sb("sub", destination_y, source_y)
    dx = sb("sub", destination_x, source_x)
    separation = sb("add", su("abs", dy), su("abs", dx))
    preference = sb(
        "add",
        sf(separation, 1, -1.0, 256.0),
        sf(col_index, 1, -1.0/1024.0)
    )

    step = vf(scaled, 3.0, 3.0)
    offsets = [
        (0, 0), (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)
    ]
    dy_offsets = de(
        step, np.asarray([d[0] for d in offsets], np.float32)[:, None]
    )
    dx_offsets = de(
        step, np.asarray([d[1] for d in offsets], np.float32)[:, None]
    )
    error = sb(
        "add",
        su("abs", sb("sub", dy, dy_offsets)),
        su("abs", sb("sub", dx, dx_offsets))
    )
    neighbours = sb(
        "mul",
        su("relu", sf(error, 9, -2.0, 1.0)),
        pair_present
    )
    neighbour_any = sclip(sumch(neighbours, 9))
    far_pairs = sb(
        "mul", pair_present, sb("sub", one_s, neighbour_any)
    )

    old_row = embed_row(previous_colours)
    old_col = embed_col(previous_colours)
    new_row = embed_row(acolour)
    new_col = embed_col(acolour)
    right_compatible = seq(old_row, new_col)
    left_compatible = seq(old_col, new_row)

    # ----------------------------------------------------------
    # Shared bit decoder.
    # Nine local candidate flags occupy integer bits; a cached
    # non-local destination occupies the fractional part.
    # ----------------------------------------------------------
    selected_groups = []
    new_counts = []
    input_code_maps = []

    for g in range(G):
        count = vs(count_vector, g, g+1)
        learned = vgt(count, 0.5)
        previous_right = vs(previous_action, 2*g, 2*g+1)
        previous_left = vs(previous_action, 2*g+1, 2*g+2)
        selected = va(previous_right, previous_left)
        selected_groups.append(selected)
        new_counts.append(va(count, selected))
        input_code = va(
            vm(stored_codes[g], learned),
            vm(vf(one_v, 511.0), vnot(learned))
        )
        input_code_maps.append(embed_row(input_code))

    decoder_input = sc(input_code_maps)
    bit_cursor_bias = np.zeros(9*G, np.float32)
    bit_shift = np.zeros((9*G, 9*G), np.float32)
    bit_expand = np.zeros((9*G, G, 1, 1), np.float32)
    for g in range(G):
        bit_cursor_bias[9*g] = 1.0
        for i in range(9):
            bit_expand[9*g+i, g, 0, 0] = 1.0
            if i:
                bit_shift[9*g+i, 9*g+i-1] = 1.0

    bit_cursor0 = de(
        "avail", np.zeros((9*G, 8), np.float32), bit_cursor_bias
    )
    bit_power0 = vf(one_v, 256.0)
    sizes["decode_cursor"] = 9*G
    sizes["decode_power"] = 1

    decoder_body = []
    output[0] = decoder_body
    digit_difference = sb("sub", "decode_remainder", "decode_power")
    digit = sgt(digit_difference, -0.5, 4.0, G)
    remainder_next = sb(
        "sub", "decode_remainder", sb("mul", digit, "decode_power")
    )
    expanded_digit = cv(digit, bit_expand)
    bits_next = sb(
        "add", "decode_bits",
        sb("mul", expanded_digit, "decode_cursor")
    )
    cursor_next = de("decode_cursor", bit_shift)
    power_next = vf("decode_power", 0.5)

    output[0] = graph
    nd(
        "loop", name="decode", T=9,
        state=[
            {"id": "decode_remainder", "init": decoder_input},
            {"id": "decode_bits", "zeros": [9*G, 64, 64]},
            {"id": "decode_cursor", "init": bit_cursor0},
            {"id": "decode_power", "init": bit_power0},
        ],
        body=decoder_body,
        next={
            "decode_remainder": remainder_next,
            "decode_bits": bits_next,
            "decode_cursor": cursor_next,
            "decode_power": power_next,
        }
    )

    # A second difference of a ramp supplies compact index lookups.
    # Its boundary artefact lies outside the active N x N square,
    # where every planner state is zero.
    row_lookup_kernel = np.zeros((N, N, 3, 3), np.float32)
    col_lookup_kernel = np.zeros_like(row_lookup_kernel)
    for i in range(N):
        row_lookup_kernel[i, i, 0, 1] = 1.0
        row_lookup_kernel[i, i, 1, 1] = -2.0
        row_lookup_kernel[i, i, 2, 1] = 1.0
        col_lookup_kernel[i, i, 1, 0] = 1.0
        col_lookup_kernel[i, i, 1, 1] = -2.0
        col_lookup_kernel[i, i, 1, 2] = 1.0

    def make_lookups(pvector):
        row_ramp = su("relu", sb("sub", row_index, pvector))
        col_ramp = su("relu", sb("sub", col_index, pvector))
        return (
            cv(row_ramp, row_lookup_kernel),
            cv(col_ramp, col_lookup_kernel),
        )

    def row_winner(score, allowed):
        best = nd("rowmax", [score])
        winner = su(
            "relu", sf(sb("sub", score, best), 1, 4096.0, 1.0)
        )
        return sb("mul", winner, allowed)

    learned_codes = []
    lookup_masks = []
    descending_powers = np.asarray(
        [2**i for i in range(8, -1, -1)], np.float32
    )

    for g in range(G):
        selected = selected_groups[g]
        previous_right = vs(previous_action, 2*g, 2*g+1)
        previous_left = vs(previous_action, 2*g+1, 2*g+2)

        old_bits = ss("decode.decode_bits", 9*g, 9*g+9)
        old_fraction = ss("decode.decode_remainder", g, g+1)
        old_long_index = sf(old_fraction, 1, 128.0)
        old_long_mask = seq(old_long_index, col_index)

        compatible = sb(
            "add",
            sb(
                "add",
                sb("mul", right_compatible, previous_right),
                sb("mul", left_compatible, previous_left)
            ),
            sb("mul", one_s, vnot(selected))
        )
        confirmations = nd(
            "rowsum", [sb("mul", neighbours, compatible)]
        )
        surviving_bits = sb("mul", old_bits, confirmations)
        surviving_bits = sb("mul", surviving_bits, present_row)
        local_allowed = sumch(
            sb("mul", neighbours, surviving_bits), 9
        )

        far_allowed = sb("mul", far_pairs, compatible)
        far_score = sb("mul", far_allowed, preference)
        far_choice = row_winner(far_score, far_allowed)
        old_long_ok = nd(
            "rowsum", [sb("mul", old_long_mask, far_allowed)]
        )
        old_long_ok = sb("min2", old_long_ok, one_v)
        long_choice = sb(
            "add",
            sb("mul", old_long_mask, old_long_ok),
            sb("mul", far_choice, sb("sub", one_s, old_long_ok))
        )

        long_index = nd(
            "rowsum", [sb("mul", long_choice, col_index)]
        )
        flag_code = cv(
            surviving_bits, descending_powers.reshape(1, 9, 1, 1)
        )
        candidate_code = read_row(
            sb("add", flag_code, sf(long_index, 1, 1.0/128.0))
        )
        learned_codes.append(
            va(
                vm(stored_codes[g], vnot(selected)),
                vm(candidate_code, selected)
            )
        )

        allowed = sb("add", local_allowed, long_choice)
        fallback = sb(
            "mul", diagonal, sb("sub", one_s, present_row)
        )
        allowed = sb("add", allowed, fallback)
        score = sb(
            "add",
            sb(
                "add",
                sb(
                    "mul", local_allowed,
                    sf(preference, 1, 1.0, 1024.0)
                ),
                sb("mul", long_choice, preference)
            ),
            sf(fallback, 1, 2048.0)
        )
        permutation = row_winner(score, allowed)
        forward_index = nd(
            "rowsum", [sb("mul", permutation, col_index)]
        )
        forward = read_row(forward_index)
        lookup_masks.append(make_lookups(forward))

        if g < 2:
            inverse_index = nd(
                "colmax", [sb("mul", permutation, row_index)]
            )
            inverse_exists = nd("colmax", [permutation])
            inverse_index = sb(
                "add", inverse_index,
                sb(
                    "mul", col_index,
                    sb("sub", one_s, inverse_exists)
                )
            )
            inverse = read_col(inverse_index)
            lookup_masks.append(make_lookups(inverse))

    # ----------------------------------------------------------
    # Current goal pair and terminal pair states.
    # ----------------------------------------------------------
    goals = va(
        veq(acolour, vf(one_v, 12.0)),
        veq(acolour, vf(one_v, 13.0))
    )
    goal_rows = embed_row(goals)
    logical_rank = sf(row_index, 1, -1.0, 64.0)

    first_rank = sb("mul", goal_rows, logical_rank)
    first_best = pool(first_rank, 1)
    first_goal = sb(
        "mul", goal_rows,
        su("relu", sf(sb("sub", first_rank, first_best), 1, 4.0, 1.0))
    )
    remaining_goals = su("relu", sb("sub", goal_rows, first_goal))
    second_rank = sb("mul", remaining_goals, logical_rank)
    second_best = pool(second_rank, 1)
    second_goal = sb(
        "mul", remaining_goals,
        su("relu", sf(sb("sub", second_rank, second_best), 1, 4.0, 1.0))
    )
    has_second = pool(second_goal, 1)
    second_goal = sb(
        "add", second_goal, sb("mul", first_goal, vnot(has_second))
    )
    second_column = embed_col(read_row(second_goal))
    current_pair = sb("mul", first_goal, second_column)

    first_type = vf(
        pool(sb("mul", new_row, first_goal), 1), 1.0, -11.0
    )
    second_type = vf(
        pool(sb("mul", new_row, second_goal), 1), 1.0, -11.0
    )
    target_row = embed_row(atarget)
    target_col = embed_col(atarget)
    terminal_row = seq(target_row, first_type)
    terminal_col = seq(target_col, second_type)
    terminal_col = sb(
        "add",
        sb("mul", terminal_col, has_second),
        sb("mul", present_col, vnot(has_second))
    )
    distinct = sb(
        "sub", one_s, sb("mul", diagonal, has_second)
    )
    terminal = sb(
        "mul",
        sb(
            "mul",
            sb("mul", terminal_row, terminal_col),
            distinct
        ),
        pair_present
    )
    reward = sf(terminal, 1, 512.0)

    # ----------------------------------------------------------
    # Bellman propagation over the joint positions of two goals.
    # ----------------------------------------------------------
    plan_body = []
    output[0] = plan_body
    q_maps = []

    for row_lookup, col_lookup in lookup_masks:
        gathered_rows = nd(
            "colsum", [sb("mul", "plan_value", row_lookup)]
        )
        remapped_rows = sumch(
            sb("mul", gathered_rows, row_basis), N
        )
        gathered_cols = nd(
            "rowsum", [sb("mul", remapped_rows, col_lookup)]
        )
        remapped = sumch(sb("mul", gathered_cols, col_basis), N)
        q_maps.append(remapped)

    best_q = q_maps[0]
    for qmap in q_maps[1:]:
        best_q = sb("max2", best_q, qmap)
    propagated = su("relu", sf(best_q, 1, 1.0, -1.0))
    next_value = sb("max2", reward, propagated)
    next_value = sb("max2", "plan_value", next_value)
    all_q = sc(q_maps)

    output[0] = graph
    nd(
        "loop", name="planner", T=96,
        state=[
            {"id": "plan_value", "init": reward},
            {"id": "plan_q", "zeros": [5, 64, 64]},
        ],
        body=plan_body,
        next={"plan_value": next_value, "plan_q": all_q}
    )

    action_values = pool(
        sb("mul", "planner.plan_q", current_pair), 5
    )
    planned_action = argmax_v(action_values)
    reachable = vgt(total(action_values), 0.5)

    # Two initial observations per available group. Additional
    # observations are requested when the current model has no route.
    count_next = vc(new_counts)
    probe_scores = []
    needs_probe = []
    for g in range(G):
        count = new_counts[g]
        active = group_present[g]
        needs_probe.append(vm(active, vnot(vgt(count, 1.5))))
        probe_scores.append(
            va(
                vf(count, -10.0, 100.0-g),
                vf(vnot(active), -10000.0)
            )
        )

    probe_group = argmax_v(vc(probe_scores))
    group_to_action = np.zeros((5, G), np.float32)
    group_to_action[0, 0] = 1.0
    group_to_action[2, 1] = 1.0
    group_to_action[4, 2] = 1.0
    probe_action = de(probe_group, group_to_action)
    probing = vclip(
        va(vgt(total(vc(needs_probe)), 0.5), vnot(reachable))
    )
    chosen_action = va(
        vm(probe_action, probing),
        vm(planned_action, vnot(probing))
    )

    # Primitive order: R0,L0,R1,L1,R2.
    button_choice = None
    for i in range(5):
        contribution = vm(
            primitive_buttons[i], vs(chosen_action, i, i+1)
        )
        button_choice = (
            contribution if button_choice is None
            else va(button_choice, contribution)
        )

    click_row = total(vm(br_pixel, button_choice))
    click_col = total(vm(bc_pixel, button_choice))
    click_error = sb(
        "add",
        su("abs", sb("sub", row_pixel, click_row)),
        su("abs", sb("sub", col_pixel, click_col))
    )
    click_point = su("relu", sf(click_error, 1, -2.0, 1.0))
    nd("mul", [click_point, button_mask], name="click")

    chosen_six = vc([chosen_action, zero_v])
    vc(
        [acolour] + learned_codes + [count_next, chosen_six],
        name="mem_next"
    )

    logits_bias = np.full(8, -10.0, np.float32)
    logits_bias[6] = 10.0
    de(
        "avail", np.zeros((8, 8), np.float32),
        logits_bias, name="logits"
    )

    # Static dead-code elimination, including loop bodies.
    def prune(nodes, required):
        needed = set(required)
        kept = []

        for item in reversed(nodes):
            if item["op"] == "loop":
                finals = {
                    item["id"] + "." + s["id"]
                    for s in item["state"]
                }
                if not (finals & needed) and item["id"] not in needed:
                    continue

                body, external = prune(
                    item["body"], set(item["next"].values())
                )
                state_ids = {s["id"] for s in item["state"]}
                needed.difference_update(finals)
                needed.discard(item["id"])
                needed.update(external - state_ids)
                for s in item["state"]:
                    if "init" in s:
                        needed.add(s["init"])
                copied = dict(item)
                copied["body"] = body
                kept.append(copied)
            elif item["id"] in needed:
                needed.remove(item["id"])
                needed.update(item.get("in", []))
                kept.append(item)

        kept.reverse()
        return kept, needed

    graph, _ = prune(graph, {"logits", "click", "mem_next"})

    used_parameters = set()

    def collect(nodes):
        for item in nodes:
            for field in ("w", "b"):
                if field in item:
                    used_parameters.add(item[field])
            if item["op"] == "loop":
                collect(item["body"])

    collect(graph)
    params = {p: a for p, a in params.items() if p in used_parameters}

    return {
        "graph": graph,
        "memory_size": M,
        "params": params,
    }
