import numpy as np


def build():
    graph = []
    params = {}
    nc = 0
    pc = 0

    def par(value):
        nonlocal pc
        pc += 1
        name = "p%d" % pc
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def op(kind, *inputs, **fields):
        nonlocal nc
        nc += 1
        name = "n%d" % nc
        nd = {"id": name, "op": kind, "in": list(inputs)}
        nd.update(fields)
        graph.append(nd)
        return name

    def dense(x, w, b=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return op("dense", x, **fields)

    def conv(x, w, b=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return op("conv", x, **fields)

    def sl(x, a, b):
        return op("slice", x, start=a, stop=b)

    def vs(x, a, b):
        return op("vslice", x, start=a, stop=b)

    def add(a, b):
        return op("add", a, b)

    def sub(a, b):
        return op("sub", a, b)

    def mul(a, b):
        return op("mul", a, b)

    def va(a, b):
        return op("vadd", a, b)

    def vm(a, b):
        return op("vmul", a, b)

    def scalar(x, factor=1.0, bias=0.0):
        return dense(x, [[factor]], [bias])

    def scale(x, size, factor=1.0, bias=0.0):
        return dense(
            x,
            np.eye(size, dtype=np.float32) * np.float32(factor),
            np.full(size, bias, dtype=np.float32),
        )

    def spatial_scalar(x, factor=1.0, bias=0.0):
        return conv(
            x,
            np.array([[[[factor]]]], dtype=np.float32),
            np.array([bias], dtype=np.float32),
        )

    def vneg(x, size):
        return scale(x, size, -1.0)

    def vd(a, b, size):
        return va(a, vneg(b, size))

    def inv(x):
        return scalar(x, -1.0, 1.0)

    def clip(x, size=1):
        r = op("vrelu", x)
        return scale(
            op("vrelu", scale(r, size, -1.0, 1.0)),
            size,
            -1.0,
            1.0,
        )

    def vabs(x, size):
        return va(op("vrelu", x), op("vrelu", vneg(x, size)))

    def vmin(a, b, size):
        return vd(a, op("vrelu", vd(a, b, size)), size)

    def vmax(a, b, size):
        return va(b, op("vrelu", vd(a, b, size)))

    def choose(flag, yes, no, size):
        return va(vm(flag, yes), vm(inv(flag), no))

    def sumvec(x, size):
        return dense(x, np.ones((1, size), dtype=np.float32))

    def norm1(x):
        return sumvec(vabs(x, 2), 2)

    def exact_zero(x):
        return clip(scalar(x, -63.0, 2.0))

    def spclip(x):
        return op("min2", op("relu", x), one)

    def position(y, x):
        return op("vconcat", y, x)

    def pick(mask, score):
        weighted = mul(mask, score)
        maximum = op("gmax", weighted)
        difference = op("abs", sub(weighted, maximum))
        chosen = op("relu", spatial_scalar(difference, -1.0e7, 1.0))
        chosen = mul(chosen, mask)
        y = op("gmax", mul(chosen, row))
        x = op("gmax", mul(chosen, col))
        return position(y, x)

    def point_map(pt):
        y = vs(pt, 0, 1)
        x = vs(pt, 1, 2)
        ym = op("relu", spatial_scalar(
            op("abs", sub(row, y)), -63.0, 1.0
        ))
        xm = op("relu", spatial_scalar(
            op("abs", sub(col, x)), -63.0, 1.0
        ))
        return mul(ym, xm)

    def exclude(mask, pt):
        return mul(mask, sub(one, point_map(pt)))

    def bbox(mask):
        ymin = inv(op("gmax", mul(
            mask, spatial_scalar(row, -1.0, 1.0)
        )))
        xmin = inv(op("gmax", mul(
            mask, spatial_scalar(col, -1.0, 1.0)
        )))
        ymax = op("gmax", mul(mask, row))
        xmax = op("gmax", mul(mask, col))
        return position(ymin, xmin), position(ymax, xmax)

    def rotate_vector(v):
        return dense(v, [[0.0, 1.0], [-1.0, 0.0]])

    offset = 0
    slots = []
    updates = {}

    def memory(name, length):
        nonlocal offset
        value = vs("mem", offset, offset + length)
        slots.append((name, length, value))
        offset += length
        return value

    ready_old = memory("ready", 1)
    mode_old = memory("mode", 1)

    u0_old = memory("u0", 2)
    u1_old = memory("u1", 2)
    umin_old = memory("umin", 2)
    umax_old = memory("umax", 2)
    t0_old = memory("t0", 2)
    t1_old = memory("t1", 2)
    flip_old = memory("flip", 1)
    uvalid_old = memory("uvalid", 1)

    stages_old = memory("stages", 4)
    triple_old = memory("triple", 1)
    a0_old = memory("a0", 2)
    a1_old = memory("a1", 2)
    amin_old = memory("amin", 2)
    amax_old = memory("amax", 2)
    b0_old = memory("b0", 2)
    b1_old = memory("b1", 2)
    bclick_old = memory("bclick", 2)
    cclick_old = memory("cclick", 2)
    bv_old = memory("bv", 2)
    cv_old = memory("cv", 2)
    bt_old = memory("bt", 1)
    ct_old = memory("ct", 1)

    fallback_target_old = memory("fallback_target", 2)
    fallback_backup_old = memory("fallback_backup", 2)

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    one = sl("coords", 2, 3)
    black = sl("x0", 0, 1)
    green = sl("x0", 3, 4)
    grey = sl("x0", 4, 5)
    blue = sl("x0", 8, 9)

    top = spclip(spatial_scalar(row, 63.0, -1.0))
    bottom = spclip(spatial_scalar(row, -63.0, 62.0))
    left = spclip(spatial_scalar(col, 63.0, -1.0))
    right = spclip(spatial_scalar(col, -63.0, 62.0))
    interior = mul(mul(top, bottom), mul(left, right))
    grey_inside = mul(grey, interior)
    has_grey = op("gmax", grey_inside)

    corner_y = op("relu", spatial_scalar(row, -63.0, 1.0))
    corner_x = op("relu", spatial_scalar(col, -63.0, 1.0))
    corner = mul(corner_y, corner_x)
    bg_vector = op("gmax", mul("x0", corner))
    bg_planes = mul("x0", bg_vector)
    bg = conv(
        bg_planes, np.ones((1, 16, 1, 1), dtype=np.float32)
    )

    selected_body = op("relu", sub(
        sub(sub(sub(one, bg), grey), blue), green
    ))
    selected_body = mul(selected_body, interior)
    near_body = conv(
        selected_body, np.ones((1, 1, 7, 7), dtype=np.float32)
    )
    near_body = spclip(near_body)

    union = spclip(add(blue, green))
    selected_union = spclip(add(blue, mul(green, near_body)))
    corner_kernel = np.zeros((1, 1, 3, 3), dtype=np.float32)
    corner_kernel[0, 0, 1, 1] = 1.0
    corner_kernel[0, 0, 0, 1] = -1.0
    corner_kernel[0, 0, 1, 0] = -1.0
    blue_corners = op("relu", conv(blue, corner_kernel))
    selected_corners = op("relu", conv(selected_union, corner_kernel))
    union_corners = op("relu", conv(union, corner_kernel))

    rank_w = np.zeros((1, 3, 1, 1), dtype=np.float32)
    rank_w[0, 0, 0, 0] = 1.0
    rank_w[0, 1, 0, 0] = 0.01
    rank = conv("coords", rank_w)
    first_score = spatial_scalar(rank, -1.0, 3.0)

    blue0 = pick(blue_corners, first_score)
    blue_rest = exclude(blue_corners, blue0)
    blue1 = pick(blue_rest, first_score)
    blue_rest2 = exclude(blue_rest, blue1)
    has_blue = op("gmax", blue_corners)
    has_blue2 = op("gmax", blue_rest)
    has_blue3 = op("gmax", blue_rest2)
    exactly_two = vm(vm(has_blue, has_blue2), inv(has_blue3))

    sel0 = pick(selected_corners, first_score)
    sel_rest = exclude(selected_corners, sel0)
    sel1 = pick(sel_rest, first_score)

    selected_pixels = spclip(add(selected_body, selected_union))
    selected_pixels = mul(selected_pixels, interior)
    selected_min, selected_max_pixels = bbox(selected_pixels)

    has_black = op("gmax", mul(black, interior))
    mode = choose(ready_old, mode_old, has_grey, 1)
    unmasked = inv(mode)
    selected_exists = va(
        vm(mode, has_blue), vm(unmasked, has_black)
    )
    init = vm(inv(ready_old), selected_exists)
    ready = vmax(ready_old, selected_exists, 1)
    updates["ready"] = ready
    updates["mode"] = choose(init, mode, mode_old, 1)

    action_w = np.zeros((3, 8), dtype=np.float32)
    cell = np.float32(3.0 / 63.0)
    action_w[0, 1] = -cell
    action_w[0, 2] = cell
    action_w[1, 3] = -cell
    action_w[1, 4] = cell
    action_w[2, 5] = 1.0
    previous = dense("aprev", action_w)
    movement = vs(previous, 0, 2)
    rotated = vs(previous, 2, 3)

    # Unmasked two-piece controller.
    near_black = conv(
        mul(black, interior), np.ones((1, 1, 7, 7), dtype=np.float32)
    )
    near_black = spclip(near_black)
    active = mul(union_corners, near_black)
    inactive = mul(blue_corners, sub(one, near_black))
    ui0 = pick(active, first_score)
    active_rest = exclude(active, ui0)
    ui1 = pick(active_rest, first_score)
    ti0 = pick(inactive, first_score)
    inactive_rest = exclude(inactive, ti0)
    ti1 = pick(inactive_rest, first_score)
    ui_valid = vm(op("gmax", active_rest), op("gmax", inactive_rest))

    unmasked_pixels = spclip(add(
        mul(black, interior), mul(union, near_black)
    ))
    unmasked_pixels = mul(unmasked_pixels, interior)
    ui_min, ui_max_pixels = bbox(unmasked_pixels)
    ui_max = scale(ui_max_pixels, 2, 1.0, -2.0 / 63.0)

    lower_move = scale(umin_old, 2, -1.0, 2.0 / 63.0)
    upper_move = scale(umax_old, 2, -1.0, 59.0 / 63.0)
    move_clamped = vmin(vmax(movement, lower_move, 2), upper_move, 2)

    moved0 = va(u0_old, move_clamped)
    moved1 = va(u1_old, move_clamped)
    moved_min = va(umin_old, move_clamped)
    moved_max = va(umax_old, move_clamped)

    box_height = vd(umax_old, umin_old, 2)
    height_y = vs(box_height, 0, 1)
    zero1 = scalar(height_y, 0.0)
    rotation_offset = position(zero1, height_y)

    rotated0 = va(
        va(umin_old, rotate_vector(vd(u0_old, umin_old, 2))),
        rotation_offset,
    )
    rotated1 = va(
        va(umin_old, rotate_vector(vd(u1_old, umin_old, 2))),
        rotation_offset,
    )
    rotated_min = umin_old
    rotated_max = va(
        umin_old,
        dense(box_height, [[0.0, 1.0], [1.0, 0.0]]),
    )
    overflow = op(
        "vrelu", scale(rotated_max, 2, 1.0, -59.0 / 63.0)
    )
    rotated0 = vd(rotated0, overflow, 2)
    rotated1 = vd(rotated1, overflow, 2)
    rotated_min = vd(rotated_min, overflow, 2)
    rotated_max = vd(rotated_max, overflow, 2)

    pred0 = choose(rotated, rotated0, moved0, 2)
    pred1 = choose(rotated, rotated1, moved1, 2)
    pred_min = choose(rotated, rotated_min, moved_min, 2)
    pred_max = choose(rotated, rotated_max, moved_max, 2)

    u0 = choose(ready_old, pred0, ui0, 2)
    u1 = choose(ready_old, pred1, ui1, 2)
    umin = choose(ready_old, pred_min, ui_min, 2)
    umax = choose(ready_old, pred_max, ui_max, 2)
    t0 = choose(ready_old, t0_old, ti0, 2)
    t1 = choose(ready_old, t1_old, ti1, 2)
    uvalid = choose(ready_old, uvalid_old, ui_valid, 1)

    udest0 = choose(flip_old, t1, t0, 2)
    udest1 = choose(flip_old, t0, t1, 2)
    uvec_error = norm1(vd(
        vd(u1, u0, 2), vd(udest1, udest0, 2), 2
    ))
    umismatch = inv(exact_zero(uvec_error))
    ualign_error = va(
        norm1(vd(u0, udest0, 2)),
        norm1(vd(u1, udest1, 2)),
    )
    ualigned = exact_zero(ualign_error)
    urotate = clip(va(umismatch, ualigned))
    udelta = vd(udest0, u0, 2)
    flip_next = choose(ualigned, inv(flip_old), flip_old, 1)

    updates["u0"] = choose(selected_exists, u0, u0_old, 2)
    updates["u1"] = choose(selected_exists, u1, u1_old, 2)
    updates["umin"] = choose(selected_exists, umin, umin_old, 2)
    updates["umax"] = choose(selected_exists, umax, umax_old, 2)
    updates["t0"] = choose(init, t0, t0_old, 2)
    updates["t1"] = choose(init, t1, t1_old, 2)
    updates["flip"] = choose(unmasked, flip_next, flip_old, 1)
    updates["uvalid"] = choose(init, uvalid, uvalid_old, 1)

    # Masked three-piece controller.
    s1 = vs(stages_old, 0, 1)
    s2 = vs(stages_old, 1, 2)
    s3 = vs(stages_old, 2, 3)
    s4 = vs(stages_old, 3, 4)
    stage_sum = sumvec(stages_old, 4)
    s0 = inv(clip(stage_sum))
    triple = choose(ready_old, triple_old, exactly_two, 1)
    updates["triple"] = choose(init, exactly_two, triple_old, 1)

    a0 = choose(s0, blue0, a0_old, 2)
    a1 = choose(s0, blue1, a1_old, 2)
    amin = choose(s0, selected_min, amin_old, 2)
    amax = choose(s0, selected_max_pixels, amax_old, 2)
    a0y = vs(a0, 0, 1)
    a0x = vs(a0, 1, 2)
    near_a = add(
        op("abs", sub(row, a0y)),
        op("abs", sub(col, a0x)),
    )
    b_score = add(
        spatial_scalar(near_a, -1.0, 3.0),
        spatial_scalar(rank, -0.0001),
    )
    b_map = mul(grey_inside, b_score)
    bclick = pick(grey_inside, b_score)

    aminy = vs(amin, 0, 1)
    aminx = vs(amin, 1, 2)
    amaxy = vs(amax, 0, 1)
    amaxx = vs(amax, 1, 2)
    outside = add(
        add(op("relu", sub(aminy, row)),
            op("relu", sub(row, amaxy))),
        add(op("relu", sub(aminx, col)),
            op("relu", sub(col, amaxx))),
    )
    outside = spclip(spatial_scalar(outside, 63.0))
    c_score = add(
        spatial_scalar(outside, 2.0, 0.1),
        spatial_scalar(near_a, 0.5),
    )
    c_score = add(c_score, spatial_scalar(rank, -0.0001))
    c_map = mul(grey_inside, c_score)
    cclick = pick(grey_inside, c_score)

    b0 = choose(s1, blue0, b0_old, 2)
    b1 = choose(s1, blue1, b1_old, 2)
    a_vec = vd(a1, a0, 2)
    b_vec = vd(b1, b0, 2)
    c_vec = vd(blue1, blue0, 2)
    vectors = op("vconcat", a_vec, b_vec, c_vec)

    rotations = [
        np.array([[1, 0], [0, 1]], dtype=np.float32),
        np.array([[0, 1], [-1, 0]], dtype=np.float32),
        np.array([[-1, 0], [0, -1]], dtype=np.float32),
        np.array([[0, -1], [1, 0]], dtype=np.float32),
    ]

    candidates = []
    for cs in (1.0, -1.0):
        for bs in (1.0, -1.0):
            for rb in range(4):
                for rc in range(4):
                    candidates.append((bs, cs, rb, rc))

    wy = np.zeros((64, 6), dtype=np.float32)
    wx = np.zeros((64, 6), dtype=np.float32)
    bwy = np.zeros((64, 2), dtype=np.float32)
    bwx = np.zeros((64, 2), dtype=np.float32)
    cwy = np.zeros((64, 2), dtype=np.float32)
    cwx = np.zeros((64, 2), dtype=np.float32)
    bturns = np.zeros((1, 64), dtype=np.float32)
    cturns = np.zeros((1, 64), dtype=np.float32)

    for i, (bs, cs, rb, rc) in enumerate(candidates):
        br = bs * rotations[rb]
        cr = cs * rotations[rc]
        wy[i, 0] = 1.0
        wx[i, 1] = 1.0
        wy[i, 2:4] = br[0]
        wx[i, 2:4] = br[1]
        wy[i, 4:6] = cr[0]
        wx[i, 4:6] = cr[1]
        bwy[i] = br[0]
        bwx[i] = br[1]
        cwy[i] = cr[0]
        cwx[i] = cr[1]
        bturns[0, i] = rb
        cturns[0, i] = rc

    ey = dense(vectors, wy)
    ex = dense(vectors, wx)
    errors = va(vabs(ey, 64), vabs(ex, 64))
    valid_candidates = clip(scale(errors, 64, -63.0, 2.0), 64)
    prefix_w = np.tril(np.ones((64, 64), dtype=np.float32), k=-1)
    prefix = dense(valid_candidates, prefix_w)
    first_candidate = vm(
        valid_candidates,
        clip(scale(prefix, 64, -1.0, 1.0), 64),
    )

    def selected_sum(values):
        return sumvec(vm(first_candidate, values), 64)

    chosen_bv = position(
        selected_sum(dense(b_vec, bwy)),
        selected_sum(dense(b_vec, bwx)),
    )
    chosen_cv = position(
        selected_sum(dense(c_vec, cwy)),
        selected_sum(dense(c_vec, cwx)),
    )
    chosen_bt = dense(first_candidate, bturns)
    chosen_ct = dense(first_candidate, cturns)

    bv = choose(s2, chosen_bv, bv_old, 2)
    cv = choose(s2, chosen_cv, cv_old, 2)
    bt_decrement = vm(s3, rotated)
    ct_decrement = vm(s4, rotated)
    bt_remaining = op("vrelu", vd(bt_old, bt_decrement, 1))
    ct_remaining = op("vrelu", vd(ct_old, ct_decrement, 1))
    bt = choose(s2, chosen_bt, bt_remaining, 1)
    ct = choose(s2, chosen_ct, ct_remaining, 1)
    b_turn_done = exact_zero(bt)
    c_turn_done = exact_zero(ct)

    current_vec = vd(sel1, sel0, 2)
    b_same = exact_zero(norm1(vd(current_vec, bv, 2)))
    c_same = exact_zero(norm1(vd(current_vec, cv, 2)))
    b_anchor = choose(b_same, sel0, sel1, 2)
    c_anchor = choose(c_same, sel0, sel1, 2)
    b_delta = vd(a1, b_anchor, 2)
    c_target = va(a1, bv)
    c_delta = vd(c_target, c_anchor, 2)
    b_aligned = vm(b_turn_done, exact_zero(norm1(b_delta)))
    c_aligned = vm(c_turn_done, exact_zero(norm1(c_delta)))

    begin = vm(s0, exactly_two)
    next_s1 = begin
    next_s2 = s1
    next_s3 = va(s2, vm(s3, inv(b_aligned)))
    next_s4 = va(s4, vm(s3, b_aligned))
    stages_next = op("vconcat", next_s1, next_s2, next_s3, next_s4)

    updates["stages"] = stages_next
    updates["a0"] = choose(begin, a0, a0_old, 2)
    updates["a1"] = choose(begin, a1, a1_old, 2)
    updates["amin"] = choose(begin, amin, amin_old, 2)
    updates["amax"] = choose(begin, amax, amax_old, 2)
    updates["b0"] = choose(s1, b0, b0_old, 2)
    updates["b1"] = choose(s1, b1, b1_old, 2)
    updates["bclick"] = choose(begin, bclick, bclick_old, 2)
    updates["cclick"] = choose(s1, cclick, cclick_old, 2)
    updates["bv"] = choose(s2, bv, bv_old, 2)
    updates["cv"] = choose(s2, cv, cv_old, 2)
    updates["bt"] = bt
    updates["ct"] = ct

    triple_click = clip(
        va(va(begin, s1), va(s2, vm(s3, b_aligned)))
    )
    triple_rotate = clip(
        va(
            vm(s3, inv(b_turn_done)),
            vm(s4, va(inv(c_turn_done), c_aligned)),
        )
    )
    triple_delta = va(vm(s3, b_delta), vm(s4, c_delta))
    triple_click_map = add(
        add(mul(b_map, begin), mul(c_map, s1)),
        add(
            mul(point_map(bclick_old), s2),
            mul(point_map(cclick_old), vm(s3, b_aligned)),
        ),
    )

    # Fallback for masked pieces with more than two connectors.
    fallback_target = choose(init, blue0, fallback_target_old, 2)
    fallback_backup = choose(init, blue1, fallback_backup_old, 2)
    fty = vs(fallback_target, 0, 1)
    ftx = vs(fallback_target, 1, 2)
    fdistance = add(
        op("abs", sub(row, fty)),
        op("abs", sub(col, ftx)),
    )
    fscore = add(
        spatial_scalar(fdistance, -1.0, 3.0),
        spatial_scalar(rank, -0.0001),
    )
    fsource = pick(blue_corners, fscore)
    fjoint = op("gmax", mul(point_map(fallback_target), green))
    fjoint = vm(fjoint, ready_old)

    # A black marker on a selected masked hub denotes another growth
    # stage. After the last growth action, select a grey piece.
    finished_growth = vm(rotated, inv(has_black))
    fallback_click = clip(va(va(init, fjoint), finished_growth))
    fallback_delta = vd(fallback_target, fsource, 2)
    fallback_map = mul(
        grey_inside,
        add(
            spatial_scalar(fdistance, 1.0, 1.0),
            spatial_scalar(rank, -0.0001),
        ),
    )
    fallback_target_next = choose(
        fjoint,
        choose(has_blue, fsource, fallback_backup, 2),
        fallback_target,
        2,
    )
    updates["fallback_target"] = fallback_target_next
    updates["fallback_backup"] = fallback_backup

    no_selection = inv(selected_exists)
    colored_piece = mul(
        interior,
        op("relu", sub(sub(sub(one, bg), blue), green)),
    )
    recovery_map = add(
        mul(grey_inside, mode),
        mul(colored_piece, unmasked),
    )
    recovery_map = mul(recovery_map, first_score)

    use_u = vm(unmasked, vm(uvalid, selected_exists))
    use_triple = vm(mode, vm(triple, selected_exists))
    use_fallback = vm(mode, vm(inv(triple), selected_exists))
    recover = no_selection

    direction = va(
        va(vm(use_u, udelta), vm(use_triple, triple_delta)),
        vm(use_fallback, fallback_delta),
    )
    rotate_flag = clip(va(
        va(vm(use_u, urotate), vm(use_triple, triple_rotate)),
        vm(use_fallback, has_black),
    ))
    click_flag = clip(va(
        recover,
        va(
            vm(use_triple, triple_click),
            vm(use_fallback, fallback_click),
        ),
    ))

    click_map = add(
        add(
            mul(recovery_map, recover),
            mul(triple_click_map, use_triple),
        ),
        mul(fallback_map, use_fallback),
    )

    # A level always starts with a selected piece. No selection on the
    # first observation therefore indicates a perturbed start.
    reset_flag = vm(inv(ready_old), no_selection)
    features = op(
        "vconcat", direction, rotate_flag, click_flag, reset_flag
    )
    logits_w = np.zeros((8, 5), dtype=np.float32)
    logits_b = np.zeros(8, dtype=np.float32)
    logits_b[0] = -1000.0
    logits_b[7] = -1000.0
    logits_w[0, 4] = 2000.0
    logits_w[1, 0] = -20.0
    logits_w[2, 0] = 20.0
    logits_w[3, 1] = -20.0
    logits_w[4, 1] = 20.0
    logits_w[5, 2] = 100.0
    logits_w[6, 3] = 100.0
    logits_b[5] = -0.5
    logits_b[6] = -0.5

    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [features],
        "w": par(logits_w),
        "b": par(logits_b),
    })
    graph.append({
        "id": "click",
        "op": "conv",
        "in": [click_map],
        "w": par(np.ones((1, 1, 1, 1), dtype=np.float32)),
    })

    memory_outputs = [
        updates.get(name, old) for name, length, old in slots
    ]
    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": memory_outputs,
    })

    return {"graph": graph, "memory_size": offset, "params": params}
