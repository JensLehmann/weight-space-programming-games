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

    colours = (1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    order_colours = (6, 7, 8, 9, 11, 12, 14, 15)
    eps = 1.0e-5

    CODE, ROW, COL = 0, 1, 2
    BP, BM, RP, RM = 3, 4, 5, 6
    TARGET, EMPTY, SOLID, RING = 7, 8, 9, 10
    SELECT_SOLID, SELECT_RING = 11, 12
    FRAME_PROBE, RIGHT_Y_PLUS, RIGHT_Y_BASE = 13, 14, 15
    LOWER_EDGE_PROBE = 27

    def first_affine(out, terms, bias=0.0):
        for inp, value in terms.items():
            p["W1"][out, inp, 2, 2] += value
        p["b1"][out] = bias

    for c in colours:
        p["W1"][CODE, c, 2, 2] = float(c)

        # Combined with W2's left shift, samples (-2, -3).
        # This distinguishes a frame boundary from a preceding tile.
        p["W1"][FRAME_PROBE, c, 0, 0] = 1.0

        # Combined with W2's (+1,+1) shift, samples (+3,+1).
        # This is the bottom edge of a 4x4 hollow tile, but lies
        # in the empty inner margin of a large program frame.
        p["W1"][LOWER_EDGE_PROBE, c, 4, 2] = 1.0

    first_affine(ROW, {16: 1.0})
    first_affine(COL, {17: 1.0})
    first_affine(BP, {16: 63.0}, -53.0)
    first_affine(BM, {16: 63.0}, -54.0)
    first_affine(RP, {17: 63.0}, -31.0)
    first_affine(RM, {17: 63.0}, -32.0)

    # Difference equals row/63 on the right half and zero on the left.
    first_affine(RIGHT_Y_PLUS, {16: 1.0, 17: 256.0}, -128.0)
    first_affine(RIGHT_Y_BASE, {17: 256.0}, -128.0)

    for i, c in enumerate(colours):
        first_affine(16 + i, {c: 1.0})

    def signature(out, tests):
        for dy, dx, allowed in tests:
            for c in allowed:
                p["W1"][out, c, 2 + dy, 2 + dx] += 1.0
        p["b1"][out] = 1.0 - len(tests)

    signature(TARGET, [
        (0, 0, colours),
        (0, 1, colours),
        (1, 0, colours),
        (1, 1, (5,)),
        (-1, 0, (5,)),
        (0, -1, (5,)),
    ])

    # Empty slots and pieces share anchor (sprite.x+1, sprite.y+1).
    signature(EMPTY, [
        (1, 1, (2,)),
        (1, 2, (2,)),
        (2, 1, (2,)),
        (2, 2, (2,)),
        (0, 1, (4,)),
        (1, 0, (4,)),
    ])

    def piece_signature(out, hollow, selected):
        edge = (0,) if selected else (0, 4)
        signature(out, [
            (0, 0, colours),
            (0, 1, colours),
            (1, 0, colours),
            (1, 1, (4,) if hollow else colours),
            (-1, 0, edge),
            (0, -1, edge),
        ])

    piece_signature(SOLID, False, False)
    piece_signature(RING, True, False)
    piece_signature(SELECT_SOLID, False, True)
    piece_signature(SELECT_RING, True, True)

    def second(out, terms, bias=0.0):
        for inp, value in terms.items():
            p["W2"][out, inp, 1, 1] += value
        p["b2"][out] = bias

    # W2:
    # 0 call destination, 1 verified board-call colour,
    # 2 target, 3 inventory literal, 4 inventory inverse-x,
    # 5 selection, 6 empty slot, 7 traversal priority,
    # 8 inverse verified board-call colour, 9 inventory x,
    # 10..20 coloured targets, 21..31 coloured inventory.
    second(0, {
        EMPTY: 1.0,
        BP: -1.0, BM: 1.0,
        RP: -1.0, RM: 1.0,
        ROW: -0.001,
        COL: -0.00001,
    })
    p["W2"][0, FRAME_PROBE, 1, 0] = -1.0

    second(1, {
        CODE: 1.0,
        RING: 32.0,
        BP: -32.0, BM: 32.0,
    }, -64.0)
    p["W2"][1, LOWER_EDGE_PROBE, 2, 2] = 32.0

    second(2, {TARGET: 1.0})

    second(3, {
        SOLID: 1.0,
        BP: 1.0, BM: -1.0,
    }, -1.0)

    stock_terms = {
        SOLID: 100.0,
        RING: 100.0,
        EMPTY: 100.0,
        BP: 100.0,
        BM: -100.0,
    }
    second(4, stock_terms, -199.0)
    p["W2"][4, COL, 1, 1] = -1.0

    second(5, {
        SELECT_SOLID: 1.0,
        SELECT_RING: 2.0,
        BP: 2.0, BM: -2.0,
    }, -2.0)

    second(6, {
        EMPTY: 1.0,
        BP: -1.0, BM: 1.0,
    })

    second(7, {
        EMPTY: 100.0,
        BP: -100.0, BM: 100.0,
        COL: -1.0,
        ROW: -0.01,
        RIGHT_Y_PLUS: 0.02,
        RIGHT_Y_BASE: -0.02,
    }, -99.0)

    second(8, {
        CODE: -1.0,
        RING: 32.0,
        BP: -32.0, BM: 32.0,
    }, -48.0)
    p["W2"][8, LOWER_EDGE_PROBE, 2, 2] = 32.0

    second(9, stock_terms, -200.0)
    p["W2"][9, COL, 1, 1] = 1.0

    for i, c in enumerate(colours):
        second(10 + i, {
            TARGET: 1.0,
            16 + i: 1.0,
            COL: -63.0 * eps,
        }, -1.0)

        second(21 + i, {
            SOLID: 1.0,
            RING: 1.0,
            BP: 1.0, BM: -1.0,
            16 + i: 1.0,
        }, -2.0)

    def local(out, inp, value=1.0):
        p["W3"][out, inp] += value

    def rowmax(out, inp, value=1.0):
        p["W3"][out, 32 + inp] += value

    def colmax(out, inp, value=1.0):
        p["W3"][out, 64 + inp] += value

    T, TC, REMAINING = 0, 1, 2
    SOURCE, SOURCE_CODE, SELECTED = 3, 4, 5
    DEST, PRIORITY, CALL_SOURCE, CALL_DEST = 6, 7, 8, 9
    AVAIL_START, ORDER_START = 10, 21
    STOCK_COUNT, CALL_COUNT_A, CALL_COUNT_B = 29, 30, 31

    colmax(T, 2)
    local(SOURCE, 3)
    rowmax(SELECTED, 5)
    local(DEST, 6)
    local(PRIORITY, 7)
    local(CALL_DEST, 0)

    for i, c in enumerate(colours):
        colmax(TC, 10 + i, float(c))
        colmax(REMAINING, 21 + i)
        local(SOURCE_CODE, 21 + i, float(c))
        local(CALL_SOURCE, 21 + i)

        out = AVAIL_START + i
        rowmax(out, 21 + i)
        colmax(out, 10 + i)
        p["b3"][out] = -1.0

    local(CALL_SOURCE, 3, -1.0)

    for j, c in enumerate(order_colours):
        i = colours.index(c)
        out = ORDER_START + j
        rowmax(out, 10 + i)
        colmax(out, 21 + i)
        p["b3"][out] = -1.0

    # Estimate regular inventory capacity from tray endpoints.
    rowmax(STOCK_COUNT, 4, 9.0)
    rowmax(STOCK_COUNT, 9, 9.0)
    p["b3"][STOCK_COUNT] = -8.0

    # Clipped column-level call count.
    for out in (CALL_COUNT_A, CALL_COUNT_B):
        colmax(out, 1)
        colmax(out, 8)
    p["b3"][CALL_COUNT_A] = -15.0
    p["b3"][CALL_COUNT_B] = -17.0

    def rec(out, inp, value, dy=0, dx=0):
        p["Wh"][out, inp, 1 + dy, 1 + dx] += value

    def drive(out, inp, value):
        p["Uh"][out, inp] += value

    # h0: target prefix rank.
    rec(0, 0, 1.0, dx=-1)
    drive(0, T, 1.0)

    # h1,h2: bidirectional inventory-plus-call counts.
    for out, dx in ((1, -1), (2, 1)):
        rec(out, out, 1.0, dx=dx)
        drive(out, REMAINING, 1.0)
        drive(out, CALL_COUNT_A, 1.0)
        drive(out, CALL_COUNT_B, -1.0)

    # h3: target-colour availability, propagated upward.
    rec(3, 3, 1.0, dy=1)
    for i in range(len(colours)):
        drive(3, AVAIL_START + i, 1.0)

    # h4: reject targets before the estimated progress index.
    drive(4, STOCK_COUNT, 4.0)
    drive(4, REMAINING, 4.0)
    drive(4, CALL_COUNT_A, 4.0)
    drive(4, CALL_COUNT_B, -4.0)
    rec(4, 1, -4.0)
    rec(4, 2, -4.0)
    rec(4, 0, -4.0)
    p["bh"][4] = 2.0

    # h5,h6: eligible targets and their prefix count.
    drive(5, T, 1.0)
    rec(5, 3, 1.0)
    rec(5, 4, -1.0)
    p["bh"][5] = -1.0

    rec(6, 6, 1.0, dx=-1)
    rec(6, 5, 1.0)

    # h7: first eligible target's colour.
    drive(7, TC, 1.0)
    rec(7, 5, 64.0)
    rec(7, 6, -32.0)
    p["bh"][7] = -32.0

    # h8,h9: requested-colour broadcast.
    rec(8, 8, 1.0, dx=-1)
    rec(9, 9, 1.0, dx=1)
    rec(8, 7, 1.0)
    rec(9, 7, 1.0)

    # h10: positive part of source colour minus requested colour.
    drive(10, SOURCE_CODE, 1.0)
    rec(10, 8, -1.0)
    rec(10, 9, -1.0)
    rec(10, 7, 1.0)

    # h11: literal-source score using an absolute colour difference.
    drive(11, SOURCE, 20.0)
    drive(11, SOURCE_CODE, 20.0)
    rec(11, 8, -20.0)
    rec(11, 9, -20.0)
    rec(11, 7, 20.0)
    rec(11, 10, -40.0)
    rec(11, 13, -1000.0)

    # h12: call destinations, gated by selected-call state.
    drive(12, CALL_DEST, 2000.0)
    rec(12, 13, 2000.0)
    p["bh"][12] = -5000.0

    # h13: selected literal=1, selected call=2.
    drive(13, SELECTED, 1.0)
    rec(13, 13, 1.0, dy=1)

    # h14: literal destinations.
    drive(14, DEST, 21.0)
    rec(14, 13, 1000.0)
    rec(14, 15, -10.0)
    p["bh"][14] = -1000.0

    # h15: target-order preference at inventory columns.
    rec(15, 15, 1.0, dy=-1)
    for j in range(len(order_colours)):
        drive(15, ORDER_START + j, 1.0)

    p["Wc"][0, CALL_SOURCE] = 100.0
    p["Wc"][0, PRIORITY] = 0.1
    p["Wc"][0, 32 + 11] = 1.0
    p["Wc"][0, 32 + 12] = 1.0
    p["Wc"][0, 32 + 14] = 1.0
    p["Wc"][0, 32 + 15] = 10.0

    # Assemble while holes remain; execute when filled.
    p["Wz"][0, 32 + DEST] = 1.0
    p["bo"][:] = -1000.0
    p["bo"][5] = 1.0
    p["bo"][6] = 0.0
    p["Wo"][6, 0] = 4.0

    # A single initial reset handles perturbed episode starts.
    p["bm"][0] = 20.0
    p["Wm"][1, 0] = 20.0
    p["bo"][0] = -100.0
    p["Wo"][0, 128] = 200.0
    p["Wo"][0, 129] = -200.0

    return p
