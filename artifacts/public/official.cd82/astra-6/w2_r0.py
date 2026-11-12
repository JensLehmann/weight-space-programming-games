import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3),
        "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {name: np.zeros(shape, np.float32)
         for name, shape in shapes.items()}

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    colours = [0, 8, 9, 11, 12, 14, 15]
    code = {c: i for i, c in enumerate(colours)}

    def k1(out, inp, dy, dx, value):
        W1[out, inp, dy + 2, dx + 2] += value

    def k2(out, inp, dy=0, dx=0, value=1.0):
        W2[out, inp, dy + 1, dx + 1] += value

    def kh(out, inp, dy=0, dx=0, value=1.0):
        Wh[out, inp, dy + 1, dx + 1] += value

    wedges = [
        (3, 4), (3, 5), (4, 6), (5, 6),
        (6, 5), (6, 4), (5, 3), (4, 3),
    ]
    representatives = [wedges[(i - 1) % 8] for i in range(8)]
    offsets = [(r - 5, c - 5) for r, c in representatives]

    # Eight local colour probes, in leading-sector order.
    for i, (dy, dx) in enumerate(offsets):
        for c, v in code.items():
            k1(i, c, dy, dx, v + 1)

    # Two coordinate hinges whose simultaneous zero set contains exactly
    # the target and canvas probe centres on the integer image grid:
    # 24*r - 31*c + 56 = 0.
    for ch, sign in ((8, 1), (27, -1)):
        k1(ch, 16, 0, 0, sign * 24 * 63)
        k1(ch, 17, 0, 0, -sign * 31 * 63)
        b1[ch] = sign * 56 - 0.25

    # Basket orientation detectors.
    inward = [
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1),
    ]
    for i, (dy, dx) in enumerate(inward):
        ch = 9 + i
        for t in (-2, -1, 0, 1):
            k1(ch, 2, t * dy, t * dx, 1)
        py, px = -dx, dy
        for yy, xx in (
            (2 * dy, 2 * dx),
            (dy + py, dx + px),
            (dy - py, dx - px),
        ):
            k1(ch, 2, yy, xx, -4)
        b1[ch] = -3

    def hinges(first, row, col):
        for ch, inp, centre, sign in (
            (first, 16, row, 1),
            (first + 1, 16, row, -1),
            (first + 2, 17, col, 1),
            (first + 3, 17, col, -1),
        ):
            k1(ch, inp, 0, 0, sign)
            b1[ch] = (
                -(centre + 0.25) / 63.0
                if sign == 1 else
                (centre - 0.25) / 63.0
            )

    hinges(17, 8, 8)
    hinges(21, 39, 32)

    # Selected-palette underline.
    for dx in range(-2, 3):
        k1(25, 0, 0, dx, 1)
    k1(25, 4, -1, 0, 1)
    b1[25] = -5

    # Palette square centre.
    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        k1(26, 4, dy, dx, 1)
    k1(26, 4, 0, 0, -4)
    b1[26] = -3

    # Small-tool detectors; paint-coloured pixels are wildcards.
    small = np.array([
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 2, 0, 0, 2, 2],
        [-1, 2, 2, 2, 2, -1],
    ], dtype=np.int8)

    tool_specs = [
        (180, (29, 18), (32, 20)),
        (270, (48, 36), (50, 38)),
        (0, (29, 55), (31, 57)),
        (90, (11, 36), (13, 38)),
    ]
    for j, (rotation, origin, click) in enumerate(tool_specs):
        image = np.rot90(small, k=(-rotation // 90) % 4)
        cx = click[0] - origin[0]
        cy = click[1] - origin[1]
        ch = 28 + j
        required = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                yy, xx = cy + dy, cx + dx
                if 0 <= yy < image.shape[0] and 0 <= xx < image.shape[1]:
                    v = int(image[yy, xx])
                    if v < 0:
                        v = 5
                else:
                    v = 5
                if v == 15:
                    k1(ch, 2, dy, dx, -2)
                    k1(ch, 5, dy, dx, -2)
                else:
                    k1(ch, v, dy, dx, 1)
                    required += 1
        b1[ch] = 1 - required

    def target_gate(out, strength=2048):
        for ch in range(17, 21):
            k2(out, ch, value=-strength)

    # Each of these channels now has TWO probes: target and canvas.
    for i in range(8):
        k2(i, i)
        k2(i, 8, value=-2048)
        k2(i, 27, value=-2048)

    stamp_probes = [
        (1, -1, 0),
        (3, 0, 1),
        (6, 1, 0),
        (0, 0, -1),
    ]
    other_probes = [
        (2, -1, 0),
        (4, 0, 1),
        (5, 1, 0),
        (7, 0, -1),
    ]
    for j, (src, dy, dx) in enumerate(stamp_probes):
        k2(8 + j, src, dy, dx)
        target_gate(8 + j)
    for j, (src, dy, dx) in enumerate(other_probes):
        k2(18 + j, src, dy, dx)
        target_gate(18 + j)

    # Exact row pulses, using the coordinate dead bands.
    for out, first in ((12, 17), (13, 21)):
        k2(out, first, value=-100)
        k2(out, first + 1, value=-100)
        b2[out] = 1

    # Ternary orientation features.
    for j in range(4):
        k2(14 + j, 9 + j)
        k2(14 + j, 13 + j, value=2)

    # Probe 1 has offset (-2,-1); this extra shift reads the swatch
    # centre three rows above the detected underline.
    k2(22, 1, -1, 1)
    k2(22, 25, value=16)
    b2[22] = -16

    k2(23, 26, value=100)
    for j in range(4):
        k2(23, 28 + j, value=90)

    k2(24, 23)
    k2(24, 24)
    k2(25, 17)
    k2(25, 18)

    for j, centre in enumerate((31, 50, 13)):
        a = 26 + 2 * j
        k2(a, 19, value=63)
        b2[a] = 8.25 - (centre + 0.25)
        k2(a + 1, 19, value=-63)
        b2[a + 1] = (centre - 0.25) - 8.25

    # Shared probe channels:
    # row 8  = target code + 8
    # row 39 = canvas code.
    # Thus max = target+8, and 64*mean = target+8+canvas.
    for ch in range(8):
        W3[ch, 32 + ch] = 1
        W3[ch, 12] = 8

    for ch in list(range(8, 12)) + list(range(18, 23)):
        W3[ch, 32 + ch] = 1
    for ch in range(14, 18):
        W3[ch, ch] = 1
    W3[23, 23] = 1

    # Reconstruct exact base-7 checksums before adding the target boost.
    # All intermediate integer magnitudes remain below 2**24.
    big = 2 ** 23
    digit_sum = sum(7 ** i for i in range(8))
    for out, pulse in ((12, 12), (13, 13)):
        for i in range(8):
            W3[out, 32 + i] = 7 ** i
        W3[out, pulse] = big
        b3[out] = -big - digit_sum

    W3[24, 25] = -100
    b3[24] = 1
    W3[29, 25] = 63

    W3[25, 24] = 63
    W3[26, 26:28] = 1
    W3[27, 28:30] = 1
    W3[28, 30:32] = 1
    W3[30, 64 + 22] = 1

    # Palette cycling: propagate a capped signal rightward.
    for out in (0, 1):
        kh(out, 0, 0, -1, 1)
        kh(out, 1, 0, -1, -1)
        Uh[out, 30] = 1
    bh[1] = -1

    kh(2, 0, 0, -1, 1)
    kh(2, 1, 0, -1, -1)
    Uh[2, 23] = 0.01
    bh[2] = -1

    # Bring the canvas checksum to the target row.
    kh(3, 3, 1, 0)
    Uh[3, 13] = 1

    kh(4, 3, 1, 0, -1)
    Uh[4, 12] = 1
    kh(5, 3, 1, 0, 1)
    Uh[5, 12] = -1
    Uh[5, 29] = -(2 ** 25)

    # Equality pulse, propagated down toward the small tools.
    kh(6, 6, -1, 0, 1)
    kh(6, 4, value=-1)
    kh(6, 5, value=-1)
    Uh[6, 24] = 1

    Uh[7, 22] = 1

    # Signed small-tool colour differences, in four vertical lanes.
    lanes = [25, 27, 26, 28]
    for j, lane in enumerate(lanes):
        pos, neg = 8 + 2 * j, 9 + 2 * j
        kh(pos, pos, -1, 0)
        kh(neg, neg, -1, 0)
        kh(pos, 7, -1, 0, -1)
        kh(neg, 7, -1, 0, 1)
        Uh[pos, 8 + j] = 1
        Uh[neg, 8 + j] = -1
        Uh[pos, lane] = -100
        Uh[neg, lane] = -100

    p["Wc"][0, 23] = 1
    p["Wc"][0, 32 + 2] = 10
    p["Wc"][0, 32 + 6] = 30
    p["Wc"][0, 32 + 8:32 + 16] = -1000

    Wz, bz = p["Wz"], p["bz"]

    def feature(unit, fch, weight):
        Wz[unit, 32 + fch] += weight

    # Target versus selected colour; remove the target's +8 boost.
    for i in range(8):
        feature(2 * i, i, 1)
        feature(2 * i, 22, -1)
        bz[2 * i] = -8
        feature(2 * i + 1, i, -1)
        feature(2 * i + 1, 22, 1)
        bz[2 * i + 1] = 8

    for j in range(4):
        feature(16 + 2 * j, 8 + j, 1)
        feature(16 + 2 * j, 22, -1)
        feature(17 + 2 * j, 8 + j, -1)
        feature(17 + 2 * j, 22, 1)

    for j in range(4):
        feature(24 + j, 14 + j, 1)
        feature(28 + j, 14 + j, 1)
        bz[28 + j] = -1

    Wz[32, 6] = 1

    for j in range(4):
        feature(33 + 2 * j, 8 + j, 1)
        feature(33 + 2 * j, 18 + j, -1)
        feature(34 + 2 * j, 8 + j, -1)
        feature(34 + 2 * j, 18 + j, 1)

    # Target versus current leading-sector colour:
    # target-current = 2*max - 64*mean - 8.
    for i in range(8):
        pos, neg = 41 + 2 * i, 42 + 2 * i
        Wz[pos, 32 + i] = 2
        Wz[pos, 64 + i] = -64
        bz[pos] = -8
        Wz[neg, 32 + i] = -2
        Wz[neg, 64 + i] = 64
        bz[neg] = 8

    Wmz, Wma = p["Wmz"], p["Wma"]
    Wm, bm = p["Wm"], p["bm"]
    scale = 10.0

    def mz(unit, zunit, value):
        Wmz[unit, zunit] += scale * value

    def orient(unit, i, value=1.0):
        j = i % 4
        if i < 4:
            mz(unit, 24 + j, value)
            mz(unit, 28 + j, -2 * value)
        else:
            mz(unit, 28 + j, value)

    def bias(unit, value):
        bm[unit] = scale * value

    # Ready to paint a large region.
    for i in range(8):
        orient(i, i)
        mz(i, 2 * i, -2)
        mz(i, 2 * i + 1, -2)
        mz(i, 32, -2)
        bias(i, -0.5)

    cardinal = [0, 2, 4, 6]

    # Ready for a correctly coloured, uniform small-tool repair.
    for j, i in enumerate(cardinal):
        u = 8 + j
        orient(u, i)
        mz(u, 32, 1)
        for zunit in (
            16 + 2 * j, 17 + 2 * j,
            33 + 2 * j, 34 + 2 * j,
        ):
            mz(u, zunit, -2)
        bias(u, -1.5)

    movement_units = []

    # Move after a large paint.
    for i in range(8):
        u = 12 + i
        orient(u, i)
        Wma[u, 5] = scale
        bias(u, -1.5)
        movement_units.append((u, i))

    # Move after a small-tool click, but not after a palette click.
    for i in range(8):
        u = 20 + i
        orient(u, i, 2)
        Wma[u, 6] = 4 * scale
        Wm[u, 8:12] = scale
        bias(u, -3)
        movement_units.append((u, i))

    # Diagonal stations have no small tool.
    for j, i in enumerate((1, 3, 5, 7)):
        u = 28 + j
        orient(u, i)
        mz(u, 32, 1)
        bias(u, -1.5)
        movement_units.append((u, i))

    # Do not stamp a nonuniform rectangular target region.
    for j, i in enumerate(cardinal):
        u = 32 + j
        orient(u, i, 10)
        mz(u, 32, 10)
        mz(u, 33 + 2 * j, 1)
        mz(u, 34 + 2 * j, 1)
        bias(u, -20.5)
        movement_units.append((u, i))

    # Revision: skip any large stroke whose leading sector is already
    # correct. This takes precedence over both painting and colour choice.
    for i in range(8):
        u = 36 + i
        orient(u, i)
        mz(u, 41 + 2 * i, -2)
        mz(u, 42 + 2 * i, -2)
        mz(u, 32, -2)
        bias(u, -0.5)
        movement_units.append((u, i))

    Wo, bo = p["Wo"], p["bo"]
    bo[:] = -1000
    bo[1:7] = 0
    bo[6] = 1

    bo[5] = 8
    Wo[5, 128:136] = 1

    clockwise_action = [4, 2, 2, 3, 3, 1, 1, 4]
    for unit, orientation in movement_units:
        action = clockwise_action[orientation]
        Wo[action, 128 + unit] += 5
        bo[action] += 5

    return p
