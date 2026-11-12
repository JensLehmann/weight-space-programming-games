import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }

    A = [
        ["00100", "00100", "01110", "00100", "11111"],
        ["00001", "00101", "11111", "10100", "10000"],
        ["11111", "10100", "10000", "10100", "11111"],
        ["00100", "11111", "10101", "10101", "00100"],
        ["10001", "11111", "00100", "11111", "10001"],
        ["11111", "00100", "11100", "10000", "11111"],
        ["10001", "11111", "10001", "10001", "11011"],
    ]
    B = [
        ["10000", "11110", "10010", "10010", "11111"],
        ["11111", "10001", "11101", "10101", "11111"],
        ["00111", "00101", "11111", "10100", "11100"],
        ["01110", "01010", "11111", "10001", "11111"],
        ["11111", "10001", "10001", "11111", "00100"],
        ["11110", "10011", "10001", "11001", "01111"],
        ["00100", "11111", "10101", "11111", "00100"],
    ]
    C = [
        ["10101", "10000", "10101", "10000", "11111"],
        ["11101", "10000", "00000", "00001", "10111"],
        ["10111", "00100", "00100", "00100", "11101"],
        ["00100", "10101", "00100", "10101", "00100"],
        ["11011", "10001", "10101", "10001", "11011"],
        ["10111", "10101", "11101", "00000", "10101"],
        ["10001", "00000", "10001", "11011", "01010"],
    ]

    def rotate(pt):
        y, x = pt
        return x, 4 - y

    def orbit(pt):
        result = []
        for _ in range(4):
            result.append(pt)
            pt = rotate(pt)
        return result

    reps = sorted({
        min(orbit((y, x)))
        for y in range(5)
        for x in range(5)
    })
    rank = {rep: i for i, rep in enumerate(reps)}
    weights = {
        (y, x): 5 ** rank[min(orbit((y, x)))]
        for y in range(5)
        for x in range(5)
    }

    pairs = [
        ((0, 0), (0, 1)),
        ((0, 2), (1, 1)),
        ((0, 2), (1, 2)),
    ]

    def signature(glyph):
        value = sum(
            weights[y, x]
            for y in range(5)
            for x in range(5)
            if glyph[y][x] == "1"
        )
        for k, (a, b) in enumerate(pairs):
            count = 0
            for _ in range(4):
                count += int(
                    glyph[a[0]][a[1]] == "1"
                    and glyph[b[0]][b[1]] == "1"
                )
                a, b = rotate(a), rotate(b)
            value += (5 ** (7 + k)) * count
        return value

    W1, b1 = p["W1"], p["b1"]

    # Rotation-invariant single-pixel orbit counts.
    for (y, x), value in weights.items():
        W1[0, 5, y, x] = value

    # Interior and border compatibility for two logical families.
    for colour in (5, 10, 11):
        W1[1, colour, :, :] = 1
    for colour in (5, 7, 10):
        W1[2, colour, :, :] = 1

    for base, colours in ((3, (10, 11)), (7, (7, 10))):
        for colour in colours:
            W1[base, colour, 0, :] = 1
            W1[base + 1, colour, 4, :] = 1
            W1[base + 2, colour, :, 0] = 1
            W1[base + 3, colour, :, 4] = 1

    # Selection bracket and lower-region background.
    W1[11, 0, 2, :] = 1
    W1[11, 0, 3, 0] = 1
    W1[11, 0, 3, 4] = 1
    b1[11] = -6
    W1[12, 3, 1:4, 1:4] = 1
    b1[12] = -8

    # Rotation-invariant pair counts.
    for k, (a, b) in enumerate(pairs):
        for r in range(4):
            ch = 13 + 4 * k + r
            W1[ch, 5, a[0], a[1]] = 1
            W1[ch, 5, b[0], b[1]] = 1
            b1[ch] = -1
            a, b = rotate(a), rotate(b)

    # Tile colour samples and left bracket end.
    W1[25, 11, 0, 2] = 1
    W1[26, 7, 0, 2] = 1
    W1[27, 0, 2, 2:5] = 1
    W1[27, 0, 3, 2] = 1
    b1[27] = -3

    W2, b2 = p["W2"], p["b2"]

    # Source family is A or C; target family is B or A.
    for family in range(2):
        for number in range(7):
            index = family * 7 + number
            compat = 1 + family
            edge = 3 if family == 0 else 7

            if family == 0:
                default = signature(A[number])
                alternate = signature(C[number])
            else:
                default = signature(B[number])
                alternate = signature(A[number])

            for side, sign in enumerate((1, -1)):
                out = 2 * index + side
                W2[out, 0, 1, 1] = sign

                for k in range(3):
                    for r in range(4):
                        W2[out, 13 + 4 * k + r, 1, 1] = (
                            sign * (5 ** (7 + k))
                        )

                W2[out, compat, 1, 1] = -1
                W2[out, edge, 0, 1] = -1
                W2[out, edge + 1, 2, 1] = -1
                W2[out, edge + 2, 1, 0] = -1
                W2[out, edge + 3, 1, 2] = -1
                b2[out] = 45 - sign * default

                correction = sign * (default - alternate)
                if family == 0:
                    W2[out, 25, 0, 1] += correction
                else:
                    b2[out] += correction
                    W2[out, 26, 0, 1] -= correction

    W2[28, 11, 1, 1] = 1
    W2[28, 27, 1, 2] = 1
    W2[29, 12, 1, 1] = 1
    W2[30, 25, 0, 1] = 1
    W2[31, 26, 0, 1] = 1

    W3, b3 = p["W3"], p["b3"]

    for number in range(7):
        # Upper source glyph.
        out = number
        W3[out, 2 * number] = -1
        W3[out, 2 * number + 1] = -1
        W3[out, 32 + 29] = -1
        W3[out, 30] = 1
        W3[out, 64 + 30] = -1
        b3[out] = 1

        # Upper target glyph.
        index = 7 + number
        out = 7 + number
        W3[out, 2 * index] = -1
        W3[out, 2 * index + 1] = -1
        W3[out, 32 + 29] = -1
        W3[out, 31] = 1
        W3[out, 64 + 30] = 1

        # Selected lower source glyph.
        out = 14 + number
        W3[out, 2 * number] = -1
        W3[out, 2 * number + 1] = -1
        W3[out, 32 + 29] = 1
        W3[out, 64 + 28] = 1
        W3[out, 30] = 1
        W3[out, 64 + 30] = -1
        b3[out] = -1

        # Selected lower target glyph.
        index = 7 + number
        out = 21 + number
        W3[out, 2 * index] = -1
        W3[out, 2 * index + 1] = -1
        W3[out, 32 + 29] = 1
        W3[out, 64 + 28] = 1
        W3[out, 31] = 1
        W3[out, 64 + 30] = 1
        b3[out] = -2

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    # Carry selected source identity upward, then horizontally.
    Wh[0, 0, 2, 1] = 1
    for i in range(7):
        Uh[0, 14 + i] = i + 1

    Wh[1, 1, 1, 0] = 1
    Wh[1, 0, 1, 0] = 1
    Wh[2, 2, 1, 2] = 1
    Wh[2, 0, 1, 2] = 1

    # Compare the carried identity with upper source glyphs.
    for channel in (0, 1, 2):
        Wh[3, channel, 1, 1] = 1
        Wh[4, channel, 1, 1] = -1
    bh[3] = bh[4] = 10
    for i in range(7):
        Uh[3, i] = -(i + 1) - 10
        Uh[4, i] = (i + 1) - 10

    Wh[5, 3, 1, 1] = -1
    Wh[5, 4, 1, 1] = -1
    bh[5] = -9
    for i in range(7):
        Uh[5, i] = 10

    # Carry a match toward a target. Stop an older signal at
    # another source glyph so adjacent matches cannot add together.
    Wh[6, 6, 1, 0] = 1
    Wh[6, 5, 1, 0] = 1
    Wh[7, 6, 1, 0] = 1
    Wh[7, 5, 1, 0] = 1
    bh[7] = -10
    for i in range(7):
        Uh[6, i] = -10
        Uh[6, 7 + i] = -10
        Uh[7, 7 + i] = 10

    # Encode matched target identity.
    Wh[8, 7, 1, 1] = 10
    bh[8] = -10
    for i in range(7):
        Uh[8, 7 + i] = i + 1

    Wz, bz = p["Wz"], p["bz"]
    Wo, bo = p["Wo"], p["bo"]
    bo[:] = -4
    bo[1:5] = 0
    bo[4] = 0.01

    # Decode cyclic changes between current and desired target glyph.
    for current in range(1, 8):
        for k in range(8):
            unit = (current - 1) * 8 + k
            Wz[unit, 8] = 1
            Wz[unit, 32 + 21 + current - 1] = 12
            bz[unit] = -12 - k

        for desired in range(1, 8):
            delta = (desired - current) % 7
            if delta == 0:
                action = 4
            elif delta <= 3:
                action = 2
            else:
                action = 1

            base = (current - 1) * 8
            Wo[action, base + desired - 1] += 1
            Wo[action, base + desired] -= 2
            if desired + 1 < 8:
                Wo[action, base + desired + 1] += 1

    # If the proposed adjustment immediately reverses the previous
    # adjustment, advance the selection rather than oscillating.
    p["Wmz"][0] = 5 * Wo[1, :128]
    p["Wma"][0, 2] = 5
    p["bm"][0] = -7.5

    p["Wmz"][1] = 5 * Wo[2, :128]
    p["Wma"][1, 1] = 5
    p["bm"][1] = -7.5

    Wo[4, 128] = 2
    Wo[4, 129] = 2
    bo[4] += 4

    return p
