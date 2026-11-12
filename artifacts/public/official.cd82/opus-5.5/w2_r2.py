import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128), "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    W1, b1, W2, W3 = p["W1"], p["b1"], p["W2"], p["W3"]

    # ---------- layer 1 ----------
    a = 1.0 / 32.0
    # 0: colour 9 inside target region (row+col<31); 1: colour 8 there
    for ch, col in ((0, 9), (1, 8)):
        W1[ch, col, 2, 2] = 1.0
        W1[ch, 16, 2, 2] = -a * 63
        W1[ch, 17, 2, 2] = -a * 63
        b1[ch] = a * 31 - 1.0
    # 2: underline end -> fires at (6, x+5): black at (r+1,c-1),(r+1,c-2), colour 3 at (r+1,c)
    W1[2, 0, 3, 1] = 1.0
    W1[2, 0, 3, 0] = 1.0
    W1[2, 3, 3, 2] = 1.0
    b1[2] = -2.5
    # 4: column coordinate passthrough
    W1[4, 17, 2, 2] = 1.0
    # 5: swatch-0 bottom border marker: colour 4 with black above
    W1[5, 4, 2, 2] = 1.0
    W1[5, 0, 1, 2] = 1.0
    b1[5] = -1.5
    # 6: small bucket horizontal pattern 2,2,[0],0,2
    W1[6, 0, 2, 2] = 1; W1[6, 0, 2, 3] = 1
    W1[6, 2, 2, 1] = 1; W1[6, 2, 2, 0] = 1; W1[6, 2, 2, 4] = 1
    b1[6] = -4.5
    # 7: small bucket vertical pattern
    W1[7, 0, 2, 2] = 1; W1[7, 0, 3, 2] = 1
    W1[7, 2, 1, 2] = 1; W1[7, 2, 0, 2] = 1; W1[7, 2, 4, 2] = 1
    b1[7] = -4.5

    # ---------- layer 2 ----------
    for c in (0, 1, 6, 7):
        W2[c, c, 1, 1] = 1.0
    # 2: next-swatch click point (shift right by one): f2(r,c) = f1[2](r,c-1)
    W2[2, 2, 1, 0] = 1.0
    # 3: U = underline is under swatch 9 (next-pattern at col 62), scaled
    W2[3, 2, 1, 1] = 4 * 2.0
    W2[3, 4, 1, 1] = 4 * 63.0 / 4.0
    p["b2"][3] = 4 * (-1.0 - 59.0 / 4.0)
    # 4: swatch-0 marker
    W2[4, 5, 1, 1] = 2.0

    # ---------- layer 3 (axial) ----------
    for c in (0, 1, 2, 6, 7):
        W3[c, c] = 1.0
    # 3: wrap click = rowmax(U) at swatch-0 marker
    W3[3, 32 + 3] = 1.0
    W3[3, 4] = 3.0
    p["b3"][3] = -4.0

    # ---------- click map ----------
    Wc = p["Wc"]
    Wc[0, 6] = 20.0
    Wc[0, 7] = 20.0
    Wc[0, 3] = 3.0
    Wc[0, 2] = 4.0

    # ---------- z: clipped detectors ----------
    Wz, bz = p["Wz"], p["bz"]
    Wz[0, 32 + 0] = 10.0
    Wz[1, 32 + 0] = 10.0; bz[1] = -1.0
    Wz[2, 32 + 1] = 10.0
    Wz[3, 32 + 1] = 10.0; bz[3] = -1.0

    # ---------- memory scripts ----------
    B = 20.0
    Wm, Wmz, Wma, bm = p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    Wo, bo = p["Wo"], p["bo"]
    s1 = [4, 2, 2, 3, 5]
    s3 = [4, 6, 6, 6, 2, 5, 1, 6, 3, 3, 2, 5, 1, 6, 6, 6, 5, 6, 4, 6]
    s5 = [3, 6, 6, 6, 5, 6, 6, 4, 4, 5, 6, 6, 6, 2, 2, 5, 6, 6, 6, 1, 1, 3, 6]

    def d9(u, w):
        Wmz[u, 0] += w; Wmz[u, 1] -= w

    def d8(u, w):
        Wmz[u, 2] += w; Wmz[u, 3] -= w

    start = 1
    for seq, kind in ((s1, 1), (s3, 3), (s5, 5)):
        h = start
        Wma[h, 1:8] = -2 * B
        if kind == 1:
            bm[h] = 0.5 * B
            d8(h, -B); d9(h, -B)
        elif kind == 3:
            bm[h] = -0.5 * B
            d8(h, B); d9(h, -B)
        else:
            bm[h] = -0.5 * B
            d9(h, B)
        for j in range(1, len(seq)):
            Wm[h + j, h + j - 1] = B
            bm[h + j] = -0.5 * B
        for j, act in enumerate(seq):
            Wo[act, 128 + h + j] += 5.0
            bo[act] += 5.0
        start += len(seq)

    bo[0] += 1.0
    return p
