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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}
    w1, b1 = p["W1"], p["b1"]
    w2, b2 = p["W2"], p["b2"]
    w3, b3 = p["W3"], p["b3"]

    # Direction order: up, down, left, right.
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    cardinal = [(0, 1), (2, 1), (1, 0), (1, 2)]

    # Broad floor. Thin wires do not qualify as floor.
    w1[0, 18, :, :] = 1
    for colour in (0, 8, 11, 15, 14):
        w1[0, colour, :, :] = -1
    b1[0] = -17

    def actor_pattern(out, colour, dy=0, dx=0,
                      scale=1.0, centre_colour=5):
        cy, cx = 2 + dy, 2 + dx
        for yy in (-1, 0, 1):
            for xx in (-1, 0, 1):
                if yy or xx:
                    w1[out, colour, cy + yy, cx + xx] += scale
        w1[out, centre_colour, cy, cx] += scale

    actor_pattern(1, 9)
    b1[1] = -8
    actor_pattern(8, 2)
    b1[8] = -8
    actor_pattern(9, 2, centre_colour=0)
    b1[9] = -8

    # Isolated goal dot.
    w1[2, 9, 2, 2] = 1
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                w1[2, 9, 2 + dy, 2 + dx] = -1

    # Door region and exact door centre. Standard doors have 23
    # coloured pixels, rather than a completely filled 5x5 square.
    for c in (8, 11):
        w1[3, c, :, :] = 1
        w1[22, c, :, :] = 1
    b1[3] = -12
    b1[22] = -21

    # Small physical switches; large door interiors are rejected.
    for c in (8, 11, 15):
        w1[4, c, 1:4, 1:4] = 1
        for yy, xx in ((0, 0), (0, 4), (4, 0), (4, 4)):
            w1[4, c, yy, xx] = -9
        w1[5, c, :, :] = 1
    b1[4] = -8

    # Shifted player detectors, occupied-switch detectors, and
    # five-pixel-wide probes.
    legal_w1 = [6, 7, 27, 28]
    for j, (dy, dx) in enumerate(dirs):
        actor_pattern(10 + j, 9, -dy, -dx)
        b1[10 + j] = -8

        for out, colours in ((14 + j, (8, 15)),
                             (18 + j, (15,))):
            actor_pattern(out, 9, -dy, -dx, scale=10)
            for c in colours:
                w1[out, c, 2 + 2 * dy, 2 + 2 * dx] += 1
            b1[out] = -90

        probe = 23 + j
        legal = legal_w1[j]
        actor_pattern(legal, 9, -dy, -dx, scale=10)
        for t in range(-2, 3):
            yy = 2 + 2 * dy + t * dx
            xx = 2 + 2 * dx - t * dy
            w1[probe, 18, yy, xx] += 1
            w1[probe, 0, yy, xx] -= 1
            w1[legal, 18, yy, xx] += 1
            w1[legal, 0, yy, xx] -= 1
        b1[probe] = -4
        b1[legal] = -94

    # Features for a white 7x7 frame.
    for yy, xx in ((0, 2), (4, 2), (2, 0), (2, 4)):
        w1[30, 15, yy, xx] = 1
    w1[31, 15, 1:4, 1:4] = 1

    def copy2(out, src, bias=0):
        w2[out, src, 1, 1] = 1
        b2[out] = bias

    copy2(0, 0)
    copy2(1, 0, -1)
    copy2(2, 1)
    copy2(3, 2)

    for out in (4, 5):
        w2[out, 3, :, :] = 1
    b2[5] = -1

    copy2(6, 4)

    for out in (8, 9):
        w2[out, 5, :, :] = 1
    b2[9] = -1

    copy2(12, 8)
    copy2(13, 9)
    copy2(22, 22)
    copy2(23, 22, -1)

    legal_w2 = [10, 11, 30, 31]
    for j, (dy, dx) in enumerate(dirs):
        # Player focus is two pixels ahead.
        w2[14 + j, 10 + j, 1 - dy, 1 - dx] = 1
        # Recover occupied-switch indicators at the player centre.
        for out in (18, 19):
            w2[out, 14 + j, 1 + dy, 1 + dx] += 1
        for out in (20, 21):
            w2[out, 18 + j, 1 + dy, 1 + dx] += 1
        # Wide probes three pixels ahead.
        w2[24 + j, 23 + j, 1 + dy, 1 + dx] = 1
        w2[legal_w2[j], legal_w1[j], 1 + dy, 1 + dx] = 1
    b2[19] = -1
    b2[21] = -1

    # Restore the floor hidden by a small coloured switch.
    for out in (28, 29):
        w2[out, 4, :, :] = 1
    b2[29] = -1

    # Exact white-frame centre: outer axes white, interior and
    # radius-two axes not white.
    for yy, xx in cardinal:
        w2[7, 30, yy, xx] += 1
    w2[7, 30, 1, 1] -= 10
    w2[7, 31, 1, 1] -= 10
    b2[7] = -3

    def diff3(out, a, b):
        w3[out, a] += 1
        w3[out, b] -= 1

    # f3:
    # 0 floor; 1 player; 2 goal; 3 broad door; 4 switch target;
    # 5 wire; 6 occupied pressure switch; 7 occupied white switch;
    # 8 visible ghost count; 9 recorded-slot count;
    # 10..13 directional player foci; 14..17 local legal probes;
    # 18 exact door; 19 pad; 20..23 wide directional probes.
    diff3(0, 0, 1)
    diff3(0, 28, 29)
    w3[1, 2] = 1
    w3[2, 3] = 1
    diff3(3, 4, 5)

    w3[4, 6] = 1
    diff3(4, 18, 19)
    w3[4, 7] = -2

    diff3(5, 8, 9)
    diff3(6, 18, 19)
    w3[6, 7] = -1
    diff3(7, 20, 21)
    w3[7, 7] -= 1

    w3[8, 12] = 1
    w3[9, 13] = 1
    for j in range(4):
        w3[10 + j, 14 + j] = 1
        w3[14 + j, legal_w2[j]] = 1
        w3[20 + j, 24 + j] = 1
    diff3(18, 22, 23)
    w3[19, 7] = 1

    wh, uh, bh = p["Wh"], p["Uh"], p["bh"]
    K = 10000.0

    def neighbours(out, a, b):
        for yy, xx in cardinal:
            wh[out, a, yy, xx] += 1
            wh[out, b, yy, xx] -= 1

    # Clipped floor wave and accumulated arrival time.
    for out in (0, 1):
        neighbours(out, 0, 1)
        wh[out, 6, 1, 1] = 1
        uh[out, 0] = K
        uh[out, 3] = -K
        uh[out, 2] = 1
        bh[out] = -K
    bh[1] -= 1

    wh[2, 2, 1, 1] = 1
    wh[2, 0, 1, 1] = 1
    wh[2, 1, 1, 1] = -1

    # Clipped wire wave.
    for out in (3, 4):
        neighbours(out, 3, 4)
        wh[out, 11, 1, 1] = 1
        uh[out, 5] = K
        bh[out] = -K
    bh[4] -= 1

    # Store opposing width-five probes. Their shifted sums examine
    # the four sides just outside a door's sprite.
    uh[7, 20] = 1
    uh[7, 21] = 1
    uh[8, 22] = 1
    uh[8, 23] = 1

    # A closed door has floor on opposing external sides. An opened
    # door displaced into a thin wire generally does not.
    wh[5, 7, 0, 1] = 1
    wh[5, 7, 2, 1] = 1
    wh[5, 8, 1, 0] = 1
    wh[5, 8, 1, 2] = 1
    wh[5, 5, 1, 1] = 1
    uh[5, 18] = K
    bh[5] = -K - 5

    # Expand the closed-door marker to its boundary.
    for out in (9, 10):
        neighbours(out, 9, 10)
        wh[out, 5, 1, 1] = 1
        uh[out, 3] = K
        bh[out] = -K
    bh[10] -= 1

    # Goal floor touching a closed door activates its wiring.
    # A delayed fallback also exposes remote closed-door dependencies.
    neighbours(11, 0, 1)
    wh[11, 9, 1, 1] = K
    wh[11, 10, 1, 1] = -K
    wh[11, 5, 1, 1] = 1
    uh[11, 18] = -48
    bh[11] = -K

    # Transfer wire reachability back to floor at switches and pads.
    wh[6, 3, 1, 1] = 1
    wh[6, 4, 1, 1] = -1
    uh[6, 4] = K
    uh[6, 19] = K
    bh[6] = -K

    # Arrival-time samples three pixels in each movement direction.
    for j, (dy, dx) in enumerate(dirs):
        out = 12 + j
        wh[out, 2, 1 + dy, 1 + dx] = 1
        uh[out, 10 + j] = K
        bh[out] = -K

    wz, bz = p["Wz"], p["bz"]
    for j in range(4):
        wz[j, 12 + j] = 1
        wz[4 + j, 32 + 14 + j] = 1

    # Recording conditions and counts. f3 means begin at index 64.
    wz[8, 32 + 6] = 1
    wz[8, 32 + 7] = -1
    wz[9, 32 + 7] = 1
    wz[10, 64 + 8] = 4096
    wz[11, 64 + 9] = 4096

    wm, wmz, wma, bm = (
        p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    )

    # Small visit counter for generating a repeated teleport trigger.
    wm[0, 0] = 1
    wmz[0, 9] = 0.12

    # White-switch recording readiness.
    wm[1, 0] = 8
    wmz[1, 9] = 8
    wmz[1, 10] = 8
    wmz[1, 11] = -8
    bm[1] = -11.2

    # Ordinary pressure-switch recording. If a ghost is hidden under
    # the player, visible ghosts are fewer than recorded UI slots.
    wmz[2, 8] = 8
    wmz[2, 10] = 8
    wmz[2, 11] = -8
    bm[2] = -4

    # At a white switch, prefer reversing the arrival move. The floor
    # field then brings the player back, producing repeated triggers.
    for j in range(4):
        unit = 3 + j
        wmz[unit, 9] = 8
        wma[unit, 1 + j] = 8
        bm[unit] = -12

    wo, bo = p["Wo"], p["bo"]
    bo[:] = -1000000
    opposite = [1, 0, 3, 2]

    for j in range(4):
        action = 1 + j
        bo[action] = 0
        wo[action, j] = 1
        wo[action, 4 + j] = 1000
        wo[action, 128 + 3 + opposite[j]] = 200

    bo[5] = 8000
    wo[5, 128 + 1] = 8000
    wo[5, 128 + 2] = 8000

    return p
