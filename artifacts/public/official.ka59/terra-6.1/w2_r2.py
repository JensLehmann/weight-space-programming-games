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

    W1 = p["W1"]
    b1 = p["b1"]
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def pixel(out, color, dy, dx, value):
        W1[out, color, dy + 2, dx + 2] += value

    def selected_marker(out):
        pixel(out, 0, 0, 0, 10)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy or dx:
                    pixel(out, 0, dy, dx, 1)
                    pixel(out, 14, dy, dx, 1)

    def unselected_marker(out):
        for color in (4, 5):
            pixel(out, color, 0, 0, 10)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy or dx:
                    for color in (0, 4, 5, 14):
                        pixel(out, color, dy, dx, 1)

    selected_marker(0)
    b1[0] = -17

    unselected_marker(1)
    b1[1] = -17

    # Empty 5x5 goal.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dy) == 2 or abs(dx) == 2:
                pixel(2, 4, dy, dx, 1)
            else:
                for color in (0, 5, 14):
                    pixel(2, color, dy, dx, -20)
    b1[2] = -11

    # Selected or any block exactly occupying a 5x5 goal.
    for out, centers in ((3, (0,)), (4, (0, 4, 5))):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if abs(dy) == 2 or abs(dx) == 2:
                    pixel(out, 4, dy, dx, 2)
        for color in centers:
            pixel(out, color, 0, 0, 4)
        b1[out] = -35

    # Wall and slide-surface footprints for a 3x3 block.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            pixel(5, 2, dy, dx, 1)
            pixel(19, 15, dy, dx, 1)

    # Selected block adjacent to a slide surface.
    for i, (dy, dx) in enumerate(directions):
        out = 6 + i
        selected_marker(out)
        pixel(out, 15, 2 * dy, 2 * dx, 10)
        b1[out] = -27

    # Unselected block adjacent to a slide surface.
    for i, (dy, dx) in enumerate(directions):
        out = 20 + i
        unselected_marker(out)
        pixel(out, 15, 2 * dy, 2 * dx, 10)
        b1[out] = -27

    # Partial corner detectors for larger rectangular goals.
    def rectangle(out, width, height, mode):
        for dx in range(-2, 3):
            pixel(out, 4, -2, dx, 4)
        for dy in range(-1, 3):
            pixel(out, 4, dy, -2, 4)

        wide = width == 8
        tall = height == 8

        pixel(out, 4, -1, 2, -4 if wide else 4)
        pixel(out, 4, 2, -1, -4 if tall else 4)
        required = 36 + 4 * int(not wide) + 4 * int(not tall)

        if mode == "empty":
            for color in (0, 5, 14):
                pixel(out, color, -1, -1, -100)
        else:
            for color in (0, 14):
                pixel(out, color, -1, -1, 4)
            centers = (0,) if mode == "selected" else (0, 4, 5)
            for color in centers:
                pixel(out, color, int(tall), int(wide), 4)
            required += 8

        b1[out] = -(required - 1)

    rectangles = ((5, 8), (8, 5), (8, 8))
    for i, (width, height) in enumerate(rectangles):
        rectangle(10 + i, width, height, "empty")
        rectangle(13 + i, width, height, "selected")
        rectangle(16 + i, width, height, "occupied")

    W2 = p["W2"]
    for c in range(24):
        W2[c, c, 1, 1] = 1

    # Shift larger-goal features toward the corresponding block marker.
    for base in (10, 13, 16):
        for i, (width, height) in enumerate(rectangles):
            c = base + i
            W2[c, c, 1, 1] = 0
            dy = int(height == 8)
            dx = int(width == 8)
            W2[c, c, 1 - dy, 1 - dx] = 1

    # Put push-position sources just behind waiting blocks.
    for i, (dy, dx) in enumerate(directions):
        c = 20 + i
        W2[c, c, 1, 1] = 0
        W2[c, c, 1 + dy, 1 + dx] = 1

    W3 = p["W3"]
    W3[0, 0] = 1                       # selected marker

    W3[1, 2] = 1                       # empty goal sources
    for c in (10, 11, 12):
        W3[1, c] = 1

    W3[2, 5] = 1                       # solid wall footprint
    for i in range(4):
        W3[3 + i, 6 + i] = 1           # selected beside slide

    W3[7, 3] = 1                       # selected block placed
    for c in (13, 14, 15):
        W3[7, c] = 1

    W3[8, 1] = 1                       # unselected marker
    W3[9, 4] = 1                       # occupied goal
    for c in (16, 17, 18):
        W3[9, c] = 1

    W3[10, 19] = 1                     # slide-surface footprint
    for i in range(4):
        W3[12 + i, 20 + i] = 1         # push-position sources

    Wh = p["Wh"]
    Uh = p["Uh"]
    bh = p["bh"]

    # Goal field propagates in three-pixel hops.
    for i, (dy, dx) in enumerate(directions):
        Wh[1 + i, 0, dy + 1, dx + 1] = 1
        Wh[5 + i, 1 + i, dy + 1, dx + 1] = 1
        Wh[0, 5 + i, dy + 1, dx + 1] = 0.245

    Uh[0, 1] = 1
    Uh[0, 2] = -1000

    # Push-position field uses one-pixel diffusion. It does not cross
    # solid walls or slide surfaces.
    for dy, dx in directions:
        Wh[13, 13, dy + 1, dx + 1] = 0.249
    for i in range(4):
        Uh[13, 12 + i] = 1
    Uh[13, 2] = -1000
    Uh[13, 10] = -1000

    # Direction scores read both fields at the selected marker.
    for i, (dy, dx) in enumerate(directions):
        Wh[9 + i, 5 + i, dy + 1, dx + 1] = 0.02
        Wh[9 + i, 13, dy + 1, dx + 1] = 100
        Uh[9 + i, 0] = 1000
        bh[9 + i] = -1000

    Wz = p["Wz"]
    bz = p["bz"]

    for i in range(4):
        # Usable direction.
        Wz[i, 9 + i] = 1
        Wz[i, 32 + 3 + i] = -100

        # Direction blocked by a slide surface. Suppress switch requests
        # while an unselected waiting block provides a push source.
        Wz[4 + i, 9 + i] = 1
        Wz[4 + i, 32 + 3 + i] = 100
        for j in range(4):
            Wz[4 + i, 32 + 12 + j] = -100
        bz[4 + i] = -100

    # Selected block placed, unless it is needed for a push.
    Wz[8, 32 + 7] = 1
    for j in range(4):
        Wz[8, 32 + 12 + j] = -100

    # Suppress clicking if no unselected marker is visible.
    Wz[9, 32 + 8] = -1
    bz[9] = 1

    # Remember whether the previous action was a click.
    p["Wma"][0, 6] = 5

    Wo = p["Wo"]
    bo = p["bo"]

    for i in range(4):
        Wo[1 + i, i] = 1
        Wo[6, 4 + i] = 1.01

    Wo[6, 8] = 100
    Wo[6, 9] = -1000
    Wo[6, 128] = -1000
    bo[6] = 0.0001

    bo[0] = -10000
    bo[5] = -10000
    bo[7] = -10000

    # Prefer unselected blocks that are not already on goals.
    p["Wc"][0, 8] = 10
    p["Wc"][0, 9] = -1

    return p
