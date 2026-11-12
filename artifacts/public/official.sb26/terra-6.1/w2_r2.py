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

    colors = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    n = len(colors)

    dot = 26
    selection = 27
    too_high = 28
    too_low = 29
    above_inventory = 30
    black_at_x_plus_2 = 31

    # Target colors on the top row and inventory-item corners.
    for i, color in enumerate(colors):
        p["W1"][i, color, 2, 2] = 1.0
        p["W1"][i, 16, 2, 2] = -15.75
        p["b1"][i] = -0.5

        j = n + i
        p["W1"][j, color, 2, 2] = 2.0
        p["W1"][j, color, 2, 3] = 2.0
        p["W1"][j, color, 3, 2] = 2.0
        p["W1"][j, color, 2, 1] = -3.0
        p["W1"][j, color, 1, 2] = -3.0
        p["b1"][j] = -5.0

    # Empty-slot marker.
    p["W1"][dot, 2, 2, 2] = 2.0
    p["W1"][dot, 2, 2, 3] = 2.0
    p["W1"][dot, 2, 3, 2] = 2.0
    p["W1"][dot, 2, 2, 1] = -3.0
    p["W1"][dot, 2, 1, 2] = -3.0
    p["b1"][dot] = -5.0

    # Candidate corner of a black selection outline.
    p["W1"][selection, 0, 2, 2] = 2.0
    p["W1"][selection, 0, 2, 3] = 2.0
    p["W1"][selection, 0, 3, 2] = 2.0
    p["W1"][selection, 0, 2, 1] = -3.0
    p["W1"][selection, 0, 1, 2] = -3.0
    p["b1"][selection] = -5.0

    # Row masks.
    p["W1"][too_high, 16, 2, 2] = -63.0
    p["b1"][too_high] = 10.0
    p["W1"][too_low, 16, 2, 2] = 63.0
    p["b1"][too_low] = -52.0
    p["W1"][above_inventory, 16, 2, 2] = -63.0
    p["b1"][above_inventory] = 55.0
    p["W1"][black_at_x_plus_2, 0, 2, 4] = 1.0

    for i in range(n):
        p["W2"][i, i, 1, 1] = 4.0
        p["W2"][n + i, n + i, 1, 1] = 1.0
        p["W2"][n + i, above_inventory, 1, 1] = -1.0

    p["W2"][dot, dot, 1, 1] = 1.0
    p["W2"][dot, too_high, 1, 1] = -1.0
    p["W2"][dot, too_low, 1, 1] = -1.0

    # Reject three-pixel frame corners.
    p["W2"][selection, selection, 1, 1] = 2.0
    p["W2"][selection, black_at_x_plus_2, 1, 2] = 1.0
    p["W2"][selection, too_high, 1, 1] = -1.0
    p["b2"][selection] = -2.0

    # f3: empty marker, target color, item marker, item color,
    # and selection indicator.
    p["W3"][0, dot] = 1.0
    for i, color in enumerate(colors):
        p["W3"][1, 64 + i] = float(color)
        p["W3"][2, n + i] = 1.0
        p["W3"][3, n + i] = float(color)
    p["W3"][4, 32 + selection] = 1.0

    wh = p["Wh"]
    uh = p["Uh"]
    bh = p["bh"]

    # First empty marker in each row.
    wh[0, 0, 1, 0] = 1.0
    uh[0, 0] = 1.0
    wh[1, 0, 1, 1] = -1.0
    uh[1, 0] = 2.0

    # Read and propagate the target color.
    wh[2, 1, 1, 1] = 16.0
    uh[2, 1] = 1.0
    bh[2] = -16.0

    wh[3, 3, 1, 0] = 1.0
    wh[3, 2, 1, 1] = 1.0
    wh[4, 4, 1, 2] = 1.0
    wh[4, 2, 1, 1] = 1.0
    wh[5, 5, 0, 1] = 1.0
    wh[5, 3, 1, 1] = 1.0
    wh[5, 4, 1, 1] = 1.0
    wh[5, 2, 1, 1] = -1.0

    # Item-versus-target color difference.
    uh[6, 3] = 1.0
    wh[6, 5, 1, 1] = -1.0
    wh[7, 5, 1, 1] = 1.0
    uh[7, 3] = -1.0

    # Propagate selection status.
    wh[8, 8, 2, 1] = 1.0
    uh[8, 4] = 1.0
    wh[9, 9, 0, 1] = 1.0
    uh[9, 4] = 1.0

    # Empty marker while an item is selected.
    uh[10, 0] = 1.0
    wh[10, 8, 1, 1] = 1.0
    bh[10] = -1.0

    # Memory shift register: six ACTION7s, then one ACTION5.
    p["bm"][0] = 10.0
    for i in range(1, 8):
        p["Wm"][i, i - 1] = 10.0

    # Normal policy: click if an empty marker exists, otherwise execute.
    p["Wz"][0, 32] = 1.0
    p["Wo"][6, 0] = 1.0
    p["Wo"][6, 128 + 7] = 2.0
    p["bo"][6] = -2.0

    # During warm-up, select ACTION7 for six steps, then ACTION5.
    p["bo"][7] = 2.0
    p["Wo"][7, 128 + 6] = -2.0
    p["bo"][5] = 0.5
    p["Wo"][5, 128 + 6] = 1.5
    p["Wo"][5, 128 + 7] = -1.5

    # Select a matching inventory item, then the first empty marker.
    p["Wc"][0, 2] = 30.0
    p["Wc"][0, 32 + 1] = 1.0
    p["Wc"][0, 32 + 6] = -1.0
    p["Wc"][0, 32 + 7] = -1.0
    p["Wc"][0, 32 + 9] = -30.0
    p["Wc"][0, 32 + 10] = 30.0

    return p
