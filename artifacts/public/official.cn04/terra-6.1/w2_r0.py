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
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    # Blue endpoints and already-overlapped green endpoints.
    p["W1"][0, 8, 2, 2] = 1.0
    p["W1"][0, 3, 2, 2] = 1.0

    # A difference of two thresholded counts detects whether an
    # endpoint is near a black selected body.
    for ch, bias in ((1, -100.0), (2, -101.0)):
        p["W1"][ch, 8, 2, 2] = 100.0
        p["W1"][ch, 3, 2, 2] = 100.0
        p["W1"][ch, 0, :, :] = 1.0
        p["b1"][ch] = bias

    # The corresponding detector for an unselected coloured body.
    for ch, bias in ((3, -100.0), (4, -101.0)):
        p["W1"][ch, 8, 2, 2] = 100.0
        p["W1"][ch, 3, 2, 2] = 100.0
        p["W1"][ch, 12, :, :] = 1.0
        p["W1"][ch, 14, :, :] = 1.0
        p["b1"][ch] = bias

    p["W1"][5, 16, 2, 2] = 1.0
    p["W1"][6, 17, 2, 2] = 1.0

    # Extend ownership across each displayed endpoint block.
    for out_a, out_b, in_a, in_b in ((0, 1, 1, 2), (2, 3, 3, 4)):
        for out_ch, bias in ((out_a, -100.0), (out_b, -101.0)):
            p["W2"][out_ch, 0, 1, 1] = 100.0
            p["W2"][out_ch, in_a, :, :] = 1.0
            p["W2"][out_ch, in_b, :, :] = -1.0
            p["b2"][out_ch] = bias

    p["W2"][4, 5, 1, 1] = 1.0
    p["W2"][5, 6, 1, 1] = 1.0

    # Max and reversed-min coordinate fields for selected and target
    # endpoints. The spatial maxima enter g at indices 32:40.
    for base, mask_a, mask_b in ((0, 0, 1), (4, 2, 3)):
        for out_ch, coord_ch, sign, bias in (
            (base,     4,  1.0, -1.0),
            (base + 1, 4, -1.0,  0.0),
            (base + 2, 5,  1.0, -1.0),
            (base + 3, 5, -1.0,  0.0),
        ):
            p["W3"][out_ch, mask_a] = 1.0
            p["W3"][out_ch, mask_b] = -1.0
            p["W3"][out_ch, coord_ch] = sign
            p["b3"][out_ch] = bias

    # ReLU hinge features for four endpoint coordinates:
    # selected row, selected column, target row, target column.
    # For a 20x20 board, the last pixel of grid cell q is 4 + 3q.
    coordinate_g = (32, 34, 36, 38)
    for group, g_index in enumerate(coordinate_g):
        for j in range(19):
            z_index = 19 * group + j
            p["Wz"][z_index, g_index] = 21.0
            p["bz"][z_index] = -4.0 / 3.0 - j

    # Vertical spans, excluding the width of one displayed cell.
    p["Wz"][76, 32] = 1.0
    p["Wz"][76, 33] = 1.0
    p["bz"][76] = -1.0 - 2.0 / 63.0
    p["Wz"][77, 36] = 1.0
    p["Wz"][77, 37] = 1.0
    p["bz"][77] = -1.0 - 2.0 / 63.0

    # Startup clock. The first three actions either rotate three times
    # or click the letterbox, depending on the orientation comparison.
    p["bm"][50] = 5.0
    p["Wm"][51, 50] = 10.0
    p["bm"][51] = -5.0
    p["Wm"][52, 51] = 10.0
    p["bm"][52] = -5.0
    p["Wm"][53, 52] = 10.0
    p["bm"][53] = -5.0

    # Latch the two signs of the initial span difference. Clear them
    # as startup ends so that they no longer affect rotation scores.
    for unit, sign in ((42, 1.0), (43, -1.0)):
        p["Wmz"][unit, 76] = 500.0 * sign
        p["Wmz"][unit, 77] = -500.0 * sign
        p["Wm"][unit, unit] = 100.0
        p["Wm"][unit, 50] = -100.0
        p["Wm"][unit, 52] = -100.0
        p["bm"][unit] = -20.0

    def add_coordinate_bits(memory_start, group, pulse_a, pulse_b):
        """Latch five binary coordinate bits on a one-step pulse."""
        for bit in range(5):
            unit = memory_start + bit
            previous_slope = 0
            for j in range(19):
                value_j = (j >> bit) & 1
                value_next = ((j + 1) >> bit) & 1
                slope = value_next - value_j
                coefficient = slope - previous_slope
                if coefficient:
                    p["Wmz"][unit, 19 * group + j] = 8.0 * coefficient
                previous_slope = slope

            p["Wm"][unit, unit] = 20.0
            p["Wm"][unit, pulse_a] = 14.0
            p["Wm"][unit, pulse_b] = -14.0
            p["bm"][unit] = -12.0

    # First snapshot, taken after the three startup actions.
    for group in range(4):
        add_coordinate_bits(5 * group, group, 52, 53)

    # Integrated movement since the first snapshot.
    p["Wm"][20, 20] = 1.0
    p["Wma"][20, 1] = -0.005
    p["Wma"][20, 2] = 0.005
    p["Wm"][21, 21] = 1.0
    p["Wma"][21, 3] = -0.005
    p["Wma"][21, 4] = 0.005

    # When an alignment attempt produces a rotation, latch a phase flag.
    # Its delayed copy produces exactly one forced second rotation.
    p["Wm"][55, 55] = 30.0
    p["Wm"][55, 53] = 30.0
    p["Wma"][55, 5] = 55.0
    p["bm"][55] = -50.0

    p["Wm"][56, 55] = 10.0
    p["bm"][56] = -5.0

    # Second selected-coordinate snapshot after the half-turn.
    add_coordinate_bits(22, 0, 55, 56)
    add_coordinate_bits(27, 1, 55, 56)

    # Copy the first target snapshot into the second bank at that pulse.
    for axis in range(2):
        for bit in range(5):
            source = 10 + 5 * axis + bit
            unit = 32 + 5 * axis + bit
            p["Wm"][unit, unit] = 20.0
            p["Wm"][unit, source] = 4.0
            p["Wm"][unit, 55] = 14.0
            p["Wm"][unit, 56] = -14.0
            p["bm"][unit] = -8.0

    # Direction scores. Each snapshot contributes target minus selected;
    # the integrated movement is subtracted from the total.
    for bit in range(5):
        weight = (1 << bit) / 2.0

        # Down and up.
        for unit, coeff in (
            (10 + bit, weight), (bit, -weight),
            (32 + bit, weight), (22 + bit, -weight),
        ):
            p["Wo"][2, 128 + unit] += coeff
            p["Wo"][1, 128 + unit] -= coeff

        # Right and left.
        for unit, coeff in (
            (15 + bit, weight), (5 + bit, -weight),
            (37 + bit, weight), (27 + bit, -weight),
        ):
            p["Wo"][4, 128 + unit] += coeff
            p["Wo"][3, 128 + unit] -= coeff

    p["Wo"][2, 128 + 20] = -200.0
    p["Wo"][1, 128 + 20] = 200.0
    p["Wo"][4, 128 + 21] = -200.0
    p["Wo"][3, 128 + 21] = 200.0

    # Startup rotation versus a harmless letterbox click.
    # After startup, the small rotation bias tries the other endpoint
    # assignment when the estimated displacement is zero.
    p["Wo"][5, 128 + 42] = 50.0
    p["Wo"][5, 128 + 43] = 50.0
    p["Wo"][5, 128 + 53] = -100.0
    p["Wo"][5, 128 + 55] = 100.0
    p["Wo"][5, 128 + 56] = -100.0
    p["bo"][5] = 200.2

    p["Wo"][6, 128 + 53] = -125.0
    p["bo"][6] = 125.0

    # A zero click map selects display cell (0, 0), in the letterbox.
    p["bo"][0] = -1000.0
    p["bo"][7] = -1000.0

    return p
