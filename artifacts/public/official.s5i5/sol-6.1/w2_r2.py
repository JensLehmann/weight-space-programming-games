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

    colours = (7, 8, 9, 10, 11, 12, 14)
    ix = {c: i for i, c in enumerate(colours)}

    def a(out, colour, dy, dx, weight):
        p["W1"][out, colour, dy + 2, dx + 2] += weight

    def b(out, channel, dy=0, dx=0, weight=1.0):
        p["W2"][out, channel, dy + 1, dx + 1] += weight

    def c(out, channel, weight=1.0, axis=None):
        offset = {None: 0, "row": 32, "col": 64}[axis]
        p["W3"][out, offset + channel] += weight

    # Colour-specific marked endpoints.
    for i, colour in enumerate(colours):
        horizontal, vertical = 2 * i, 2 * i + 1
        sign = -1 if colour == 12 else 1

        for out in (horizontal, vertical):
            a(out, 13, 0, 0, 12.0)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy or dx:
                        a(out, 13, dy, dx, -12.0)
            p["b1"][out] = -15.0

        for dy in (-1, 1):
            a(horizontal, colour, dy, 0, 1.0)
        for dy in (-1, 0, 1):
            a(horizontal, colour, dy, -2 * sign, 2.0 / 3.0)
            a(horizontal, colour, dy, 2 * sign, -2.0 / 3.0)
        a(horizontal, 3, 0, -sign, 2.0)
        a(horizontal, 3, 0, sign, -2.0)

        for dx in (-1, 1):
            a(vertical, colour, 0, dx, 1.0)
        for dx in (-1, 0, 1):
            a(vertical, colour, -2, dx, 2.0 / 3.0)
            a(vertical, colour, 2, dx, -2.0 / 3.0)
        a(vertical, 3, -1, 0, 2.0)
        a(vertical, 3, 1, 0, -2.0)

        a(14 + i, colour, 0, 0, 1.0)

    # A vertical colour-12 parent directly above a carried marker.
    parent12 = 2 * ix[12] + 1
    p["W1"][parent12] = 0.0
    p["b1"][parent12] = -12.0
    a(parent12, 13, 0, 0, 12.0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                a(parent12, 13, dy, dx, -12.0)
    for dx in (-1, 0, 1):
        a(parent12, 12, -2, dx, 1.0 / 3.0)

    # Stop direct leftward extension immediately before a solid wall.
    a(2 * ix[12], 15, 0, -2, -12.0)

    # Repurpose endpoint orientations for local clearance predicates.
    front14_first = 2 * ix[7]
    top8_first = 2 * ix[8]
    down14_first = 2 * ix[9]
    down_rigid_first = 2 * ix[9] + 1
    left9_first = 2 * ix[11]
    right_rigid_first = 2 * ix[14] + 1

    for channel in (
        front14_first, top8_first, down14_first,
        down_rigid_first, left9_first, right_rigid_first,
    ):
        p["W1"][channel] = 0.0
        p["b1"][channel] = 0.0

    # A marked horizontal endpoint facing a wall.
    a(front14_first, 13, 0, 0, 1.0)
    a(front14_first, 15, 0, 2, 1.0)
    p["b1"][front14_first] = -1.0

    # Nonminimal vertical arm below an attached horizontal rigid bar.
    a(top8_first, 8, 0, 0, 1.0)
    a(top8_first, 8, 2, 0, 1.0)
    a(top8_first, 1, -1, 0, 1.0)
    p["b1"][top8_first] = -2.0

    # A horizontal carried arm facing a wall below.
    a(down14_first, 14, 0, 0, 1.0)
    a(down14_first, 15, 2, 0, 1.0)
    p["b1"][down14_first] = -1.0

    # Imminent downward contact of a rigid bar.
    a(down_rigid_first, 1, -1, 0, 1.0)
    for colour in (7, 9, 10, 11, 12, 14, 15):
        a(down_rigid_first, colour, 2, 0, 1.0)
    p["b1"][down_rigid_first] = -1.0

    # Nonminimal horizontal arm beside an attached vertical rigid bar.
    a(left9_first, 9, 0, 0, 1.0)
    a(left9_first, 9, 0, 2, 1.0)
    a(left9_first, 1, 0, -1, 1.0)
    p["b1"][left9_first] = -2.0

    # Imminent rightward contact of a rigid bar.
    a(right_rigid_first, 1, 0, -1, 1.0)
    for colour in (7, 8, 10, 11, 12, 14, 15):
        a(right_rigid_first, colour, 0, 2, 1.0)
    p["b1"][right_rigid_first] = -1.0

    # Four length-control glyph orientations.
    glyphs = (
        (21, ((0, 0), (-1, 0), (1, 0), (-1, 1), (1, 1)), (0, 1)),
        (22, ((0, 0), (0, -1), (0, 1), (1, -1), (1, 1)), (1, 0)),
        (23, ((0, 0), (-1, 0), (1, 0), (-1, -1), (1, -1)), (0, -1)),
        (24, ((0, 0), (0, -1), (0, 1), (-1, -1), (-1, 1)), (-1, 0)),
    )
    for out, points, hole in glyphs:
        for dy, dx in points:
            for colour in colours:
                a(out, colour, dy, dx, 1.0)
        a(out, 4, hole[0], hole[1], 1.0)
        p["b1"][out] = -5.0

    # Unfilled target, coordinate sum, and row coordinate.
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        a(25, 13, dy, dx, 1.0)
    a(25, 13, 0, 0, -4.0)
    p["b1"][25] = -3.0

    a(26, 16, 0, 0, 1.0)
    a(26, 17, 0, 0, 1.0)
    a(27, 16, 0, 0, 1.0)

    # Nonminimal colour-12 arm attached to a vertical rigid bar.
    a(28, 12, 0, 0, 1.0)
    a(28, 12, 0, 2, 1.0)
    a(28, 1, 0, -1, 1.0)
    p["b1"][28] = -2.0

    # Isolated carried marker.
    a(29, 13, 0, 0, 1.0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                a(29, 13, dy, dx, -1.0)

    # Rotation-button centre.
    for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        for colour in colours:
            a(30, colour, dy, dx, 1.0)
    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        a(30, 2, dy, dx, 1.0)
    p["b1"][30] = -8.0

    # Upward contact with a horizontal colour-10 obstruction.
    a(31, 9, 0, 0, 1.0)
    a(31, 1, 0, 0, 1.0)
    a(31, 10, -2, 0, 1.0)
    p["b1"][31] = -1.0

    # Second layer: coordinate and two presence channels per colour.
    # Presence channels also identify colour-specific panel glyphs.
    disabled_h = {7, 8, 9, 11}
    disabled_v = {9, 14}

    for i, colour in enumerate(colours):
        horizontal, vertical = 2 * i, 2 * i + 1
        coordinate, hp, vp = 3 * i, 3 * i + 1, 3 * i + 2

        if colour not in disabled_h:
            b(coordinate, horizontal, weight=4.0)
        if colour not in disabled_v:
            b(coordinate, vertical, weight=4.0)
        b(coordinate, 26)
        p["b2"][coordinate] = -4.0

        for out, endpoint, enabled in (
            (hp, horizontal, colour not in disabled_h),
            (vp, vertical, colour not in disabled_v),
        ):
            b(out, 14 + i)
            for role in (21, 22, 23, 24):
                b(out, role)
            if enabled:
                b(out, endpoint, weight=2.0)
            p["b2"][out] = -1.0

    # Encode two clearance predicates in otherwise unused horizontal
    # presence channels. Panel contributions are removed in W3.
    b(3 * ix[7] + 1, front14_first, weight=2.0)
    b(3 * ix[9] + 1, down14_first, weight=2.0)

    # Unused coordinate channels carry upward contact and rotation.
    upward_contact = 3 * ix[8]
    rotation_second = 3 * ix[9]
    for out in (upward_contact, rotation_second):
        p["W2"][out] = 0.0
        p["b2"][out] = 0.0
    b(upward_contact, 31)
    b(rotation_second, 30)

    marker_row, target_row = 21, 22
    top8, down_rigid, left9, right_rigid, long12 = 23, 24, 25, 26, 27
    decrement, increment, target, target_sum = 28, 29, 30, 31

    b(marker_row, 29, weight=4.0)
    b(marker_row, 27)
    p["b2"][marker_row] = -3.0

    b(target_row, 25, weight=4.0)
    b(target_row, 27)
    p["b2"][target_row] = -3.0

    b(top8, top8_first)
    b(down_rigid, down_rigid_first)
    b(down_rigid, down_rigid_first, dy=1)
    b(left9, left9_first)
    b(right_rigid, right_rigid_first)
    b(long12, 28)

    b(decrement, 21)
    b(decrement, 22)
    b(increment, 23)
    b(increment, 24)
    b(target, 25)
    b(target_sum, 25, weight=2.0)
    b(target_sum, 26, weight=0.5)
    p["b2"][target_sum] = -2.0

    # Third-layer channel assignments.
    grow = {colour: 2 * i for i, colour in enumerate(colours)}
    shrink = {colour: 2 * i + 1 for i, colour in enumerate(colours)}
    vertical = {7: 14, 9: 15, 11: 16, 12: 17}
    rotation_third = 18
    selector = {colour: 19 + i for i, colour in enumerate(colours)}
    dec3, inc3 = 26, 27
    clearance8, clearance9, clearance12 = 28, 29, 30
    lateral_clearance = 31

    # Signed alignment error on the carried marker's current axis.
    for colour in colours:
        if colour in (8, 9):
            continue
        i = ix[colour]
        coordinate, hp, vp = 3 * i, 3 * i + 1, 3 * i + 2

        for out, direction in (
            (grow[colour], 1.0),
            (shrink[colour], -1.0),
        ):
            if colour == 12:
                direction = -direction
            c(out, hp, 3.0, "row")
            c(out, vp, 3.0, "col")
            for role in (decrement, increment):
                c(out, role, -3.0, "row")
                c(out, role, -3.0, "col")
            c(out, target)
            c(out, target_sum, direction)
            c(out, coordinate, -0.5 * direction, "row")
            c(out, coordinate, -0.5 * direction, "col")
            p["b3"][out] = -4.0

    for colour, out in selector.items():
        i = ix[colour]
        c(out, 3 * i + 1)
        c(out, 3 * i + 2)
        c(out, decrement)
        c(out, increment)
        p["b3"][out] = -2.0

    for colour, out in vertical.items():
        vp = 3 * ix[colour] + 2
        c(out, vp, 3.0, "col")
        for role in (decrement, increment):
            c(out, role, -3.0, "col")
        c(out, target, 3.0, "row")
        c(out, target_row, 1.0, "row")
        c(out, marker_row, -1.0, "col")
        p["b3"][out] = -6.0

    c(rotation_third, rotation_second)
    c(dec3, decrement)
    c(inc3, increment)

    # Clearance requests for unmarked attached rigid bars.
    c(clearance8, top8)
    c(clearance8, down_rigid, -10.0, "row")
    c(clearance8, marker_row, -4.0, "row")
    c(clearance8, marker_row, -4.0, "col")

    c(clearance9, left9)
    c(clearance9, right_rigid, -10.0, "col")
    c(clearance9, marker_row, -4.0, "row")
    c(clearance9, marker_row, -4.0, "col")

    c(clearance12, long12)
    c(clearance12, right_rigid, -10.0, "col")
    c(clearance12, marker_row, -4.0, "row")
    c(clearance12, marker_row, -4.0, "col")

    c(lateral_clearance, down_rigid)
    c(lateral_clearance, top8, 1.0, "row")
    p["b3"][lateral_clearance] = -1.0

    # Freed channels 2..5:
    # rightward rigid contact, upward contact, lower wall contact,
    # and marked-endpoint front contact.
    right_contact3 = 2
    upward_contact3 = 3
    lower_wall3 = 4
    front_wall3 = 5

    c(right_contact3, right_rigid)
    c(right_contact3, left9, 1.0, "col")
    c(right_contact3, long12, 1.0, "col")
    p["b3"][right_contact3] = -1.0

    c(upward_contact3, upward_contact)

    c(lower_wall3, 3 * ix[9] + 1)
    c(front_wall3, 3 * ix[7] + 1)
    for role in (decrement, increment):
        c(lower_wall3, role, -1.0)
        c(front_wall3, role, -1.0)

    # Recurrent command fields. Channel 1 and channel 15 are
    # obstruction fields rather than click-command fields.
    gate = np.float32(2.0 ** 120)
    stencil = ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2))
    command_channels = [h for h in range(14) if h != 1]

    for h in range(14):
        for y, x in stencil:
            p["Wh"][h, h, y, x] = 0.22
    for y, x in stencil:
        p["Wh"][15, 15, y, x] = 0.22

    for i, colour in enumerate(colours):
        gh, sh = i, 7 + i

        if gh != 1:
            p["Uh"][gh, selector[colour]] = gate
            p["Uh"][gh, inc3] = -gate
            p["Uh"][gh, dec3] = -2.0 * gate

        p["Uh"][sh, selector[colour]] = gate
        p["Uh"][sh, dec3] = -gate
        p["Uh"][sh, inc3] = -2.0 * gate

        if colour not in (8, 9):
            p["Uh"][gh, grow[colour]] = 1.0
            p["Uh"][sh, shrink[colour]] = 1.0
        if colour in vertical:
            p["Uh"][gh, vertical[colour]] = (
                1.0e3 if colour == 12 else 1.0
            )

    # Clearance precedence is analytic and independent of coordinates.
    p["Uh"][7 + ix[8], clearance8] = 1.0e6
    p["Uh"][7 + ix[9], clearance9] = 1.0e8
    p["Uh"][7 + ix[12], clearance12] = 1.0e12
    p["Uh"][ix[9], lateral_clearance] = 1.0e14
    p["Uh"][ix[14], right_contact3] = 1.0e16
    p["Uh"][7 + ix[10], upward_contact3] = 1.0e20

    # A low-priority shortening request frees a carried horizontal arm
    # before its vertical parent moves it through a wall.
    p["Uh"][7 + ix[14], lower_wall3] = 1.0e-10

    # Obstruction fields inhibit the corresponding growth commands.
    p["Uh"][1] = 0.0
    p["Uh"][1, front_wall3] = 1.0
    p["Uh"][15, lower_wall3] = 1.0
    for y, x in stencil:
        p["Wh"][ix[14], 1, y, x] = -1.0e30
        p["Wh"][ix[12], 15, y, x] = -1.0e24

    # Channel 14 upper-bounds summed command activity off panels.
    # Inhibitory cross-connections are deliberately omitted from this
    # reconstruction, so suppression cannot create off-panel clicks.
    for h in command_channels:
        for y, x in stencil:
            p["Wh"][14, h, y, x] = 0.22
    p["Uh"][14] = np.sum(
        p["Uh"][command_channels], axis=0, dtype=np.float32
    )

    for h in command_channels:
        p["Wc"][0, 32 + h] = np.float32(1.0e-10)
    p["Wc"][0, 46] = np.float32(-2.0e-10)
    p["Wc"][0, rotation_third] = np.float32(1.0e-38)

    p["bo"][6] = 1.0
    return p
