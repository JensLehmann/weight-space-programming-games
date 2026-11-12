"""Figure: one state of the public game ls20 under its own rendering and under an illustrative template.

The template is constructed for this figure only (it is not one of the study's renderer templates and
was never shown to a writer or actor). It changes the three things the private templates vary:
object and HUD style (sprite redraws), geometry (a quarter turn of the whole frame, which moves the HUD
bar to the side) and palette (a permutation of the 16 colour ids). Both frames are produced by the game
engine's own camera; the native one is checked against the observation the actors receive.

Usage: fig_template_example.py [--out fig_template_example.png]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arc3_games.games.gallery import PALETTE  # noqa: E402
from wsp.envs import enter_level, obs_of  # noqa: E402

GAME, LEVEL = "official.ls20", 1          # level 2 (0-based 1), level start
SCALE, GAP = 8, 3                          # pixels per cell, white gap in cells

# Sprite redraws (-1 = transparent). Names are the game's own sprite identifiers.
REDRAW = {
    "sfqyzhzkij": [[-1, 12, 12, 12, -1], [12, 12, 9, 12, 12], [12, 9, 9, 9, 12],
                   [12, 12, 9, 12, 12], [-1, 12, 12, 12, -1]],                     # player
    "npxgalaybz": [[-1, 11, -1], [11, 11, 11], [-1, 11, -1]],                    # collectible
    "ihdgageizm": [[4, 4, 4, 4, 4], [4, 2, 4, 2, 4], [4, 4, 4, 4, 4],
                   [4, 2, 4, 2, 4], [4, 4, 4, 4, 4]],                              # wall tile
    "vjotnebuqo": [[5, -1, 5, -1, 5, -1, 5], [-1] * 7, [5] + [-1] * 5 + [5], [-1] * 7,
                   [5] + [-1] * 5 + [5], [-1] * 7, [5, -1, 5, -1, 5, -1, 5]],      # key frame
}
ROLE_COLOURS = {4: 1, 3: 0, 5: 4, 2: 3, 9: 14, 12: 8, 11: 6, 8: 15, 0: 5, 1: 2}


def palette_permutation() -> dict:
    m = dict(ROLE_COLOURS)
    free = [c for c in range(16) if c not in m.values()]
    for c in range(16):
        if c not in m:
            m[c] = free.pop(0)
    assert sorted(m.values()) == list(range(16))
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_template_example.png")
    args = ap.parse_args()

    game = enter_level(GAME, 0, LEVEL)
    engine, level = game.engine, game.engine.current_level
    native = obs_of(game)
    assert (np.asarray(engine.camera.render(level.get_sprites())) == native).all()

    for sprite in level.get_sprites():
        if sprite.name in REDRAW:
            sprite.pixels = np.array(REDRAW[sprite.name], dtype=np.int8)
    variant = np.rot90(np.asarray(engine.camera.render(level.get_sprites())).astype(int), k=-1)
    perm = palette_permutation()
    variant = np.vectorize(perm.get)(variant)
    print(f"cells changed: {(variant != native).mean():.1%}")

    pal = np.array(PALETTE, dtype=np.uint8)
    gap = np.full((64, GAP, 3), 255, np.uint8)
    img = np.concatenate([pal[native], gap, pal[variant]], 1).repeat(SCALE, 0).repeat(SCALE, 1)
    Image.fromarray(img).save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
