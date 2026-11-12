"""Source-licensed data augmentation for imitation baselines (BC-aug, distill-aug).

Two transformation families, both applied to training minibatches only:

* ``palette``  - a random permutation of the 16 colour ids, applied consistently to every frame of
  a sample (one step, or one BPTT window). Action labels are unchanged.
* ``dihedral`` - one of the 8 elements of the dihedral group D4 acting on the 64x64 frame. Element
  ``e = k + 4 f`` is "transpose if f, then rotate by k quarter turns" with ``np.rot90`` semantics
  (counter-clockwise as displayed, ``new[i, j] = old[j, N-1-i]``). Labels are remapped:
  ACTION1-4 are the screen directions up/down/left/right and are permuted by the linear part of the
  transform; ACTION6 click coordinates are mapped like pixels; RESET/ACTION5/ACTION7 are unchanged.
  The available-action mask and the previous-action one-hot are permuted the same way.

Which transformations are licensed is a per-family decision: ``augmentations_for(family)`` reads
``FAMILY_AUGMENT``. Palette permutation is licensed for every private family, the dihedral group in
addition for the families in ``DIHEDRAL_FAMILIES`` (criteria below). Families not in the table
raise ``KeyError``: each must be licensed explicitly.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from .envs import ACTIONS, A_INDEX

N = 64
from . import private as _private

PALETTE, DIHEDRAL = "palette", "dihedral"
KINDS = frozenset({PALETTE, DIHEDRAL})

# Private tier: the 20 families of the study.
PRIVATE_FAMILIES = ("ztl", "wih", "qcn", "lys", "esm", "aiv", "nkp", "obn", "qwv", "ruz", "vvf",
                  "ubr", "yuz", "bvn", "trx", "yiy", "crg", "qpv", "ydv", "mds")
DEFAULT_AUGMENT = frozenset({PALETTE})
# The dihedral group is licensed for a family when (a) its own renderer templates apply board
# symmetries that generate D4 and (b) ACTION1-4 are absent or are screen directions (checked against
# the sources before training). These are the 13 families listed in the paper's appendix.
DIHEDRAL_FAMILIES: frozenset = frozenset(_private.get("DIHEDRAL_FAMILIES", (
    "lys", "esm", "aiv", "obn", "qwv", "ubr", "trx", "qpv", "yiy", "crg", "bvn", "yuz", "mds")))
FAMILY_AUGMENT: dict[str, frozenset] = {
    f: DEFAULT_AUGMENT | ({DIHEDRAL} if f in DIHEDRAL_FAMILIES else set()) for f in PRIVATE_FAMILIES
}
FAMILY_AUGMENT["toy_reach"] = frozenset({PALETTE, DIHEDRAL})   # verified in tests/test_augment.py
# Public tier (official.<id>): verdicts from a survey of the 25 public game sources.
# Palette only where the survey verdict is "safe"; dihedral only where all 8 D4 elements are safe
# (subsets are not supported, so "subset" verdicts are not licensed). All 25 games of the tier are listed.
_PUBLIC_PALETTE = {"ar25", "cd82", "lp85", "ls20", "m0r0", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80",
                   "su15", "tn36", "tr87", "vc33"}
_PUBLIC_DIHEDRAL = {"cd82", "ft09", "ka59", "tu93"}
for _g in ("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 "
           "sp80 su15 tn36 tr87 tu93 vc33 wa30").split():
    FAMILY_AUGMENT[f"official.{_g}"] = frozenset(({PALETTE} if _g in _PUBLIC_PALETTE else set())
                                                 | ({DIHEDRAL} if _g in _PUBLIC_DIHEDRAL else set()))


def augmentations_for(family: str) -> set:
    if family not in FAMILY_AUGMENT:
        raise KeyError(f"no augmentation licence recorded for family {family!r}; add it to "
                       f"wsp.augment.FAMILY_AUGMENT (use an empty set if none is licensed)")
    return set(FAMILY_AUGMENT[family])


def parse_kinds(spec) -> set | None:
    """'none'/''/None -> None; 'palette,dihedral' -> set; iterables pass through (validated)."""
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = [s.strip() for s in spec.split(",") if s.strip() and s.strip() != "none"]
    kinds = set(spec)
    bad = kinds - KINDS
    if bad:
        raise ValueError(f"unknown augmentation kinds {sorted(bad)}; known: {sorted(KINDS)}")
    return kinds or None


# --------------------------------------------------------------------------- dihedral group D4
_R = np.array([[0, -1], [1, 0]])        # quarter turn acting on (row, col) offsets: (dr,dc)->(-dc,dr)
_TR = np.array([[0, 1], [1, 0]])        # transpose: (dr,dc)->(dc,dr)
ELEMENTS = tuple(range(8))
IDENTITY = 0


def matrix(e: int) -> np.ndarray:
    """Linear part of element e on (row, col) offsets: M = R^k @ TR^f."""
    k, f = e % 4, e // 4
    return np.linalg.matrix_power(_R, k) @ np.linalg.matrix_power(_TR, f)


_MATS = [matrix(e) for e in ELEMENTS]


def _element_of(m: np.ndarray) -> int:
    for e, me in enumerate(_MATS):
        if np.array_equal(me, m):
            return e
    raise ValueError("not a D4 element")


def compose(a: int, b: int) -> int:
    """Element c with T_c = T_a o T_b (apply b first, then a)."""
    return _element_of(_MATS[a] @ _MATS[b])


def inverse(e: int) -> int:
    return _element_of(_MATS[e].T)


def transform_cell(row: int, col: int, e: int, n: int = N) -> tuple[int, int]:
    """Pixel (row, col) of the original frame -> its position in the transformed frame."""
    k, f = e % 4, e // 4
    r, c = (col, row) if f else (row, col)
    for _ in range(k):
        r, c = n - 1 - c, r
    return int(r), int(c)


def transform_grid(grid: np.ndarray, e: int) -> np.ndarray:
    """Transform the last two axes of `grid` (consistent with `transform_cell`)."""
    k, f = e % 4, e // 4
    g = np.swapaxes(grid, -2, -1) if f else grid
    return np.ascontiguousarray(np.rot90(g, k, axes=(-2, -1)))


# directions of the screen-directional actions as (d_row, d_col)
DIRECTIONS = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}
_DIR_ACTION = {v: k for k, v in DIRECTIONS.items()}


def action_perm(e: int) -> np.ndarray:
    """perm[i] = index of the action that action i becomes under element e."""
    perm = np.arange(len(ACTIONS))
    for a, d in DIRECTIONS.items():
        d2 = tuple(int(x) for x in _MATS[e] @ np.array(d))
        perm[A_INDEX[a]] = A_INDEX[_DIR_ACTION[d2]]
    return perm


_PERM = np.stack([action_perm(e) for e in ELEMENTS])                       # (8, 8)
_INV_PERM = np.stack([np.argsort(p) for p in _PERM])                       # new idx -> old idx


def _cell_map(e: int) -> np.ndarray:
    """flat old cell index -> flat new cell index (equivalently: transform of an index grid)."""
    idx = np.arange(N * N).reshape(N, N)
    out = np.empty(N * N, np.int64)
    out[transform_grid(idx, e).ravel()] = np.arange(N * N)
    return out


_CELL_MAP = np.stack([_cell_map(e) for e in ELEMENTS])                     # (8, 4096)


def transform_move(move: tuple, e: int) -> tuple:
    a, r, c = move
    if a == "ACTION6":
        r2, c2 = transform_cell(int(r), int(c), e)
        return (a, r2, c2)
    return (ACTIONS[_PERM[e][A_INDEX[a]]], r, c)


def apply_palette(grid: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return np.asarray(perm)[np.asarray(grid)]


# --------------------------------------------------------------------------- batched (torch)
def _t_grid(x: torch.Tensor, e: int) -> torch.Tensor:
    k, f = e % 4, e // 4
    if f:
        x = x.transpose(-2, -1)
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
    return x


def augment_batch(obs: torch.Tensor, avail: torch.Tensor, prev: torch.Tensor, act: torch.Tensor,
                  cell: torch.Tensor, kinds: Iterable[str] | None, rng: np.random.Generator):
    """Augment a minibatch. One random transformation per leading index b, shared by all frames
    under it (so a BPTT window of shape (B, L, ...) is transformed consistently).

    obs (B,*S,64,64) long; avail (B,*S,8) bool; prev (B,*S,8) float one-hot; act (B,*S) long with
    -1 = unlabelled; cell (B,*S) long flat click index (row*64+col) or -1. Returns new tensors.
    """
    kinds = set(kinds or ())
    if not kinds:
        return obs, avail, prev, act, cell
    bad = kinds - KINDS
    if bad:
        raise ValueError(f"unknown augmentation kinds {sorted(bad)}")
    B, dev = obs.shape[0], obs.device
    if PALETTE in kinds:
        perms = torch.as_tensor(np.argsort(rng.random((B, 16)), axis=1), device=dev)
        obs = torch.gather(perms, 1, obs.long().reshape(B, -1)).reshape(obs.shape)
    if DIHEDRAL in kinds:
        es = rng.integers(0, 8, B)
        obs, avail, prev, act, cell = obs.clone(), avail.clone(), prev.clone(), act.clone(), cell.clone()
        perm_t = torch.as_tensor(_PERM, device=dev)
        inv_t = torch.as_tensor(_INV_PERM, device=dev)
        cmap_t = torch.as_tensor(_CELL_MAP, device=dev)
        for e in np.unique(es):
            e = int(e)
            if e == IDENTITY:
                continue
            sel = torch.as_tensor(es == e, device=dev)
            obs[sel] = _t_grid(obs[sel], e)
            avail[sel] = avail[sel][..., inv_t[e]]
            prev[sel] = prev[sel][..., inv_t[e]]
            a = act[sel]
            act[sel] = torch.where(a >= 0, perm_t[e][a.clamp(min=0)], a)
            c = cell[sel]
            cell[sel] = torch.where(c >= 0, cmap_t[e][c.clamp(min=0)], c)
    return obs, avail, prev, act, cell
