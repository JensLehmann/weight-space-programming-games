"""Per-family overrides for the revision-2 redaction (`wsp.redact2`): public stub.

The overrides used for the paper are regex substitutions, applied in list order to every redacted
file of a family. Their patterns and replacements quote the private game sources verbatim (from
the redaction audits, explicit move-budget tables and `_engine_*` hook
definitions), so they are not part of this release.

If ``$WSP_PRIVATE_REDACTION`` names the private module (see `wsp.private`; the as-run version of
this file), its `overrides_for`, `DOC_ROOTS`, `MS_SOLVER_ROOTS` and `FAMILIES` are used unchanged.
Otherwise every per-family table is empty and `overrides_for` returns only the generic G2 entry
below (G1 is done in `redact2._insert_withheld`).

Interface (unchanged): `Ov(pattern, repl, conds, note, optional)`; `Ov.conds` says in which
conditions an entry runs: "M" (mechanics only) and/or "MS" (mechanics + solver). Every entry must
match at least once in each condition it runs in, unless `optional=True`;
`scripts/check_redaction.py` reports the per-entry match counts.
"""
from __future__ import annotations

from typing import NamedTuple

from . import private


class Ov(NamedTuple):
    pattern: str
    repl: str
    conds: tuple = ("M",)
    note: str = ""
    optional: bool = False


M_ONLY, BOTH, MS_ONLY = ("M",), ("M", "MS"), ("MS",)

# Generic entries that apply to every family (not game-specific).
_COMMON = [
    # G1: `WITHHELD = None` is prepended before the module docstring and `from __future__`; compile() fails.
    (r"\AWITHHELD = None  # placeholder for an evaluation-only entry you cannot see\n((?s:.*?)from __future__ import annotations\n)",
     r"\1WITHHELD = None  # placeholder for an evaluation-only entry you cannot see\n"),
    # G2: `tuple(t.key for t in X_TEMPLATES)` raises on the WITHHELD (None) entries at class creation.
    (r"tuple\((\w+)\.key for \1 in (\w+TEMPLATES)\)", r"tuple(\1.key for \1 in \2 if \1 is not WITHHELD)"),
]


def _G1(p: str) -> bool:
    return p.startswith(r"\AWITHHELD = None")


def _generic_overrides_for(family: str) -> list[Ov]:
    return [Ov(p, r, BOTH, "G2", optional=True) for p, r in _COMMON if not _G1(p)]


_priv = private.module()
if _priv is not None and hasattr(_priv, "overrides_for"):
    overrides_for = _priv.overrides_for
    DOC_ROOTS: dict[str, dict[str, str]] = getattr(_priv, "DOC_ROOTS", {})
    MS_SOLVER_ROOTS: dict[str, set[str]] = getattr(_priv, "MS_SOLVER_ROOTS", {})
    FAMILIES: tuple = tuple(getattr(_priv, "FAMILIES", ()))
else:
    overrides_for = _generic_overrides_for
    # both conditions: unreferenced definitions kept on purpose, with the reason (private; empty here)
    DOC_ROOTS = {}
    # condition MS: extra solver entry points for the tree shaker (empty in the paper's runs too)
    MS_SOLVER_ROOTS = {}
    FAMILIES = ()
