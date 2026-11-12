"""Paths, hashes and access to the game-collection package."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests"
# Run outputs live outside Git (they contain private material); WSP_RUNS overrides the location.
RUNS = Path(os.environ.get("WSP_RUNS") or ROOT.parent / "runs")
GAMES_COMMIT = "88425fbecb0cd4d44ce18bcf8adde185e019ae02"  # game-collection commit of this study (freeze manifests)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def games_package():
    """The game-collection package (`arc3_games`, not part of this release; see README)."""
    try:
        import arc3_games
    except ImportError as exc:
        raise ImportError("the game-collection package `arc3_games` is not installed; the private families "
                          "(their games and attempt detector) and the planner need it (see README)") from exc
    return arc3_games
