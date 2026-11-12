"""Loader for private per-family material that is not part of this release.

Some harness tables quote the private game families' sources verbatim (regex overrides of the
redaction, engine-hook adapters, per-family file, identifier and attribute names) or state facts
about them (mechanism predictions, augmentation licences, renderer-template counts). They are kept in one private Python module
whose file path is given by ``$WSP_PRIVATE_REDACTION``. Without it every such table is empty and
only the generic machinery is available; the public tier never needs it.

Names the private module may define (all optional; see README "What is not included"):
  overrides_for, DOC_ROOTS, MS_SOLVER_ROOTS, FAMILIES   -> wsp.redact2_overrides (revision-2 redaction)
  ADAPTERS, VOCAB_ALLOW_EXTRA, RESOURCE_ATTRS_EXTRA     -> scripts/check_redaction.py
  MECHANISM_ARTIFACTS                                   -> scripts/mechanism_rev2.py
  DIHEDRAL_FAMILIES                                     -> wsp.augment (dihedral licence verdicts)
  N_TEMPLATES                                           -> scripts/analyze_rev2.py (renderer counts != 10)
"""
from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

ENV = "WSP_PRIVATE_REDACTION"


@lru_cache(maxsize=1)
def module():
    """The private module, or None if ``$WSP_PRIVATE_REDACTION`` is unset."""
    path = os.environ.get(ENV)
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{ENV}={path} does not name a file")
    spec = importlib.util.spec_from_file_location("wsp_private_redaction", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def get(name: str, default):
    """Attribute `name` of the private module, or `default` if there is no module or no such name."""
    mod = module()
    return getattr(mod, name, default) if mod is not None else default


def require(name: str):
    """Attribute `name` of the private module; raises with a pointer to the README otherwise."""
    value = get(name, None)
    if value is None:
        raise LookupError(f"{name} is private per-family material that is not part of this release; "
                          f"set {ENV} to the private module that defines it (see README)")
    return value
