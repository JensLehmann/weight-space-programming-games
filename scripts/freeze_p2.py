"""Freeze protocol P2: hash manifests, redacted sections, every prompt the runner will send, and the
harness source. Writes manifests/rev2/freeze.json and prints a short summary.

Usage: freeze_p2.py [--families ...] [--check]   (--check verifies an existing freeze)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_rev2 import section, support_observations  # noqa: E402
from wsp.manifest import GAMES_COMMIT, ROOT  # noqa: E402
from wsp.prompts import build_prompt_v2  # noqa: E402
from wsp.queries import load_sets, manifest_path  # noqa: E402

PRIVATE20 = ["ztl", "wih", "qcn", "lys", "esm", "aiv", "nkp", "obn", "qwv", "ruz", "vvf", "ubr", "yuz", "bvn",
           "trx", "yiy", "crg", "qpv", "ydv", "mds"]
PROMPTS = [("weights", "M", "full"), ("python", "M", "full"), ("pyreact", "M", "full"),
           ("weights", "MS", "full"), ("weights", "M", "noaxial"), ("weights", "M", "t8")]
CODE = ["wsp/actor.py", "wsp/envs.py", "wsp/manifest.py", "wsp/prompts.py", "wsp/queries.py", "wsp/redact2.py",
        "wsp/redact2_overrides.py", "wsp/augment.py",
        "wsp/sandbox.py", "wsp/_sandbox_runner.py", "wsp/session.py", "wsp/writer.py", "scripts/run_rev2.py",
        "scripts/eval_artifacts.py", "scripts/diag_eval.py", "scripts/analyze_rev2.py"]


def h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build(families):
    code_root = Path(__file__).resolve().parent.parent
    out = {"games_commit": GAMES_COMMIT, "families": {}, "code": {}}
    for c in CODE:
        out["code"][c] = h((code_root / c).read_bytes())
    for f in families:
        sets = load_sets(f, rev=2)
        man = manifest_path(f, rev=2)
        fam = {"manifest_sha256": h(man.read_bytes()), "n": {k: len(v) for k, v in sets.items()},
               "sections": {}, "prompts": {}}
        obs = support_observations(sets)
        for info in ("M", "MS"):
            sec, rep = section(f, info)
            fam["sections"][info] = h(sec.encode())
        for cond, info, variant in PROMPTS:
            sec, _ = section(f, info)
            p = build_prompt_v2(sec, cond, obs, variant=variant)
            fam["prompts"][f"{cond}:{info}:{variant}"] = {"sha256": h(p.encode()), "chars": len(p)}
        out["families"][f] = fam
        print(f, fam["n"], flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(PRIVATE20))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    fams = a.families.split(",")
    target = Path(__file__).resolve().parent.parent / "manifests" / "rev2" / "freeze.json"
    cur = build(fams)
    if a.check:
        old = json.loads(target.read_text())
        bad = [f for f in fams if old["families"].get(f) != cur["families"][f]]
        code_bad = [c for c in CODE if old["code"].get(c) != cur["code"][c]]
        print("families changed:", bad or "none", "| code changed:", code_bad or "none")
        sys.exit(1 if bad else 0)
    cur["freeze_sha256"] = h(json.dumps(cur, sort_keys=True).encode())
    target.write_text(json.dumps(cur, indent=1, sort_keys=True))
    print("freeze", cur["freeze_sha256"][:16], "->", target)
