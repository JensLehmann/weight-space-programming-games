"""Freeze the free-architecture (FA) condition: hash every FA prompt the runner will send (6 private families)
and the FA code (new files plus the shared files they extend). Writes manifests/rev2/freeze_fa.json.

Usage: freeze_fa.py [--families ztl,lys,esm,ruz,crg,mds] [--check]   (--check verifies an existing freeze)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_freearch import FA_FAMILIES, INFO, prompt_for  # noqa: E402
from wsp import freearch as FA  # noqa: E402
from wsp.manifest import GAMES_COMMIT  # noqa: E402
from wsp.queries import manifest_path  # noqa: E402

CODE_NEW = ["wsp/freearch.py", "wsp/prompts_fa.py", "wsp/_sandbox_runner_fa.py", "scripts/run_freearch.py",
            "scripts/freeze_fa.py", "scripts/smoke_fa.py", "tests/test_freearch.py"]
CODE_SHARED = ["wsp/actor.py", "wsp/envs.py", "wsp/prompts.py", "wsp/sandbox.py", "wsp/_sandbox_runner.py",
               "wsp/session.py", "wsp/writer.py", "scripts/run_rev2.py", "scripts/eval_artifacts.py",
               "scripts/diag_eval.py", "scripts/postdiag_parallel.py", "scripts/blind_eval.py"]


def h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build(families):
    root = Path(__file__).resolve().parent.parent
    out = {"games_commit": GAMES_COMMIT, "condition": "freearch", "info": INFO,
           "budget": {"params": FA.MAX_PARAMS, "macs_per_step": FA.MAX_MACS},
           "limits": {"memory": FA.MAX_MEM, "T": FA.MAX_T, "nodes": FA.MAX_NODES, "node_evals": FA.MAX_EVALS,
                      "channels": FA.MAX_CH, "total_channels": FA.MAX_TOTAL_CH, "vector": FA.MAX_VEC},
           "families": {}, "code": {}}
    for c in CODE_NEW + CODE_SHARED:
        out["code"][c] = h((root / c).read_bytes())
    for f in families:
        prompt, _ = prompt_for(f)
        out["families"][f] = {"manifest_sha256": h(manifest_path(f, rev=2).read_bytes()),
                              "prompt": {"sha256": h(prompt.encode()), "chars": len(prompt)}}
        print(f, out["families"][f]["prompt"], flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(FA_FAMILIES))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    fams = a.families.split(",")
    target = Path(__file__).resolve().parent.parent / "manifests" / "rev2" / "freeze_fa.json"
    cur = build(fams)
    if a.check:
        old = json.loads(target.read_text())
        bad = [f for f in fams if old["families"].get(f) != cur["families"][f]]
        code_bad = [c for c in cur["code"] if old["code"].get(c) != cur["code"][c]]
        print("families changed:", bad or "none", "| code changed:", code_bad or "none")
        sys.exit(1 if bad or code_bad else 0)
    cur["freeze_sha256"] = h(json.dumps(cur, sort_keys=True).encode())
    target.write_text(json.dumps(cur, indent=1, sort_keys=True))
    print("freeze", cur["freeze_sha256"][:16], "->", target)
