"""Freeze the bounded program player (pbounded) condition: hash every prompt the runner will send (20 private families)
and the code involved (new files plus the shared files they use). Writes manifests/rev2/freeze_pbounded.json, in the
layout of freeze_fa_controls.json ("code", "prompts" by condition and family, "freeze_sha256") plus per-family
manifest hashes and prompt sizes, as freeze_fa.json.

Before writing, every family is checked against the frozen FA prompt (freeze_fa.json): the FA prompt rebuilt now
must still have its frozen hash, and the pbounded prompt must equal it except for the title line and the
player/task sections (same game section, episode rules and support observations).

Usage: freeze_pbounded.py [--families ztl,...] [--check]   (--check verifies an existing freeze)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_freearch as RF  # noqa: E402
from run_pbounded import COND, INFO, PB_FAMILIES, prompt_for  # noqa: E402
from wsp import freearch as FA  # noqa: E402
from wsp import pbounded as PB  # noqa: E402
from wsp.manifest import GAMES_COMMIT  # noqa: E402
from wsp.prompts_fa import FA_SPEC, FA_TASK  # noqa: E402
from wsp.prompts_pb import PB_SPEC, PB_TASK  # noqa: E402
from wsp.queries import manifest_path  # noqa: E402

CODE_NEW = ["wsp/pbounded.py", "wsp/_pb_runtime.py", "wsp/_sandbox_runner_pb.py", "wsp/prompts_pb.py",
            "scripts/run_pbounded.py", "scripts/freeze_pbounded.py", "scripts/smoke_pbounded.py",
            "tests/test_pbounded.py"]
CODE_SHARED = ["wsp/freearch.py", "wsp/prompts.py", "wsp/prompts_fa.py", "wsp/actor.py", "wsp/envs.py",
               "wsp/sandbox.py", "wsp/_sandbox_runner.py", "wsp/session.py", "wsp/writer.py", "wsp/queries.py",
               "wsp/redact2.py", "wsp/redact2_overrides.py", "scripts/run_rev2.py", "scripts/run_freearch.py",
               "scripts/eval_artifacts.py"]
ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "manifests" / "rev2" / "freeze_pbounded.json"


def h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check_against_fa(fam, prompt, fa_frozen):
    FA.CONTROL = None
    fa_prompt, _ = RF.prompt_for(fam)
    if fam in fa_frozen and h(fa_prompt.encode()) != fa_frozen[fam]["prompt"]["sha256"]:
        raise SystemExit(f"{fam}: the FA prompt no longer matches freeze_fa.json")
    head_pb = prompt[: -len(PB_SPEC + "\n\n" + PB_TASK)]
    head_fa = fa_prompt[: -len(FA_SPEC + "\n\n" + FA_TASK)]
    if head_pb.split("\n", 1)[1] != head_fa.split("\n", 1)[1]:
        raise SystemExit(f"{fam}: game section, episode rules or support observations differ from the FA prompt")


def build(families):
    fa_frozen = json.loads((ROOT / "manifests" / "rev2" / "freeze_fa.json").read_text())["families"]
    out = {"games_commit": GAMES_COMMIT, "condition": COND, "info": INFO, "budget": PB.budget(),
           "caps": PB.CAPS, "code": {}, "families": {}, "prompts": {COND: {}}}
    for c in CODE_NEW + CODE_SHARED:
        out["code"][c] = h((ROOT / c).read_bytes())
    for f in families:
        prompt, _ = prompt_for(f)
        check_against_fa(f, prompt, fa_frozen)
        sha = h(prompt.encode())
        out["families"][f] = {"manifest_sha256": h(manifest_path(f, rev=2).read_bytes()),
                              "prompt": {"sha256": sha, "chars": len(prompt)},
                              "fa_prompt_sha256": fa_frozen.get(f, {}).get("prompt", {}).get("sha256")}
        out["prompts"][COND][f] = sha
        print(f, out["families"][f]["prompt"], flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(PB_FAMILIES))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    fams = a.families.split(",")
    cur = build(fams)
    if a.check:
        old = json.loads(TARGET.read_text())
        bad = [f for f in fams if old["families"].get(f) != cur["families"][f]]
        code_bad = [c for c in cur["code"] if old["code"].get(c) != cur["code"][c]]
        print("families changed:", bad or "none", "| code changed:", code_bad or "none")
        sys.exit(1 if bad or code_bad else 0)
    cur["freeze_sha256"] = h(json.dumps(cur, sort_keys=True).encode())
    TARGET.write_text(json.dumps(cur, indent=1, sort_keys=True))
    print("freeze", cur["freeze_sha256"][:16], "->", TARGET)
