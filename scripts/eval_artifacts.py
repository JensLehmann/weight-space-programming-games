"""Evaluate final artifacts of writer sessions (or trained baselines) on the frozen query sets.

Usage: eval_artifacts.py <root> [--sets S-start,Q-lvl,...] [--gpus N] [--py-workers N]
                         [--device cpu] [--batch 1] [--out DIR]
Finds every */session.json (writer) or */student.npz (baseline) below <root> lacking eval.json;
also */student.graph.json (learned free-architecture baseline, wsp.train_bc.save_fa_student).
Bounded program players (condition pbounded, wsp.pbounded) run on CPU like Python players: NumPy float32,
decoded as FA networks.
Released public artifacts (artifacts/public/<game>/<writer>/{w2,p2,fa_w2}_r<k>.py; <root> may be one such
file or a directory above them) are rebuilt with wsp.session.materialize and evaluated the same way; their
eval.json goes to <out>/<game>/<writer>/<file stem>/ and also holds q_lvl_level_balanced.
--device and --batch (episodes per batch of W2 and FA networks) default to the study's cuda-if-available and 64.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

SETS = ["S-start", "Q-lvl", "Q-tpl", "Q-off", "Q-pal"]
RELEASED = re.compile(r"(w2|p2|fa_w2)_r\d+\.py")                       # artifacts/public/<game>/<writer>/
RELEASED_CONDITION = {"w2": "weights", "p2": "python", "fa_w2": "freearch"}


def released_condition(path: Path) -> str | None:
    m = RELEASED.fullmatch(path.name)
    return RELEASED_CONDITION[m.group(1)] if m else None


PRIVATE = {"ztl", "lys", "esm", "toy_reach", "wih", "qcn", "aiv", "nkp", "obn", "qwv", "ruz",
         "vvf", "ubr", "yuz", "bvn", "trx", "yiy", "crg", "qpv", "ydv", "mds"}


def family_of(path: Path) -> str:
    for part in path.parts:
        if part in PRIVATE or part.startswith("official."):
            return part
    raise ValueError(path)


def released_out(path: Path, out_dir: str) -> Path:
    """<out>/<game>/<writer>/<stem> for artifacts/public/<game>/<writer>/<stem>.py."""
    return Path(out_dir) / path.parent.parent.name / path.parent.name / path.stem


def rev_of(path: Path, rec: dict | None) -> int:
    """Query-manifest split version of a run: the session's split_version, else 2 (the only one)."""
    if rec and rec.get("split_version"):
        return int(rec["split_version"])
    return 2


def work(args):
    d, gpu, sets, device, batch, out_dir = args
    d = Path(d)
    try:
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        import torch
        # exact FP32 semantics (TF32 convolutions flip argmax decisions of high-gain written actors)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        from wsp import actor as A
        from wsp.envs import summarize
        from wsp.queries import load_sets
        from wsp.session import Artifact, evaluate, load_final, materialize
        fam = family_of(d)
        rec = json.loads((d / "session.json").read_text()) if (d / "session.json").exists() else None
        qs = load_sets(fam, rev=rev_of(d, rec))
        released = d.is_file()
        if released:                                 # released public artifact: rebuild it from the program
            cond = released_condition(d)
            art = materialize(cond, d.read_text())
        elif rec is not None:
            art = load_final(d, rec["condition"])
            cond = rec["condition"]
        elif (d / "student.graph.json").exists():    # learned FA student: student.graph.json + student_fa.npz
            from wsp.train_bc import load_fa_student
            arch, params = load_fa_student(d)
            art = Artifact("freearch", "", params=params, graph=arch)
            cond = "freearch"
        else:
            meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
            art = Artifact("weights", "", params=A.load_npz(d / "student.npz"), variant=meta.get("variant", "full"))
            cond = "weights"
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        out = {"dir": str(d), "family": fam, "condition": cond, "sets": {}, "episodes": {}}
        t0 = time.time()
        for s in sets:
            specs = qs.get(s, [])
            if not specs:
                continue
            if art is None:
                out["sets"][s] = {"n": len(specs), "solve_rate": 0.0, "no_artifact": True}
                out["episodes"][s] = [[sp.key, 0, 0, 0, 0] for sp in specs]
                continue
            res = evaluate(art, specs, device=device, batch=batch)
            out["sets"][s] = summarize(res)
            out["episodes"][s] = [[r.key, int(r.success), r.actions, r.resets, r.invalid] for r in res]
        out["eval_wall_s"] = time.time() - t0
        dst = d
        if released:
            out.update(sha256=hashlib.sha256(d.read_bytes()).hexdigest(), device=device, batch=batch,
                       materialize_error=art.error)
            if "Q-lvl" in out["episodes"]:             # level-balanced Q-lvl, as in data/sessions.csv
                level = {sp.key: sp.level for sp in qs["Q-lvl"]}
                per = defaultdict(list)
                for e in out["episodes"]["Q-lvl"]:
                    per[level[e[0]]].append(e[1])
                out["q_lvl_level_balanced"] = sum(sum(v) / len(v) for v in per.values()) / len(per)
            dst = released_out(d, out_dir)
            dst.mkdir(parents=True, exist_ok=True)
        (dst / "eval.json").write_text(json.dumps(out))
        res = {k: round(v.get("solve_rate", 0), 3) for k, v in out["sets"].items()}
        if "q_lvl_level_balanced" in out:
            res["q_lvl_level_balanced"] = round(out["q_lvl_level_balanced"], 6)
        return str(d), res
    except Exception:
        return str(d), "ERROR " + traceback.format_exc()[-800:]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--py-workers", type=int, default=16)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--device", default=None, help="torch device of W2 and FA networks (default: cuda if available, else cpu)")
    ap.add_argument("--batch", type=int, default=64, help="episodes per batch of W2 and FA networks")
    ap.add_argument("--out", default=None, help="output directory for released artifacts")
    a = ap.parse_args()
    sets = a.sets.split(",")
    root = Path(a.root)
    files = [root] if root.is_file() else sorted(root.rglob("*.py"))
    files = [p for p in files if released_condition(p) and any(x.startswith("official.") for x in p.parts)]
    if files and not a.out:
        ap.error("released artifacts need --out DIR")
    dirs = sorted({p.parent for p in root.rglob("session.json")} |
                  {p.parent for p in root.rglob("student.npz")} |
                  {p.parent for p in root.rglob("student.graph.json")}) if root.is_dir() else []
    dirs = [d for d in dirs if a.redo or not (d / "eval.json").exists()]
    dirs += [p for p in files if a.redo or not (released_out(p, a.out) / "eval.json").exists()]
    weights, python = [], []
    for d in dirs:
        cond = (released_condition(d) if d.is_file() else
                json.loads((d / "session.json").read_text())["condition"] if (d / "session.json").exists() else "weights")
        (python if cond in ("python", "pyreact", "pbounded") else weights).append(d)
    print(f"{len(weights)} weight artifacts, {len(python)} python artifacts", flush=True)
    ctx = mp.get_context("spawn")
    opts = (a.device, a.batch, a.out)
    jobs = [(str(d), (i % a.gpus) if a.gpus else None, sets, *opts) for i, d in enumerate(weights)]
    jobs += [(str(d), None, sets, *opts) for d in python]
    nproc = max(1, (a.gpus * 2 if a.gpus else 2)) + a.py_workers
    with ctx.Pool(min(nproc, max(1, len(jobs)))) as pool:
        for d, res in pool.imap_unordered(work, jobs):
            print(json.dumps({"dir": d, "res": res}), flush=True)
