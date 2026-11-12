"""Mechanism evidence for written actors (rev 2): targeted ablations vs pre-registered predictions.

For five strong written artifacts (GPT-6 Astra / GPT-6.1 Sol, W2 = weights_feedback) we zero the
channels/units that implement a component named in the writer's own explanation, or switch to the
`t8` operator variant for iteration-dependent claims, and evaluate on S-start and a fixed Q-lvl
subsample. Each ablation is paired with a control that zeroes an equally sized random set of
*unused* channels (no incoming or no outgoing weights) from the same layer, or from the nearest layer
that has enough unused channels when the component's own layer is fully used.

The predictions (table ARTIFACTS, from the private module) were fixed before any ablation was run
(intact eval.json results were known).

Usage: mechanism_rev2.py [--qlvl-n 120] [--only ztl,lys] [--out runs/analysis_rev2/mechanism.json]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from wsp import actor as A
from wsp import private
from wsp.envs import run_episodes, summarize
from wsp.manifest import RUNS as RUNS_ROOT
from wsp.queries import load_sets

RUNS = RUNS_ROOT / "rev2_private"
SEED = 20261105

# --------------------------------------------------------------------------- pre-registered design
# Prediction keys (absolute success rates of the ablated actor):
#   Q / S          : [lo, hi] range for Q-lvl (subsample) / S-start
#   Q_unpert       : [lo, hi] for unperturbed Q-lvl starts (prefix == ())
#   Q_pert         : [lo, hi] for perturbed Q-lvl starts
#   same_as        : component id whose per-episode outcomes should be identical (fraction >= 0.95)
#   near           : (component id, tol) -- |Q - Q(other)| <= tol
# A prediction is confirmed iff every listed condition holds. Controls are predicted |dQ|,|dS| <= 0.03.
# Per family: {"dir": run dir under runs/rev2_private, "stratum": ..., "components": [{"id", "name",
# "claim", "prediction", "pred", and "ablate" ({layer: [channels]}) or "variant"}, ...]}. The names,
# claims and predictions describe the private games' mechanics, so the table is read from the
# private module (wsp.private: MECHANISM_ARTIFACTS) and is empty without it.
ARTIFACTS: dict = private.get("MECHANISM_ARTIFACTS", {})

# --------------------------------------------------------------------------- ablation machinery
def _in_out(p: dict, layer: str, c: int):
    """(incoming, outgoing) slices for channel/unit c of `layer` as lists of (tensor, index)."""
    if layer == "F1":
        return [("W1", np.s_[c]), ("b1", np.s_[c])], [("W2", np.s_[:, c])]
    if layer == "F2":
        return ([("W2", np.s_[c]), ("b2", np.s_[c])],
                [("W3", np.s_[:, c]), ("W3", np.s_[:, 32 + c]), ("W3", np.s_[:, 64 + c])])
    if layer == "F3":
        return ([("W3", np.s_[c]), ("b3", np.s_[c])],
                [("Uh", np.s_[:, c]), ("Wz", np.s_[:, 32 + c]), ("Wz", np.s_[:, 64 + c]), ("Wc", np.s_[0, c])])
    if layer == "H":
        return ([("Wh", np.s_[c]), ("Uh", np.s_[c]), ("bh", np.s_[c])],
                [("Wh", np.s_[:, c]), ("Wz", np.s_[:, c]), ("Wz", np.s_[:, 16 + c]), ("Wc", np.s_[0, 32 + c])])
    if layer == "z":
        return [("Wz", np.s_[c]), ("bz", np.s_[c])], [("Wmz", np.s_[:, c]), ("Wo", np.s_[:, c])]
    if layer == "m":
        return ([("Wm", np.s_[c]), ("Wmz", np.s_[c]), ("Wma", np.s_[c]), ("bm", np.s_[c])],
                [("Wm", np.s_[:, c]), ("Wo", np.s_[:, 128 + c])])
    if layer == "Wc":
        return [("Wc", np.s_[0, c])], []
    raise KeyError(layer)


LAYER_SIZE = {"F1": 32, "F2": 32, "F3": 32, "H": 16, "z": 128, "m": 64, "Wc": 48}
FALLBACK = {"H": ["F3", "F2", "F1"], "m": ["z"], "z": ["m"], "F3": ["F2", "F1"], "F2": ["F3", "F1"],
            "Wc": []}


def used(p: dict, layer: str, c: int) -> bool:
    ins, outs = _in_out(p, layer, c)
    if layer == "Wc":
        return bool(np.abs(p["Wc"][0, c]) > 0)
    if layer == "H":   # self-loops do not count as outgoing use
        out = sum(float(np.abs(p[t][s]).sum()) for t, s in outs[1:]) + \
            float(np.abs(np.delete(p["Wh"][:, c], c, axis=0)).sum())
    elif layer == "m":
        out = float(np.abs(np.delete(p["Wm"][:, c], c)).sum()) + float(np.abs(p["Wo"][:, 128 + c]).sum())
    else:
        out = sum(float(np.abs(p[t][s]).sum()) for t, s in outs)
    inn = sum(float(np.abs(p[t][s]).sum()) for t, s in ins)
    return inn > 0 and out > 0


def ablate(p: dict, spec: dict) -> dict:
    q = {k: v.copy() for k, v in p.items()}
    for layer, idx in spec.items():
        for c in idx:
            ins, outs = _in_out(q, layer, c)
            for t, s in ins + outs:
                q[t][s] = 0.0
    return q


def control_for(p: dict, spec: dict, rng: random.Random) -> dict:
    """Equally sized random set of unused channels: same layer, else the nearest fallback layer."""
    ctrl = {}
    for layer, idx in spec.items():
        need = len(idx)
        for cand in [layer] + FALLBACK[layer]:
            if need == 0:
                break
            pool = [c for c in range(LAYER_SIZE[cand]) if not used(p, cand, c)
                    and not (cand == layer and c in idx) and c not in ctrl.get(cand, [])]
            take = sorted(rng.sample(pool, min(need, len(pool))))
            if take:
                ctrl.setdefault(cand, []).extend(take)
            need -= len(take)
    return ctrl


def sparsity(p: dict, variant_T: int = 128) -> dict:
    per = {k: round(float((v != 0).mean()), 5) for k, v in p.items()}
    nnz = int(sum((v != 0).sum() for v in p.values()))
    active = {L: [c for c in range(LAYER_SIZE[L]) if used(p, L, c)] for L in ("F1", "F2", "F3", "H", "z", "m")}
    return {"nonzero_frac": per, "nonzero_total": nnz, "nonzero_frac_total": round(nnz / A.N_PARAMS, 5),
            "active_counts": {L: len(v) for L, v in active.items()}, "active_H": active["H"],
            "active_m": active["m"]}


def functional_H(p: dict, specs, device: str) -> list[int]:
    """Tied-block channels with nonzero activation on any S-start initial frame."""
    from wsp.envs import build_start, obs_of
    net = A.Actor(p).to(device).eval()
    obs = torch.as_tensor(np.stack([obs_of(build_start(s)) for s in specs]), device=device)
    with torch.no_grad():
        _, h = net.encode(obs)
    mx = h.abs().amax((0, 2, 3)).cpu().numpy()
    return [int(c) for c in range(16) if mx[c] > 0]


# --------------------------------------------------------------------------- evaluation
def run(p: dict, specs, device: str, variant: str = "full") -> list:
    pol = A.NeuralPolicy(A.Actor(p, variant=variant), device=device)
    return run_episodes(specs, pol, batch=64, keep_trace=False)


def stats(res, specs) -> dict:
    s = summarize(res)
    unp = [r.success for r, sp in zip(res, specs) if not sp.prefix]
    per = [r.success for r, sp in zip(res, specs) if sp.prefix]
    lv = {}
    for r, sp in zip(res, specs):
        lv.setdefault(str(sp.level), []).append(r.success)
    return {"n": s["n"], "solve": round(s["solve_rate"], 4), "noop_rate": round(s["noop_rate"], 4),
            "invalid_rate": round(s["invalid_rate"], 4), "resets_per_ep": round(s["resets"] / s["n"], 3),
            "lost_attempts_per_ep": round(float(np.mean([r.attempts_lost for r in res])), 3),
            "solved_after_lost_attempt": round(float(np.mean([r.success and r.attempts_lost > 0 for r in res])), 4),
            "unpert": round(float(np.mean(unp)), 4) if unp else None, "n_unpert": len(unp),
            "pert": round(float(np.mean(per)), 4) if per else None, "n_pert": len(per),
            "by_level": {k: round(float(np.mean(v)), 3) for k, v in sorted(lv.items(), key=lambda x: int(x[0]))},
            "success": [int(r.success) for r in res]}


def check(pred: dict, ab: dict, intact: dict, others: dict) -> tuple[bool, list]:
    ok, notes = True, []

    def rng_(key, val):
        nonlocal ok
        lo, hi = pred[key]
        good = val is not None and lo - 1e-9 <= val <= hi + 1e-9
        ok &= good
        notes.append(f"{key}={val} in [{lo},{hi}]: {good}")

    if "Q" in pred:
        rng_("Q", ab["Q"]["solve"])
    if "S" in pred:
        rng_("S", ab["S"]["solve"])
    if "Q_unpert" in pred:
        rng_("Q_unpert", ab["Q"]["unpert"])
    if "Q_pert" in pred:
        rng_("Q_pert", ab["Q"]["pert"])
    if "Q_drop_max" in pred:
        good = intact["Q"]["solve"] - ab["Q"]["solve"] <= pred["Q_drop_max"] + 1e-9
        ok &= good
        notes.append(f"Q drop {intact['Q']['solve'] - ab['Q']['solve']:.3f} <= {pred['Q_drop_max']}: {good}")
    if "Q_ratio_min" in pred:
        r = ab["Q"]["solve"] / max(1e-9, intact["Q"]["solve"])
        good = r >= pred["Q_ratio_min"] - 1e-9
        ok &= good
        notes.append(f"Q ratio {r:.3f} >= {pred['Q_ratio_min']}: {good}")
    if "same_as" in pred:
        o = others[pred["same_as"]]
        a = np.array(ab["Q"]["success"] + ab["S"]["success"])
        b = np.array(o["Q"]["success"] + o["S"]["success"])
        agree = float((a == b).mean())
        good = agree >= 0.95
        ok &= good
        notes.append(f"episode agreement with {pred['same_as']} = {agree:.3f} >= 0.95: {good}")
    if "near" in pred:
        oid, tol = pred["near"]
        d = abs(ab["Q"]["solve"] - others[oid]["Q"]["solve"])
        good = d <= tol + 1e-9
        ok &= good
        notes.append(f"|Q - Q({oid})| = {d:.3f} <= {tol}: {good}")
    return bool(ok), notes


def eval_json_check(d: Path, specs_by_set: dict, intact: dict) -> dict:
    """Agreement of our intact re-evaluation with the recorded eval.json per-episode outcomes."""
    ev = json.loads((d / "eval.json").read_text())
    out = {}
    for s, key in (("S-start", "S"), ("Q-lvl", "Q")):
        rec = {e[0]: e[1] for e in ev["episodes"][s]}
        ours = intact[key]["success"]
        keys = [sp.key for sp in specs_by_set[key]]
        agree = [int(rec[k] == o) for k, o in zip(keys, ours) if k in rec]
        out[s] = {"recorded_full_set": round(ev["sets"][s]["solve_rate"], 4),
                  "recorded_on_subsample": round(float(np.mean([rec[k] for k in keys if k in rec])), 4),
                  "episode_agreement": round(float(np.mean(agree)), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlvl-n", type=int, default=120)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(RUNS_ROOT / "analysis_rev2" / "mechanism.json"))
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if not ARTIFACTS:
        raise SystemExit("no mechanism artifacts configured (private; set WSP_PRIVATE_REDACTION, see README)")
    names = [n for n in ARTIFACTS if not a.only or n in a.only.split(",")]
    out_path = Path(a.out)
    result = json.loads(out_path.read_text()) if out_path.exists() else {"artifacts": {}}
    result.update({"seed": SEED, "qlvl_n": a.qlvl_n, "device": device,
                   "note": "predictions fixed in scripts/mechanism_rev2.py before ablations were run"})
    for fam in names:
        cfg = ARTIFACTS[fam]
        t0 = time.time()
        d = RUNS / cfg["dir"]
        rec = json.loads((d / "session.json").read_text())
        p = A.load_npz(d / f"round{rec['final_round']}.npz")
        qs = load_sets(fam, rev=2)
        rng = random.Random(f"{SEED}-{fam}")
        ql = qs["Q-lvl"]
        idx = sorted(rng.sample(range(len(ql)), min(a.qlvl_n, len(ql))))
        specs = {"S": qs["S-start"], "Q": [ql[i] for i in idx]}

        def ev(params, variant="full"):
            return {k: stats(run(params, sp, device, variant), sp) for k, sp in specs.items()}

        intact = ev(p)
        art = {"dir": cfg["dir"], "stratum": cfg["stratum"], "model": cfg["dir"].split("/")[1],
               "final_round": rec["final_round"], "intact": intact,
               "eval_json_check": eval_json_check(d, specs, intact),
               "sparsity": sparsity(p), "functional_H_on_S_start": functional_H(p, qs["S-start"], device),
               "components": []}
        done = {}
        crng = random.Random(f"{SEED}-{fam}-control")
        for comp in cfg["components"]:
            row = {k: comp[k] for k in ("id", "name", "claim", "prediction", "pred")}
            if "variant" in comp:
                row["ablation"] = {"variant": comp["variant"]}
                ab = ev(p, comp["variant"])
                row["control"] = None
            else:
                row["ablation"] = comp["ablate"]
                ab = ev(ablate(p, comp["ablate"]))
                ctrl = control_for(p, comp["ablate"], crng)
                cres = ev(ablate(p, ctrl))
                row["control"] = {"channels": ctrl,
                                  "size": sum(len(v) for v in ctrl.values()),
                                  "Q": {k: v for k, v in cres["Q"].items() if k != "success"},
                                  "S": {k: v for k, v in cres["S"].items() if k != "success"},
                                  "dQ": round(cres["Q"]["solve"] - intact["Q"]["solve"], 4),
                                  "dS": round(cres["S"]["solve"] - intact["S"]["solve"], 4)}
                row["control"]["null"] = abs(row["control"]["dQ"]) <= 0.03 and abs(row["control"]["dS"]) <= 0.03
            done[comp["id"]] = ab
            row["confirmed"], row["checks"] = check(comp["pred"], ab, intact, done)
            row["result"] = ab
            row["dQ"] = round(ab["Q"]["solve"] - intact["Q"]["solve"], 4)
            row["dS"] = round(ab["S"]["solve"] - intact["S"]["solve"], 4)
            art["components"].append(row)
            print(f"{comp['id']:16s} Q {intact['Q']['solve']:.3f}->{ab['Q']['solve']:.3f} "
                  f"S {intact['S']['solve']:.3f}->{ab['S']['solve']:.3f} "
                  f"ctrl {row['control'] and (row['control']['dQ'], row['control']['dS'])} "
                  f"confirmed={row['confirmed']}", flush=True)
        art["wall_s"] = round(time.time() - t0, 1)
        result["artifacts"][fam] = art
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=1))
        print(fam, "done", art["wall_s"], "s", art["eval_json_check"], flush=True)


if __name__ == "__main__":
    sys.exit(main())
