"""Perturbation control for the RL result: does PPO erase written circuits, or does any
small update? No writer (model) calls.

Actors: the final W2 fixed-actor weights (run 0) of GPT-6 Astra, GPT-6.1 Sol and GPT-6.1 Terra on the six
ablation families (ztl lys esm ruz crg mds), i.e. the 18 initialisations of Stage 2.

  noise    For every tensor k: theta_k' = theta_k + eps * rms(theta_k) * z_k, z_k ~ N(0, I), rms over all
           entries (all-zero tensors stay zero), so ||theta_k' - theta_k|| / ||theta_k|| ~= eps per tensor and
           overall. eps in {1e-4, 1e-3, 1e-2, 1e-1} as planned, extended by {1e-7, 1e-6, 1e-5} because the
           actors already collapse at 1e-4 (float32 resolution is ~6e-8 relative); noise seeds {0, 1}; within a seed the direction z is the
           same for every eps. Exact-FP32 greedy evaluation of S-start (full set) and Q-lvl on a fixed
           subsample (QLVL_PER_ACTOR episodes, the same number per held-out level, uniform within level,
           seed 0; the same subsample for every eps, writer and seed); Q-lvl is scored level-balanced as in
           analyze_fa_all.score. eps = 0 is evaluated on the subsample and on the full sets; the full-set run
           must equal the actor's eval.json episode by episode.
  ppo      The Stage-2 runs saved only init.npz and the final student.npz (303k steps); the weights at the
           greedy checkpoints (16k, 41k, 82k, ...) were evaluated but not saved. They are reproduced with
           the Stage-2 code (wsp.train_ppo.ppo) and the options in meta.json of the seed-0 sparse-reward
           run, from its init.npz, for exactly the steps up to the checkpoint: standard PPO to its first
           checkpoint (16,384 steps, 2 PPO updates; all 18 actors); structure-preserving PPO to its first
           checkpoint after the critic warm-up (81,920 steps, 10 updates; Astra on ztl, lys, esm; its actor
           is not updated before 50k steps, so it is unchanged at 16k and 41k). SGD (lr 1e-5) is not
           re-run: gradient clipping (norm 0.5, no momentum) bounds its update at 16k exactly. Differences from the original runs: device (MPS instead of CUDA, so
           a different random stream; same seed), exact FP32 instead of TF32, gradient checkpointing of the
           tied block (memory only), no training-time evaluation (it does not touch the training RNG).
  evalckpt Exact evaluation of every reproduced checkpoint (same sets as the noise) and of a size-matched
           random control: theta_0 + a random direction with the support (changed entries) and per-tensor
           norm of that checkpoint's update, 2 seeds. It separates "any update this small erases the
           circuits" (control collapses too) from "the RL direction does" (control keeps the init's score).
  finals   The same size-matched random control for the saved final students (303k steps) of the seed-0
           sparse runs of SGD and structure-preserving PPO (no training needed).
  analyze  Relative update ||theta - theta_0|| / ||theta_0|| per tensor and overall (the parametrisation of
           the noise), for the reproduced checkpoints and for all saved final students; exact evaluation of
           the reproduced checkpoints on the same sets as the noise; family-bootstrap CIs. Writes
           runs/analysis_rev2/perturbation.json and runs/analysis_rev2/tables/perturbation.tex.

Usage: perturbation_rev2.py ppo | noise | evalckpt | finals [--workers 3] [--device mps], then analyze (mps or cpu;
CPU FP32 differs from the CUDA reference on some episodes, MPS reproduces it, see the eps = 0 check).
"""
import argparse
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs"
WORK = R / "rev2_perturbation"
FAMS = ["ztl", "lys", "esm", "ruz", "crg", "mds"]
WRITERS = ["astra-6", "sol-6.1", "terra-6.1"]
EPS_PLANNED = [1e-4, 1e-3, 1e-2, 1e-1]
EPS = [1e-7, 1e-6, 1e-5] + EPS_PLANNED       # extended downwards: every actor already collapses at 1e-4
NOISE_SEEDS = [0, 1]
QLVL_PER_ACTOR = 60
# reproduced checkpoints: variant -> (steps, writers, families). Cost bound: one PPO update of a written
# actor backpropagates through the 128-step tied block, so standard PPO is
# reproduced for all 18 actors and structure-preserving PPO (10 updates) for three Astra actors; SGD is
# covered by the exact clipping bound at 16k and by its saved final students (see analyze).
PPO_JOBS = {"std": (16384, WRITERS, FAMS), "sp": (81920, ["astra-6"], ["ztl", "lys", "esm"])}
LABEL = {"std": "Standard PPO", "sp": "Structure-preserving PPO", "sgd": "SGD, lr $10^{-5}$"}


def actor_dir(fam, w):
    return R / "rev2_private" / fam / w / "weights_feedback_M_full_r0"


def stage2_dir(fam, w, variant, reward="sparse", seed=0):
    return R / "rev2_stage2" / fam / f"{w}_{variant}_{reward}_s{seed}"


def setup_torch(threads=2):
    import torch
    from wsp.freearch import exact_fp32
    exact_fp32()                                     # no TF32 (CUDA); MPS and CPU compute in FP32
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(threads)


def load_actor(fam, w):
    from wsp import actor as A
    d = actor_dir(fam, w)
    rec = json.loads((d / "session.json").read_text())
    return A.load_npz(d / f"round{rec['final_round']}.npz")


# --------------------------------------------------------------------------- sets and scoring
def eval_sets(fam):
    """{'S-start': full, 'Q-lvl': fixed level-stratified subsample, 'Q-lvl-full': full}."""
    from wsp.queries import load_sets
    qs = load_sets(fam, rev=2)
    by_level = defaultdict(list)
    for s in qs["Q-lvl"]:
        by_level[s.level].append(s)
    per = QLVL_PER_ACTOR // len(by_level)
    rng = np.random.default_rng([0, zlib.crc32(fam.encode())])
    sub = []
    for lv in sorted(by_level):
        idx = sorted(rng.choice(len(by_level[lv]), min(per, len(by_level[lv])), replace=False))
        sub += [by_level[lv][i] for i in idx]
    return {"S-start": qs["S-start"], "Q-lvl": sub, "Q-lvl-full": qs["Q-lvl"]}


_LV = {}


def levels(fam):
    if fam not in _LV:
        from wsp.queries import load_sets
        _LV[fam] = {s.key: s.level for s in load_sets(fam, rev=2)["Q-lvl"]}
    return _LV[fam]


def qlvl_balanced(fam, episodes):
    """Mean over held-out levels of per-level success (analyze_fa_all.score)."""
    lv, per = levels(fam), defaultdict(list)
    for e in episodes:
        per[lv[e[0]]].append(e[1])
    return float(np.mean([np.mean(v) for v in per.values()])) if per else None


def rate(episodes):
    return float(np.mean([e[1] for e in episodes])) if episodes else None


def run_eval(params, sets, device):
    """Exact-FP32 greedy episodes [[key, success, actions, resets, invalid], ...] per set (session.evaluate)."""
    from wsp import actor as A
    from wsp.envs import run_episodes
    if params is None or not all(np.isfinite(v).all() for v in params.values()):
        return {s: [[sp.key, 0, 0, 0, 0] for sp in specs] for s, specs in sets.items()}, True
    out = {}
    for s, specs in sets.items():
        pol = A.NeuralPolicy(A.Actor(params), device)
        res = run_episodes(specs, pol, batch=64, keep_trace=False)
        out[s] = [[r.key, int(r.success), r.actions, r.resets, r.invalid] for r in res]
    return out, False


# --------------------------------------------------------------------------- perturbation
def noise_direction(params, seed, tag):
    rng = np.random.default_rng([seed, zlib.crc32(tag.encode())])
    return {k: rng.standard_normal(params[k].shape) for k in sorted(params)}


def perturb(params, eps, z):
    out = {}
    for k, v in params.items():
        v64 = v.astype(np.float64)
        rms = float(np.sqrt(np.mean(v64 ** 2)))
        out[k] = (v64 + eps * rms * z[k]).astype(np.float32)
    return out


def rel_change(p0, p1):
    """Per-tensor and overall ||p1 - p0|| / ||p0|| (float64), plus structural counts."""
    per, num, den = {}, 0.0, 0.0
    changed = zero = zero_changed = n = 0
    finite = all(np.isfinite(p1[k]).all() for k in p0)
    for k in sorted(p0):
        a, b = p0[k].astype(np.float64), p1[k].astype(np.float64)
        d = float(np.linalg.norm(b - a)) if finite else math.inf
        n0 = float(np.linalg.norm(a))
        per[k] = {"abs": d, "rel": (d / n0) if n0 > 0 else (0.0 if d == 0 else math.inf), "init_norm": n0}
        num += d * d
        den += n0 * n0
        if finite:
            ch = b != a
            changed += int(ch.sum())
            zero += int((a == 0).sum())
            zero_changed += int((ch & (a == 0)).sum())
        n += a.size
    rel_nz = [v["rel"] for v in per.values() if v["init_norm"] > 0]
    return {"overall": math.sqrt(num / den) if finite else math.inf, "finite": finite,
            "max_tensor": max(rel_nz) if finite else math.inf,
            "max_tensor_name": max((k for k in per if per[k]["init_norm"] > 0), key=lambda k: per[k]["rel"]),
            "median_tensor": float(np.median(rel_nz)) if finite else math.inf,
            "zero_tensors_changed": [k for k, v in per.items() if v["init_norm"] == 0 and v["abs"] > 0],
            "frac_entries_changed": changed / n if finite else None,
            "frac_zero_entries_changed": zero_changed / max(1, zero) if finite else None,
            "per_tensor": per}


def noise_job(args):
    fam, w, eps, seed, which, device = args
    out = WORK / "noise" / fam / w / (f"eps0_{which}.json" if eps == 0 else f"eps{eps:g}_s{seed}.json")
    if out.exists():
        return str(out), "cached"
    try:
        setup_torch()
        t0 = time.time()
        p0 = load_actor(fam, w)
        sets = eval_sets(fam)
        if eps == 0:
            p, size = p0, None
            sets = {"S-start": sets["S-start"], "Q-lvl": sets["Q-lvl-full" if which == "full" else "Q-lvl"]}
        else:
            p = perturb(p0, eps, noise_direction(p0, seed, f"{fam}/{w}"))
            size = {k: v for k, v in rel_change(p0, p).items() if k != "per_tensor"}
            sets = {"S-start": sets["S-start"], "Q-lvl": sets["Q-lvl"]}
        eps_res, nonfinite = run_eval(p, sets, device)
        rec = {"family": fam, "writer": w, "eps": eps, "seed": seed, "which": which, "device": device,
               "episodes": eps_res, "nonfinite_params": nonfinite, "size": size, "wall_s": time.time() - t0}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec))
        return str(out), {"S-start": rate(eps_res["S-start"]), "Q-lvl": qlvl_balanced(fam, eps_res["Q-lvl"]),
                          "wall_s": round(rec["wall_s"])}
    except Exception:
        return str(out), "ERROR " + traceback.format_exc()[-1500:]


# --------------------------------------------------------------------------- PPO checkpoint reproduction
def ppo_job(args):
    fam, w, variant, steps, device = args
    out = WORK / "ppo" / fam / f"{w}_{variant}_sparse_s0_{steps}"
    if (out / "ckpt.npz").exists():
        return str(out), "cached"
    try:
        setup_torch(threads=4)
        from wsp import actor as A
        from wsp.queries import load_sets
        from wsp import train_ppo
        src = stage2_dir(fam, w, variant)
        meta = json.loads((src / "meta.json").read_text())
        kw = dict(meta["ppo_options"])
        for k in ("eval_at", "eval_every"):
            kw.pop(k, None)
        kw["grad_ckpt"] = 16                               # memory only (torch.utils.checkpoint)
        init = A.load_npz(src / "init.npz")
        sets = load_sets(fam, rev=2)
        out.mkdir(parents=True, exist_ok=True)
        saved = {}

        def save_raw(params, path):                         # keep non-finite (diverged) weights too
            saved["params"] = {k: np.asarray(v, np.float32) for k, v in params.items()}
            np.savez_compressed(path, **saved["params"])
        train_ppo.A.save_npz = save_raw
        t0 = time.time()
        res = train_ppo.ppo(sets["S-start"] + sets["S-dev"], out, seed=meta["seed"], total_steps=steps,
                            device=device, init=init, eval_specs=None, **kw)
        np.savez_compressed(out / "ckpt.npz", **{k: np.asarray(v, np.float32) for k, v in res["params"].items()})
        (out / "repro_meta.json").write_text(json.dumps({
            "source": str(src.relative_to(ROOT)), "family": fam, "writer": w, "variant": variant, "steps": steps,
            "device": device, "options": kw, "wall_s": time.time() - t0, "log": res["log"]}))
        for f in ("student.npz", "student_latest.npz"):
            (out / f).unlink(missing_ok=True)
        return str(out), {"wall_s": round(time.time() - t0)}
    except Exception:
        return str(out), "ERROR " + traceback.format_exc()[-1500:]


def matched_update(p0, p1, seed, tag):
    """Random update with the support and per-tensor norm of the RL update p1 - p0 (random direction)."""
    rng = np.random.default_rng([seed, zlib.crc32(tag.encode())])
    out = {}
    for k in sorted(p0):
        a = p0[k].astype(np.float64)
        dl = p1[k].astype(np.float64) - a
        supp = dl != 0
        z = np.zeros_like(a)
        z[supp] = rng.standard_normal(int(supp.sum()))
        nzz = float(np.linalg.norm(z))
        out[k] = (a + (float(np.linalg.norm(dl)) / nzz) * z if nzz > 0 else a).astype(np.float32)
    return out


def evalckpt_job(args):
    """kind 'ckpt': exact evaluation of a reproduced checkpoint; 'matched': of a random update with the
    same support and per-tensor norms as that checkpoint's update (seed)."""
    path, kind, seed, device = args
    d = Path(path)
    out = d / ("eval_exact.json" if kind == "ckpt" else f"matched_s{seed}.json")
    if out.exists():
        return str(out), "cached"
    try:
        setup_torch()
        from wsp import actor as A
        m = json.loads((d / "repro_meta.json").read_text())
        with np.load(d / "ckpt.npz") as z:
            p = {k: z[k] for k in z.files}
        size = None
        if kind == "matched":
            p0 = A.load_npz(ROOT / m["source"] / "init.npz")
            if not all(np.isfinite(v).all() for v in p.values()):
                out.write_text(json.dumps({"skipped": "non-finite checkpoint"}))
                return str(out), "skipped"
            p = matched_update(p0, p, seed, f"{m['family']}/{m['writer']}/{m['variant']}")
            size = {k: v for k, v in rel_change(p0, p).items() if k != "per_tensor"}
        sets = eval_sets(m["family"])
        res, nonfinite = run_eval(p, {"S-start": sets["S-start"], "Q-lvl": sets["Q-lvl"]}, device)
        out.write_text(json.dumps({"episodes": res, "nonfinite_params": nonfinite, "device": device, "size": size}))
        return str(out), {"S-start": rate(res["S-start"]), "Q-lvl": qlvl_balanced(m["family"], res["Q-lvl"])}
    except Exception:
        return str(out), "ERROR " + traceback.format_exc()[-1500:]


def final_matched_job(args):
    """Size-matched random control for a saved final student (303k steps) of a seed-0 sparse run."""
    fam, w, variant, seed, device = args
    out = WORK / "final_matched" / fam / f"{w}_{variant}_sparse_s0_m{seed}.json"
    if out.exists():
        return str(out), "cached"
    try:
        setup_torch()
        from wsp import actor as A
        src = stage2_dir(fam, w, variant)
        p0, p1 = A.load_npz(src / "init.npz"), A.load_npz(src / "student.npz")
        p = matched_update(p0, p1, seed, f"{fam}/{w}/{variant}/final")
        size = {k: v for k, v in rel_change(p0, p).items() if k != "per_tensor"}
        sets = eval_sets(fam)
        res, nonfinite = run_eval(p, {"S-start": sets["S-start"], "Q-lvl": sets["Q-lvl"]}, device)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"episodes": res, "nonfinite_params": nonfinite, "device": device, "size": size}))
        return str(out), {"S-start": rate(res["S-start"]), "Q-lvl": qlvl_balanced(fam, res["Q-lvl"])}
    except Exception:
        return str(out), "ERROR " + traceback.format_exc()[-1500:]


def run_pool(fn, jobs, workers):
    ctx = mp.get_context("spawn")
    with ctx.Pool(max(1, min(workers, len(jobs)))) as pool:
        for d, res in pool.imap_unordered(fn, jobs):
            print(json.dumps({"job": d, "res": res}), flush=True)


# --------------------------------------------------------------------------- analysis
def boot_family(vals, B=10000):
    """vals: {family: [per-actor values]}. Mean of family means and family-bootstrap 95% CI."""
    fams = sorted(vals)
    fm = np.array([np.mean(vals[f]) for f in fams], float)
    if not len(fm):
        return {"mean": None, "ci": [None, None], "n_families": 0}
    b = fm[np.random.default_rng(0).integers(0, len(fm), (B, len(fm)))].mean(1)
    return {"mean": float(fm.mean()), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "n_families": len(fm), "n_actors": int(sum(len(vals[f]) for f in fams))}


def logged(d, steps):
    """Greedy checkpoint success at `steps` in ppo_eval.json (raw solve rate over the full set)."""
    p = d / "ppo_eval.json"
    if not p.exists():
        return {}
    return {e["set"]: e["solve_rate"] for e in json.loads(p.read_text())
            if e["eval_steps"] == steps and e["decoding"] == "greedy"}


def nz(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


def analyze():
    from wsp import actor as A
    res = {"design": __doc__.split("\n\n")[1].strip() if __doc__ else "", "actors": {}, "noise": {}, "rl_first_checkpoint": {},
           "rl_final_saved": {}, "sanity": {}}
    # ---- noise
    actor_scores = defaultdict(dict)             # (eps) -> {(fam, w): {"Q-lvl", "S-start"}}
    mism = []
    for fam in FAMS:
        for w in WRITERS:
            nd = WORK / "noise" / fam / w
            ev = json.loads((actor_dir(fam, w) / "eval.json").read_text())
            full = json.loads((nd / "eps0_full.json").read_text())
            ref = {s: {e[0]: e[1:] for e in ev["episodes"][s]} for s in ("S-start", "Q-lvl")}
            bad = {s: sum(e[1:] != ref[s][e[0]] for e in full["episodes"][s]) for s in ("S-start", "Q-lvl")}
            bad_succ = {s: sum(e[1] != ref[s][e[0]][0] for e in full["episodes"][s]) for s in ("S-start", "Q-lvl")}
            sub = json.loads((nd / "eps0_sub.json").read_text())
            bad_sub = sum(e[1:] != ref["Q-lvl"][e[0]] for e in sub["episodes"]["Q-lvl"])
            mism.append({"family": fam, "writer": w, "full_episode_mismatch": bad, "full_success_mismatch": bad_succ,
                         "subsample_episode_mismatch": bad_sub,
                         "full_qlvl_balanced": qlvl_balanced(fam, full["episodes"]["Q-lvl"]),
                         "evaljson_qlvl_balanced": qlvl_balanced(fam, ev["episodes"]["Q-lvl"]),
                         "evaljson_qlvl_raw": ev["sets"]["Q-lvl"]["solve_rate"],
                         "evaljson_sstart": ev["sets"]["S-start"]["solve_rate"]})
            actor_scores[0.0][(fam, w)] = {"Q-lvl": qlvl_balanced(fam, sub["episodes"]["Q-lvl"]),
                                           "S-start": rate(sub["episodes"]["S-start"]),
                                           "Q-lvl-full": qlvl_balanced(fam, full["episodes"]["Q-lvl"])}
            for eps in EPS:
                rs = [json.loads((nd / f"eps{eps:g}_s{s}.json").read_text()) for s in NOISE_SEEDS]
                actor_scores[eps][(fam, w)] = {
                    "Q-lvl": float(np.mean([qlvl_balanced(fam, r["episodes"]["Q-lvl"]) for r in rs])),
                    "S-start": float(np.mean([rate(r["episodes"]["S-start"]) for r in rs])),
                    "size_overall": float(np.mean([r["size"]["overall"] for r in rs])),
                    "size_max_tensor": float(np.mean([r["size"]["max_tensor"] for r in rs])),
                    "by_seed": [{"Q-lvl": qlvl_balanced(fam, r["episodes"]["Q-lvl"]),
                                 "S-start": rate(r["episodes"]["S-start"])} for r in rs]}
            res["actors"][f"{fam}/{w}"] = {str(e): actor_scores[e][(fam, w)] for e in [0.0] + EPS}
    res["sanity"]["eps0_vs_evaljson"] = mism
    res["sanity"]["eps0_all_episodes_identical"] = all(sum(m["full_episode_mismatch"].values()) == 0 for m in mism)

    def pooled(scores, key, subset=None):
        vals = defaultdict(list)
        for (fam, w), v in scores.items():
            if subset is None or w in subset:
                vals[fam].append(v[key])
        return boot_family(vals)

    def paired(scores, base, key, subset=None):
        vals = defaultdict(list)
        for (fam, w), v in scores.items():
            if subset is None or w in subset:
                vals[fam].append(v[key] - base[(fam, w)][key])
        return boot_family(vals)

    for eps in [0.0] + EPS:
        sc = actor_scores[eps]
        res["noise"][str(eps)] = {"Q-lvl": pooled(sc, "Q-lvl"), "S-start": pooled(sc, "S-start"),
                                  "dQ-lvl": paired(sc, actor_scores[0.0], "Q-lvl"),
                                  "dS-start": paired(sc, actor_scores[0.0], "S-start"),
                                  "astra_Q-lvl": pooled(sc, "Q-lvl", ["astra-6"]),
                                  "astra_S-start": pooled(sc, "S-start", ["astra-6"])}
        if eps:
            res["noise"][str(eps)]["realised_rel_size_overall_mean"] = float(np.mean([v["size_overall"] for v in sc.values()]))
            res["noise"][str(eps)]["realised_rel_size_max_tensor_mean"] = float(np.mean([v["size_max_tensor"] for v in sc.values()]))
    res["noise"]["0.0"]["Q-lvl-full"] = pooled(actor_scores[0.0], "Q-lvl-full")

    # ---- reproduced first checkpoints
    rl_rows = {}
    for variant, (steps, writers, fams) in PPO_JOBS.items():
        rows = []
        for fam in fams:
            for w in writers:
                d = WORK / "ppo" / fam / f"{w}_{variant}_sparse_s0_{steps}"
                if not (d / "ckpt.npz").exists():
                    continue
                src = stage2_dir(fam, w, variant)
                p0 = A.load_npz(src / "init.npz")
                written = load_actor(fam, w)
                same_init = all(np.array_equal(p0[k], written[k]) for k in p0)
                with np.load(d / "ckpt.npz") as z:
                    p1 = {k: z[k] for k in z.files}
                size = rel_change(p0, p1)
                m = json.loads((d / "repro_meta.json").read_text())
                ev = json.loads((d / "eval_exact.json").read_text()) if (d / "eval_exact.json").exists() else None
                last = m["log"][-1] if m["log"] else {}
                exact_dir = R / "rev2_stage2_exact" / fam / f"{w}_{variant}_sparse_s0"
                mt = [json.loads((d / f"matched_s{s}.json").read_text()) for s in NOISE_SEEDS
                      if (d / f"matched_s{s}.json").exists()]
                mt = [x for x in mt if "episodes" in x]
                rows.append({"family": fam, "writer": w, "steps": steps, "init_is_written_actor": same_init,
                             "size": size, "kl_init": last.get("kl_init"), "margin": last.get("margin"),
                             "Q-lvl": qlvl_balanced(fam, ev["episodes"]["Q-lvl"]) if ev else None,
                             "S-start": rate(ev["episodes"]["S-start"]) if ev else None,
                             "matched_Q-lvl": _mean([qlvl_balanced(fam, x["episodes"]["Q-lvl"]) for x in mt]) if mt else None,
                             "matched_S-start": _mean([rate(x["episodes"]["S-start"]) for x in mt]) if mt else None,
                             "matched_rel_overall": _mean([x["size"]["overall"] for x in mt]) if mt else None,
                             "matched_n_seeds": len(mt),
                             "eps0_Q-lvl": actor_scores[0.0][(fam, w)]["Q-lvl"],
                             "eps0_S-start": actor_scores[0.0][(fam, w)]["S-start"],
                             "logged_at_ckpt_tf32_raw": logged(src, steps),
                             "logged_at_ckpt_exact_raw": logged(exact_dir, steps) if exact_dir.exists() else None,
                             "wall_s": m["wall_s"]})
        if not rows:
            continue
        sc = {(r["family"], r["writer"]): {"Q-lvl": r["Q-lvl"], "S-start": r["S-start"]} for r in rows if r["Q-lvl"] is not None}
        msc = {(r["family"], r["writer"]): {"Q-lvl": r["matched_Q-lvl"], "S-start": r["matched_S-start"]}
               for r in rows if r["matched_Q-lvl"] is not None}
        ov = [r["size"]["overall"] for r in rows]
        mx = [r["size"]["max_tensor"] for r in rows]
        base = {k: {"Q-lvl": actor_scores[0.0][k]["Q-lvl"], "S-start": actor_scores[0.0][k]["S-start"]} for k in sc}
        rl_rows[variant] = rows
        res["rl_first_checkpoint"][variant] = {
            "steps": steps, "n_actors": len(rows), "writers": writers,
            "rel_overall_median": float(np.median(ov)), "rel_overall_range": [float(min(ov)), float(max(ov))],
            "rel_max_tensor_median": float(np.median(mx)), "rel_max_tensor_range": [float(min(mx)), float(max(mx))],
            "max_tensor_names": sorted({r["size"]["max_tensor_name"] for r in rows}),
            "n_nonfinite": sum(not r["size"]["finite"] for r in rows),
            "frac_entries_changed_median": nz(float(np.median([r["size"]["frac_entries_changed"] for r in rows if r["size"]["finite"]]))),
            "frac_zero_entries_changed_median": nz(float(np.median([r["size"]["frac_zero_entries_changed"] for r in rows if r["size"]["finite"]]))),
            "kl_init_median": float(np.median([r["kl_init"] for r in rows if r["kl_init"] is not None])) if any(r["kl_init"] is not None for r in rows) else None,
            "Q-lvl": boot_family(_by_fam(sc, "Q-lvl")), "S-start": boot_family(_by_fam(sc, "S-start")),
            "matched_random": {"n_actors": len(msc), "Q-lvl": boot_family(_by_fam(msc, "Q-lvl")),
                               "S-start": boot_family(_by_fam(msc, "S-start")),
                               "note": "random direction with the support and per-tensor norms of each actor's "
                                       "update (finite checkpoints only), mean of 2 seeds"},
            "eps0_same_actors_Q-lvl": boot_family(_by_fam(base, "Q-lvl")),
            "eps0_same_actors_S-start": boot_family(_by_fam(base, "S-start")),
            "logged_Q-lvl_raw_mean_tf32": _mean([r["logged_at_ckpt_tf32_raw"].get("Q-lvl") for r in rows]),
            "logged_Q-lvl_raw_mean_exact": _mean([(r["logged_at_ckpt_exact_raw"] or {}).get("Q-lvl") for r in rows]),
            "rows": rows}
    # structure-preserving PPO: unchanged actor at its first two checkpoints (critic warm-up 50k steps)
    sp_logged = [logged(stage2_dir(f, w, "sp", rw, s), st) for f in FAMS for w in WRITERS for rw in ("sparse", "shaped")
                 for s in range(3) for st in (16384, 40960)]
    sp_step0 = [logged(stage2_dir(f, w, "sp", rw, s), 0) for f in FAMS for w in WRITERS for rw in ("sparse", "shaped")
                for s in range(3) for _ in (16384, 40960)]
    res["rl_first_checkpoint"]["sp_16k"] = {
        "steps": 16384, "rel_overall": 0.0,
        "note": "critic warm-up: the actor optimiser is not stepped while steps_before < 50000, so theta is "
                "identical to theta_0 at the 16k and 41k checkpoints (by construction of wsp.train_ppo.ppo)",
        "logged_equal_to_step0": sum(a == b for a, b in zip(sp_logged, sp_step0)), "logged_n": len(sp_logged)}
    # SGD: exact bound from gradient clipping (clip_grad_norm_ over actor + critic to 0.5, momentum 0)
    n_steps_16k = 2 * 4 * ((64 * (128 // 16)) // (1024 // 16))
    bnd = {}
    for fam in FAMS:
        for w in WRITERS:
            p0 = load_actor(fam, w)
            bnd[f"{fam}/{w}"] = n_steps_16k * 1e-5 * 0.5 / math.sqrt(sum(float((v.astype(np.float64) ** 2).sum()) for v in p0.values()))
    res["rl_first_checkpoint"]["sgd_bound_16k"] = {
        "actor_steps": n_steps_16k, "abs_bound": n_steps_16k * 1e-5 * 0.5,
        "rel_overall_bound_max_over_actors": max(bnd.values()), "rel_overall_bound_median": float(np.median(list(bnd.values()))),
        "per_actor": bnd}
    # ---- final saved students (303k steps), all seeds and rewards
    for variant in ("std", "sp", "sgd"):
        rows = []
        for fam in FAMS:
            for w in WRITERS:
                for rw in ("sparse", "shaped"):
                    for s in range(3):
                        d = stage2_dir(fam, w, variant, rw, s)
                        if not (d / "meta.json").exists():
                            continue
                        p0 = A.load_npz(d / "init.npz")
                        if not (d / "student.npz").exists():
                            rows.append({"run": d.name, "family": fam, "diverged": True})
                            continue
                        sz = rel_change(p0, A.load_npz(d / "student.npz"))
                        ev = json.loads((d / "eval.json").read_text()) if (d / "eval.json").exists() else None
                        rows.append({"run": d.name, "family": fam, "diverged": False, "rel_overall": sz["overall"],
                                     "rel_max_tensor": sz["max_tensor"], "max_tensor_name": sz["max_tensor_name"],
                                     "frac_zero_entries_changed": sz["frac_zero_entries_changed"],
                                     "Q-lvl": qlvl_balanced(fam, ev["episodes"]["Q-lvl"]) if ev else None})
        ok = [r for r in rows if not r["diverged"]]
        res["rl_final_saved"][variant] = {
            "steps": 303104, "runs": len(rows), "diverged": len(rows) - len(ok),
            "rel_overall_median": float(np.median([r["rel_overall"] for r in ok])),
            "rel_overall_range": [float(min(r["rel_overall"] for r in ok)), float(max(r["rel_overall"] for r in ok))],
            "rel_max_tensor_median": float(np.median([r["rel_max_tensor"] for r in ok])),
            "Q-lvl_balanced_mean_nondiverged": _mean([r["Q-lvl"] for r in ok]), "rows": rows}
    # size-matched random controls for the saved final students (seed-0 sparse runs; no training)
    sub_keys = {fam: {s.key for s in eval_sets(fam)["Q-lvl"]} for fam in FAMS}
    for variant in ("std", "sp", "sgd"):
        rows = []
        for fam in FAMS:
            for w in WRITERS:
                src = stage2_dir(fam, w, variant)
                ms = [WORK / "final_matched" / fam / f"{w}_{variant}_sparse_s0_m{s}.json" for s in NOISE_SEEDS]
                ms = [json.loads(m.read_text()) for m in ms if m.exists()]
                if not ms or not (src / "eval.json").exists():
                    continue
                ev = json.loads((src / "eval.json").read_text())
                sz = rel_change(A.load_npz(src / "init.npz"), A.load_npz(src / "student.npz"))
                rows.append({"family": fam, "writer": w, "rel_overall": sz["overall"], "rel_max_tensor": sz["max_tensor"],
                             "student_Q-lvl": qlvl_balanced(fam, [e for e in ev["episodes"]["Q-lvl"] if e[0] in sub_keys[fam]]),
                             "student_S-start": rate(ev["episodes"]["S-start"]),
                             "matched_Q-lvl": _mean([qlvl_balanced(fam, m["episodes"]["Q-lvl"]) for m in ms]),
                             "matched_S-start": _mean([rate(m["episodes"]["S-start"]) for m in ms]),
                             "eps0_Q-lvl": actor_scores[0.0][(fam, w)]["Q-lvl"],
                             "eps0_S-start": actor_scores[0.0][(fam, w)]["S-start"]})
        if not rows:
            continue
        pick = lambda key: boot_family(_by_fam({(r["family"], r["writer"]): {key: r[key]} for r in rows}, key))
        res["rl_final_saved"][variant]["matched_seed0_sparse"] = {
            "n_actors": len(rows), "rel_overall_median": float(np.median([r["rel_overall"] for r in rows])),
            "rel_max_tensor_median": float(np.median([r["rel_max_tensor"] for r in rows])),
            **{k: pick(k) for k in ("student_Q-lvl", "student_S-start", "matched_Q-lvl", "matched_S-start",
                                    "eps0_Q-lvl", "eps0_S-start")},
            "note": "final students (303k) of the seed-0 sparse runs that did not diverge; student scores from their "
                    "exact eval.json restricted to the Q-lvl subsample; matched = random direction with the "
                    "student update's support and per-tensor norms (2 seeds)", "rows": rows}
    res["tolerance"] = {"largest_eps_keeping_half_of_qlvl": tolerance(res, 0.5),
                        "largest_eps_keeping_90pct_of_qlvl": tolerance(res, 0.9)}
    res["table_tex"] = table(res)
    res["rl_line"] = rl_line(res)
    (R / "analysis_rev2" / "perturbation.json").write_text(json.dumps(res, indent=1, default=float))
    (R / "analysis_rev2" / "tables" / "perturbation.tex").write_text(res["table_tex"])
    print(res["table_tex"])
    print(res["rl_line"])
    print("eps0 sanity (all episodes identical to eval.json):", res["sanity"]["eps0_all_episodes_identical"],
          "| mismatching episodes:", sum(sum(m["full_episode_mismatch"].values()) for m in res["sanity"]["eps0_vs_evaljson"]),
          "| tolerance:", res["tolerance"])


def _by_fam(sc, key):
    out = defaultdict(list)
    for (fam, w), v in sc.items():
        out[fam].append(v[key])
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def f2(c):
    return f"{c['mean']:.2f} [{c['ci'][0]:.2f}, {c['ci'][1]:.2f}]"


def sci(x):
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f"${m:.1f}\\cdot10^{{{e}}}$" if abs(m - 1) > 0.05 else f"$10^{{{e}}}$"


def _rl_rows(res):
    """(label, rel overall, rel largest tensor, Q-lvl, S-start, matched Q-lvl, matched S-start, init Q, init S, n)."""
    out = []
    for v in ("std", "sp"):
        r = res["rl_first_checkpoint"].get(v)
        if r:
            mr = r["matched_random"]
            out.append((f"{LABEL[v]}, {'first ckpt' if v == 'std' else 'first ckpt after warm-up'} {round(r['steps'] / 1000)}k", r["rel_overall_median"], r["rel_max_tensor_median"],
                        r["Q-lvl"], r["S-start"], mr["Q-lvl"] if mr["n_actors"] else None,
                        mr["S-start"] if mr["n_actors"] else None, r["eps0_same_actors_Q-lvl"],
                        r["eps0_same_actors_S-start"], r["n_actors"], mr["n_actors"]))
    for v in ("sgd", "sp"):
        f = res["rl_final_saved"].get(v, {}).get("matched_seed0_sparse")
        if f:
            out.append((f"{LABEL[v]}, final 303k", f["rel_overall_median"], f["rel_max_tensor_median"], f["student_Q-lvl"],
                        f["student_S-start"], f["matched_Q-lvl"], f["matched_S-start"], f["eps0_Q-lvl"], f["eps0_S-start"],
                        f["n_actors"], f["n_actors"]))
    return out


def table(res):
    L = [r"\begin{tabular}{lrrll}", r"\toprule",
         r"Weights ($n$ actors) & \multicolumn{2}{c}{$\|\Delta\theta\|/\|\theta\|$} & Q-lvl [95\% CI] & S-start [95\% CI]\\",
         r" & overall & max.\ tensor & & \\", r"\midrule"]
    for eps in [0.0] + EPS:
        n = res["noise"][str(eps)]
        name = "Written W2 actor (18)" if eps == 0 else f"\\quad + isotropic noise, $\\epsilon={sci(eps)[1:-1]}$"
        sz = "0" if eps == 0 else sci(n["realised_rel_size_overall_mean"])
        mx = "0" if eps == 0 else sci(n["realised_rel_size_max_tensor_mean"])
        L.append(f"{name} & {sz} & {mx} & {f2(n['Q-lvl'])} & {f2(n['S-start'])}\\\\")
    for (lab, ov, mx, q, st, mq, ms, iq, is_, n, nm) in _rl_rows(res):
        L.append(r"\midrule")
        L.append(f"{lab} ({n}) & {sci(ov)} & {sci(mx)} & {f2(q)} & {f2(st)}\\\\")
        if mq is not None:
            L.append(f"\\quad random direction, same size ({nm}) & & & {f2(mq)} & {f2(ms)}\\\\")
        if n != 18:
            L.append(f"\\quad same actors at initialisation & & & {f2(iq)} & {f2(is_)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def tolerance(res, frac=0.5):
    """Largest eps whose pooled Q-lvl keeps at least `frac` of the unperturbed pooled Q-lvl."""
    q0 = res["noise"]["0.0"]["Q-lvl"]["mean"]
    ok = [e for e in EPS if res["noise"][str(e)]["Q-lvl"]["mean"] >= frac * q0]
    return max(ok) if ok else None


def rl_line(res):
    parts = []
    for (lab, ov, mx, q, st, mq, ms, iq, is_, n, nm) in _rl_rows(res):
        lab = lab.replace("$10^{-5}$", "1e-5")
        t = (f"{lab} ({n} actors): relative update {ov:.1e} overall (median; {mx:.1e} in the most-changed tensor), "
             f"held-out Q-lvl {q['mean']:.2f} (init {iq['mean']:.2f}), S-start {st['mean']:.2f} (init {is_['mean']:.2f})")
        if mq is not None:
            t += f"; random update of the same size and support: Q-lvl {mq['mean']:.2f}, S-start {ms['mean']:.2f}"
        parts.append(t)
    b = res["rl_first_checkpoint"].get("sgd_bound_16k")
    if b:
        parts.append(f"SGD at its first checkpoint (16k): relative update <= {b['rel_overall_bound_max_over_actors']:.1e} "
                     f"overall (exact bound from gradient clipping, {b['actor_steps']} steps x lr 1e-5 x 0.5)")
    parts.append("Structure-preserving PPO at 16k and 41k: actor unchanged (critic warm-up to 50k steps)")
    return "; ".join(parts) + "."


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ppo", "noise", "evalckpt", "finals", "analyze"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if a.cmd == "ppo":
        jobs = [(f, w, v, st, a.device) for v, (st, ws, fs) in PPO_JOBS.items() for w in ws for f in fs]
        run_pool(ppo_job, jobs, a.workers)
    elif a.cmd == "noise":
        jobs = [(f, w, 0.0, 0, which, a.device) for f in FAMS for w in WRITERS for which in ("full", "sub")]
        jobs += [(f, w, e, s, "sub", a.device) for e in EPS for s in NOISE_SEEDS for f in FAMS for w in WRITERS]
        run_pool(noise_job, jobs, a.workers)
    elif a.cmd == "evalckpt":
        dirs = [str(p.parent) for p in sorted((WORK / "ppo").glob("*/*/ckpt.npz"))]
        jobs = [(d, "ckpt", 0, a.device) for d in dirs] + [(d, "matched", s, a.device) for d in dirs for s in NOISE_SEEDS]
        run_pool(evalckpt_job, jobs, a.workers)
    elif a.cmd == "finals":
        jobs = [(f, w, v, s, a.device) for v in ("sgd", "sp") for f in FAMS for w in WRITERS for s in NOISE_SEEDS
                if (stage2_dir(f, w, v) / "student.npz").exists()]
        run_pool(final_matched_job, jobs, a.workers)
    else:
        analyze()
