"""Learned baselines into the fixed actor (one process per run).

kinds: bc_support (planner DAgger on support), bc_aug (bc_support + the family's licensed
augmentation, wsp.augment.augmentations_for), bc_oracle (planner DAgger on all levels/templates;
capacity ceiling only), distill:<session_dir> (writer player teacher; add --augment auto for
distill-aug), ppo (sparse reward, support), ppo_shaped (ppo + planner-potential shaping).

All training options default to off (the trainers' default behaviour). The planner teacher uses the
certification budget (5000 ms) and no RESET fallback; `--teacher-budget-ms` and
`--teacher-reset-fallback` change this. Use --tag to keep option variants apart.
"""
import argparse, json, os, random, sys
from dataclasses import replace
from pathlib import Path

def oracle_specs(fam, seed=7):
    from wsp.envs import new_game
    from wsp.queries import reviewed_start, perturbed_starts
    n = new_game(fam, 0).n_levels
    rng = random.Random(f"oracle:{fam}:{seed}")
    specs = []
    for t in range(10):
        for l in range(n):
            specs.append(reviewed_start(fam, t, l, "oracle"))
            ps, _ = perturbed_starts(fam, t, l, 3, rng, set())
            specs += ps
    return specs


def ints(s):
    return [int(float(x)) for x in s.split(",") if x.strip()] if s else []


def add_ppo_args(ap):
    g = ap.add_argument_group("PPO")
    g.add_argument("--lr", type=float, default=3e-4); g.add_argument("--ent", type=float, default=0.01)
    g.add_argument("--n-envs", type=int, default=64); g.add_argument("--horizon", type=int, default=128)
    g.add_argument("--minibatch", type=int, default=1024); g.add_argument("--ppo-epochs", type=int, default=4)
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--bptt", type=int, default=None, help="recurrent PPO window length (e.g. 16)")
    g.add_argument("--grad-ckpt", type=int, default=0, help="checkpoint tied block every N iterations")
    g.add_argument("--shaping", choices=("none", "planner"), default=None)
    g.add_argument("--shaping-budget-ms", type=float, default=200)
    g.add_argument("--shaping-scale", type=float, default=1.0)
    g.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    g.add_argument("--critic-lr", type=float, default=None)
    g.add_argument("--sgd-momentum", type=float, default=0.0)
    g.add_argument("--masked", action="store_true", help="update only entries nonzero in the init")
    g.add_argument("--kl-to-init", type=float, default=0.0)
    g.add_argument("--critic-warmup", type=int, default=0, help="env steps of critic-only training")
    g.add_argument("--eval-sets", default="", help="e.g. S-start,Q-lvl (greedy checkpoints)")
    g.add_argument("--eval-at", default="0,10000,40000,80000,160000")
    g.add_argument("--eval-every", type=int, default=200_000)


def ppo_kwargs(a, sets, shaping="none"):
    return dict(lr=a.lr, ent=a.ent, n_envs=a.n_envs, horizon=a.horizon, minibatch=a.minibatch,
                epochs=a.ppo_epochs, log_every=a.log_every, bptt=a.bptt, grad_ckpt=a.grad_ckpt,
                shaping=a.shaping or shaping, shaping_budget_ms=a.shaping_budget_ms,
                shaping_scale=a.shaping_scale, optimizer=a.optimizer, critic_lr=a.critic_lr,
                sgd_momentum=a.sgd_momentum, masked=a.masked, kl_to_init=a.kl_to_init,
                critic_warmup=a.critic_warmup,
                eval_specs={k: sets[k] for k in a.eval_sets.split(",") if k} or None,
                eval_at=ints(a.eval_at), eval_every=a.eval_every or None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind"); ap.add_argument("family"); ap.add_argument("seed", type=int)
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--root", default="baselines")
    ap.add_argument("--tag", default="", help="suffix of the output directory name")
    ap.add_argument("--split-rev", type=int, default=None, help="query manifest revision (default: current)")
    ap.add_argument("--device", default="cuda")
    g = ap.add_argument_group("DAgger")
    g.add_argument("--iters", type=int, default=6); g.add_argument("--epochs", type=int, default=4)
    g.add_argument("--batch", type=int, default=64); g.add_argument("--bc-lr", type=float, default=1e-3)
    g.add_argument("--teacher-budget-ms", type=float, default=5000)
    g.add_argument("--teacher-reset-fallback", action="store_true", help="RESET labels for states the planner cannot label")
    g.add_argument("--augment", default=None,
                   help="none | auto (family licence) | palette | palette,dihedral (bc_aug: auto)")
    g.add_argument("--bc-bptt", type=int, default=None, help="BPTT window length (e.g. 16)")
    g.add_argument("--progressive", action="store_true", help="progressive loss for the tied block")
    g.add_argument("--bc-grad-ckpt", type=int, default=0)
    g.add_argument("--aug-memory", choices=("stored", "recompute"), default="stored",
                   help="memory entering a training window/step: stored (as run: collection-time "
                        "memory of the untransformed frames) | recompute (one transform per episode "
                        "per epoch, memory recomputed by the current student on the transformed "
                        "episode)")
    g.add_argument("--student-arch", default=None,
                   help="learned free-architecture student: a writer FA graph round<k>.graph.json "
                        "(params round<k>.npz next to it); its weights are re-initialised "
                        "(wsp.train_bc.FA_INIT) and trained instead of the fixed actor")
    g.add_argument("--student-bias-init", choices=("zero", "uniform"), default="zero",
                   help="biases of a re-initialised FA student: zero (as the fixed actor) | uniform "
                        "(PyTorch default U(+-1/sqrt(fan_in)))")
    g.add_argument("--time-cap-h", type=float, default=None,
                   help="wall-clock cap in hours (from the start of DAgger): stop collection/training, save "
                        "the current student and mark dagger_summary.json truncated (default: no cap)")
    add_ppo_args(ap)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from wsp.augment import augmentations_for, parse_kinds
    from wsp.manifest import RUNS
    from wsp.queries import load_sets
    from wsp.train_bc import dagger, PlannerTeacher, PlayerTeacher
    sets = load_sets(a.family) if a.split_rev is None else load_sets(a.family, rev=a.split_rev)
    support = sets["S-start"] + sets["S-dev"]
    kind = a.kind.split(":")[0]
    out = RUNS / a.root / a.family / f"{kind}{a.tag}_s{a.seed}"
    aug_spec = a.augment if a.augment is not None else ("auto" if kind == "bc_aug" else "none")
    augment = (augmentations_for(a.family) or None) if aug_spec == "auto" else parse_kinds(aug_spec)
    bc = dict(seed=a.seed, iters=a.iters, epochs=a.epochs, batch=a.batch, lr=a.bc_lr, device=a.device,
              augment=augment, bptt=a.bc_bptt, progressive=a.progressive, grad_ckpt=a.bc_grad_ckpt,
              aug_memory=a.aug_memory, student_arch=a.student_arch, student_bias_init=a.student_bias_init,
              time_cap_h=a.time_cap_h)
    teacher = lambda: PlannerTeacher(budget_ms=a.teacher_budget_ms, reset_fallback=a.teacher_reset_fallback)
    if kind == "distill":
        sess = Path(a.kind.split(":", 1)[1])
        out = sess / f"distill{a.tag}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cli.json").write_text(json.dumps({**vars(a), "augment_resolved": sorted(augment or [])}, indent=1))
    if kind in ("bc_support", "bc_aug"):
        dagger(support, teacher(), out, **bc)
    elif kind == "bc_oracle":
        dagger(oracle_specs(a.family), teacher(), out, **bc)
    elif kind in ("ppo", "ppo_shaped"):
        from wsp.train_ppo import ppo
        ppo(support, out, seed=a.seed, total_steps=a.steps, device=a.device,
            **ppo_kwargs(a, sets, "planner" if kind == "ppo_shaped" else "none"))
    elif kind == "distill":
        rec = json.loads((sess / "session.json").read_text())
        code = (sess / f"round{rec['final_round']}.py").read_text()
        dagger(support, PlayerTeacher(code), out, **bc)
    else:
        sys.exit(f"unknown kind {a.kind}")
    print("DONE", out, flush=True)
