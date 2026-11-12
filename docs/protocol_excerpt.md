# Protocol excerpt

Excerpt of the study protocol. Descriptions of the private games are not included, and bracketed
text is editorial.

## Part 1. Protocol P2 as frozen

### Protocol P2 (revision 2): frozen before the first revision-2 writer call

Implements revision 2 of the experiment plan (Part 2). The hashes of every prompt, query manifest and
redacted game section are recorded in `manifests/rev2/freeze.json` at freeze time. Changes made
during the run are listed under "Scientific amendments".

#### Inputs

| Item | Value |
|---|---|
| Games package | commit `88425fbecb0cd4d44ce18bcf8adde185e019ae02` |
| Private families (20) | easy: `ztl`, `wih`, `qcn`; medium: `lys`, `esm`, `aiv`, `nkp`, `obn`, `qwv`, `ruz`, `vvf`, `ubr`, `yuz`, `bvn`; hard: `trx`, `yiy`, `crg`, `qpv`, `ydv`; very hard: `mds` |
| Public tier (25) | official ARC3 games, separate protocol addendum after the source survey |
| Level split | 1-based held-out levels n, n−2, …, down to 2 (`queries.held_out_levels(n, version=2)`); level 1 always seen; final level always held out |
| Templates | support seeds 0–4; held-out templates 5–9; palette seeds 10–19 (Q-pal, where palettes differ) |
| Sets | S-start (support starts), S-dev (support perturbed, for feedback/BC only), Q-lvl (held-out levels × 10 templates × [1 reviewed + 5 certified perturbed]), Q-tpl, Q-off, Q-pal; manifests `manifests/rev2/queries_<family>.json` (master seed 20261104) |
| Perturbed starts | 1–6 random effective actions; kept only if the prefix neither solves the level nor ends an attempt, and the planner's direct plan (no RESET) replays to completion in the native environment |
| Episode | ends on level completion, GAME_OVER, or 3 × certified plan length + 10 actions; RESET allowed and counted |

#### Information conditions

- **M (primary): mechanics-only source** (`wsp/redact2.py`, condition `M`). The prompt contains:
  - state, transition, goal and rendering code;
  - support-level data;
  - the engine base class.

  It does not contain:
  - held-out level data, branches and comments;
  - held-out templates;
  - solver, planner, oracle and solution routines, and statements that store solver output;
  - solution-revealing comments;
  - classes of other families that share the source file.

  Per-family overrides (`OVERRIDES`) replace solver-derived quantities used by the mechanics.
  Every family has passed an automated audit plus an independent manual audit [carried out by separate agent instances, plus a mechanical
  functional-equivalence check].
- **MS (ablation): mechanics plus solver.** Same redaction of levels and templates; solver code
  kept, as in [the preliminary study].

#### Writers and settings

| Writer | Model | Settings |
|---|---|---|
| astra-6 | GPT-6 Astra | effort `high`, 128k output cap, background mode |
| sol-6.1 | GPT-6.1 Sol | as above |
| terra-6.1 | GPT-6.1 Terra | as above |
| luna-6 | GPT-6 Luna | as above |
| opus-5.5 | Claude Opus 5.5 | effort `high`, 128k cap, advisory task budget 96k |
| sonnet-5.5 | Claude Sonnet 5.5 | as above |
| haiku-5.5 | Claude Haiku 5.5 | as above |
| sol-5.6, opus-5 (anchors) | GPT-5.6 Sol, Claude Opus 5 | as the respective vendor's models above |

A 90-minute wall limit applies to every request. It is enforced for both transports; a request
that exceeds it is recorded as `unknown` and never resubmitted.

#### Conditions

| Arm | Condition | Writers | Families | Runs |
|---|---|---|---|---|
| Main | W1, P1 (M, full actor) | all 7 current | 20 private | 3 |
| Main | W2, P2 (M, full actor; ≤3 feedback rounds on S-start) | Astra, Sol 6.1, Opus 5.5, Terra | 20 private | 3 |
| Anchors | W1, P1 (M) | Sol 5.6, Opus 5 | 20 private | 3 |
| P-react | Python player, reactive, 50 ms per step (100 ms hard limit) | all 7 current | 20 private | 1 |
| Effort symmetry | W1 at effort `medium` | all 7 current | `ztl`, `lys`, `esm`, `ruz`, `crg`, `mds` | 1 |
| Solver ablation | W1, P1 (MS) | Astra, Sol 6.1, Opus 5.5 | same 6 | 1 |
| Operator ablation | W1 (M) with actor `noaxial` and with actor `t8` | Astra, Sol 6.1, Opus 5.5 | same 6 | 1 |

- **Final artifact:** one-shot runs keep their only artifact. Feedback runs keep the best S-start
  solve count, ties going to the latest round. Every candidate is kept.
- **Failures:** refusal, truncation, unknown or stall, and invalid artifact all score 0 in the
  primary analysis. A sensitivity analysis excludes them. Artifact-delivery rate is reported
  separately from performance conditional on delivery.

#### Evaluation and diagnostics

- **Primary:** Q-lvl solve rate, split into Q-lvl-mid (held-out levels between seen ones) and
  Q-lvl-final (the final level). Also reported separately for seen and held-out templates.
- **Secondary:** Q-tpl, Q-off, Q-pal, S-start; actions, resets, invalid and non-finite actions.
- **Memorisation and shortcut controls:**
  - zero-frame and first-frame blindness;
  - counterfactual-frame sensitivity;
  - RESET-disabled perturbed starts (the primary perturbed metric counts RESET-free successes);
  - per-family replay floor and best open-loop schedule (`scripts/diag_eval.py`).

  A family whose best open-loop schedule reaches ≥0.5 on Q-lvl is flagged blind-solvable and
  excluded from headline macros.
- **Deployment cost:** latency, memory and MACs per action for written actors and Python players.

#### Learned baselines and RL

As specified in the experiment plan:
- BC-support, BC-aug, BC-oracle and distill-aug with the fixed teacher, augmentation, truncated
  BPTT and progressive training;
- PPO-sparse and PPO-shaped (planner potential);
- WSP→RL with standard, structure-preserving and SGD variants, 3 seeds, both rewards, plus
  diagnostics;
- RL→WSP with the inspection arm and a budget-matched fresh-synthesis control.

Exact hyperparameters are summarised in the paper's reproducibility appendix.

#### Analysis

- **Unit:** the family. Pooled numbers are equal-weight means over families, with runs averaged
  within a family first.
- **Uncertainty:** hierarchical bootstrap over families → held-out levels → runs (10,000
  resamples). Per cell, k/n runs with Q-lvl ≥0.5 (Wilson).
- **Paired comparisons:** W vs P; W2 vs best-of-k(W1) at matched tokens; W vs the best learned
  baseline.
- Results are reported per difficulty stratum. A missingness table is included.

#### Decisions recorded before the freeze

**Redaction (private tier, condition M).** Two independent manual audits [carried out by separate agent instances, plus a mechanical
functional-equivalence check] found leaks or broken code in 18 of the 20 automatically redacted families. The worst cases:
- solvers kept because analysis code called them;
- per-level oracle-depth tables for held-out levels;
- held-out builder functions shipped in full (one family);
- core mechanics removed (one family).

The fixes:
- Per-family overrides (`wsp/redact2_overrides.py`).
- Prose rule (strict): all prose that describes support levels or strategy is removed in every
  family; level data stays.
- Slack fields: support-level `slack` fields are dropped, because budget − slack would reveal the
  optimum.
- Move budgets: budgets that were derived from the oracle become explicit per-level tables. Support
  values are copied from the original simulator; held-out entries are `WITHHELD`.
- Engine hooks: loss rules and animations that need a search call
  engine-provided `_engine_*` hooks. A hook's implementation is withheld; its
  docstring states exactly what it computes.
- Renderer branches: generic branches used only by held-out templates are kept. Their parameters
  are withheld.
- trx selects its renderer by seed % 8, so Q-tpl seeds 8 and 9 reuse support templates. trx's
  Q-tpl and template-split metrics count only seeds 5–7.

Each redacted family must:
- compile;
- have no undefined names;
- reproduce the original simulator frame-for-frame on every support level and support template,
  under certified, perturbed and random action sequences;
- refuse every held-out level and template (`scripts/check_redaction.py`).

**Public tier (addendum).** The 25 official games are provisioned at
pinned versions. Every level start certifies; 92.4% of perturbed starts do.

Redaction and checks:
- Mechanics-only sections come from reviewed per-game configs (`wsp/redact_public_configs.py`)
  plus a traced excerpt of the engine API.
- The same functional equivalence check applies (`scripts/check_redaction_public.py`). All 25
  pass, and an independent full read found and fixed 10 more leaks.
- Kept: generic rules that only held-out levels use, and range gates that include a support level.
  The latter can reveal some held-out budgets.
- Withheld: dc22's teleporter cycle, which is reachable only from held-out L6.

Exclusions:
- **ka59** is excluded from writer runs: its section is 126k tokens, mostly wall pixel arrays.
  [Amendment, 7 November 2026: ka59 was added to all public-tier writer arms and learned baselines, with its prompts frozen separately before its first writer call; the public tier comprises 25 games.]
- Games whose held-out levels lack a certified start are excluded from the tier macro.

Manifests: Q-lvl = 1 reviewed + 20 perturbed starts per held-out level; S-dev = 4 perturbed starts
per support level; Q-off = 6 per support level; no Q-tpl or Q-pal. The harness counts lost lives
via `lives_map.json` (ls20, r11l).

**Public tier, as run (frozen in `manifests/rev2/freeze_public.json`, freeze `a1203ec3…`):**
- Runner: `scripts/run_public.py`, a wrapper around the private-tier runner.
- Feedback rounds evaluate on S-start plus S-dev (11–25 episodes).
- The task text drops the mention of renderer templates.
- The engine switch `ONLY_RESET_LEVELS` is set, so RESET restarts only the current level, as in
  the private tier and as stated in the episode rules. Without it, a RESET at a level start would restart the
  game at level 1.
- Lost lives count as ended attempts (ls20, r11l).
- Floors: Q-lvl replay and best open-loop are 0.00 for all 25 games.

**Augmentation licensing (BC-aug, distill-aug).**
- Palette permutation: all private families.
- Dihedral group (with action and click remapping): only where the family's own templates apply
  board symmetries that generate D4 and ACTION1–4 are absent or screen directions. That holds for
  13 of the 20 families. The per-family list follows from the private sources and is withheld, as in
  `wsp/augment.py`.
- Public tier: licensed per game from the survey, kept conservative.

**Baseline teacher.** The planner teacher uses a 5 s budget with no RESET fallback; unlabelled
states are dropped and counted. BC uses BPTT-16 and progressive training. PPO uses BPTT-16. PPO
hyperparameters come from a lr ∈ {1e-4, 3e-4} × entropy ∈ {0.003, 0.01} sweep (1M steps, on lys,
ruz and crg), separately for sparse and shaped reward; the best setting is then used everywhere.

**Floors measured before any writer call.** No private family reaches the
blind-solvable threshold on Q-lvl. wih comes closest: its best open-loop schedule solves 0.46.

**Harness checks.**
- The batched torch actor agrees with the float32 NumPy reference on 1,513 decisions from 40
  [preliminary-study] artifacts: 1 argmax tie, maximum relative logit error 1.1e-6.
- Actor cost per action: 1.32 GMAC for `full`, 1.31 GMAC for `noaxial` and 0.19 GMAC for `t8`.

## Part 2. From the governing experiment plan (revision 2)

#### 2. Research question and claims

> **When the rules of a game are known, can frontier models compile them directly into the weights
> of a compact neural agent that generalises to new instances? How does this compare with
> writing a program, or with learning the same network from data?**

Claims are restricted to generalisation within a game: one actor per game, mechanics given as source code
(without solver routines; §4.0), tested on new
levels, layouts, renderer variants, palettes and states of the same game. "Unseen" refers to the
writer: private games cannot be in any training data. Public ARC3 games form a separate, explicitly "possibly seen" tier.

Pre-registered hypotheses:

| ID | Hypothesis | Primary test |
|---|---|---|
| H1 | Frontier written weights solve held-out instances well above floors on unseen games | Q-lvl vs scripted/random/replay floors, per difficulty stratum; Q-lvl-mid and Q-lvl-final reported separately |
| H2 | Written weights generalise better than *fairly trained* learned weights in the same actor, given the same support data plus source-licensed augmentation | Paired gap (Q-lvl, Q-tpl; Q-lvl only on the public tier) between W2 and the best of {BC-aug, distill-aug, tuned PPO}; family-level bootstrap |
| H3 | Weight writing degrades with game difficulty faster than program writing | Difficulty × {W, P} interaction; slopes over strata |
| H4 | Written actors are closed-loop and do not replay support sequences | Zero- and first-frame blindness; replay floor |

Secondary, reported with CIs and no strong claims: feedback vs compute-matched best-of-k;
writer and vendor differences; elicitation settings; the RL interaction (§6).

#### 6. Weight writing × RL (excerpt)

- **Decision rule (pre-registered):** a hybrid benefit means beating both RL(θ₀) and the written
  initialisation on Q-lvl, with non-overlapping seed ranges.

#### 7. Statistics and reporting

- **Unit of analysis:** the family. Uncertainty comes from a hierarchical
  bootstrap over families → held-out levels → runs. Per-cell success is reported as k/n runs with
  Q-lvl ≥0.5, with Wilson intervals.
- **Aggregation rule:** every pooled number is an equal-weight mean over families. Within a
  family, runs and seeds are averaged first. Missing runs are listed in a missingness table
  (planned vs completed, per family and condition).
- **Matched comparisons:** palette, template and perturbation effects are estimated on matched
  start sets, never by comparing different start sets. The support-vs-held-out claim is shown at
  artifact level (pairs), not via aggregate means.
- **Paired comparisons:** W vs P; W2 vs best-of-k(W1) at matched tokens; W vs the best learned
  baseline. All on identical episodes.
- **Strata:** results per difficulty stratum and macro over strata; interpolation (Q-lvl-mid) and
  extrapolation (Q-lvl-final) separately. Blind-solvable families are reported separately.
- **Figures:** a hero figure of per-artifact support vs held-out success for every policy source,
  and difficulty curves for W vs P vs learned.
- **Mechanism evidence:** causal ablations for 1–2 artifacts per stratum (zero the component the
  writer names; test the predicted behaviour change).
- Every deviation is logged [see "Scientific amendments"].

## Scientific amendments

- **5 November 2026, evaluation numerics.** Every evaluation of a weight artifact uses exact FP32 (TF32 disabled), matching the float32 NumPy reference; earlier TF32 evaluations were recomputed, and training stays on TF32.
- **5–7 November 2026, exploratory arms.** Added and analysed as exploratory, each writer arm with its prompts frozen before its first call: the free architecture (FA) and its controls A (fixed graph) and B (fixed operations), 5 November; weaker and previous-generation writers in FA and W2, 6 November; the bounded program player, 7 November; and learned FA networks (GPT-6 Astra's FA graphs re-initialised and trained by BC-support and distillation), 7 November.
- **7 November 2026, learned baselines.** All DAgger learners use window-start memory recomputed by the current student on the same (augmented) frames; only these runs are reported.
- **7 November 2026, ka59.** ka59 is included in every public-tier writer arm and learned baseline, with its prompts frozen before its first writer call; the public tier has 25 games.
- **5 and 8 November 2026, re-runs.** A session is re-run from scratch, without regard to its outcome, after an interruption; from 8 November this includes an API failure of its first request that produced no output. Only the final attempt is scored.
