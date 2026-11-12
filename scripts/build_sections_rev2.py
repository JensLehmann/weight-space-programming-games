"""Generate revision-2 writer game sections (M and MS) + audits for all private families.

Check the result with scripts/check_redaction.py (reads the files written here)."""
import json, sys
from wsp.envs import new_game
from wsp.manifest import RUNS
from wsp.redact2 import game_section_v2
FAMS = sys.argv[1:] or "ztl wih qcn lys esm aiv nkp obn qwv ruz vvf ubr yuz bvn trx yiy crg qpv ydv mds".split()
out = RUNS / "private" / "game_sections_rev2"; out.mkdir(parents=True, exist_ok=True)
summary = {}
for fam in FAMS:
    n = new_game(fam, 0).n_levels
    for cond in ("M", "MS"):
        try:
            sec, rep = game_section_v2(fam, n, cond)
        except Exception as exc:
            print("FAILED", fam, cond, repr(exc)[:300]); continue
        (out / f"{fam}_{cond}.md").write_text(sec)
        (out / f"{fam}_{cond}.audit.json").write_text(json.dumps(rep, indent=1))
        a = [f["audit"] for f in rep["files"]]
        if cond == "M":
            summary[fam] = {"files": [f["file"] for f in rep["files"]], "chars": len(sec),
                            "levels_withheld": sum(len(x["withheld_levels"]) for x in a),
                            "if_branches": sum(len(x["if_level_branches"]) for x in a),
                            "n_templates": rep["n_templates"], "held_templates": rep["held_templates"],
                            "templates_withheld": sum(len(x["withheld_templates"]) for x in a),
                            "solvers": sorted({s for x in a for s in x["solvers_removed"]}),
                            "gated": sum(len(x["level_gated"]) for x in a),
                            "manual": [m for x in a for m in x["solver_refs_kept"]],
                            "suspicious": sum(len(x["remaining_suspicious"]) for x in a),
                            "hooks": rep["hooks"], "notes": rep["notes"]}
        summary.setdefault(fam, {})[f"{cond}_overrides_applied"] = sum(1 for c in rep["override_counts"] if c["count"])
        summary[fam][f"{cond}_overrides_unmatched"] = len(rep["override_unmatched"])
        summary[fam][f"{cond}_tree_shaken"] = sum(len(x["tree_shaken"]) for x in a)
        summary[fam][f"{cond}_chars"] = len(sec)
(out / "summary.json").write_text(json.dumps(summary, indent=1))
for k, v in summary.items():
    if "levels_withheld" not in v:
        print(k, "M FAILED"); continue
    print(f"{k:4s} lv={v['levels_withheld']:2d} if={v['if_branches']:2d} tpl={v['templates_withheld']:2d} gated={v['gated']:2d} susp={v['suspicious']:2d} solvers={v['solvers'][:5]} manual={v['manual'][:2]}")
