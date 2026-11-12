"""Generate the public-tier writer game sections (condition M) and their audit records.

Writes runs/private/game_sections_public/<id>_M.md, <id>_M.audit.json and summary.json.
Prompt size is reported as tokens ~= chars / 4. Usage: build_sections_public.py [game ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp.redact_public import GAMES, RUNS, public_section  # noqa: E402

OVER = 60_000
out = RUNS / "private" / "game_sections_public"
out.mkdir(parents=True, exist_ok=True)
games = sys.argv[1:] or GAMES
spath = out / "summary.json"
summary = json.loads(spath.read_text()) if spath.exists() and sys.argv[1:] else {}
for g in games:
    try:
        md, rep = public_section(g, "M")
    except Exception as exc:  # noqa: BLE001
        print("FAILED", g, repr(exc)[:400])
        summary[g] = {"error": repr(exc)[:400]}
        continue
    (out / f"{g}_M.md").write_text(md)
    (out / f"{g}_M.audit.json").write_text(json.dumps(rep, indent=1, default=str))
    libs = rep["generic"]["libraries"]
    summary[g] = {
        "n_levels": rep["n_levels"], "held_levels": rep["held_levels"], "support_levels": rep["support_levels"],
        "levels_withheld": len(rep["generic"]["levels_withheld"]),
        "sprites_omitted": sum(len(v["omitted"]) for v in libs.values()),
        "sprites_kept": sum(v["kept"] for v in libs.values()),
        "held_comments_removed": rep["generic"]["held_comments_removed"],
        "config_edits": sum(1 for c in rep["config"] if c["kind"] in ("Items", "Branch", "Body", "Sub", "Drop")),
        "config_entries": len(rep["config"]) + len(rep["notes"]),
        "level_index_compares": len(rep["level_index_uses"]["compares"]),
        "level_index_subscripts": len(rep["level_index_uses"]["subscripts"]),
        "held_refs": rep["level_index_uses"]["held_refs"],
        "problems": rep["problems"],
        "chars_source": rep["chars_source"], "chars_redacted": rep["chars_redacted"],
        "chars_engine": rep["chars_engine"], "chars_section": rep["chars_section"],
        "tokens_section": rep["tokens_section"], "over_60k_tokens": rep["tokens_section"] > OVER,
    }
spath.write_text(json.dumps(summary, indent=1))
for g in GAMES:
    v = summary.get(g)
    if not v or "error" in v:
        continue
    print(f"{g} n={v['n_levels']} held={v['held_levels']} edits={v['config_edits']} sprites-={v['sprites_omitted']} "
          f"tokens={v['tokens_section']}{' OVER' if v['over_60k_tokens'] else ''} problems={v['problems']} "
          f"held_refs={v['held_refs']}")
