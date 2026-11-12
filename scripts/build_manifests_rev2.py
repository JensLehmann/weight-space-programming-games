import json, sys, time
from wsp.queries import write_family_sets
fam = sys.argv[1]
t = time.time()
d = write_family_sets(fam, rev=2)
print(json.dumps({"family": fam, "n_levels": d["n_levels"], "held": [h + 1 for h in d["held_out_levels"]],
                  "sizes": {k: len(v) for k, v in d["sets"].items()}, "palette_distinct": d["palette_distinct"],
                  "stats": d["construction_stats"], "s": round(time.time() - t)}), flush=True)
