"""Guard against public-tier aggregates over a partial cohort (scripts/analyze_rev2.py, analyze_fa_all.py,
tables_rev2.py): every game of the tier must have all sessions of the five public arms (W1, P1 with the
seven writers; W2, P2, FA W2 with the four frontier writers; three runs each), each with session.json
and eval.json."""
from pathlib import Path
from .manifest import RUNS

CURRENT = ('astra-6','sol-6.1','terra-6.1','luna-6','opus-5.5','sonnet-5.5','haiku-5.5')
FRONTIER = ('astra-6','sol-6.1','terra-6.1','opus-5.5')

def assert_public_cohort_complete(runs=None):
    runs=Path(runs) if runs is not None else RUNS
    # applies once the 25th game (official.ka59, frozen in freeze_public_ka59.json) is present
    begun=any((runs/tag/'official.ka59').exists() for tag in ('rev2_public','rev2_fa_public'))
    if not begun:return
    from .redact_public import GAMES
    missing=[]
    for gid in GAMES:
        family='official.'+gid
        for cond,mode,models in [('weights','oneshot',CURRENT),('python','oneshot',CURRENT),
                                 ('weights','feedback',FRONTIER),('python','feedback',FRONTIER),
                                 ('freearch','feedback',FRONTIER)]:
            tag='rev2_fa_public' if cond=='freearch' else 'rev2_public'
            for model in models:
                for rep in range(3):
                    name=f'freearch_{mode}_M_r{rep}' if cond=='freearch' else f'{cond}_{mode}_M_full_r{rep}'
                    d=runs/tag/family/model/name
                    for file in ('session.json','eval.json'):
                        if not (d/file).is_file():missing.append(str(d/file))
    if missing:
        raise RuntimeError(f'Public cohort incomplete: {len(missing)} required session/evaluation files missing. '
                           f'Public aggregates need all 25 games. First missing: {missing[0]}')
