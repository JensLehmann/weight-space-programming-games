"""Public tier (official ARC3 games): construction, continuous entry, RESET and attempt ends."""
import json

import pytest

from wsp import envs
from wsp.envs import EpisodeSpec, build_start, enter_level, new_game, run_episodes
from wsp.manifest import MANIFESTS

P = pytest.importorskip("wsp.public")


def _games_available() -> bool:
    try:
        return (P.source_root() / "ls20").is_dir()
    except FileNotFoundError:
        return False


def test_released_routes_match_query_manifests():
    """The released routes and lives map are those the frozen query manifests were built with."""
    for fam in P.public_families():
        man = json.loads((MANIFESTS / "rev2" / f"queries_{fam}.json").read_text())
        assert man["env"]["route_sha256"] == P.route_sha256(fam)
        assert man["env"]["source_sha256"] == P._routes()[P.game_id(fam)]["source_sha256"]
        assert man["env"]["lives_exprs"] == list(P.lives_exprs(fam))
        assert man["n_levels"] == P.n_levels(fam)


needs_games = pytest.mark.skipif(not _games_available(), reason="official game files not provisioned")


class Repeat:
    def __init__(self, move):
        self.move = move

    def begin(self, slots):
        pass

    def act(self, slots, obs, avail):
        return [self.move for _ in slots]


@needs_games
def test_ids_and_levels():
    released = {p.name[len("queries_"):-len(".json")] for p in (MANIFESTS / "rev2").glob("queries_official.*.json")}
    assert len(P.public_ids()) == 25 and set(P.public_families()) == released
    tab = P.levels_table()
    assert tab["official.ls20"] == 7 and tab["official.lf52"] == 10
    assert new_game("official.ls20", 0).n_levels == 7
    with pytest.raises(ValueError):
        new_game("official.ls20", 1)


@needs_games
def test_enter_level_is_continuous_route_play():
    g = enter_level("official.ls20", 0, 3)
    ref = new_game("official.ls20", 0)
    for lv in P.route("official.ls20")[:3]:
        for a, r, c in lv:
            ref.step(a, r, c)
    assert g.level == ref.level == 3 and g.observe() == ref.observe()
    s = build_start(EpisodeSpec("official.ls20", 0, 3, (("ACTION1", None, None),)))
    assert s.level == 3


@needs_games
def test_reset_restarts_current_level():
    g = enter_level("official.ft09", 0, 2)
    o = g.observe()
    g.step("RESET")
    g.step("RESET")
    assert g.level == 2 and g.observe() == o


@needs_games
def test_ls20_life_loss_is_an_ended_attempt():
    spec = EpisodeSpec("official.ls20", 0, 0, (), 0, 50, "t")
    (res,) = run_episodes([spec], Repeat(("ACTION2", None, None)), keep_trace=False)
    assert res.attempts_lost == 1 and not res.game_over and not res.success


@needs_games
def test_release_adapter_matches_game_package():
    """wsp.public_games.OfficialGame steps exactly like the package's environment (routes + random moves)."""
    import random
    pytest.importorskip("arc3_games")
    from wsp.public_games import OfficialGame
    rng = random.Random(0)
    for fam in ("official.ls20", "official.ft09", "official.r11l"):
        gid = P.game_id(fam)
        a = new_game(fam, 0)
        b = OfficialGame(gid, P.source_path(fam), P._routes()[gid]["source_sha256"])
        moves = [m for lv in P.route(fam)[:2] for m in lv]
        moves += [(x, rng.randrange(64), rng.randrange(64)) if x == "ACTION6" else (x, None, None)
                  for x in rng.choices(envs.ACTIONS, k=60)]
        assert a.observe() == b.observe()
        for m in moves:
            assert a.step(*m) == b.step(*m), (fam, m)
            assert (a.level, a.state, a.available_actions, P.attempt_marker(a)) == \
                   (b.level, b.state, b.available_actions, P.attempt_marker(b))


def test_private_detector_unchanged():
    pytest.importorskip("arc3_games")
    from arc3_games.games.base import attempt_marker
    marker, _ended = envs.attempt_detector("lys")
    assert marker is attempt_marker
    marker, _ended = envs.attempt_detector("official.ls20")
    assert marker is P.attempt_marker
