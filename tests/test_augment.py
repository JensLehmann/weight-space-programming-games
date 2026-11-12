import itertools

import numpy as np
import pytest
import torch

from wsp import augment as G
from wsp.envs import ACTIONS, A_INDEX
from wsp.toy import PLAYER, ToyReach


def test_group_closure_identity_associativity_inverse():
    els = G.ELEMENTS
    assert len(set(map(tuple, (G.matrix(e).ravel() for e in els)))) == 8
    for a, b in itertools.product(els, els):
        assert G.compose(a, b) in els
    for a in els:
        assert G.compose(a, G.IDENTITY) == a == G.compose(G.IDENTITY, a)
        assert G.compose(a, G.inverse(a)) == G.IDENTITY == G.compose(G.inverse(a), a)
    for a, b, c in itertools.product(els, els, els):
        assert G.compose(G.compose(a, b), c) == G.compose(a, G.compose(b, c))


def test_grid_transform_is_a_group_action():
    g = np.random.default_rng(0).integers(0, 16, (64, 64))
    for a, b in itertools.product(G.ELEMENTS, G.ELEMENTS):
        lhs = G.transform_grid(G.transform_grid(g, b), a)
        assert np.array_equal(lhs, G.transform_grid(g, G.compose(a, b)))
    for e in G.ELEMENTS:
        assert np.array_equal(G.transform_grid(G.transform_grid(g, e), G.inverse(e)), g)
    assert np.array_equal(G.transform_grid(g, 1), np.rot90(g))
    assert np.array_equal(G.transform_grid(g, 4), g.T)


def test_transformed_click_hits_transformed_cell():
    rng = np.random.default_rng(1)
    for e in G.ELEMENTS:
        for _ in range(20):
            r, c = (int(x) for x in rng.integers(0, 64, 2))
            g = np.zeros((64, 64), int)
            g[r, c] = 7
            r2, c2 = G.transform_cell(r, c, e)
            assert G.transform_grid(g, e)[r2, c2] == 7
            assert G.transform_move(("ACTION6", r, c), e) == ("ACTION6", r2, c2)
            assert G._CELL_MAP[e][r * 64 + c] == r2 * 64 + c2


def test_action_permutation():
    for e in G.ELEMENTS:
        p = G.action_perm(e)
        assert sorted(p) == list(range(8))
        for a in ("RESET", "ACTION5", "ACTION6", "ACTION7"):
            assert p[A_INDEX[a]] == A_INDEX[a]
        # moving a pixel by d then transforming == transforming then moving by the mapped action
        for a, (dr, dc) in G.DIRECTIONS.items():
            r, c = 30, 20
            r1, c1 = G.transform_cell(r, c, e)
            r2, c2 = G.transform_cell(r + dr, c + dc, e)
            assert G.DIRECTIONS[ACTIONS[p[A_INDEX[a]]]] == (r2 - r1, c2 - c1)
    for a, b in itertools.product(G.ELEMENTS, G.ELEMENTS):
        assert np.array_equal(G.action_perm(G.compose(a, b)), G.action_perm(a)[G.action_perm(b)])
    assert G.transform_move(("ACTION1", None, None), 1) == ("ACTION3", None, None)  # up -> left (CCW)
    assert G.transform_move(("ACTION1", None, None), 4) == ("ACTION3", None, None)  # transpose


def _block_tl(cells, e):
    """Top-left corner of the transformed 2x2 block whose top-left is `cells`."""
    r, c = cells
    pts = [G.transform_cell(r + i, c + j, e) for i in (0, 1) for j in (0, 1)]
    return (min(p[0] for p in pts), min(p[1] for p in pts))


def _transformed_toy(toy: ToyReach, e: int) -> ToyReach:
    t = ToyReach(toy.seed)
    t.level, t.state = toy.level, toy.state
    t.walls = {_block_tl(w, e) for w in toy.walls}
    t.pos = _block_tl(toy.pos, e)
    t.goal = _block_tl(toy.goal, e)
    t._render()
    return t


def test_dihedral_commutes_with_toy_dynamics():
    """toy_reach is D4-symmetric: transformed state + transformed action -> transformed successor."""
    for seed in range(3):
        base = ToyReach(seed)
        for e in G.ELEMENTS:
            for a in G.DIRECTIONS:
                g = ToyReach(seed)
                g.walls, g.pos, g.goal = set(base.walls), base.pos, base.goal
                g._render()
                t = _transformed_toy(g, e)
                assert np.array_equal(np.array(t.observe()), G.transform_grid(np.array(g.observe()), e))
                _, _, _, ch1, _ = g.step(a)
                _, _, _, ch2, _ = t.step(G.transform_move((a, None, None), e)[0])
                assert ch1 == ch2
                assert np.array_equal(np.array(t.observe()), G.transform_grid(np.array(g.observe()), e))


def test_augment_batch_dihedral_remaps_labels_consistently():
    rng = np.random.default_rng(0)
    B = 64
    obs = torch.zeros(B, 64, 64, dtype=torch.long)
    act = torch.empty(B, dtype=torch.long)
    cell = torch.full((B,), -1, dtype=torch.long)
    avail = torch.zeros(B, 8, dtype=torch.bool)
    prev = torch.zeros(B, 8)
    for b in range(B):
        a = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"][b % 5]
        r, c = (int(x) for x in rng.integers(4, 60, 2))
        obs[b, r, c] = 1
        act[b] = A_INDEX[a]
        avail[b, A_INDEX[a]] = True
        prev[b, A_INDEX[a]] = 1
        if a == "ACTION6":
            cell[b] = r * 64 + c
        else:
            dr, dc = G.DIRECTIONS[a]
            obs[b, r + 2 * dr, c + 2 * dc] = 2
    o2, av2, p2, a2, c2 = G.augment_batch(obs, avail, prev, act, cell, {"dihedral"}, rng)
    assert not torch.equal(o2, obs)                     # something was transformed
    for b in range(B):
        k = int(a2[b])
        assert av2[b].nonzero().flatten().tolist() == [k]
        assert p2[b].nonzero().flatten().tolist() == [k]
        r1, cc1 = (int(x) for x in (o2[b] == 1).nonzero()[0])
        if ACTIONS[k] == "ACTION6":
            assert int(c2[b]) == r1 * 64 + cc1
        else:
            r2, cc2 = (int(x) for x in (o2[b] == 2).nonzero()[0])
            assert G.DIRECTIONS[ACTIONS[k]] == ((r2 - r1) // 2, (cc2 - cc1) // 2)
            assert int(c2[b]) == -1


def test_augment_batch_palette_consistent_over_window():
    rng = np.random.default_rng(3)
    B, L = 4, 5
    base = torch.as_tensor(np.random.default_rng(1).integers(0, 16, (64, 64)))
    obs = base.expand(B, L, 64, 64).clone()
    act = torch.full((B, L), A_INDEX["ACTION2"])
    act[:, 0] = -1
    cell = torch.full((B, L), -1)
    avail = torch.ones(B, L, 8, dtype=torch.bool)
    prev = torch.zeros(B, L, 8)
    o2, av2, p2, a2, c2 = G.augment_batch(obs, avail, prev, act, cell, {"palette"}, rng)
    assert torch.equal(a2, act) and torch.equal(av2, avail) and torch.equal(c2, cell)
    for b in range(B):
        for t in range(L):
            assert torch.equal(o2[b, t], o2[b, 0])       # same permutation on every frame
        mapping = {}
        for x, y in zip(base.flatten().tolist(), o2[b, 0].flatten().tolist()):
            assert mapping.setdefault(x, y) == y
        assert len(set(mapping.values())) == len(mapping)  # bijective


def test_augment_batch_window_dihedral_shared_and_unlabelled_kept():
    rng = np.random.default_rng(5)
    B, L = 8, 3
    obs = torch.zeros(B, L, 64, 64, dtype=torch.long)
    obs[:, :, 10, 20] = 3
    act = torch.full((B, L), A_INDEX["ACTION1"])
    act[:, 1] = -1
    cell = torch.full((B, L), -1)
    avail = torch.ones(B, L, 8, dtype=torch.bool)
    prev = torch.zeros(B, L, 8)
    o2, _, _, a2, _ = G.augment_batch(obs, avail, prev, act, cell, {"palette", "dihedral"}, rng)
    assert (a2[:, 1] == -1).all()
    for b in range(B):
        assert torch.equal(o2[b, 0], o2[b, 1]) and torch.equal(o2[b, 0], o2[b, 2])
        assert a2[b, 0] == a2[b, 2]


def test_family_table():
    for f in G.PRIVATE_FAMILIES:                   # palette always; dihedral exactly where licensed
        assert G.augmentations_for(f) == {"palette"} | ({"dihedral"} if f in G.DIHEDRAL_FAMILIES else set())
    assert G.augmentations_for("official.cd82") == {"palette", "dihedral"}
    assert G.augmentations_for("official.bp35") == set()
    assert "dihedral" in G.augmentations_for("toy_reach")
    with pytest.raises(KeyError):
        G.augmentations_for("not_a_family")
    assert G.parse_kinds("none") is None and G.parse_kinds(None) is None
    assert G.parse_kinds("palette,dihedral") == {"palette", "dihedral"}
    with pytest.raises(ValueError):
        G.parse_kinds("flip")
