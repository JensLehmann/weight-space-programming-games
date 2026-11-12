"""Per-game redaction configs for the official public ARC3 games (condition M; `wsp.redact_public`).

Every entry is an explicit, reviewed edit with a one-line reason (`why`). The generic passes in
`wsp.redact_public` (G1 `levels` entries, G2 sprite-library pruning, G3 held-level comments) run for
every game; the entries below add what the generic passes cannot know. Edit kinds:

  Items(target, ...)     a list/tuple/dict literal with one entry per level (module-level name, or
                         "Class.method:var" for an assignment inside a method): the entries of held
                         levels become WITHHELD. `key` maps a 0-based level to the dict key (dicts),
                         `extra` lists further keys/indices to withhold (non-level entries);
                         `omit=True` drops dict entries (with a comment) where code iterates the dict.
  Branch(func, test)     the `if`/`elif` whose test is `test` (compared after ast.unparse) inside
                         `func` ("Class.method"): its body becomes `raise LookupError(...)`
                         (action="raise") or the whole branch is removed (action="drop", elif only).
  Body(func)             the whole body of `func` ("Class.method") becomes `raise LookupError(...)`
                         (code whose only entry point is withheld).
  Sub(old, new, n)       exact text substitution on the redacted text (applied after the AST edits),
                         expected to match exactly n times.
  Keep(pattern)          sprite-library keys matching the regex are kept although G2 found no use.
  Drop(pattern)          sprite-library keys matching the regex are omitted although code names them
                         (only as a name/tag in comparisons, never as a library lookup) and no support
                         level places them.
  Allow(pattern)         a static-check hit (held-level number in a string/comment) matching the regex
                         is accepted.
  Note()                 a judgement call that needs no edit (kept on purpose).
  LevelExpr(src, base)   an expression that holds the level number (base 0 or 1), so that the audit
                         can find comparisons/subscripts that gate on it.
  Library(name)          a further module-level dict of templates pruned by the G2 rule.

Level numbers in `why` texts are 1-based (L1 = level index 0).
"""
from __future__ import annotations

from typing import Callable, NamedTuple


class Items(NamedTuple):
    target: str
    why: str
    key: Callable[[int], object] | None = None      # 0-based level -> dict key / list index
    extra: tuple = ()                                # further keys / indices to withhold
    extra_why: str = ""
    omit: bool = False                               # dicts only: drop the entries instead of WITHHELD


class Branch(NamedTuple):
    func: str
    test: str
    why: str
    action: str = "raise"
    n: int = 1


class Body(NamedTuple):
    func: str
    why: str


class Sub(NamedTuple):
    old: str
    new: str
    why: str
    n: int = 1


class Keep(NamedTuple):
    pattern: str
    why: str


class Drop(NamedTuple):
    pattern: str
    why: str


class Allow(NamedTuple):
    pattern: str
    why: str


class Note(NamedTuple):
    why: str


class LevelExpr(NamedTuple):
    src: str
    base: int
    why: str


class Library(NamedTuple):
    name: str
    why: str


RAISE = "raise LookupError('level withheld for evaluation')"


def _grid(level0: int) -> str:
    return f"grid{level0 + 1}"


LP85_LEVEL_NAMES = ["kdrsqrvpwb", "cecdsipmha", "vvnbesozdj", "ihpokmcrjm", "qibynbipmw", "ksgcbuhgwz",
                    "yzzznxxvju", "hmzsasouat"]           # Level.data["level_name"] of L1..L8

CONFIGS: dict[str, list] = {
    # ------------------------------------------------------------------------------------------
    "ar25": [
        Branch("Ar25.step", "self.level_index == 1 and self.hsiusrsrdkswnt == 0 and "
                            "(self.lelsvjlwneo.current_steps < 50) and (not self.qehjebksqcm)",
               "L2-only tutorial trigger (flash a glyph at the piece once <50 steps remain): held-level hint"),
        Sub("if self.level_index == 1 and self.hsiusrsrdkswnt == 0 and (self.lelsvjlwneo.current_steps < 50) and "
            "(not self.qehjebksqcm):", "if self.level_index == 1:",
            "the rest of the withheld trigger's test describes when the L2 hint fires"),
        Branch("Ar25.step", "self.hsiusrsrdkswnt > 0",
               "the glyph-flash animation that only the L2 trigger starts: held-level hint code"),
        Note("glyph sprite 0004afwyadxelg kept: on_set_level clones it into every level off-screen (x=500), "
             "where it is inert; only its (withheld) L2 trigger ever shows it"),
        Note("StepCounter budgets and the unused '0003uqrdzdofso' data key live in the Level entries, so held "
             "values go with G1"),
        Note("latent mechanics (tags 'reflect_horizontal_only', '0038pnuzypawco', rotate-on-distance tags "
             "0040/0044) kept: generic rules, no level places such pieces"),
        Note("fixed-mirror tag 0056icpryeujyf kept: generic rule; L1 (support) uses it"),
    ],
    # ------------------------------------------------------------------------------------------
    "bp35": [
        LevelExpr("self.qswcochjodb", 1, "1-based level number (= _current_level_index + 1, set in on_set_level)"),
        LevelExpr("self.qivnakklopy.qswcochjodb", 1, "same, read from the HUD"),
        LevelExpr("self.oztjzzyqoek.qswcochjodb", 1, "same, set in Bp35.on_set_level"),
        Items("tjdtolkmxo", "real layouts grid1..grid9 (the `levels` entries are 1x1 dummies), selected by "
                            "f'grid{level}': held grids -> WITHHELD", key=_grid, extra=("grid10",),
              extra_why="grid10 is an unused 10th layout (the game has 9 levels): not support data, withheld"),
        Note("tile library ymmwcccrhb kept whole: generic tile art (walls, blocks, spikes, player, gem), looked "
             "up through grid legends; not level layouts"),
        Note("rising hazard gate `qswcochjodb > 3` (hazard only in L1-3) kept: gated on support levels L1-L2"),
        Note("step budget `qswcochjodb <= 6` -> 64 else 128 kept: both bands contain support levels (L1-6, L8)"),
        Branch("qmjscfjptx.render_interface", "self.qivnakklopy.qswcochjodb == 10",
               "dead HUD branch (192-step budget bar) keyed on the withheld unused grid10 layout"),
        Note("budget bands (`<= 6`: 64, else: 128) and the rising-hazard band (`> 3` off) kept: each band contains "
             "support levels; they imply held L3/L5 have 64 steps, L7/L9 128, and that L3 has the hazard"),
        Note("dead BFS helper klmsuijofik.nqybsxhxoy (never called, flood fill) kept: not a planner"),
    ],
    # ------------------------------------------------------------------------------------------
    "cd82": [
        Note("per-level target images eoqnvkspoa-pqwme{2,4,6}-1 are omitted by G2 (only held levels place them)"),
        Note("nozzle mechanic (enabled by sprite presence, L3-6) kept: support levels L3, L5 use it"),
        Note("unused click-enumeration helpers yrfgxhebei / bmwcxxvjum kept: candidate clicks, not solutions"),
    ],
    # ------------------------------------------------------------------------------------------
    "cn04": [
        Note("each piece sprite is used in one level; held-level pieces are omitted by G2"),
        Note("GreyMasking (L3-6) and co-located variant stacks (L5-6) kept: data-flag/layout mechanics that "
             "support levels L3, L5 use"),
    ],
    # ------------------------------------------------------------------------------------------
    "dc22": [
        Branch("Dc22.on_set_level", "self.level_index == 5",
               "L6-only hack (teleporter at (18,48) becomes colour-cyclable, joins group 'd'): held-level code"),
        Keep(r"^(refgps-plelvb|tovemc-plelvb|brixto|tewfut|tovemc-plelvb-p-|tovemc-plelvb_plong_|"
             r"drfztmbrixto-|moxubw-plelvb-)\d$",
             "state variants: on_set_level counts `prefix+str(k) in sprites` (k=2..8) for every digit-named "
             "'tovemc' sprite of the level, and button presses clone them; prefixes computed from L1/L3/L5"),
        Keep(r"^crzsjq-\d$", "crane animation frames f'{crane}-{k}' (crane 'crzsjq', L5)"),
        Note("crane mode 'brixtocrzsjq' (track-following crane; only L6 places it, sprite omitted by G2) kept: "
             "a generic rule selected by sprite name, not by level index"),
        Branch("Dc22.step", "iophjflwsn in gpttjaogqn.tags",
               "colour-cycle button: only L6 places such a button and only the L6 hack makes a teleporter "
               "cyclable, so this path exists only for a held level"),
        Body("Dc22.kiwniaoeyu", "teleporter colour cycling, reachable only from the withheld branch above"),
        Body("Dc22.mzuiagpcmy", "next colour of the L6 cycle; only kiwniaoeyu calls it"),
        Body("Dc22.qiwzdkqjja", "membership in the L6 colour cycle; only kiwniaoeyu calls it"),
        Sub('awhuyiogsr = ["tewfutpibpar", "tewfutrefgps", "tewfutyefmyf", "tewfutblrmbx"]',
            "awhuyiogsr = WITHHELD", "the L6 teleporter colour-cycle order (used only by the withheld code)"),
        Sub('qiukbrokfa = "tewfut-color-cycle"', "qiukbrokfa = WITHHELD",
            "tag name of the L6-only cyclable teleporter (now used only by withheld code)"),
        Sub('iophjflwsn = "tewfut-color-buezna"', "iophjflwsn = WITHHELD",
            "tag name of the L6-only colour-cycle button; its one remaining use (`in tags`) is always False "
            "on support levels, as before"),
        Note("normal teleporter pairing pmmrygpdgl kept: support L3/L5 teleporters use it"),
        Note("pressure plates (njvd-rolo) and key pickups (piyqze) kept: generic rules enabled by sprite presence"),
        Note("L1 tutorial outline sprite-12 (tag 'aybe') kept: tutorial of a support level"),
        Note("L6 legend icon renrjo-buezna is omitted by G2 (only L6 places it)"),
    ],
    # ------------------------------------------------------------------------------------------
    "ft09": [
        Note("tutorial gates `level_index == 0` (hint frame UEq, worked examples) kept: L1 is a support level"),
        Note("level-name comments ('# hxv', '# oea', '# DFx') of held levels go with G1 (comment above entry)"),
        Note("glyph tiles NTi/ZkU (L5 INW support, L6 DFx held) kept: generic rule"),
    ],
    # ------------------------------------------------------------------------------------------
    "g50t": [
        Note("levels are pure sprite placements; no level-index code exists"),
    ],
    # ------------------------------------------------------------------------------------------
    "ka59": [
        Keep(r"^explode-6-\d$", "bomb explosion frames looked up as f'explode-{width}-{index}'; support L6 has "
                                "only 6-wide bombs, so the 3-wide frames (only held-level bombs) are omitted"),
        Note("per-level wall and outer-wall sprites of held levels are omitted by G2; the support levels' "
             "wall sprites (most of the remaining text) are support-level data"),
        Note("dead 'Enemy' chaser code (no sprite has the tag) kept: generic, unused"),
    ],
    # ------------------------------------------------------------------------------------------
    "lf52": [
        LevelExpr("self.whtqurkphir", 1, "1-based level number (= _current_level_index + 1)"),
        LevelExpr("self.ikhhdzfmarl.whtqurkphir", 1, "same, read from Lf52"),
        Items("kciatvszkc", "real boards grid1..grid10 (the `levels` entries are dummies), selected by "
                            "f'grid{level}': held boards -> WITHHELD", key=_grid),
        Note("tile library lebqfosqjk kept whole: generic tile art looked up through board legends and tile "
             "properties (.get(name)), so per-tile use cannot be established statically; not level layouts"),
        Branch("equnaohchtj.qjwmwkhrml", "self.whtqurkphir == 2", "L2 initial board offset"),
        Branch("equnaohchtj.qjwmwkhrml", "self.whtqurkphir == 10", "L10 initial board offset"),
        Branch("equnaohchtj.tmhxwcojkh", "self.whtqurkphir == 4 and i.grid_y >= 11", "L4 platform-scroll case"),
        Sub("elif self.whtqurkphir == 4 and i.grid_y >= 11:", "elif self.whtqurkphir == 4:",
            "the extra test names an L4 board row"),
        Branch("equnaohchtj.tmhxwcojkh", "self.whtqurkphir == 6", "L6 platform-scroll vector"),
        Branch("equnaohchtj.tmhxwcojkh", "self.whtqurkphir == 8", "L8 platform-scroll vector"),
        Sub("elif self.whtqurkphir == 7 or self.whtqurkphir == 10:", "elif self.whtqurkphir == 7:",
            "shared L7/L10 scroll case: L7 (support) kept, L10 (held) withheld below"),
        Sub(f"elif self.whtqurkphir == 8:\n                                {RAISE}\n",
            f"elif self.whtqurkphir == 8:\n                                {RAISE}\n"
            f"                            elif self.whtqurkphir == 10:\n                                {RAISE}\n",
            "L10 scroll case withheld; placed after L8 so its position does not suggest L7's behaviour"),
        Branch("equnaohchtj.cfilhtifcb", "self.whtqurkphir == 2", "L2 dead-end landing cells"),
        Branch("equnaohchtj.cfilhtifcb", "self.whtqurkphir == 4", "L4 scripted camera pans"),
        Branch("equnaohchtj.cfilhtifcb", "self.whtqurkphir == 6", "L6 dead-end cell and camera pans"),
        Sub("self.whtqurkphir in [6, 7] and ddaguepwkt == 2", "self.whtqurkphir == 7 and ddaguepwkt == 2",
            "win rule 'two pegs may remain' is index-keyed for L6 (held) and L7 (support): L6 removed"),
        Note("board-offset default (5,5) and scroll `else` vector kept: they now cover only support levels "
             "(L5/L7/L9 and L9)"),
        Note("blue-peg exclusion `whtqurkphir >= 8` kept: band contains support level L9"),
        Note("scroll gate `whtqurkphir <= 2` (no scroll) kept: band contains support level L1"),
        Note("move budgets by band (L1: 64, `< 6`: 320, `>= 6`: 640) kept: every band contains a support level"),
        Note("L1 hint circles (sgxkqallyv) and the L1/L3 dead-end tables and L5/L7/L9 camera pans kept: "
             "support-level content the game shows on screen (needed for equivalence)"),
        Note("unseeded np.random.shuffle (cosmetic dissolve) kept: mechanics"),
    ],
    # ------------------------------------------------------------------------------------------
    "lp85": [
        Items("izutyjcpih", "track tables keyed by Level.data['level_name']: held levels' tables -> WITHHELD",
              key=lambda l: LP85_LEVEL_NAMES[l], extra=("asymmetry", "concentric", "metronome"),
              extra_why="three tables no level selects (developer prototypes): not support data, withheld",
              omit=True),
        Note("izutyjcpih entries are omitted rather than WITHHELD: qfvvosdkqr preprocesses every table at "
             "start-up (iterates the dict)"),
        Note("level-progress pips (len(levels), level_index) kept: UI only"),
    ],
    # ------------------------------------------------------------------------------------------
    "ls20": [
        Note("L1 tutorial gate `level_index > 0` (lock-match hint, skipped budget tick) kept: L1 is a support level"),
        Note("Fog rule (Level.data['Fog'], only L7 sets it) kept: generic rule; the flag itself goes with G1"),
        Note("changer patrols and pushers kept: generic rules enabled by layout (L5-7, L3-7; support L4, L6)"),
    ],
    # ------------------------------------------------------------------------------------------
    "m0r0": [
        Allow(r"^wahtyt-Level\d+$", "maze sprite names carry shuffled numbers unrelated to the level order "
                                    "(L1 uses 'Level6', L5 'Level2'); the kept ones are placed by support levels"),
        Drop(r"^pikgci-boweok-(leklkn|rivmdg)$", "vertical-mirror token templates no level places; code only "
                                                 "lists their names"),
        Drop(r"^spswjz$", "hazard tile placed only by L2/L4/L6 (held); code matches it by name "
                          "(get_sprites_by_name/get_sprite_at), never looks the template up"),
        Note("hazard mechanic (name 'spswjz') kept: generic rule"),
        Note("unused maze sprites are omitted by G2"),
    ],
    # ------------------------------------------------------------------------------------------
    "r11l": [
        Note("hazard blobs, colour pickups and decoy targets are enabled by sprite presence/name prefix; held-only "
             "sprites are omitted by G2, the generic rules stay"),
        Note("unused click lattice jtqexauuzid kept: candidate clicks, not a solution"),
    ],
    # ------------------------------------------------------------------------------------------
    "re86": [Note("levels are sprite placements plus StepCounter; no level-index code exists")],
    "s5i5": [
        Drop(r"^0058exfukigofv$", "arm piece placed only by L4 (held); support L7's data names it only in a "
                                  "dangling Children link (no-op); code never looks it up"),
        Note("rotate buttons (L6-8) and Children links (L2-8) kept: generic rules used by support L7, L3-7"),
    ],
    "sb26": [Note("indicator choice `len(targets) < 12` kept: cosmetic, keyed on a count, not a level index")],
    # ------------------------------------------------------------------------------------------
    "sc25": [
        Drop(r"^crzdcq$", "obstacle template no level places; code only compares sprite names with it"),
        Note("hint gate `_current_level_index < 3` (paint the first spell's pattern) kept: band contains "
             "support levels L1, L3"),
        Note("L1 demo animation `_current_level_index == 0` kept: L1 is a support level"),
        Note("spell pattern table zzpoabuniyn kept: generic rules; every spell is castable in support level L5 "
             "and the spellbook icons show the patterns in every level"),
        Note("unused helpers (wxohoxzlfh auto-fill, bmmtkvkbcdd click list, debug strings) kept: not solutions "
             "of any level"),
    ],
    "sk48": [Note("the only level-index use re-clones the level on entry")],
    "sp80": [Note("display rotation and budget come from Level.data (held values go with G1)")],
    # ------------------------------------------------------------------------------------------
    "su15": [
        Note("L1 tutorial gates `level_index == 0` (crosshair, guide path) kept: L1 is a support level"),
        Note("feasibility test xspiwkmfbs kept: it recolours the goal zones (visible mechanic), not a solver"),
        Note("candidate click lattice self.actions kept: not used by the rules, not a solution"),
    ],
    # ------------------------------------------------------------------------------------------
    "tn36": [
        Drop(r"^laycmu(ofkkgm|ommm)$", "toggling-hazard templates placed only by L7 (held); code matches their "
                                      "tag, never looks the templates up"),
        Note("budget doubling `level_index >= 5` kept: band contains support level L6"),
        Note("L1 pre-filled program kept: support-level content"),
        Note("demo-tab programs are Level.data (held values go with G1)"),
    ],
    # ------------------------------------------------------------------------------------------
    "tr87": [
        Items("Tr87.on_set_level:ofysoutulp", "per-level RNG seeds for the glyph rotations: held -> WITHHELD"),
        Items("Tr87.on_set_level:ripmydnety", "per-level RNG seeds for the start scramble: held -> WITHHELD"),
        Sub("self.vfpimnmtnta = 128 if self._current_level_index <= 4 else 256",
            "self.vfpimnmtnta = 128 if self._current_level_index <= 4 else WITHHELD",
            "the `else` budget applies only to L6 (held)"),
        Keep(r"^nxkictbbvzt[ABC][1-7]$", "glyph tiles are cycled by name arithmetic (wpbnovjwkv)"),
        Keep(r"^qvtymdcqear[1-3]$", "cursor brackets looked up as f'qvtymdcqear{len}'"),
        Note("rule flags alter_rules (L5-6), double_translation (L4, L6) and tree_translation (L6) kept: "
             "generic rules switched by Level.data; double/tree translation are exercised only by held levels"),
        Note("dead link-marker code (tjaqvwdgkxe) kept: no level uses it"),
    ],
    # ------------------------------------------------------------------------------------------
    "tu93": [Note("one maze sprite per level; held mazes are omitted by G2")],
    "vc33": [Note("gravity special case `Gravity[0] == -3` (only L6, support) kept")],
    "wa30": [
        Note("NPC BFS routines (czrprbohhe, cyjrduhzmz, zauouvdhta, egqayvffim) kept: they are the NPCs' "
             "transition rule, called every step; removing them changes the mechanics"),
    ],
}


def config_for(game_id: str) -> list:
    return CONFIGS.get(game_id, [])
