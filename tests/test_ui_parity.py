"""Functional parity across the P3 restructure: no control may vanish silently.

The 15-tab UI had 213 interactive controls. `ui_migration_map.py` records where each one goes. This
test holds the invariant at EVERY point of the strangler migration, while some views are new and
others are still the old panes:

    a legacy control is either still present, or its named replacement is present — never neither.

A control with no `target_id` yet must still exist, so a pane cannot be deleted before its
replacement has been named. Retiring is allowed; it just has to be written down.

Run: pytest tests/test_ui_parity.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

from ui_migration_map import DISPOSITION, GAINED, LEGACY_CONTROLS  # noqa: pytest puts tests/ on sys.path

WEB = Path(__file__).resolve().parents[1] / "chevron" / "web"
ELEMENT_ID = re.compile(r'\bid="([^"]+)"')


def _present_ids() -> set[str]:
    """Ids present anywhere the browser will see them: the served pages, plus ids the JS injects."""
    ids: set[str] = set()
    for p in list(WEB.glob("*.html")) + [p for p in WEB.rglob("*.js") if "vendor" not in p.parts]:
        ids |= set(ELEMENT_ID.findall(p.read_text()))
    return ids


def test_every_legacy_control_has_a_disposition():
    """Guards the decision, not the code: a control cannot be quietly forgotten about."""
    missing = sorted(LEGACY_CONTROLS - set(DISPOSITION))
    assert not missing, f"legacy controls with no recorded disposition: {missing}"


def test_no_control_is_lost_in_transit():
    """THE parity invariant. Fails the moment a control disappears without a live replacement."""
    present = _present_ids()
    lost = []
    for legacy, (kind, target, dest, note) in DISPOSITION.items():
        if legacy in present:
            continue                                    # not migrated yet — fine at any point
        if target and target in present:
            continue                                    # migrated to its named replacement — fine
        lost.append(f"{legacy} ({kind} -> {dest}"
                    + (f", target {target!r} not found" if target else ", no target named") + ")")
    assert not lost, (
        "controls removed from the UI with no live replacement:\n  " + "\n  ".join(sorted(lost)))


def test_named_targets_are_real_once_declared():
    """A `target_id` is a promise that an element by that name exists. Catches a typo'd target,
    which would otherwise let the parity check pass against nothing."""
    present = _present_ids()
    bad = sorted({t for _, (_, t, _, _) in ((k, v) for k, v in DISPOSITION.items())
                  if t and t not in present})
    assert not bad, f"declared migration targets that do not exist in the UI: {bad}"


def test_merged_controls_declare_where_they_collapse_to():
    """MERGED is the risky disposition — 48 controls collapse into ~14. Each must say which."""
    vague = sorted(k for k, (kind, _, dest, _) in DISPOSITION.items() if kind == "MERGED" and not dest)
    assert not vague, f"MERGED controls with no destination recorded: {vague}"


def test_gained_controls_are_documented():
    """The three endpoints that had no UI. Once built, they must be real elements."""
    present = _present_ids()
    for cid, why in GAINED.items():
        assert why, f"{cid} must record why it exists"
        if cid in present:
            assert "/api/" in why or "-" in why


def test_counts_match_the_audit():
    """A snapshot of the audit, so an accidental edit to the map is visible."""
    from collections import Counter
    c = Counter(kind for kind, _, _, _ in DISPOSITION.values())
    assert len(DISPOSITION) == 213
    assert c == {"MOVED": 162, "MERGED": 48, "BECOMES": 3}
    assert "REMOVED" not in c, "nothing was removed; if that changes, say so explicitly here"
