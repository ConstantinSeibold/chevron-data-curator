"""Every pane can be scrolled to its bottom.

`body` is `overflow:hidden` — the shell is a fixed frame, and each pane is expected to bring its
own scroller. That works while the pane fits: a `.grid` or `#plist` inside it takes the slack and
scrolls. It stops working on a short window, where the fixed chrome (toolbars that wrap, notes,
the button rows under a log) is taller than the frame on its own. The pane then overflowed with
`overflow:visible` and nothing above it scrolled either, so the controls at its bottom — "Adopt
trained model", the release pager, the classifier's reject row — were unreachable at any window
size, with no scrollbar to hint that anything had been cut.

These are CSS-level assertions on index.html because the layout has no browser in the test rig;
they pin the two rules that together make the overflow reachable rather than clipped.

Run: pytest tests/test_pane_scroll.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "chevron" / "web" / "index.html"


@pytest.fixture(scope="module")
def css() -> str:
    html = INDEX.read_text()
    return html[html.index("<style>"):html.index("</style>")]


def rule(css: str, selector: str) -> str:
    """The declaration block of the first rule whose selector list starts with `selector`."""
    m = re.search(r"(?m)^\s*" + re.escape(selector) + r"\s*(?:,[^{]*)?\{([^}]*)\}", css)
    assert m, f"no rule for {selector!r}"
    return m.group(1)


def test_the_frame_is_still_fixed(css):
    """The premise. If body ever scrolls, the pane rules below stop being the only way out."""
    assert "overflow:hidden" in rule(css, "body")


def test_a_pane_shrinks_with_the_frame_and_scrolls_its_overflow(css):
    pane = rule(css, ".pane")
    assert "min-height:0" in pane, "min-height:auto lets the pane push its tail past the frame"
    assert "overflow:auto" in pane, "a pane taller than the frame needs a scroller of its own"


@pytest.mark.parametrize("selector", [".grid", ".rgrid", ".mcards", "#plist"])
def test_scroll_regions_keep_a_floor(css, selector):
    """Without a floor a region collapses to nothing on a short window: the pane then fits, so the
    pane scroller never appears, and the region shows zero rows with no way to reach them."""
    body = rule(css, selector)
    assert "min-height:var(--pane-floor)" in body, f"{selector} would collapse instead of overflowing"


def test_the_sub_nav_scrolls_sideways(css):
    """15 tab buttons in one row: on a narrow window they used to be squeezed off the edge."""
    assert "overflow-x:auto" in rule(css, "nav")
    assert "flex:0 0 auto" in rule(css, "nav button")


def test_the_area_rail_scrolls(css):
    assert "overflow-y:auto" in rule(css, ".areas")
