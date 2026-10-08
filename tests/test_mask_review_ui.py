"""The Refine pane in a real browser: the re-mask review queue (pick / keys / accept), the one-instance
focus, and "apply to others" (sampled preview gates the commit). Skips without Playwright + Chromium."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from test_shape_transfer import _circle, _engine   # noqa: pytest puts tests/ on sys.path


def _rle(m):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(m.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii")
    return r


def _with_candidates(eng, iuids, *, backend="sam_auto", box_only=()):
    """What remask_instances leaves behind: the best candidate applied, the alternatives queued."""
    for u in iuids:
        orig = eng._mask(u)
        a, b = _circle(r=12), _circle(r=26)
        eng._mask_cands[u] = {"backend": backend, "original": _rle(orig), "reviewed": False,
                              "original_box_only": u in box_only,
                              "cands": [{"rle": _rle(a), "score": 0.9, "by": backend.split("+")},
                                        {"rle": _rle(b), "score": 0.6, "by": backend.split("+")[-1:]}]}
        eng._set_mask_nohist(u, a, op={"name": "remask", "kw": {"backend": backend}})


@pytest.fixture(scope="module")
def ui(tmp_path_factory):
    sync_api = pytest.importorskip("playwright.sync_api")
    uvicorn = pytest.importorskip("uvicorn")
    from chevron.server import create_app
    eng, order = _engine(tmp_path_factory.mktemp("rf"), [_circle(), _circle(cx=40, cy=40), _circle(cx=90, cy=90)])
    eng.assign(order, "A")
    _with_candidates(eng, order[:2], box_only={order[1]})
    srv = uvicorn.Server(uvicorn.Config(create_app(engine=eng), port=8797, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    pw = sync_api.sync_playwright().start()
    try:
        browser = pw.chromium.launch()
    except Exception as e:                              # browsers not installed
        pw.stop(); srv.should_exit = True
        pytest.skip(f"chromium unavailable: {e}")
    page = browser.new_page(viewport={"width": 1500, "height": 950})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://127.0.0.1:8797/"); page.wait_for_timeout(800)
    page.evaluate("document.querySelector('nav button[data-tab=\"refine\"]').click()")
    page.wait_for_function("RF.cur !== ''")
    yield eng, order, page, errors
    browser.close(); pw.stop(); srv.should_exit = True


def _settle(page, ms=400):
    page.wait_for_timeout(ms)


def test_it_opens_on_the_review_queue_with_every_choice_original_first(ui):
    eng, order, page, errors = ui
    assert page.evaluate("RF.src") == "review" and page.evaluate("RF.items.length") == 2
    assert page.evaluate("RF.cur") == order[0]
    figs = page.locator("#mcView figure")
    assert figs.count() == 3 and "original" in figs.nth(0).inner_text()
    assert "now" in figs.nth(1).inner_text()                    # candidate 1 is the applied one
    assert page.is_visible("#rpCandsSec") and page.text_content("#rpFixT") == "Or fix it"
    assert not errors


def test_a_number_key_picks_but_never_while_typing(ui):
    eng, order, page, _ = ui
    page.click('#rpMode button[data-mode="recipe"]')
    page.focus("#rfParams .rfp"); page.keyboard.press("2"); _settle(page)
    assert eng._mask_cands[order[0]]["reviewed"] is False           # typed into a field: no pick
    page.focus("body"); page.evaluate("document.activeElement.blur()")
    page.keyboard.press("2"); _settle(page, 700)
    assert eng._mask_cands[order[0]]["reviewed"] is True and eng._current_candidate(order[0]) == 2
    assert page.evaluate("RF.cur") == order[1]                        # moved on
    assert "box only" in page.inner_text("#mcView")                   # a box-only original says so


def test_enter_accepts_marks_reviewed_and_keeps_the_mask(ui):
    eng, order, page, _ = ui
    before = eng._mask(order[1]).copy()
    page.keyboard.press("Enter"); _settle(page, 700)
    assert eng._mask_cands[order[1]]["reviewed"] is True
    assert np.array_equal(eng._mask(order[1]), before)
    assert "End of the queue" in page.text_content("#rfHint") or page.evaluate("RF.items.length") == 0


def test_group_queue_and_apply_to_others_is_gated_by_a_sampled_preview(ui):
    eng, order, page, _ = ui
    page.evaluate(f"rfOpen('{order[2]}')"); page.click('#rqSrc button[data-src="group"]'); _settle(page, 700)
    assert page.evaluate("RF.items.length") == 3 and "class A" in page.text_content("#rqInfo")
    page.click('#rpMode button[data-mode="recipe"]')
    page.select_option("#rfOp", "dilate"); page.fill('#rfParams .rfp[data-k="max_contrast"]', "1")
    page.click("#rfAdd"); _settle(page)
    assert page.text_content("#rpAccept").startswith("Apply dilate")
    page.click("#rpApply summary"); page.select_option("#raScope", "group")
    assert page.is_disabled("#raApply")                               # preview first
    page.click("#raPreview"); page.wait_for_selector("#rsSample figure")
    assert page.locator("#rsSample figure").count() == 2               # the other two members
    assert not page.is_disabled("#raApply") and "Apply to 2" in page.text_content("#raApply")
    page.select_option("#raScope", "similar"); page.fill("#raTau", "0.3")
    assert page.is_disabled("#raApply")                               # changed settings: preview again
    page.select_option("#raScope", "group")
    assert not page.is_disabled("#raApply")
    page.once("dialog", lambda d: d.accept())
    page.click("#raApply"); _settle(page, 800)
    assert all(eng.state.meta[u].rule_ops and eng.state.meta[u].rule_ops[0]["name"] == "dilate"
               for u in (order[0], order[1]))
    assert "Applied to 2" in page.text_content("#raSummary")
