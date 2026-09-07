"""Device selection: the same code has to run on CUDA, Apple MPS and CPU.

The bug this module exists to prevent was not a crash — it was silence. Nine copies of
`"cuda" if torch.cuda.is_available() else "cpu"` meant every Apple-silicon user got the CPU, with
nothing anywhere saying so, and `RadDinoExtractor(device="cuda")` was a hard crash on a machine
without CUDA rather than a fallback.

Only one of the three targets can be exercised on any given machine, so the interesting tests inject
the availability probes instead of asking the hardware. That is deliberate: it means the MPS
behaviour is checked on this Linux box, on CI, and on a Mac, rather than only where someone happens
to have the silicon.

Run: pytest tests/test_device.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from chevron import device as D

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """A pristine environment per test: the warn-once cache and the memoised probe are process
    state, and would otherwise make these order-dependent."""
    monkeypatch.delenv(D.ENV_DEVICE, raising=False)
    monkeypatch.delenv(D.ENV_AMP, raising=False)
    D._warned.clear()
    D._described = None


def _have(monkeypatch, *, cuda=False, mps=False):
    monkeypatch.setattr(D, "_cuda_ok", lambda: cuda)
    monkeypatch.setattr(D, "_mps_ok", lambda: mps)


# --------------------------------------------------------------------------- automatic selection
def test_apple_silicon_gets_its_gpu(monkeypatch):
    """The whole point: a Mac with Metal must not silently land on the CPU."""
    _have(monkeypatch, mps=True)
    assert D.available_devices() == ["mps", "cpu"]
    assert D.resolve_device() == "mps"


def test_cuda_preferred_over_mps_when_both_somehow_exist(monkeypatch):
    _have(monkeypatch, cuda=True, mps=True)
    assert D.resolve_device() == "cuda"


def test_plain_cpu_box(monkeypatch):
    _have(monkeypatch)
    assert D.available_devices() == ["cpu"]
    assert D.resolve_device() == "cpu"


def test_no_torch_at_all_is_cpu(monkeypatch):
    """The base install is torch-free; asking for a device must answer, not explode."""
    monkeypatch.setattr(D, "_torch", lambda: None)
    assert D.resolve_device() == "cpu"
    assert D.amp_dtype("cpu") is None


# --------------------------------------------------------------------------- impossible requests
def test_requesting_cuda_on_a_mac_degrades_instead_of_crashing(monkeypatch):
    """A project configured on a CUDA box, reopened on a laptop."""
    _have(monkeypatch, mps=True)
    with pytest.warns(RuntimeWarning, match="CUDA is not available"):
        assert D.resolve_device("cuda") == "mps"


def test_requesting_mps_on_linux_degrades(monkeypatch):
    _have(monkeypatch, cuda=True)
    with pytest.warns(RuntimeWarning, match="MPS is not available"):
        assert D.resolve_device("mps") == "cuda"


def test_nonsense_device_falls_back_loudly(monkeypatch):
    _have(monkeypatch)
    with pytest.warns(RuntimeWarning, match="unknown device"):
        assert D.resolve_device("tpu") == "cpu"


def test_a_multi_gpu_index_survives(monkeypatch):
    _have(monkeypatch, cuda=True)
    assert D.resolve_device("cuda:1") == "cuda:1"
    assert D.device_type("cuda:1") == "cuda"


def test_the_warning_is_said_once_not_per_image(monkeypatch):
    """These resolve inside per-image loops; a warning per instance is a broken console."""
    _have(monkeypatch)
    import warnings
    with pytest.warns(RuntimeWarning):
        D.resolve_device("cuda")
    with warnings.catch_warnings(record=True) as later:
        warnings.simplefilter("always")
        for _ in range(5):
            assert D.resolve_device("cuda") == "cpu"
    assert not [w for w in later if issubclass(w.category, RuntimeWarning)], \
        "warned more than once for the same fallback"


# --------------------------------------------------------------------------- the escape hatch
def test_env_forces_cpu(monkeypatch):
    _have(monkeypatch, cuda=True)
    monkeypatch.setenv(D.ENV_DEVICE, "cpu")
    assert D.resolve_device() == "cpu"
    assert D.resolve_device("cuda") == "cpu", "the override must beat a caller asking for CUDA"


def test_env_does_not_drag_deliberate_cpu_work_onto_a_gpu(monkeypatch):
    """`subcluster` and the shape priors pass "cpu" because a tiny net is slower on a GPU than the
    transfer costs. CHEVRON_DEVICE is an escape hatch, not a scheduler."""
    _have(monkeypatch, cuda=True)
    monkeypatch.setenv(D.ENV_DEVICE, "cuda")
    assert D.resolve_device("cpu") == "cpu"
    assert D.resolve_device() == "cuda"


def test_env_can_pick_between_accelerators(monkeypatch):
    _have(monkeypatch, cuda=True, mps=True)
    monkeypatch.setenv(D.ENV_DEVICE, "mps")
    assert D.resolve_device() == "mps"


# --------------------------------------------------------------------------- mixed precision
def test_cpu_never_autocasts(monkeypatch):
    _have(monkeypatch)
    assert D.amp_dtype("cpu") is None
    from contextlib import nullcontext
    assert isinstance(D.autocast_ctx("cpu"), type(nullcontext()))


def test_mps_is_fp32_unless_asked(monkeypatch):
    """fp16 on Metal has shipped real numerical differences, and a pooled embedding goes straight
    into `feats` where a bad block poisons clustering, kNN and the map for the whole project."""
    _have(monkeypatch, mps=True)
    assert D.amp_dtype("mps") is None
    monkeypatch.setenv(D.ENV_AMP, "1")
    import torch
    assert D.amp_dtype("mps") is torch.float16


def test_cuda_picks_bf16_only_where_the_card_supports_it(monkeypatch):
    import torch
    _have(monkeypatch, cuda=True)

    class _FakeCuda:
        @staticmethod
        def is_bf16_supported():
            return False

    class _FakeTorch:
        cuda, float16, bfloat16 = _FakeCuda(), torch.float16, torch.bfloat16

    monkeypatch.setattr(D, "_torch", lambda: _FakeTorch())
    assert D.amp_dtype("cuda") is torch.float16, "pre-Ampere bf16 is emulated and can lose to fp32"
    _FakeCuda.is_bf16_supported = staticmethod(lambda: True)
    assert D.amp_dtype("cuda") is torch.bfloat16


def test_amp_can_be_forced_off_everywhere(monkeypatch):
    _have(monkeypatch, cuda=True)
    monkeypatch.setenv(D.ENV_AMP, "0")
    assert D.amp_dtype("cuda") is None


def test_autocast_targets_the_device_family_not_the_index(monkeypatch):
    """`torch.autocast(device_type=...)` wants "cuda", never "cuda:1"."""
    _have(monkeypatch, cuda=True)
    ctx = D.autocast_ctx("cuda:1")
    assert getattr(ctx, "device", None) in ("cuda", None)


# --------------------------------------------------------------------------- MPS op gaps
REAL_MPS_MESSAGE = ("The operator 'aten::_upsample_bicubic2d_aa.out' is not currently implemented "
                    "for the MPS device. If you want this op to be considered for addition please "
                    "comment on https://github.com/pytorch/pytorch/issues/77764.")


def test_the_real_torch_message_is_recognised():
    assert D.is_unsupported_op_error(NotImplementedError(REAL_MPS_MESSAGE))
    assert D.is_unsupported_op_error(RuntimeError("Could not run 'aten::foo' with MPS backend"))


REAL_MPS_FLOAT64_MESSAGE = ("Cannot convert a MPS Tensor to float64 dtype as the MPS framework "
                            "doesn't support float64. Please use float32 instead.")


def test_metals_missing_float64_counts_as_a_gap():
    """Not a missing kernel but the same consequence: a library handing torch a float64 array cannot
    run on Metal at all, and it arrives as a TypeError rather than an op-gap error."""
    assert D.is_unsupported_op_error(TypeError(REAL_MPS_FLOAT64_MESSAGE))


def test_an_ordinary_bug_is_not_mistaken_for_an_op_gap():
    """The matcher has to be narrow: swallowing our own exceptions and retrying on the CPU would
    turn a real defect into a mystery slowdown."""
    assert not D.is_unsupported_op_error(RuntimeError("shape '[2, 3]' is invalid for input of size 7"))
    assert not D.is_unsupported_op_error(ValueError("no foreground patches in the support set"))
    assert not D.is_unsupported_op_error(RuntimeError("CUDA out of memory"))


def test_a_missing_operator_costs_speed_not_the_run():
    calls, demoted = [], []

    def fn():
        calls.append(1)
        if len(calls) == 1:
            raise NotImplementedError(REAL_MPS_MESSAGE)
        return "ok"

    with pytest.warns(RuntimeWarning, match="does not implement"):
        out = D.run_or_fallback(fn, device="mps", demote=lambda: demoted.append(1), what="a model")
    assert out == "ok" and len(calls) == 2 and demoted == [1]


def test_a_real_failure_still_propagates():
    def fn():
        raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        D.run_or_fallback(fn, device="mps", demote=lambda: pytest.fail("must not demote"))


def test_only_mps_gets_the_fallback():
    """The same message on CUDA is not an MPS op gap and must not be silently retried."""
    def fn():
        raise NotImplementedError(REAL_MPS_MESSAGE)

    with pytest.raises(NotImplementedError):
        D.run_or_fallback(fn, device="cuda", demote=lambda: pytest.fail("must not demote"))


def test_success_costs_nothing():
    assert D.run_or_fallback(lambda: 42, device="mps") == 42


def test_move_to_degrades_rather_than_failing():
    class _M:
        def __init__(self):
            self.where = None

        def to(self, dev, **kw):
            if dev != "cpu":
                raise RuntimeError("MPS backend out of memory")
            self.where = dev
            return self

    with pytest.warns(RuntimeWarning, match="could not place a model"):
        m = D.move_to(_M(), "mps")
    assert m.where == "cpu"


def test_empty_cache_is_safe_to_call_anywhere():
    D.empty_cache()
    D.empty_cache("cpu")
    D.empty_cache("mps")


# --------------------------------------------------------------------------- the extractor wiring
def test_a_demoted_extractor_reruns_with_cpu_inputs():
    """The subtle half of the fallback: retrying is useless unless the inputs are re-placed on the
    NEW device. This asserts the second attempt sees "cpu", not the device that just failed."""
    torch = pytest.importorskip("torch")
    from chevron.extractors.base import HFPatchGridExtractor

    seen = []

    class _Pixels:                       # stands in for the processor output, so no real device is touched
        def __init__(self, t):
            self.t = t
            self.shape = t.shape         # grid_batch reads the input size off it

        def to(self, dev, **kw):
            seen.append(dev)
            return self.t

    class _Model:
        def to(self, dev, **kw):
            return self

    class _Flaky(HFPatchGridExtractor):
        def _load(self):                     # idempotent, like the real one — grid_batch calls it
            if self.model is not None:
                return
            self.model = _Model()
            self.proc = lambda images, return_tensors: {
                "pixel_values": _Pixels(torch.zeros(1, 3, 4, 4))}
            self.device, self._channels_last = "mps", False

        def _forward_tokens(self, px):
            if len(seen) == 1:
                raise NotImplementedError(REAL_MPS_MESSAGE)
            return torch.zeros(1, 5, 8)      # 1 CLS + 4 patches (2x2), C=8

    e = _Flaky("fake/model", "fake", "Fake")

    with pytest.warns(RuntimeWarning):
        g = e.grid_batch([object()])

    assert seen == ["mps", "cpu"], f"retry did not re-place the inputs: {seen}"
    assert e.device == "cpu", "the extractor must stay demoted, not retry the GPU every batch"
    assert tuple(g.shape) == (1, 8, 2, 2)


# --------------------------------------------------------------------------- the standing guard
_BANNED = re.compile(r"""(torch\.)?cuda\.is_available|backends\.mps\.is_available|"""
                     r"""["'](cuda(:\d+)?|mps)["']""")


def test_device_selection_never_gets_copy_pasted_again():
    """The regression that motivated `chevron.device`: nine call sites each deciding for themselves,
    disagreeing, and none of them able to pick MPS. Selection belongs in exactly one module.

    A new call site with a legitimate need should route through `chevron.device`; if some future
    case genuinely cannot, add it here with a reason rather than deleting the guard.
    """
    offenders = []
    for p in sorted((ROOT / "chevron").rglob("*.py")):
        if p.name == "device.py":
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if _BANNED.search(line):
                offenders.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not offenders, ("device selection outside chevron/device.py:\n  " + "\n  ".join(offenders))


def test_the_guard_would_actually_catch_a_regression():
    """A guard nobody has seen fail is not a guard."""
    assert _BANNED.search('dev = "cuda" if torch.cuda.is_available() else "cpu"')
    assert _BANNED.search('model.to("mps")')
    assert _BANNED.search("sam.to('cuda')")
    assert not _BANNED.search("# CUDA and MPS are both supported")       # prose stays legal


def test_starting_the_server_does_not_import_torch(tmp_path):
    """Probing a device means importing torch. `/api/state` is POLLED, so putting the device there
    would make every session — including one that only reviews and exports an existing project —
    pay a multi-second torch import it never needed. The answer is available from `/api/device`,
    which is asked for rather than pushed.

    A subprocess because torch is long since imported inside this test session.
    """
    import subprocess
    import sys
    script = f"""
import sys
from pathlib import Path
from chevron.engine import CuratorEngine
from chevron.server import create_app
from fastapi.testclient import TestClient
eng = CuratorEngine(Path({str(tmp_path / "p")!r})); eng.init_project({{}})
app = create_app(engine=eng)
assert "torch" not in sys.modules, "create_app imported torch"
c = TestClient(app)
assert c.get("/api/state").status_code == 200
assert "torch" not in sys.modules, "/api/state imported torch"
body = c.get("/api/device").json()
assert "device" in body and "available" in body
print("OK")
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                       cwd=str(ROOT))
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_describe_answers_without_a_gpu(monkeypatch):
    _have(monkeypatch)
    d = D.describe(refresh=True)
    assert d["device"] == "cpu" and d["available"] == ["cpu"] and d["amp"] is None
    assert set(d) == {"device", "type", "available", "amp", "torch", "override"}
