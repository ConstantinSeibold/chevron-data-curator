"""Who is acting, and through which view: the per-request context of a shared engine.

One `CuratorEngine` serves everyone who has its project open. The DATA it holds (labels, taxonomy,
masks, the clustering, the classifier) is shared on purpose. What each person is LOOKING at is not:
the ingest scope, the facet filters, the FINCH level and the sub-cluster under study. Those used to
be plain engine attributes, so one person narrowing their view narrowed everyone's.

They now live on a `ViewState`, and the engine reaches them through `ViewAttr` descriptors that read
the view of whoever is making the current request. The server binds that per request with `use()`.
Outside any request (tests, scripts, local single-user mode) there is no context, and the engine
falls back to its own default view: exactly the old behaviour.

Scope tokens come from one process-wide counter, so two views can never share a token while
showing different things; caches keyed on the token stay correct with any number of views.
"""
from __future__ import annotations

import contextlib
import itertools
import weakref
from contextvars import ContextVar
from dataclasses import dataclass, field, replace

_TOKENS = itertools.count(1)
_REVS = itertools.count(1)


_INTERNED: dict = {}


def new_scope_token() -> int:
    return next(_TOKENS)


def scope_token_for(key) -> int:
    """One token per distinct narrowing (0 = none), so two views showing the same thing share caches and
    a view that returns to a narrowing finds what it left there."""
    if key is None:
        return 0
    tok = _INTERNED.get(key)
    if tok is None:
        tok = _INTERNED.setdefault(key, next(_TOKENS))
    return tok


@dataclass
class ViewState:
    scope_bids: set | None = None
    scope_id: str | None = None
    scope_token: int = 0                       # 0 = nothing narrowed; else unique process-wide
    source_filter: set | None = None
    method_filter: set | None = None
    granularity_filter: set | None = None
    modality_filter: set | None = None
    subcluster: dict | None = None             # within-class substructure under study
    levels: dict = field(default_factory=dict)   # clustering rev -> the FINCH level this view shows


class StaleCluster(Exception):
    """The client acted on a partition of a clustering that has since been replaced."""


@dataclass
class Ctx:
    actor: str | None = None
    views: "weakref.WeakKeyDictionary | None" = field(default_factory=weakref.WeakKeyDictionary)
    cluster_rev: str | None = None             # "<rev>:<level>" the client's partition ids refer to
    session: object = None                     # the server's session record (owns jobs, open project)
    user: object = None                        # the signed-in chevron.auth.User (hosted mode)
    project: str | None = None                 # the project the requesting TAB shows (?p= / header)
    masks: bool = True                         # the client's mask switch ('m'): False draws previews bare


_CTX: ContextVar[Ctx | None] = ContextVar("chevron_ctx", default=None)


def current() -> Ctx | None:
    return _CTX.get()


def masks_shown() -> bool:
    """Whether previews drawn for this request carry their masks. One switch in the client hides
    them in every view at once, so the server reads it here instead of per endpoint."""
    c = _CTX.get()
    return c.masks if c else True


@contextlib.contextmanager
def masks(on: bool):
    """Override the mask switch for a block (an endpoint that takes it as an explicit parameter)."""
    c = _CTX.get()
    token = _CTX.set(replace(c, masks=bool(on)) if c else Ctx(views=None, masks=bool(on)))
    try:
        yield
    finally:
        _CTX.reset(token)


def actor() -> str | None:
    c = _CTX.get()
    return c.actor if c else None


@contextlib.contextmanager
def use(ctx: Ctx | None):
    token = _CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CTX.reset(token)


def view_of(engine) -> ViewState:
    c = _CTX.get()
    if c is None or c.views is None:           # no request, or single-user mode: the engine's own view
        return engine._default_view
    v = c.views.get(engine)
    if v is None:
        v = c.views[engine] = ViewState()
    return v


class ViewAttr:
    """An engine attribute that lives on the current view instead of the engine."""

    def __init__(self, field_name: str):
        self.field = field_name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return getattr(view_of(obj), self.field)

    def __set__(self, obj, value):
        setattr(view_of(obj), self.field, value)


class SharedClustering(dict):
    """A clustering everyone shares, whose "level" is chosen per view.

    FINCH computes every level at once, so the partitions are shared data, but which level someone is
    browsing is theirs. Reading or writing "level" goes to the current view (falling back to the level
    the clustering was made with); every other key is an ordinary shared dict entry. `rev` identifies
    this clustering for the life of the process, which is what partition ids are checked against.
    """

    def __init__(self, data: dict, owner):
        super().__init__(data)
        self.rev = next(_REVS)
        self._owner = weakref.ref(owner)

    def _view(self) -> ViewState | None:
        o = self._owner()
        return view_of(o) if o is not None else None

    def __getitem__(self, k):
        if k == "level":
            v = self._view()
            if v is not None and self.rev in v.levels:
                return v.levels[self.rev]
        return super().__getitem__(k)

    def get(self, k, default=None):
        return self[k] if k in self else default

    def __setitem__(self, k, value):
        if k == "level":
            v = self._view()
            if v is not None:
                v.levels[self.rev] = value
                return
        super().__setitem__(k, value)

    def token(self) -> str:
        """What a partition id is relative to, for this view: the clustering and the level shown."""
        return f"{self.rev}:{self['level']}"

    def check_rev(self) -> None:
        """Raise StaleCluster when the request's partition ids were read off another clustering or level."""
        c = _CTX.get()
        if c is not None and c.cluster_rev is not None and c.cluster_rev != self.token():
            raise StaleCluster("the clusters changed since this view loaded them; reload to continue")


def check_partition_token(clustering) -> None:
    """A request that names its partitions' clustering must still find that clustering. Without one
    (someone's reset, an evicted project) a numeric pid would quietly resolve to nothing at all."""
    c = _CTX.get()
    if c is None or c.cluster_rev is None:
        return
    if clustering is None:
        raise StaleCluster("the clusters this view showed are gone; reload to continue")
    clustering.check_rev()
