# Multi-user hosting

Goal: run Chevron on a server behind a domain, so a small team can sign in and curate the same
projects at the same time. Local use (`chevron` on 127.0.0.1) keeps working exactly as before.

## Decisions

- Audience: a small invited team. Every member sees and edits every project. No public sign-up.
- Images: only from one server-side data folder (`--data-root`), filled by an admin. No uploads.
- Sign-in: built-in accounts (username + password), managed by admins.
- Collaboration: data is shared (labels, taxonomy, masks, clustering, classifier); the view is per
  browser session (open project, scope, filters, cluster level). Undo history is per user: your own
  steps, across your tabs and devices. Each tab is pinned to its project (`/app?p=<id>`).
- Hosting target: a dedicated Linux box with a GPU, behind a TLS reverse proxy or a Cloudflare tunnel.

## Modes

`chevron --root R` stays the local, no-login mode. Adding `--data-root D` switches on hosted mode:
login required, every client-supplied path confined to `D` or `R`, and the server refuses to start
until at least one admin exists (`chevron user add NAME --admin --root R`).

## Components

- `chevron/auth.py`: user store (`<root>/.chevron-users.json`, scrypt hashes, roles admin|member),
  session store (`<root>/.chevron-sessions.json`, token stored as sha256, 14 days sliding), login
  throttle (5 failures per user+IP per 10 minutes).
- `chevron/context.py`: per-request context (user, session, open project, view) in a contextvar.
  Engine code reads the current view and actor from it. Outside a request, a default context makes
  the engine behave exactly as before, which keeps tests and scripts unchanged.
- `chevron/pool.py`: replaces `_ActiveProject`. Several engines can be resident; each session
  points at one project. Least-recently-used projects not serving a request are closed when
  `max_open` is exceeded (local mode: 1, so behaviour is unchanged), and idle ones after 30 minutes
  in hosted mode.
- `chevron/jobs.py`: ASGI middleware that detaches slow JSON POSTs. A request still running after
  20 s answers `202 {"job": id}`; the client polls `GET /api/jobs/{id}` and receives the original
  response. This keeps long runs alive behind proxies with request timeouts (Cloudflare: 100 s).
  Heavy endpoints (propose, features, inference, training) additionally share one server-wide slot.
- `chevron/pathguard.py`: in hosted mode, rejects (403) any request whose query or JSON body carries
  a path-like key (`path`, `paths`, `root`, `image_root`, `coco_path`, `dir`, `out_dir`, `ckpt`,
  `extra_image_root`, ...) resolving outside the allowed roots.

## Engine changes

- View attributes (`_scope_bids`, `_scope_id`, `_scope_token`, the four facet filters, the cluster
  and sub-cluster level) become properties backed by the current `ViewState`. Scope tokens come
  from one global counter so view-dependent caches never collide. The membership index is cached
  per view key instead of in one slot.
- Partition ids only mean something for one clustering at one level. The client sends
  `X-Chevron-Cluster: <rev>:<level>`; resolving a numeric partition against a different clustering
  answers 409 instead of acting on the wrong instances.
- Undo/redo stacks are per actor. Undoing is refused when someone else has since changed an
  instance or class the command touched. `history.jsonl` records the actor.
- Clusterings are kept per scope: every unnarrowed view shares one; narrowing your scope gives you
  your own and never replaces the shared one.

## UI

- Login page (role model: Grafana/Jupyter sign-in card: product name, username, password, one button).
- Account menu top right with sign out, and an avatar stack of who else is in the project
  (role model: Google Docs presence).
- Admin page for users (role model: Grafana users table: list, add, reset password, role, remove).
- Activity log shows who did what.

## Out of scope

Per-project permissions, uploads, public sign-up, real-time cursors, editing locks.

## Testing

Unit tests per module; server tests with two TestClients as two users; the existing suite must stay
green; a browser run of login, two users in one project, and a detached long job.
