// Account corner for hosted mode, shared by the launcher and the curator. Renders nothing locally.
// Role models: Google's avatar menu (top right: initial in a coloured circle, menu with sign out),
// Google Docs' presence row (who else has this open, as small avatars left of your own), and
// Grafana's users table for admins (name, role, reset password, remove; add row at the bottom).
const Account = (() => {
  const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const hue = name => [...name].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 7);
  const avatar = (name, size = 28) =>
    `<span class="acct-av" style="width:${size}px;height:${size}px;font-size:${Math.round(size * .45)}px;` +
    `background:hsl(${hue(name)} 55% 42%)" title="${esc(name)}">${esc(name[0].toUpperCase())}</span>`;
  const call = async (url, opts = {}) => {
    const r = await fetch(url, {headers: {"Content-Type": "application/json"}, ...opts});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || r.statusText);
    return j;
  };
  let ME = null;

  function css() {
    if (document.getElementById("acct-css")) return;
    const st = document.createElement("style");
    st.id = "acct-css";
    st.textContent = `
      .acct { display:flex; align-items:center; gap:6px; margin-left:auto; position:relative; }
      .acct-av { display:inline-grid; place-items:center; border-radius:50%; color:#fff; font-weight:600;
                 flex:0 0 auto; user-select:none; }
      .acct-peers { display:flex; }
      .acct-peers .acct-av { margin-left:-6px; box-shadow:0 0 0 2px var(--panel,#171a21); }
      .acct-me { background:none; border:0; padding:0; cursor:pointer; border-radius:50%; }
      .acct-me:focus-visible { outline:2px solid var(--acc,#4f8cff); outline-offset:2px; }
      .acct-menu { position:absolute; top:calc(100% + 8px); right:0; min-width:200px; z-index:50;
                   background:var(--panel,#171a21); border:1px solid var(--line,#2a2f3a); border-radius:10px;
                   padding:6px; box-shadow:0 8px 28px rgba(0,0,0,.45); display:none; }
      .acct-menu.open { display:block; }
      .acct-menu .who { display:flex; gap:10px; align-items:center; padding:8px 8px 10px;
                        border-bottom:1px solid var(--line,#2a2f3a); margin-bottom:4px; }
      .acct-menu .who small { display:block; color:var(--mut,#9aa3b2); }
      .acct-menu button { display:block; width:100%; text-align:left; background:none; border:0;
                          border-radius:6px; padding:7px 8px; color:inherit; cursor:pointer; font:inherit; }
      .acct-menu button:hover { background:var(--panel-2,#1c2230); }
      dialog.acct-dlg { background:var(--panel,#171a21); color:var(--fg,#e6e8ec); border:1px solid var(--line,#2a2f3a);
                        border-radius:12px; padding:20px; min-width:320px; }
      dialog.acct-dlg::backdrop { background:rgba(0,0,0,.55); }
      .acct-dlg h3 { margin:0 0 14px; font-size:16px; }
      .acct-dlg form { display:flex; flex-direction:column; gap:10px; }
      .acct-dlg input, .acct-dlg select { font:inherit; color:inherit; background:#0c0e12;
                        border:1px solid var(--line,#2a2f3a); border-radius:6px; padding:6px 9px; }
      .acct-dlg table { border-collapse:collapse; width:100%; margin-bottom:12px; }
      .acct-dlg td { padding:6px 6px; border-bottom:1px solid var(--line,#2a2f3a); vertical-align:middle; }
      .acct-dlg td.n { display:flex; gap:8px; align-items:center; }
      .acct-dlg .row { display:flex; gap:6px; }
      .acct-dlg .row input { flex:1; min-width:0; }
      .acct-dlg .acts { display:flex; justify-content:flex-end; gap:8px; margin-top:6px; }
      .acct-dlg .err { color:var(--warn,#e0533d); min-height:16px; font-size:12px; }
      .acct-dlg button { font:inherit; }`;
    document.head.appendChild(st);
  }

  function dialog(html) {
    const d = document.createElement("dialog");
    d.className = "acct-dlg";
    d.innerHTML = html;
    document.body.appendChild(d);
    d.addEventListener("close", () => d.remove());
    d.showModal();
    return d;
  }

  function passwordDialog() {
    const d = dialog(`<h3>Change password</h3><form method="dialog">
      <input type="password" name="cur" placeholder="Current password" autocomplete="current-password" required>
      <input type="password" name="pw" placeholder="New password" autocomplete="new-password" minlength="8" required>
      <input type="password" name="pw2" placeholder="New password again" autocomplete="new-password" required>
      <div class="err"></div>
      <div class="acts"><button type="button" value="cancel">Cancel</button><button class="primary">Change</button></div></form>`);
    const f = d.querySelector("form"), err = d.querySelector(".err");
    d.querySelector('[value="cancel"]').onclick = () => d.close();
    f.onsubmit = async e => {
      e.preventDefault();
      if (f.pw.value !== f.pw2.value) { err.textContent = "The new passwords differ."; return; }
      try { await call("/api/auth/password", {method: "POST", body: JSON.stringify({current: f.cur.value, new: f.pw.value})}); d.close(); }
      catch (x) { err.textContent = x.message; }
    };
  }

  async function usersDialog() {
    const d = dialog(`<h3>Users</h3><table></table>
      <form class="row"><input name="name" placeholder="Username" autocapitalize="none" required>
      <input name="pw" type="password" placeholder="Password" autocomplete="new-password" minlength="8" required>
      <select name="role"><option>member</option><option>admin</option></select><button class="primary">Add</button></form>
      <div class="err"></div><div class="acts"><button type="button" value="done">Done</button></div>`);
    const tbl = d.querySelector("table"), err = d.querySelector(".err"), add = d.querySelector("form");
    d.querySelector('[value="done"]').onclick = () => d.close();
    const act = async (fn) => { err.textContent = ""; try { await fn(); await render(); } catch (x) { err.textContent = x.message; } };
    async function render() {
      const r = await call("/api/admin/users");
      tbl.innerHTML = r.users.map(u => `<tr data-u="${esc(u.name)}">
        <td class="n">${avatar(u.name, 24)}${esc(u.name)}${u.name === r.me ? ' <small style="color:var(--mut,#9aa3b2)">you</small>' : ""}</td>
        <td><select data-act="role"><option${u.role === "member" ? " selected" : ""}>member</option>
            <option${u.role === "admin" ? " selected" : ""}>admin</option></select></td>
        <td><button data-act="pw" title="Reset password">🔑</button></td>
        <td><button data-act="rm" title="Remove">✕</button></td></tr>`).join("");
    }
    tbl.onchange = e => {
      const u = e.target.closest("tr")?.dataset.u;
      if (e.target.dataset.act === "role") act(() => call(`/api/admin/users/${encodeURIComponent(u)}`,
        {method: "POST", body: JSON.stringify({role: e.target.value})}));
    };
    tbl.onclick = e => {
      const b = e.target.closest("button"), u = b?.closest("tr")?.dataset.u;
      if (!b) return;
      if (b.dataset.act === "pw") {
        const pw = prompt(`New password for ${u}`);
        if (pw) act(() => call(`/api/admin/users/${encodeURIComponent(u)}`, {method: "POST", body: JSON.stringify({password: pw})}));
      } else if (b.dataset.act === "rm" && confirm(`Remove ${u}?`)) {
        act(() => call(`/api/admin/users/${encodeURIComponent(u)}`, {method: "DELETE"}));
      }
    };
    add.onsubmit = e => {
      e.preventDefault();
      act(async () => { await call("/api/admin/users", {method: "POST",
        body: JSON.stringify({name: add.name.value, password: add.pw.value, role: add.role.value})}); add.reset(); });
    };
    render().catch(x => err.textContent = x.message);
  }

  async function signOut() {
    await fetch("/api/auth/logout", {method: "POST"}).catch(() => {});
    location.href = "/login";
  }

  // `host`: the element the corner is appended to (a header). `presence`: show who else is in the project.
  async function init(host, {presence = false} = {}) {
    try { ME = await call("/api/auth/me"); } catch (_) { return null; }
    if (!ME.hosted || !ME.user) return ME;
    css();
    const box = document.createElement("div");
    box.className = "acct";
    box.innerHTML = `<span class="acct-peers"></span>
      <button class="acct-me" aria-haspopup="menu" title="${esc(ME.user)}">${avatar(ME.user, 30)}</button>
      <div class="acct-menu" role="menu">
        <div class="who">${avatar(ME.user, 34)}<div>${esc(ME.user)}<small>${esc(ME.role)}</small></div></div>
        <button data-m="pw" role="menuitem">Change password</button>
        ${ME.role === "admin" ? '<button data-m="users" role="menuitem">Users</button>' : ""}
        <button data-m="out" role="menuitem">Sign out</button></div>`;
    host.appendChild(box);
    const menu = box.querySelector(".acct-menu");
    box.querySelector(".acct-me").onclick = e => { e.stopPropagation(); menu.classList.toggle("open"); };
    document.addEventListener("click", e => { if (!box.contains(e.target)) menu.classList.remove("open"); });
    document.addEventListener("keydown", e => { if (e.key === "Escape") menu.classList.remove("open"); });
    menu.onclick = e => {
      const m = e.target.closest("button")?.dataset.m;
      menu.classList.remove("open");
      if (m === "pw") passwordDialog(); else if (m === "users") usersDialog(); else if (m === "out") signOut();
    };
    if (presence) {
      const peers = box.querySelector(".acct-peers");
      const tick = async () => {
        if (document.hidden) return;
        try { peers.innerHTML = (await call("/api/presence")).users.map(u => avatar(u.user, 26)).join(""); } catch (_) {}
      };
      tick(); setInterval(tick, 15000);
    }
    return ME;
  }

  return {init, me: () => ME};
})();
