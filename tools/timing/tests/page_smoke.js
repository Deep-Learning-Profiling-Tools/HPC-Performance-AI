// tools/timing/tests/page_smoke.js -- drive a generated timing page (index.html) without a browser.
//
//   node tools/timing/tests/page_smoke.js <index.html> <checks.json>
//
// A minimal DOM stands in for the browser: the page's own inlined script runs against it and the
// test clicks the page's real controls (campaign tabs, level tabs, application list, input x platform
// cells) and checks the rendered TEXT. It proves the data model and the view logic, not the visual
// layout. checks.json: [{"campaign": 0, "level": "2", "app": "remhos", "input": "periodic-hexagon-p0",
// "platform": "...", "expect": ["2.38 s", ...], "absent": [...]}, {"campaign": 0, "level": "1",
// "overview": true, "expect": [...]}, ...]; exit 1 on the first failure.
"use strict";
const fs = require("fs");

class Node_ {
  constructor(tag) { this.tagName = (tag || "").toUpperCase(); this.children = []; this.attrs = {}; this.listeners = {};
                     this._text = null; this.className = ""; this.style = {}; }
  appendChild(c) { this.children.push(c); c.parent = this; return c; }
  set textContent(v) { this.children = []; this._text = String(v); }
  get textContent() { return this._text !== null && !this.children.length ? this._text
                             : (this._text || "") + this.children.map(c => c.textContent).join(" "); }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === "class") this.className = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  click() { (this.listeners.click || []).forEach(f => f({})); }
  scrollIntoView() {}
  getBoundingClientRect() { return {top: 0}; }
  get value() { return this._value || ""; }
  set value(v) { this._value = v; }
  all() { return [this].concat(...this.children.map(c => c.all ? c.all() : [])); }
  matches(sel) {     // "tag", ".a.b", "tag.a", "[id=x]" only -- what report.js uses
    const m = sel.match(/^([a-z]*)((?:\.[\w-]+)*)$/);
    if (!m) return false;
    if (m[1] && this.tagName !== m[1].toUpperCase()) return false;
    const cls = (this.className || "").split(/\s+/);
    return m[2].split(".").filter(Boolean).every(c => cls.includes(c));
  }
  querySelectorAll(sel) {       // "A B" descendant combinator, one level
    const parts = sel.trim().split(/\s+/);
    let cur = [this];
    for (const p of parts) {
      const next = [];
      cur.forEach(n => n.all().slice(n === this && parts.length > 1 ? 0 : 1).forEach(d => { if (d.matches(p) && !next.includes(d)) next.push(d); }));
      cur = next;
    }
    return cur;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}

function makeDocument(html) {
  const doc = new Node_("document");
  const byId = {};
  const mk = (tag, id, cls) => { const n = new Node_(tag); if (id) { n.attrs.id = id; byId[id] = n; } if (cls) n.className = cls; return n; };
  const data = html.match(/<script type="application\/json" id="timing-data">([\s\S]*?)<\/script>/)[1];
  const dataNode = mk("script", "timing-data"); dataNode._text = data; doc.appendChild(dataNode);
  // the static header: campaign tabs + level tabs, as report.py writes them
  const ctabs = [...html.matchAll(/<button type="button" role="tab" data-campaign="(\d+)"[^>]*>/g)];
  if (ctabs.length) {
    const nav = mk("nav", null, "tabs campaigns"); doc.appendChild(nav);
    ctabs.forEach(m => { const b = mk("button"); b.attrs["data-campaign"] = m[1]; nav.appendChild(b); });
  }
  const lnav = mk("nav", null, "tabs levels"); doc.appendChild(lnav);
  ["1", "2"].forEach(l => { const b = mk("button"); b.attrs["data-level"] = l; b.appendChild(mk("span")); lnav.appendChild(b); });
  ["app-filter", "app-list", "main"].forEach(id => doc.appendChild(mk(id === "app-filter" ? "input" : id === "app-list" ? "ul" : "main", id)));
  doc.getElementById = id => byId[id] || doc.all().find(n => n.attrs.id === id) || null;
  doc.createElement = tag => new Node_(tag);
  doc.createTextNode = t => { const n = new Node_("#text"); n._text = String(t); return n; };
  return doc;
}

function run(html, checks) {
  const js = html.match(/<script>\n([\s\S]*?)<\/script>/)[1];
  let fails = 0;
  for (const c of checks) {
    const doc = makeDocument(html);
    const ctx = {document: doc, location: {hash: ""}};
    ctx.window = {history: {replaceState: (a, b, h) => { ctx.location.hash = h; }}};
    new Function("document", "location", "window", js)(ctx.document, ctx.location, ctx.window);
    const label = JSON.stringify(c);
    try {
      if (c.campaign) doc.querySelectorAll(".tabs.campaigns button").find(b => b.attrs["data-campaign"] === String(c.campaign)).click();
      if (c.level && c.level !== "1") doc.querySelectorAll(".tabs.levels button").find(b => b.attrs["data-level"] === c.level).click();
      if (c.app) {
        const btn = doc.getElementById("app-list").all().find(n => n.tagName === "BUTTON" && n.children[0] && n.children[0]._text === c.app);
        if (!btn) throw new Error("application not listed: " + c.app);
        btn.click();
      }
      if (c.input) {
        const want = c.app + " " + c.input + " on " + c.platform;
        const cell = doc.getElementById("main").all().find(n => n.attrs["aria-label"] === want);
        if (!cell) throw new Error("no cell " + want);
        cell.click();
      }
      const text = doc.getElementById("main").textContent.replace(/\s+/g, " ");
      for (const e of c.expect || []) if (!text.includes(e)) throw new Error("missing '" + e + "'");
      for (const e of c.absent || []) if (text.includes(e)) throw new Error("unexpected '" + e + "'");
      if (c.hash && ctx.location.hash !== c.hash) throw new Error("hash " + ctx.location.hash + " != " + c.hash);
      console.log("ok   " + (c.name || label));
    } catch (e) {
      fails++; console.log("FAIL " + (c.name || label) + ": " + e.message);
    }
  }
  return fails;
}

const [page, checksFile] = process.argv.slice(2);
process.exit(run(fs.readFileSync(page, "utf8"), JSON.parse(fs.readFileSync(checksFile, "utf8"))) ? 1 : 0);
