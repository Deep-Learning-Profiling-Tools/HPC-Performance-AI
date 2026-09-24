// tools/timing/report_assets/report.js -- inlined by tools/timing/report.py.
// Campaign -> level tab -> application -> (input x platform) -> the measurement. Every value from
// the data is inserted as text (textContent), never as markup. The selection is kept in the URL hash
// (#L2/remhos/periodic-hexagon-p0/nvidia-b200.cuda13.2; an earlier campaign is prefixed, #c1/L2/...)
// so a view can be linked. Campaign kinds: "registry" (registered inputs, current result + history
// chosen by tools/timing/registry_view.py) and "cases" (case tables; also every earlier snapshot).
(function () {
  "use strict";
  var BUNDLE = JSON.parse(document.getElementById("timing-data").textContent);
  var CAMPS = BUNDLE.campaigns || [BUNDLE];
  var D = CAMPS[0];
  var PLATS = D.platforms || [];
  var CATS = ["compute", "copy_h2d", "copy_d2h", "copy_d2d", "copy_other", "fill", "collective", "other"];
  var CAT_NAME = {compute: "compute (kernels)", copy_h2d: "copy host to device", copy_d2h: "copy device to host",
                  copy_d2d: "copy device to device", copy_other: "copy, other", fill: "fill (memset)",
                  collective: "collective", other: "other"};
  var st = {camp: 0, level: "1", app: null, kase: null, plat: null};
  function setCamp(i) { st.camp = i; D = CAMPS[i]; PLATS = D.platforms || []; }
  function isReg() { return D.kind === "registry"; }
  function rangeSpread(runs, med) {
    if (!runs || runs.length < 2 || !isNum(med) || !med) return null;
    return (Math.max.apply(null, runs) - Math.min.apply(null, runs)) / med;
  }

  // ---------------------------------------------------------------- helpers
  function el(tag, props, kids) {
    var e = document.createElement(tag);
    Object.keys(props || {}).forEach(function (k) {
      var v = props[k];
      if (v === null || v === undefined || v === false) return;
      if (k === "text") e.textContent = v;
      else if (k === "cls") e.className = v;
      else if (k.indexOf("on") === 0) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    });
    (kids || []).forEach(function (c) {
      if (c === null || c === undefined) return;
      e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return e;
  }
  function isNum(x) { return typeof x === "number" && isFinite(x); }
  function p3(v) { return String(Number(v.toPrecision(3))); }
  function fmtT(s) {
    if (!isNum(s)) return "null";
    var a = Math.abs(s);
    if (a < 1e-3) return (s * 1e6).toFixed(0) + " µs";
    if (a < 1) { var v = s * 1e3; return (Math.abs(v) < 100 ? p3(v) : v.toFixed(0)) + " ms"; }
    return (a < 100 ? p3(s) : s.toFixed(1)) + " s";
  }
  function pct(x, d, cap) {
    if (!isNum(x)) return "null";
    return (100 * (cap ? Math.min(x, 1) : x)).toFixed(d || 0) + "%";
  }
  function num(x) { return isNum(x) ? x.toLocaleString("en-US") : "null"; }
  function bytes(b) {
    if (!isNum(b)) return "null";
    var u = ["B", "KB", "MB", "GB", "TB"], i = 0;
    while (Math.abs(b) >= 1000 && i < u.length - 1) { b /= 1000; i++; }
    return (i ? p3(b) : String(b)) + " " + u[i];
  }
  function nul(text) { return el("span", {cls: "nul", text: text || "null"}); }
  function valOrNull(text) { return text === "null" ? nul() : text; }
  function apps() { return D.levels[st.level] || []; }
  function findApp(name) { return apps().filter(function (a) { return a.app === name; })[0] || null; }
  function platById(id) { return PLATS.filter(function (p) { return p.id === id; })[0] || null; }
  function rowsOf(app) { return isReg() ? app.inputs : app.cases; }
  function keyOf(row) { return isReg() ? row.input_id : row.case; }
  function rowOf(app, kase) { return app ? rowsOf(app).filter(function (c) { return keyOf(c) === kase; })[0] || null : null; }
  function cellOf(app, kase, plat) {
    var c = rowOf(app, kase);
    if (!c || !c.cells) return null;
    if (isReg()) return plat ? {reg: true, row: c, run: c.cells[plat] || null} : null;
    return c.cells[plat] || null;
  }
  function measured(app) {
    var n = 0;
    rowsOf(app).forEach(function (c) { PLATS.forEach(function (p) { if (c.cells[p.id]) n++; }); });
    return n;
  }
  function inputText(inp) {
    var parts = [];
    if (inp.env) parts.push(inp.env.split(";").join(" "));
    if (inp.args) parts.push(inp.args);
    if (inp.gpus && inp.gpus !== "1") parts.push("GPUs " + inp.gpus);
    if (!parts.length) return inp.undeclared ? "not in the case tables" : "application defaults";
    return parts.join("  ");
  }
  function kv(rows) {
    return el("table", {cls: "kv"}, [el("tbody", {}, rows.map(function (r) {
      return el("tr", {}, [el("th", {text: r[0]}), el("td", {}, [typeof r[1] === "string" ? valOrNull(r[1]) : r[1]])]);
    }))]);
  }

  // ---------------------------------------------------------------- state <-> hash
  function writeHash() {
    var parts = (st.camp ? ["c" + st.camp] : []).concat(["L" + st.level]);
    if (st.app) parts.push(st.app);
    if (st.app && st.kase && st.plat) { parts.push(st.kase); parts.push(st.plat); }
    // window.history: the local function history() below would shadow the global
    try { window.history.replaceState(null, "", "#" + parts.join("/")); } catch (e) { /* file:// in some browsers */ }
  }
  function readHash() {
    var h = (location.hash || "").replace(/^#/, "").split("/");
    if (/^c[0-9]+$/.test(h[0]) && CAMPS[+h[0].slice(1)]) { setCamp(+h[0].slice(1)); h.shift(); }
    if (h[0] === "L1" || h[0] === "L2") st.level = h[0].slice(1);
    if (h[1] && findApp(h[1])) {
      st.app = h[1];
      if (h[2] && h[3] && cellOf(findApp(h[1]), h[2], h[3])) { st.kase = h[2]; st.plat = h[3]; }
    }
  }

  // ---------------------------------------------------------------- level tabs + list
  function renderTabs() {
    document.querySelectorAll(".tabs.levels button").forEach(function (b) {
      var lv = b.getAttribute("data-level");
      b.setAttribute("aria-selected", lv === st.level ? "true" : "false");
      var span = b.querySelector("span");
      if (span) span.textContent = (D.levels[lv] || []).length + (lv === "1" ? " benchmarks" : " applications");
    });
    document.querySelectorAll(".tabs.campaigns button").forEach(function (b) {
      b.setAttribute("aria-selected", +b.getAttribute("data-campaign") === st.camp ? "true" : "false");
    });
  }
  function renderList() {
    var ul = document.getElementById("app-list");
    var q = document.getElementById("app-filter").value.trim().toLowerCase();
    ul.textContent = "";
    var shown = 0;
    apps().forEach(function (a) {
      if (q && a.app.toLowerCase().indexOf(q) === -1 && (a.suite || "").toLowerCase().indexOf(q) === -1) return;
      shown++;
      var rows = rowsOf(a);
      var meta;
      if (isReg()) {
        var bad = rows.filter(function (r) { return r.status !== "SUCCESS"; }).length;
        var uns = rows.filter(function (r) { return PLATS.some(function (p) { var m = r.cells[p.id]; return m && !m.set.stable; }); }).length;
        meta = (a.suite ? a.suite + " · " : "") + rows.length + (rows.length === 1 ? " input" : " inputs") +
               (bad ? " · " + bad + " not measured" : " · all measured") + (uns ? " · " + uns + " UNSTABLE" : "");
      } else {
        meta = (a.suite ? a.suite + " · " : "") + rows.length + (rows.length === 1 ? " input" : " inputs") +
               " · " + measured(a) + "/" + rows.length * PLATS.length + " measured";
      }
      ul.appendChild(el("li", {}, [el("button", {type: "button", "aria-current": a.app === st.app ? "true" : null,
        onclick: function () { st.app = a.app; st.kase = null; st.plat = null; update(true); }},
        [a.app, el("small", {text: meta})])]));
    });
    if (!shown) ul.appendChild(el("li", {cls: "none", text: "No application matches."}));
  }

  // ---------------------------------------------------------------- main area
  function renderMain() {
    var main = document.getElementById("main");
    main.textContent = "";
    var app = findApp(st.app);
    if (isReg()) { renderMainReg(main, app); return; }
    if (D.historical) {
      main.appendChild(el("div", {cls: "banner", text: "Earlier published snapshot (" + (D.records || 0) + " records as of " +
        (D.generated_from || "-") + "), kept verbatim. It is a different campaign: its numbers are not part of the " +
        "current results and are never compared with them."}));
    }
    if (!app) {
      var list = apps(), inputs = 0, meas = 0;
      list.forEach(function (a) { inputs += a.cases.length; meas += measured(a); });
      main.appendChild(el("div", {cls: "placeholder"}, [
        el("p", {}, ["Choose ", el("b", {text: st.level === "1" ? "a benchmark" : "an application"}),
                     " on the left to see its inputs and the platforms it was measured on."]),
        el("div", {cls: "counts"}, [
          el("span", {}, [el("b", {text: String(list.length)}), st.level === "1" ? " benchmarks" : " applications"]),
          el("span", {}, [el("b", {text: String(inputs)}), " inputs"]),
          el("span", {}, [el("b", {text: String(PLATS.length)}), PLATS.length === 1 ? " platform" : " platforms"]),
          el("span", {}, [el("b", {text: meas + "/" + inputs * PLATS.length}), " combinations measured"]),
          el("span", {}, ["data as of ", el("b", {text: D.generated_from || "-"})])])]));
      return;
    }
    main.appendChild(el("div", {cls: "apphead"}, [el("h2", {text: app.app}),
      el("span", {cls: "meta", text: (app.suite ? app.suite + " · " : "") + "Level " + st.level})]));
    main.appendChild(el("div", {cls: "step", text: "Inputs × platforms — choose a measured combination"}));
    main.appendChild(matrix(app));
    var cell = cellOf(app, st.kase, st.plat);
    if (cell) main.appendChild(detail(app, st.kase, st.plat, cell));
    else main.appendChild(el("p", {cls: "lede", text: "The measurement appears here once an input and a platform " +
                                                      "are chosen. null: never measured on that platform."}));
  }

  function matrix(app) {
    var head = el("tr", {}, [el("th", {text: "input"})].concat(PLATS.map(function (p) {
      return el("th", {cls: "plat"}, [p.label + (p.runtime ? " · " + p.runtime : ""), el("small", {text: p.id})]);
    })));
    var body = app.cases.map(function (c) {
      var tds = PLATS.map(function (p) {
        var cell = c.cells[p.id];
        if (!cell) return el("td", {}, [nul()]);
        var run = cell.run, ok = run.status === "ok";
        var sel = c.case === st.kase && p.id === st.plat;
        var note = cell.history.length + (cell.history.length === 1 ? " run" : " runs") +
                   (cell.latest_status !== "ok" ? " · latest " + cell.latest_status : "");
        return el("td", {}, [el("button", {type: "button", cls: "cellbtn" + (ok ? "" : " bad"),
          "aria-pressed": sel ? "true" : "false",
          "aria-label": app.app + " " + c.case + " on " + p.id,
          onclick: function () { st.kase = c.case; st.plat = p.id; update(false);
                                 var d = document.getElementById("detail"); if (d) d.scrollIntoView({block: "start"}); }},
          [ok ? fmtT(run.roi.wall_s) : run.status, el("small", {text: note})])]);
      });
      return el("tr", {}, [el("td", {cls: "inp"}, [el("b", {text: c.case}), el("br"),
                                                   el("code", {text: inputText(c.input)})])].concat(tds));
    });
    return el("div", {cls: "tscroll"}, [el("table", {cls: "matrix"}, [el("thead", {}, [head]), el("tbody", {}, body)])]);
  }

  // ---------------------------------------------------------------- one measurement
  function fig(v, k) {
    return el("div", {cls: "fig"}, [el("div", {cls: "v" + (v === "null" ? " nul" : ""), text: v}),
                                    el("div", {cls: "k", text: k})]);
  }

  function detail(app, kase, platId, cell) {
    var run = cell.run, R = run.roi, dv = run.device, ctx = run.context, plat = platById(platId) || {id: platId};
    var box = el("section", {cls: "detail", id: "detail", "aria-label": "measurement"});
    box.appendChild(el("h2", {text: app.app + " · " + kase + " · " + (plat.label || platId)}));
    box.appendChild(el("div", {cls: "runline", text: "run " + run.run_id + " · " + (run.utc || "") +
      " · source " + (run.git_commit || "-") + " · collector " +
      ((run.measurement.collector || {}).name || "-") + " · platform " + platId}));
    if (cell.latest_status !== "ok") {
      box.appendChild(el("div", {cls: "banner", text: run.status === "ok"
        ? "The latest run of this combination ended with status " + cell.latest_status + "; showing the last successful run."
        : "No successful run of this combination: status " + run.status + "."}));
    }
    if (run.status !== "ok") { box.appendChild(history(cell)); return box; }

    var wall = R.wall_s, proc = ctx.process_wall_s;
    var cv = (isNum(R.wall_s_stddev) && (R.runs_s || []).length > 1 && wall) ? R.wall_s_stddev / wall : null;
    var busy = dv ? dv.busy_frac_of_roi : null;
    var figs = [fig(fmtT(wall), "ROI (median of " + (R.runs_s || []).length + " clean)"),
                fig(pct(rangeSpread(R.runs_s, wall), 1), "spread (max−min)/median"),
                fig(pct(cv, 1), "CV (stddev/median)"),
                fig(pct(busy, 0, true), "device busy in the ROI"),
                fig(fmtT(dv && isNum(dv.host_gap_s) ? Math.max(dv.host_gap_s, 0) : null), "host gap in the ROI"),
                fig(pct(wall && proc ? wall / proc : null, 1), "ROI share of the process"),
                fig(isNum(R.profiler_inflation) ? R.profiler_inflation.toFixed(2) + "×" : "null", "profiler inflation")];
    if (st.level === "2") {
      var f = run.fom || {};
      figs.push(fig(isNum(f.value) ? (Math.abs(f.value) >= 1e5 || Math.abs(f.value) < 1e-2 ? f.value.toPrecision(4)
                     : f.value.toLocaleString("en-US", {maximumFractionDigits: 1})) : "null",
                    f.name ? "FOM: " + f.name + (f.unit ? " (" + f.unit + ")" : "") : "FOM: none printed"));
    }
    box.appendChild(el("div", {cls: "figs"}, figs));

    // process breakdown
    var pre = Math.max(ctx.pre_roi_s || 0, 0), post = Math.max(ctx.post_roi_s || 0, 0);
    var between = Math.max((proc || 0) - pre - post - wall, 0);
    var segs = isNum(busy)
      ? [["s-pre", pre], ["s-busy", Math.min(wall, busy * wall)], ["s-idle", wall - Math.min(wall, busy * wall)],
         ["s-excl", between], ["s-post", post]]
      : [["s-pre", pre], ["s-roi", wall], ["s-excl", between], ["s-post", post]];
    var tot = segs.reduce(function (s, x) { return s + x[1]; }, 0) || 1;
    var bar = el("div", {cls: "pbar", role: "img", "aria-label": "process breakdown"}, segs.filter(function (x) {
      return x[1] > 0; }).map(function (x) { return el("i", {cls: x[0], style: "width:" + (100 * x[1] / tot).toFixed(2) + "%"}); }));
    box.appendChild(el("div", {cls: "panel"}, [el("h3", {text: "Where the process spends its time"}), bar,
      el("div", {cls: "legend"}, [
        el("span", {}, [el("b", {cls: "s-pre"}), "before the ROI " + fmtT(pre)]),
        el("span", {}, [el("b", {cls: "s-busy"}), "ROI, device busy"]),
        el("span", {}, [el("b", {cls: "s-idle"}), "ROI, host gap"]),
        el("span", {}, [el("b", {cls: "s-excl"}), "excluded / between entries " + fmtT(between)]),
        el("span", {}, [el("b", {cls: "s-post"}), "after the ROI " + fmtT(post)]),
        el("span", {}, ["process " + fmtT(proc)])])]));

    // ROI + device side by side
    var roiRows = [["median", fmtT(wall)], ["min / max", fmtT(R.wall_s_min) + " / " + fmtT(R.wall_s_max)],
      ["clean runs", (R.runs_s || []).map(fmtT).join(", ") || "null"], ["entries", num(R.entries)],
      ["excluded inside", fmtT(R.excluded_s)], ["processes", num(R.processes)],
      ["profiled ROI", fmtT(R.profiled_wall_s)], ["process wall clock", fmtT(proc)]];
    var devTable;
    if (!dv) {
      devTable = el("p", {cls: "lede"}, ["Device activity: ", nul("null"),
        " — no collector observed this run (ROI time and FOM only)."]);
    } else {
      var rows = CATS.map(function (c) {
        return el("tr", {}, [el("th", {text: CAT_NAME[c]}), el("td", {cls: "n"}, [valOrNull(fmtT(dv[c + "_s"]))]),
          el("td", {cls: "n"}, [valOrNull(num(dv[c + "_ops"]))]),
          el("td", {cls: "n"}, [dv.hasOwnProperty(c + "_bytes") ? valOrNull(bytes(dv[c + "_bytes"])) : nul("–")])]);
      });
      rows.push(el("tr", {}, [el("th", {text: "busy (union)"}), el("td", {cls: "n em", text: fmtT(dv.busy_s)}),
        el("td", {}), el("td", {})]));
      rows.push(el("tr", {}, [el("th", {text: "op time sum / overlap"}),
        el("td", {cls: "n", text: fmtT(dv.op_time_sum_s) + " / " + fmtT(dv.overlap_s)}), el("td", {}), el("td", {})]));
      devTable = el("div", {cls: "tscroll"}, [el("table", {}, [el("thead", {}, [el("tr", {}, [el("th", {text: "category"}),
        el("th", {cls: "n", text: "time"}), el("th", {cls: "n", text: "ops"}), el("th", {cls: "n", text: "bytes"})])]),
        el("tbody", {}, rows)])]);
    }
    box.appendChild(el("div", {cls: "panel two"}, [
      el("div", {}, [el("h3", {text: "ROI"}), kv(roiRows)]),
      el("div", {}, [el("h3", {text: "Device activity inside the ROI"}), devTable])]));

    // top operations
    if (run.ops && run.ops.length) {
      var opsRows = run.ops.map(function (o) {
        return el("tr", {}, [el("td", {cls: "opname", title: o.name, text: o.name}), el("td", {text: o.category}),
          el("td", {cls: "n", text: num(o.count)}), el("td", {cls: "n em", text: fmtT(o.total_s)}),
          el("td", {cls: "n", text: fmtT(o.avg_s)}), el("td", {cls: "n", text: pct(o.share, 1)})]);
      });
      box.appendChild(el("div", {cls: "panel"}, [el("h3", {text: "Top operations inside the ROI" +
        (run.ops_total > run.ops.length ? " (" + run.ops.length + " of " + run.ops_total + ")" : "")}),
        el("div", {cls: "tscroll"}, [el("table", {}, [el("thead", {}, [el("tr", {}, ["operation", "category", "count",
          "total", "average", "share"].map(function (h, i) { return el("th", {cls: i > 1 ? "n" : null, text: h}); }))]),
          el("tbody", {}, opsRows)])])]));
    }

    // runtime API + checks
    var rt = run.runtime_api, rtRoi = rt && rt.roi, rtAll = rt && rt.whole;
    var t = run.app_timer;
    var timerCell = !t ? nul("no own timer for this region")
      : !isNum(t.value_s) ? nul(t.status)
      : el("span", {}, [fmtT(t.value_s) + "  ", el("span", {cls: Math.abs(t.roi_diff_frac) <= 0.02 ? "chk" : "off",
          text: "ROI " + (t.roi_diff_frac >= 0 ? "+" : "") + (100 * t.roi_diff_frac).toFixed(3) + "%"})]);
    var auditCell = run.audit_ok === true ? el("span", {cls: "pill ok", text: "clean"})
      : run.audit_ok === false ? el("span", {cls: "pill bad", text: "not clean"})
      : el("span", {cls: "pill na", text: "no launcher"});
    var conf = plat.conformance;
    box.appendChild(el("div", {cls: "panel two"}, [
      el("div", {}, [el("h3", {text: "Runtime API calls" + (rt ? " (" + rt.name + ")" : "")}), kv([
        ["inside the ROI", rtRoi ? num(rtRoi.calls) + " calls, " + fmtT(rtRoi.time_s) : "null"],
        ["synchronizing, in ROI", rtRoi ? num(rtRoi.sync_calls) + " calls, " + fmtT(rtRoi.sync_s) : "null"],
        ["whole process", rtAll ? num(rtAll.calls) + " calls, " + fmtT(rtAll.time_s) : "null"],
        ["kernels: ROI / process", dv ? num(dv.compute_ops) + " / " + num(ctx.whole.compute_ops) : "null"]])]),
      el("div", {}, [el("h3", {text: "Checks"}), kv([
        ["application's own timer", timerCell],
        ["launcher GPU audit", auditCell],
        ["verification vs ROI", run.measurement.verify_vs_roi || "null"],
        ["excluded inside the ROI", run.measurement.roi_excludes || "nothing"],
        ["platform conformance", conf ? el("span", {cls: "pill " + (conf.status === "pass" ? "ok" : "warn"),
          text: conf.status + " " + conf.checks + " · " + conf.date}) : nul("no record")]])])]));

    // input + measurement
    var env = run.inputs.declared_env || {};
    var envText = Object.keys(env).sort().map(function (k) { return k + "=" + env[k]; }).join(" ");
    var argv = run.inputs.argv;
    var proto = run.measurement.protocol || {};
    var coll = run.measurement.collector || {};
    var di = run.device_info || {};
    box.appendChild(el("div", {cls: "panel two"}, [
      el("div", {}, [el("h3", {text: "Input as run"}), kv([
        ["declared variables", envText || "none (application defaults)"],
        ["command", el("code", {text: Array.isArray(argv) ? argv.join(" ") : (argv || "null")})],
        ["GPUs / processes", (run.measurement.gpus || "1") + " / " + num(run.inputs.processes)],
        ["verification skipped", run.measurement.skip_verify ? "yes (outside the ROI anyway)" : "no"]])]),
      el("div", {}, [el("h3", {text: "Measurement and device"}), kv([
        ["protocol", (proto.warmup_runs || 0) + " warm-up, " + (proto.clean_runs || 0) + " clean, " +
                     (proto.profiled_runs || 0) + " profiled"],
        ["collector", (coll.name || "null") + (coll.version ? " " + coll.version : "")],
        ["device", [di.count_visible, "×", di.product, di.arch].filter(function (x) { return x !== null && x !== undefined; }).join(" ") || "null"],
        ["driver", di.driver_version || "null"],
        ["clocks at start (core / memory)", (isNum(di.core_clock_mhz) ? di.core_clock_mhz + " MHz" : "null") + " / " +
                                            (isNum(di.mem_clock_mhz) ? di.mem_clock_mhz + " MHz" : "null")],
        ["host CPU", run.host_cpu || "null"]])])]));

    if (run.caveats && run.caveats.length) {
      box.appendChild(el("div", {cls: "panel"}, [el("h3", {text: "Caveats"}),
        el("ul", {cls: "caveats"}, run.caveats.map(function (c) { return el("li", {text: c}); }))]));
    }
    box.appendChild(history(cell));
    return box;
  }

  function history(cell) {
    var okPrev = null;
    var rows = cell.history.map(function (h) {
      var change = "";
      if (h.status === "ok" && isNum(h.roi_s)) {
        change = okPrev === null ? "first" : ((h.roi_s - okPrev) / okPrev >= 0 ? "+" : "") +
                 (100 * (h.roi_s - okPrev) / okPrev).toFixed(1) + "%";
        okPrev = h.roi_s;
      }
      return el("tr", {}, [el("td", {text: h.run_id}), el("td", {text: (h.utc || "").replace("T", " ").replace("Z", "")}),
        el("td", {}, [el("span", {cls: "pill " + (h.status === "ok" ? "ok" : "bad"), text: h.status})]),
        el("td", {cls: "n em", text: h.status === "ok" ? fmtT(h.roi_s) : "–"}), el("td", {cls: "n", text: change})]);
    }).reverse();
    return el("div", {cls: "panel"}, [el("h3", {text: "Runs of this combination"}),
      el("div", {cls: "tscroll"}, [el("table", {}, [el("thead", {}, [el("tr", {}, ["run", "UTC", "status", "ROI",
        "vs previous"].map(function (h, i) { return el("th", {cls: i > 2 ? "n" : null, text: h}); }))]),
        el("tbody", {}, rows)])])]);
  }


  // ---------------------------------------------------------------- registered inputs
  var STATUS_TEXT = {SUCCESS: "ROI timing SUCCESS", RUN_FAILED: "run failed", NOT_MEASURED: "not measured"};
  function pill(text, kind) { return el("span", {cls: "pill " + kind, text: text}); }
  function statusPill(s) { return pill(STATUS_TEXT[s] || s, s === "SUCCESS" ? "ok" : "bad"); }
  function verdictPill(v) {
    return !v ? nul("none") : pill(v, v === "PASS" ? "ok" : (v === "SUPERSEDED" || v === "INCOMPLETE") ? "warn" : v === "NOT_RUN" ? "na" : "bad");
  }
  function paramsText(i) {
    var p = i.params || {}, parts = Object.keys(p).sort().filter(function (k) { return k.charAt(0) !== "_"; })
      .map(function (k) { return k + "=" + (typeof p[k] === "object" ? JSON.stringify(p[k]) : p[k]); });
    return parts.join("  ") || "application defaults";
  }
  function renderMainReg(main, app) {
    var C = D.counts || {}, camp = D.campaign || {};
    if (!app) {
      var list = apps(), rows = [];
      list.forEach(function (a) { rows = rows.concat(a.inputs); });
      var ok = rows.filter(function (r) { return r.status === "SUCCESS"; }).length;
      var ver = rows.filter(function (r) { return r.run_verification === "PASS"; }).length;
      var uns = rows.filter(function (r) { return PLATS.some(function (p) { var m = r.cells[p.id]; return m && !m.set.stable; }); }).length;
      var corr = {PASS: 0, INCOMPLETE: 0, FAIL: 0, none: 0};
      rows.forEach(function (r) { corr[r.correctness || "none"] = (corr[r.correctness || "none"] || 0) + 1; });
      main.appendChild(el("div", {cls: "placeholder"}, [
        el("p", {}, [el("b", {text: camp.title || "Registered inputs"}), " — every input registered in level" + st.level +
          "/*/inputs.yaml, measured with measure_level" + st.level + ".sh --registry. Choose ",
          el("b", {text: st.level === "1" ? "a benchmark" : "an application"}), " on the left."]),
        el("div", {cls: "counts"}, [
          el("span", {}, [el("b", {text: String(rows.length)}), " registered inputs"]),
          el("span", {}, [el("b", {text: String(ok)}), " ROI timing SUCCESS"]),
          el("span", {}, [el("b", {text: String(rows.length - ok)}), " not measured successfully"]),
          el("span", {}, [el("b", {text: String(ver)}), " run verification PASS"]),
          el("span", {}, [el("b", {text: String(uns)}), " UNSTABLE"]),
          el("span", {}, ["measured ", el("b", {text: (D.measured_from || "-") + " .. " + (D.generated_from || "-")})])]),
        kv([["protocol", (camp.protocol || {})["level" + st.level] || "null"],
            ["platform", camp.platform_note || "null"],
            ["not collected", camp.not_collected || "–"],
            ["scientific correctness", "PASS " + corr.PASS + " · INCOMPLETE " + corr.INCOMPLETE + " · FAIL " + corr.FAIL +
              " · none " + corr.none + " — evidence from outside the timing runs (basis per input); not re-verified by these ROI runs"],
            ["Level 3", camp.level3 || (D.level3_inputs + " registered inputs without ROI support")],
            ["spread", "(max − min) / median of all clean-run samples; stable when ≤ 10 %. CV = stddev / median, shown separately."]]),
        (D.notes && D.notes.length) ? el("ul", {cls: "caveats"}, D.notes.map(function (n) { return el("li", {text: n}); })) : null]));
      return;
    }
    main.appendChild(el("div", {cls: "apphead"}, [el("h2", {text: app.app}),
      el("span", {cls: "meta", text: (app.suite ? app.suite + " · " : "") + "Level " + st.level + " · registered inputs"})]));
    main.appendChild(el("div", {cls: "step", text: "Registered inputs × platforms — choose one"}));
    main.appendChild(matrixReg(app));
    var row = rowOf(app, st.kase);
    if (row && st.plat) main.appendChild(detailReg(app, row, st.plat));
    else main.appendChild(el("p", {cls: "lede", text: "The measurement appears here once an input and a platform are chosen."}));
  }
  function matrixReg(app) {
    var head = el("tr", {}, [el("th", {text: "input"})].concat(PLATS.map(function (p) {
      return el("th", {cls: "plat"}, [p.label + (p.runtime ? " · " + p.runtime : ""), el("small", {text: p.id})]);
    })).concat([el("th", {text: "run verification"}), el("th", {text: "correctness"})]));
    var body = app.inputs.map(function (r) {
      var tds = PLATS.map(function (p) {
        var m = r.cells[p.id], sel = r.input_id === st.kase && p.id === st.plat;
        var label = m ? [fmtT(m.roi.wall_s), el("small", {text: m.set.n + " samples · spread " + pct(m.set.spread, 1) +
                                                               (m.set.stable ? "" : " · UNSTABLE") + (m.set.adaptive ? " · 3+2" : "")})]
                      : [STATUS_TEXT[r.status] || r.status, el("small", {text: r.attempts.length + (r.attempts.length === 1 ? " attempt" : " attempts")})];
        return el("td", {}, [el("button", {type: "button", cls: "cellbtn" + (m ? (m.set.stable ? "" : " warn") : " bad"),
          "aria-pressed": sel ? "true" : "false", "aria-label": app.app + " " + r.input_id + " on " + p.id,
          onclick: function () { st.kase = r.input_id; st.plat = p.id; update(false);
                                 var d = document.getElementById("detail"); if (d) d.scrollIntoView({block: "start"}); }}, label)]);
      });
      return el("tr", {}, [el("td", {cls: "inp"}, [el("b", {text: r.input_id}), " ", el("small", {text: [r.case, r.variant].filter(Boolean).join(" · ")}),
        el("br"), el("code", {text: paramsText(r)})])].concat(tds).concat([
        el("td", {}, [verdictPill(r.run_verification)]), el("td", {}, [verdictPill(r.correctness)])]));
    });
    return el("div", {cls: "tscroll"}, [el("table", {cls: "matrix"}, [el("thead", {}, [head]), el("tbody", {}, body)])]);
  }
  function detailReg(app, row, platId) {
    var run = row.cells[platId], plat = platById(platId) || {id: platId};
    var box = el("section", {cls: "detail", id: "detail", "aria-label": "measurement"});
    box.appendChild(el("h2", {text: app.app + " · " + row.input_id + " · " + (plat.label || platId)}));
    var checks = kv([["ROI timing", statusPill(row.status)],
      ["run verification", el("span", {}, [verdictPill(row.run_verification), " did the run get this registered input (argv, cwd, executable, files)"])],
      ["input-file identity", !row.file_identity ? nul("–") : row.file_identity.indexOf("insufficient") >= 0
        ? pill("INSUFFICIENT: the files were not identified at measurement time", "warn")
        : row.file_identity.indexOf("supplement") >= 0
        ? el("span", {}, [pill("supplement", "warn"), " established after the measurement from the run's own argv / log and run-directory copies unchanged since before the run (the stored identity did not name these files)"])
        : pill("recorded at measurement", "ok")],
      ["scientific correctness", el("span", {}, [verdictPill(row.correctness), " " + (row.correctness_basis || "")])]]);
    if (!run) {
      box.appendChild(el("div", {cls: "banner", text: row.blocker ? row.blocker :
        "No verified successful measurement of this input on this platform (" + (STATUS_TEXT[row.status] || row.status) + ")."}));
      box.appendChild(el("div", {cls: "panel"}, [el("h3", {text: "Status"}), checks]));
      box.appendChild(inputPanel(row, null));
      box.appendChild(historyReg(row));
      return box;
    }
    var R = run.roi, S = run.set, ctx = run.context;
    box.appendChild(el("div", {cls: "runline", text: "runs " + S.run_ids.join(" + ") + " · " + S.utc_first + " .. " + S.utc_last +
      " · source " + (S.git_commit || "-") + " · executable sha256 " + (S.exe_sha256 || "-") + "… · collector " +
      ((run.measurement.collector || {}).name || "-") + " · platform " + platId}));
    var figs = [fig(fmtT(R.wall_s), "ROI (median of " + S.n + " clean)"),
                fig(pct(S.spread, 1), "spread (max−min)/median"),
                fig(pct(S.cv, 1), "CV (stddev/median)"),
                fig(S.stable ? "stable" : "UNSTABLE", "stability (spread ≤ 10 %)"),
                fig(pct(R.wall_s && ctx.process_wall_s ? R.wall_s / ctx.process_wall_s : null, 1), "ROI share of the process"),
                fig(run.device ? pct(run.device.busy_frac_of_roi, 0, true) : "null", run.device ? "device busy in the ROI" : "device busy: not collected")];
    if (st.level === "2") {
      var f = run.fom || {};
      figs.push(fig(isNum(f.value) ? (Math.abs(f.value) >= 1e5 || Math.abs(f.value) < 1e-2 ? f.value.toPrecision(4)
                     : f.value.toLocaleString("en-US", {maximumFractionDigits: 1})) : "null",
                    (f.name ? "FOM: " + f.name + (f.unit ? " (" + f.unit + ")" : "") : "FOM: none printed") + " · run " + S.run_ids[S.run_ids.length - 1]));
    }
    box.appendChild(el("div", {cls: "figs"}, figs));
    var sampleRows = S.per_record.map(function (p) {
      var pr = p.protocol || {};
      return ["run " + p.run_id, (p.runs_s || []).map(fmtT).join(", ") + "   (" + (pr.warmup_runs || 0) + " warm-up, " +
              (pr.clean_runs || 0) + " clean, " + (pr.profiled_runs || 0) + " profiled)"];
    });
    box.appendChild(el("div", {cls: "panel two"}, [
      el("div", {}, [el("h3", {text: "Clean-run samples" + (S.adaptive ? " (3 + 2 adaptive extension, pooled by measurement group " + S.group.id + ")" : "")}),
        kv(sampleRows.concat([["median / min / max", fmtT(R.wall_s) + " / " + fmtT(R.wall_s_min) + " / " + fmtT(R.wall_s_max)],
          ["spread", "(max − min) / median = " + pct(S.spread, 2)], ["CV", "sample stddev / median = " + pct(S.cv, 2)]]))]),
      el("div", {}, [el("h3", {text: "Status"}), checks])]));
    var t = run.app_timer;
    box.appendChild(el("div", {cls: "panel two"}, [
      el("div", {}, [el("h3", {text: "Device activity inside the ROI"}), run.device ? el("p", {text: "see the case-table view"}) :
        el("p", {cls: "lede"}, [nul("not collected"), " — " + ((D.campaign || {}).not_collected || "no collector observed this run")])]),
      el("div", {}, [el("h3", {text: "Checks"}), kv([
        ["application's own timer", !t ? nul("no own timer for this region") : !isNum(t.value_s) ? nul(t.status)
          : el("span", {}, [fmtT(t.value_s) + "  ", el("span", {cls: Math.abs(t.roi_diff_frac) <= 0.02 ? "chk" : "off",
              text: "ROI " + (t.roi_diff_frac >= 0 ? "+" : "") + (100 * t.roi_diff_frac).toFixed(3) + "% (run " + S.run_ids[S.run_ids.length - 1] + ")"})])],
        ["launcher GPU audit", run.audit_ok === true ? pill("clean", "ok") : run.audit_ok === false ? pill("not clean", "bad") : pill("no launcher", "na")],
        ["excluded inside the ROI", run.measurement.roi_excludes || "nothing"],
        ["profiler inflation", isNum(run.roi.profiler_inflation) ? run.roi.profiler_inflation.toFixed(2) + "×" : nul("not collected")]])])]));
    box.appendChild(inputPanel(row, run));
    if (run.caveats && run.caveats.length) {
      box.appendChild(el("div", {cls: "panel"}, [el("h3", {text: "Caveats (run " + S.run_ids[S.run_ids.length - 1] + ")"}),
        el("ul", {cls: "caveats"}, run.caveats.map(function (c) { return el("li", {text: c}); }))]));
    }
    box.appendChild(historyReg(row));
    return box;
  }
  function inputPanel(row, run) {
    var env = row.env || {}, di = run ? run.device_info || {} : {};
    var left = kv([["registered as", [row.case, row.variant, row.source_kind, row.input_form].filter(Boolean).join(" · ")],
      ["parameters", el("code", {text: paramsText(row)})],
      ["registry arguments", (row.args || []).length ? el("code", {text: row.args.join(" ")}) : "none"],
      ["registry variables", Object.keys(env).sort().map(function (k) { return k + "=" + env[k]; }).join(" ") || "none"],
      ["command as run", run ? el("code", {text: Array.isArray(run.inputs.argv) ? run.inputs.argv.join(" ") : (run.inputs.argv || "null")}) : "–"]]);
    var right = run ? kv([["collector", ((run.measurement.collector || {}).name || "null")],
      ["device", [di.count_visible, "×", di.product, di.arch].filter(function (x) { return x !== null && x !== undefined; }).join(" ") || "null"],
      ["driver", di.driver_version || "null"], ["host CPU", run.host_cpu || "null"],
      ["GPUs / processes", (run.measurement.gpus || "1") + " / " + num(run.inputs.processes)]]) : el("p", {cls: "lede", text: "no measurement"});
    return el("div", {cls: "panel two"}, [el("div", {}, [el("h3", {text: "Input"}), left]),
                                          el("div", {}, [el("h3", {text: "Measurement and device"}), right])]);
  }
  function historyReg(row) {
    var sets = row.sets.map(function (s) {
      return el("tr", {}, [el("td", {text: s.run_ids.join(" + ")}), el("td", {text: (s.utc_last || "").replace("T", " ").replace("Z", "")}),
        el("td", {}, [verdictPill(s.verdict)]), el("td", {text: s.current_definition ? "current definition" : "earlier definition"}),
        el("td", {cls: "n", text: String(s.n)}), el("td", {cls: "n em", text: fmtT(s.median)}), el("td", {cls: "n", text: pct(s.spread, 1)}),
        el("td", {text: s.git_commit}),
        el("td", {cls: "n", text: isNum(s.vs_previous) ? (s.vs_previous >= 0 ? "+" : "") + (100 * s.vs_previous).toFixed(1) + "%" : "–"})]);
    }).reverse();
    var att = row.attempts.map(function (a) {
      return el("tr", {}, [el("td", {text: a.run_id}), el("td", {text: (a.utc || "").replace("T", " ").replace("Z", "")}),
        el("td", {}, [pill(a.status, a.status === "ok" ? "ok" : "bad")]), el("td", {}, [verdictPill(a.verdict)]),
        el("td", {cls: "n", text: String(a.clean_runs)}), el("td", {cls: "n", text: a.status === "ok" ? fmtT(a.roi_s) : "–"}),
        el("td", {text: a.git_commit}), el("td", {text: (a.problems || []).join("; ")})]);
    }).reverse();
    function tbl(h, rows) {
      return el("div", {cls: "tscroll"}, [el("table", {}, [el("thead", {}, [el("tr", {}, h.map(function (x) { return el("th", {text: x}); }))]),
        el("tbody", {}, rows)])]);
    }
    var linkNote = (row.link_problems || []).length ? el("ul", {cls: "caveats"}, row.link_problems.map(function (p) { return el("li", {text: p}); })) : null;
    return el("div", {cls: "panel"}, [el("h3", {text: "Measurements of this input (pooled only by an explicit measurement group)"}), linkNote,
      sets.length ? tbl(["runs", "UTC", "verdict", "workload", "samples", "median", "spread", "source", "vs previous (same workload)"], sets)
                  : el("p", {cls: "lede", text: "none"}),
      el("h3", {text: "Every attempt (record)"}),
      tbl(["run", "UTC", "status", "verification", "clean runs", "ROI median", "source", "notes"], att)]);
  }

  // ---------------------------------------------------------------- wiring
  function update(scrollTop) {
    writeHash(); renderTabs(); renderList(); renderMain();
    if (scrollTop) { var m = document.getElementById("main"); if (m && m.getBoundingClientRect().top < 0) m.scrollIntoView(); }
  }
  document.querySelectorAll(".tabs.levels button").forEach(function (b) {
    b.addEventListener("click", function () {
      if (st.level === b.getAttribute("data-level")) return;
      st.level = b.getAttribute("data-level"); st.app = null; st.kase = null; st.plat = null;
      document.getElementById("app-filter").value = "";
      update(false);
    });
  });
  document.querySelectorAll(".tabs.campaigns button").forEach(function (b) {
    b.addEventListener("click", function () {
      var i = +b.getAttribute("data-campaign");
      if (i === st.camp) return;
      setCamp(i); st.app = null; st.kase = null; st.plat = null;
      document.getElementById("app-filter").value = "";
      update(false);
    });
  });
  document.getElementById("app-filter").addEventListener("input", renderList);
  readHash();
  update(false);
})();
