// tools/timing/report_assets/report.js -- inlined by tools/timing/report.py.
// Optional: without it the page is complete; with it, tables sort and rows filter.
// A header with data-type="num" sorts its cells' data-v numerically; missing values go last.
(function () {
  "use strict";
  function sortTable(table, index, th) {
    var dir = th.getAttribute("aria-sort") === "ascending" ? -1 : 1;
    var numeric = th.getAttribute("data-type") === "num";
    table.querySelectorAll("thead th").forEach(function (h) { h.removeAttribute("aria-sort"); });
    th.setAttribute("aria-sort", dir === 1 ? "ascending" : "descending");
    var body = table.tBodies[0];
    var rows = Array.prototype.slice.call(body.rows);
    rows.sort(function (a, b) {
      var x = a.cells[index].getAttribute("data-v") || "", y = b.cells[index].getAttribute("data-v") || "";
      if (x === "" || y === "") return x === y ? 0 : (x === "" ? 1 : -1);
      return dir * (numeric ? parseFloat(x) - parseFloat(y) : x.localeCompare(y));
    });
    rows.forEach(function (r) { body.appendChild(r); });
  }
  document.querySelectorAll("table.sortable").forEach(function (table) {
    table.querySelectorAll("thead th").forEach(function (th, i) {
      th.tabIndex = 0;
      th.addEventListener("click", function () { sortTable(table, i, th); });
      th.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sortTable(table, i, th); }
      });
    });
  });
  var box = document.getElementById("filter");
  if (box) {
    box.addEventListener("input", function () {
      var q = box.value.trim().toLowerCase();
      document.querySelectorAll("[data-key]").forEach(function (el) {
        el.hidden = q !== "" && el.getAttribute("data-key").indexOf(q) === -1;
      });
    });
  }
})();
