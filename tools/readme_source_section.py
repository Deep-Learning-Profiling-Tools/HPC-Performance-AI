#!/usr/bin/env python3
"""readme_source_section -- keep the "Source distribution (frozen bundle)" section of every level3/<app>/README.md
and the LOC table of level3/APPLICATION_AUDIT.md in sync with provenance/source.lock*.yaml, LOC*.json and
benchmark.yaml. The section is delimited by marker comments and rewritten in place (created at the end of the
README when absent); nothing else in the files is touched.

    readme_source_section.py [--apps a,b,...]
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hpcperf_source as hs  # noqa: E402

R = os.path.abspath(os.path.join(HERE, ".."))
APPS = ["lammps", "sparta", "warpx", "specfem3d", "nekrs", "nyx", "cp2k", "qmcpack", "dftfe", "geos"]
BEGIN, END = "<!-- hpcperf:source-section:begin -->", "<!-- hpcperf:source-section:end -->"
ABEGIN, AEND = "<!-- hpcperf:loc-table:begin -->", "<!-- hpcperf:loc-table:end -->"


def replace_section(text, begin, end, body):
    block = f"{begin}\n{body.rstrip()}\n{end}\n"
    if begin in text and end in text:
        return re.sub(re.escape(begin) + r".*?" + re.escape(end) + r"\n?", block, text, flags=re.S)
    return text.rstrip("\n") + "\n\n" + block


def app_data(app):
    d = os.path.join(R, "level3", app)
    by = hs.load_yaml(os.path.join(d, "benchmark.yaml")) if os.path.isfile(os.path.join(d, "benchmark.yaml")) else {}
    variants = list(by.get("variants", {}).keys()) if by.get("variants") else [None]
    out = []
    for v in variants:
        suffix = f".{v}" if v else ""
        lock_p = os.path.join(d, "provenance", f"source.lock{suffix}.yaml")
        lock = hs.load_yaml(lock_p) if os.path.isfile(lock_p) else {}
        ident = by.get("variants", {}).get(v, {}) if v else (by.get("source_bundle") or {})
        loc_p = os.path.join(d, "provenance", f"LOC{suffix}.json")
        loc = json.load(open(loc_p))["summary"] if os.path.isfile(loc_p) else {}
        out.append((v, by, lock, ident, loc))
    return out


def section(app):
    rows = app_data(app)
    lines = ["## Source distribution (frozen bundle, 2026-09-08)", "",
             "The application source is no longer read from `_upstream/`: `tools/prepare_benchmark.sh level3 " + app +
             "` materializes the frozen bundle into `src/` (+ `deps/`), the only source `build.sh`/`run.sh`/`validate.sh` use. "
             "Identity, patch series, licenses and the equivalence proof against the tree the results above were validated from are under `provenance/` "
             "(`source.lock*.yaml`, `patch_series*.txt`, `original_vs_baseline*.diff`, `LICENSES*.md`, `equivalence*.md`, `LOC*.md`); "
             "what an optimization agent may modify is in `optimization_scope.yaml`; `benchmark.yaml` is the machine-readable contract.", ""]
    lines += ["| variant | archive | compressed / uncompressed | files | source_tree_sha256 | archive sha256 | upstream | patches (pre-applied) | equivalence | LOC app-owned / agent-modifiable / bundled deps / benchmark deps / tests / total |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for v, by, lock, ident, loc in rows:
        eq = lock.get("equivalence", [])
        eqs = "SKIPPED" if eq == "SKIPPED" else (", ".join(f"{e.get('archive_path')}: {e.get('status')}" for e in eq) if eq else "not frozen")
        up = lock.get("upstream", {})
        pats = ", ".join(os.path.basename(p["path"]) for p in lock.get("patches", [])) or "none"
        c, u = ident.get("compressed_size"), ident.get("uncompressed_size")
        lines.append(f"| {v or '-'} | `{ident.get('archive', '-')}` | {hs.human(c) if c else '-'} / {hs.human(u) if u else '-'} | {ident.get('file_count', '-')} | `{str(ident.get('source_tree_sha256', '-'))}` | `{str(ident.get('archive_sha256', '-'))}` | {up.get('tag', '')} `{str(up.get('commit', ''))[:12]}` | {pats} | {eqs} | "
                     f"{loc.get('application_owned_code_loc', '-')} / {loc.get('agent_modifiable_code_loc', '-')} / {loc.get('bundled_dependency_code_loc', '-')} / {loc.get('benchmark_specific_dependency_code_loc', '-')} / {loc.get('test_code_loc', '-')} / {loc.get('total_materialized_code_loc', '-')} |")
    lines += ["", "LOC = cloc 2.06 code lines of the materialized tree (no blank/comment lines, documentation and data excluded); categories from `optimization_scope.yaml` (`loc_categories`). "
              "The validated results recorded above were produced from trees proven content-equivalent to these bundles (`provenance/equivalence*.md`); they are not re-run by the migration."]
    return "\n".join(lines)


def audit_table():
    lines = ["## Materialized source LOC per application (frozen bundles, 2026-09-08)", "",
             "cloc 2.06 code lines (no blank/comment lines; documentation, examples/data, build output excluded) of the frozen source bundles "
             "(`level3/<app>/provenance/LOC*.json`, categories from `optimization_scope.yaml`). This replaces the whole-checkout `wc -l` figures: "
             "the +9,972 lines of the integration PR are the harness, not application code.", "",
             "| Application | variant | application_owned_code_loc | bundled_dependency_code_loc | benchmark_specific_dependency_code_loc | test_loc | total_materialized_code_loc | agent_modifiable_code_loc |",
             "|---|---|---|---|---|---|---|---|"]
    for app in APPS:
        for v, by, lock, ident, loc in app_data(app):
            lines.append(f"| {by.get('application', app)} | {v or '-'} | {loc.get('application_owned_code_loc', '-')} | {loc.get('bundled_dependency_code_loc', '-')} | {loc.get('benchmark_specific_dependency_code_loc', '-')} | {loc.get('test_code_loc', '-')} | {loc.get('total_materialized_code_loc', '-')} | {loc.get('agent_modifiable_code_loc', '-')} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", default=",".join(APPS))
    a = ap.parse_args()
    for app in a.apps.split(","):
        p = os.path.join(R, "level3", app, "README.md")
        text = open(p).read()
        new = replace_section(text, BEGIN, END, section(app))
        if new != text:
            open(p, "w").write(new); print(f"updated {os.path.relpath(p, R)}")
    p = os.path.join(R, "level3", "APPLICATION_AUDIT.md")
    text = open(p).read()
    new = replace_section(text, ABEGIN, AEND, audit_table())
    if new != text:
        open(p, "w").write(new); print(f"updated {os.path.relpath(p, R)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
