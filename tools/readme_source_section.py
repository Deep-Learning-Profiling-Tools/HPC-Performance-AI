#!/usr/bin/env python3
"""readme_source_section -- keep the "Source distribution (frozen source artifact)" section of every level3/<app>/README.md
and the LOC table of level3/APPLICATION_AUDIT.md in sync with provenance/source.lock*.yaml (schema hpcperf-source-lock-2),
LOC*.json and benchmark.yaml. The section is delimited by marker comments and rewritten in place (created at the end of the
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
import hpcperf_lock as hl  # noqa: E402

R = os.path.abspath(os.path.join(HERE, ".."))
APPS = ["lammps", "sparta", "warpx", "specfem3d", "nekrs", "nyx", "cp2k", "qmcpack", "dftfe", "geos", "exaca"]
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
        ident = hl.identity_from_benchmark(by, v)
        loc_p = os.path.join(d, "provenance", f"LOC{suffix}.json")
        loc = json.load(open(loc_p))["summary"] if os.path.isfile(loc_p) else {}
        out.append((v, by, lock, ident, loc))
    return out


def section(app):
    rows = app_data(app)
    by0 = rows[0][1] if rows else {}
    suite = by0.get("suite_status", "retained")
    lines = ["## Source distribution (frozen source artifact, scheme 3, 2026-09-10)", ""]
    if suite == "retired":
        lines += ["**RETIRED_FROM_DEFAULT_SUITE**: this application is no longer part of the Level 3 default suite (dependency redistribution/licensing constraints: ParMETIS 4.0.3 in the third-party dependency set, and the project decision to replace the application). "
                  "No source artifact is staged or published for it; the code, provenance and historical results below stay in git as a record. The local materialized tree, if present, is research-only.", ""]
    elif suite == "candidate":
        lines += ["**Replacement candidate** for the tenth slot of the default suite; admission depends on the bring-up criteria recorded in this README.", ""]
    lines += ["The application source is not in git and not read from `_upstream/`: `tools/prepare_benchmark.sh level3 " + app +
              "` materializes the frozen source artifact (`<app>[-<variant>]-<source_version>.tar.zst`, found in the local content-addressed cache `.artifacts/sha256/` or downloaded from the immutable URL recorded in `provenance/source.lock*.yaml` once published; `--artifact FILE` for a local copy) into `src/` (+ `deps/`), the only source `build.sh`/`run.sh`/`validate.sh` use. "
              "Archive size + sha256 and `source_tree_sha256` are verified before anything is placed. Identity, patch series, licenses, redistribution status and the equivalence proof against the tree the results above were validated from are under `provenance/` "
              "(`source.lock*.yaml`, `patch_series*.txt`, `original_vs_baseline*.diff`, `LICENSES*.md`, `equivalence*.md`, `LOC*.md`); `optimization_scope.yaml` says what an agent may modify; `benchmark.yaml` is the machine-readable contract. Remote status: see `level3/SOURCE_ARTIFACTS.md`.", ""]
    lines += ["| variant | artifact | source version | compressed / uncompressed | entries | source_tree_sha256 | archive sha256 | upstream | patches (pre-applied) | redistribution | equivalence | remote | LOC app-owned / agent-modifiable / bundled deps / benchmark deps / tests / total |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, by, lock, ident, loc in rows:
        eq = lock.get("equivalence", [])
        eqs = "SKIPPED" if eq == "SKIPPED" else (", ".join(f"{e.get('archive_path')}: {e.get('status')}" for e in eq) if eq else "not frozen")
        up = lock.get("upstream", {})
        pats = ", ".join(os.path.basename(p["path"]) for p in lock.get("patches", [])) or "none"
        c, u = ident.get("size"), ident.get("uncompressed_size")
        prim = (lock.get("artifact") or {}).get("primary") or {}
        lines.append(f"| {v or '-'} | `{ident.get('filename', '-')}` | {ident.get('source_version', '-')} | {hs.human(c) if c else '-'} / {hs.human(u) if u else '-'} | {ident.get('file_count', '-')} | `{str(ident.get('source_tree_sha256', '-'))}` | `{str(ident.get('archive_sha256', '-'))}` | {up.get('tag', '')} `{str(up.get('commit', ''))[:12]}` | {pats} | {lock.get('redistribution_status', '-')} | {eqs} | {'REMOTE_FETCH_VERIFIED' if prim.get('status') == 'published' else 'REMOTE_ARTIFACT_UNPUBLISHED'} | "
                     f"{loc.get('application_owned_code_loc', '-')} / {loc.get('agent_modifiable_code_loc', '-')} / {loc.get('bundled_dependency_code_loc', '-')} / {loc.get('benchmark_specific_dependency_code_loc', '-')} / {loc.get('test_code_loc', '-')} / {loc.get('total_materialized_code_loc', '-')} |")
    lines += ["", "LOC = cloc 2.06 code lines of the materialized tree (no blank/comment lines, documentation and data excluded); categories from `optimization_scope.yaml` (`loc_categories`). "
              "The validated results recorded above were produced from trees proven content-equivalent to this artifact (`provenance/equivalence*.md`); they are not re-run by the migration."]
    return "\n".join(lines)


def audit_table():
    lines = ["## Materialized source LOC per application (frozen source artifacts, 2026-09-10)", "",
             "cloc 2.06 code lines (no blank/comment lines; documentation, examples/data, build output excluded) of the frozen source artifacts "
             "(`level3/<app>/provenance/LOC*.json`, categories from `optimization_scope.yaml`). This replaces the whole-checkout `wc -l` figures: "
             "the +9,972 lines of the integration PR are the harness, not application code. Suite status: retained = default suite; "
             "retired = GEOS (not counted in the suite totals); candidate = ExaCA (admission pending).", "",
             "| Application | variant | suite | application_owned_code_loc | bundled_dependency_code_loc | benchmark_specific_dependency_code_loc | test_loc | total_materialized_code_loc | agent_modifiable_code_loc |",
             "|---|---|---|---|---|---|---|---|---|"]
    for app in APPS:
        for v, by, lock, ident, loc in app_data(app):
            lines.append(f"| {by.get('application', app)} | {v or '-'} | {by.get('suite_status', '-')} | {loc.get('application_owned_code_loc', '-')} | {loc.get('bundled_dependency_code_loc', '-')} | {loc.get('benchmark_specific_dependency_code_loc', '-')} | {loc.get('test_code_loc', '-')} | {loc.get('total_materialized_code_loc', '-')} | {loc.get('agent_modifiable_code_loc', '-')} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", default=",".join(APPS))
    a = ap.parse_args()
    for app in a.apps.split(","):
        p = os.path.join(R, "level3", app, "README.md")
        if not os.path.isfile(p):
            continue
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
