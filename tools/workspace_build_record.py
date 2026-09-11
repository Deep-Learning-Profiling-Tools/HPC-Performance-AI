#!/usr/bin/env python3
"""workspace_build_record -- append a trusted build record for one agent-workspace build.

    workspace_build_record.py --workspace WS --app APP --iteration N --backend CUDA --variant V
                              --source-hash SHA --build-log REL --records FILE

Records which source hash was compiled, for which backend/variant, and the sha256 of every executable the
build produced under <WS>/build/level3/<app>/ and <WS>/.deps/level3/<app>/ (installed binaries included).
tools/validate_workspace.sh consults these records when a later iteration runs with --skip-build: a binary
whose sha256 appears in a record for the same source hash, backend and variant is `verified_from_build_record`,
anything else stays UNVERIFIED. Written by the trusted harness, outside the agent's working directory.
"""
import argparse
import datetime
import hashlib
import json
import os

MAX_BIN_BYTES = 4 << 30


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def executables(roots):
    out = {}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ("CMakeFiles", ".git", "run", "logs")]
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if not (st.st_mode & 0o111) or st.st_size < (1 << 20) or st.st_size > MAX_BIN_BYTES:
                    continue
                if fn.endswith((".sh", ".py", ".cmake", ".so", ".a")):
                    continue
                out[sha256(p)] = os.path.relpath(p, os.path.dirname(root.rstrip("/")))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for o in ("--workspace", "--app", "--iteration", "--backend", "--source-hash", "--records"):
        ap.add_argument(o, required=True)
    ap.add_argument("--variant", default=""); ap.add_argument("--build-log", default="")
    a = ap.parse_args()
    roots = [os.path.join(a.workspace, "build", "level3", a.app), os.path.join(a.workspace, ".deps", "level3", a.app)]
    bins = executables([r for r in roots if os.path.isdir(r)])
    rec = {"iteration": int(a.iteration), "backend": a.backend, "variant": a.variant or None, "source_hash": a.source_hash,
           "build_log": a.build_log, "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "executables": [{"sha256": s, "path": p} for s, p in sorted(bins.items(), key=lambda kv: kv[1])]}
    data = {"schema": "hpcperf-workspace-build-records-1", "records": []}
    if os.path.isfile(a.records):
        try:
            data = json.load(open(a.records))
        except ValueError:
            pass
    data.setdefault("records", []).append(rec)
    with open(a.records, "w") as f:
        json.dump(data, f, indent=1)
    print(f"validate_workspace: build record: source {a.source_hash[:16]}... backend {a.backend}"
          f"{' variant ' + a.variant if a.variant else ''}, {len(bins)} executable(s) recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
