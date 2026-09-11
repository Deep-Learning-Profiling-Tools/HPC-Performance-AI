#!/usr/bin/env python3
"""workspace_iteration_record -- write the verdict record of one agent-workspace validation iteration.

    workspace_iteration_record.py --workspace WS --app APP --benchmark-dir DIR --iteration N --run-id ID
                                  --backend CUDA --variant V --verdict PASS --exit-code 0
                                  --check iter-N.check.json --snapshot-before FILE --records build-records.json
                                  --built-this-iteration 0|1 --out iter-N.verdict.yaml

The runs of THIS iteration are the run_manifest.txt files that appeared or changed while the validator ran
(compared with the snapshot taken before it): every one is listed with its run id, ranks, exit code, binary and
binary sha256, plus the GPU-binding audit counts of its stdout log when one is next to it. No historical
manifest is consulted and no `tail -1` guess is made; when no manifest changed, the record says so instead of
attributing an older run to this iteration.

Build provenance of the validated binaries: `built_this_iteration` (the harness built in this iteration),
`verified_from_build_record` (a trusted build record for the same source hash, backend and variant contains the
binary sha256) or `UNVERIFIED` (`--skip-build` without such a record -- the record then states explicitly that
the current source modification is not proven to be compiled into the validated binary).
"""
import argparse
import datetime
import hashlib
import json
import os
import re

AUDIT = re.compile(r"audit summary: (\d+) verified, (\d+) mismatch, (\d+) unverified")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


KEYS = ("run_id", "app", "backend", "mode", "case", "ranks", "exit_code", "binary", "binary_sha256", "input", "input_sha256", "utc")


def split_records(text):
    """run.sh APPENDS to run_manifest.txt; every record starts with its run_id= line. Returns one dict per record
    (a leading block without run_id, e.g. a partially read tail, is kept as an anonymous record)."""
    recs, cur = [], {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k == "run_id" and cur:
            recs.append(cur); cur = {}
        if k in KEYS and k not in cur:
            cur[k] = v
    if cur:
        recs.append(cur)
    return recs


def parse_manifest(path, offset):
    """Parse only what this iteration appended: everything from `offset` (the file's size before the run); the
    whole file when it is new or was rewritten shorter than before."""
    size = os.path.getsize(path)
    with open(path, errors="replace") as f:
        if offset and size >= offset:
            f.seek(offset)
        text = f.read()
    recs = split_records(text)
    if not recs and offset:            # appended bytes carried no complete record: fall back to the whole file
        recs = split_records(open(path, errors="replace").read())[-1:]
    out = dict(recs[-1]) if recs else {}
    if len(recs) > 1:
        out["additional_records_this_iteration"] = [r.get("run_id") for r in recs[:-1] if r.get("run_id")]
    out["manifest"] = path
    d = os.path.dirname(path)
    for cand in ("stdout.log",) + tuple(f for f in sorted(os.listdir(d)) if f.endswith((".log", ".stdout"))):
        p = os.path.join(d, cand)
        if os.path.isfile(p):
            m = AUDIT.findall(open(p, errors="replace").read())
            if m:
                out["gpu_binding_audits"] = [{"verified": int(a), "mismatch": int(b), "unverified": int(c)} for a, b, c in m]
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for o in ("--workspace", "--app", "--benchmark-dir", "--iteration", "--run-id", "--backend", "--verdict", "--exit-code", "--check", "--snapshot-before", "--out"):
        ap.add_argument(o, required=True)
    ap.add_argument("--variant", default=""); ap.add_argument("--records", default=""); ap.add_argument("--built-this-iteration", default="0")
    a = ap.parse_args()
    chk = json.load(open(a.check))
    before = {}
    if os.path.isfile(a.snapshot_before):
        for line in open(a.snapshot_before):
            parts = line.strip().split(" ", 2)
            if len(parts) == 3:
                before[parts[2]] = (parts[0], int(parts[1]))
    now = {}
    build_root = os.path.join(a.workspace, "build")
    for dirpath, _dirs, files in os.walk(build_root):
        if "run_manifest.txt" in files:
            p = os.path.join(dirpath, "run_manifest.txt")
            now[p] = sha256(p)
    changed = sorted(p for p, h in now.items() if before.get(p, ("", 0))[0] != h)
    runs = []
    for p in changed:
        prev_size = before.get(p, ("", 0))[1]
        rec = parse_manifest(p, prev_size)
        rec["records_appended_this_iteration"] = True
        runs.append(rec)
    for r in runs:
        r["manifest"] = os.path.relpath(r["manifest"], a.workspace)
    bin_shas = {r.get("binary_sha256") for r in runs if r.get("binary_sha256")}
    # build provenance
    prov, prov_detail = "UNVERIFIED", ""
    if a.built_this_iteration == "1":
        prov, prov_detail = "built_this_iteration", "the harness built the workspace source in this iteration before validating"
    elif a.records and os.path.isfile(a.records) and bin_shas:
        recs = json.load(open(a.records)).get("records", [])
        match = [r for r in recs if r.get("source_hash") == chk["current_source_hash"] and r.get("backend") == a.backend
                 and (r.get("variant") or "") == (a.variant or "")
                 and bin_shas.issubset({e["sha256"] for e in r.get("executables", [])})]
        if match:
            prov = "verified_from_build_record"
            prov_detail = f"build record of iteration {match[-1]['iteration']} covers the validated binary for this source hash, backend and variant"
        else:
            prov_detail = ("--skip-build was used and no trusted build record for this source hash, backend and variant contains the validated "
                           "binary: the current source modification is NOT proven to be compiled into it")
    else:
        prov_detail = ("--skip-build was used and no run manifest or build record is available to link the validated binary to the current source: "
                       "the current source modification is NOT proven to be compiled into it")
    out = {
        "run_id": a.run_id, "iteration": int(a.iteration), "layer": "numerical", "verdict": a.verdict,
        "exit_code": int(a.exit_code), "validate_exit_code": int(a.exit_code),
        "backend": a.backend, "variant": a.variant or None,
        "baseline_origin": chk.get("baseline_origin"),
        "initial_source_hash": chk["initial_source_hash"], "current_source_hash": chk["current_source_hash"],
        "modified_files": chk["modified_files"], "added_files": chk["added_files"], "deleted_files": chk["deleted_files"],
        "build_provenance": prov, "build_provenance_detail": prov_detail,
        "runs_this_iteration": runs,
        "runs_this_iteration_count": len(runs),
        "validated_binaries": sorted(bin_shas),
        "note": ("runs_this_iteration lists exactly the run manifests that appeared or changed while this iteration's validator ran"
                 if runs else "no run manifest changed during this iteration: no run of this iteration is claimed"),
        "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    import yaml
    yaml.safe_dump(out, open(a.out, "w"), sort_keys=False, width=120)
    binlist = ", ".join(s[:16] + "..." for s in sorted(bin_shas)) or "none recorded"
    print(f"validate_workspace: {a.verdict} (numerical layer, iteration {a.iteration}, backend {a.backend}"
          f"{', variant ' + a.variant if a.variant else ''}, source {chk['current_source_hash'][:16]}..., "
          f"{len(runs)} run(s) this iteration, binary {binlist}, build provenance {prov}, "
          f"{len(chk['modified_files'])} modified file(s), baseline {chk.get('baseline_origin')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
