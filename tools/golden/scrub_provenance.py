#!/usr/bin/env python3
"""Remove machine identity (host name, GPU UUID / index, batch job, private log location) from the provenance of frozen
reference JSONs -- and nothing else.

    scrub_provenance.py plan   [--host NAME ...] FILE...                  what would be removed (nothing is written)
    scrub_provenance.py apply  [--host NAME ...] FILE...                  rewrite in place (deterministic, idempotent)
    scrub_provenance.py verify [--host NAME ...] --against REV FILE...    prove a rewrite against the file at git REV

Which fields go is fixed per schema (RULES); everything a reproduction or a check needs stays: the platform (GPU model,
driver, CUDA Toolkit), git head, binary / tool / log sha256, the repository-relative command and run directories, dates,
and -- untouched, byte for byte -- every key outside the provenance block (workload, spec, outputs with their sha256,
blobs, rules, d0 and tol_effective; Nyx archive and comparison; Level 2 cases / final_step / tolerances). The block gets
`scrubbed` = {by, rule, removed: [field paths], redacted_text: [fields], why}: WHAT was removed, never the values.

A file is refused (exit 2, not written) when its schema has no rule, when it is not exactly the json.dumps output of its
own content (re-serialising it would change bytes outside the provenance), when the provenance block is missing or a
field to remove sits under a non-mapping, or when the self-check fails. `apply` on a scrubbed file changes nothing.
`verify` exits 0 only when every FILE is byte-identical to the deterministic scrub of its content at REV, its keys outside
the provenance set are byte-identical to REV's and its provenance lost only the declared fields; 1 otherwise.
Free-text provenance fields of the Level 2 schemas (`note`) lose the host names the file itself records (its removed
`host` fields) and those named with --host: " on <host>" is dropped, any other mention becomes "the reference node".
CPU only, python stdlib; the comparison data are never read for anything but the equality checks.
"""
import argparse, copy, hashlib, json, os, re, subprocess, sys
from pathlib import Path

SCRUB_RULE = "hpcperf-provenance-scrub-1"
WHY = ("machine identity (host name, GPU UUID / index, batch job, private log location) is not needed to reproduce or "
       "check this reference; the platform (GPU model, driver, CUDA Toolkit), git head, sha256 values and every "
       "comparison datum are unchanged")
# schema -> the provenance block, the fields removed from it (dotted paths inside the block), the top-level free-text
# provenance fields whose host-name mentions are redacted. Nothing outside block + text is ever modified.
RULES = {
    "hpcperf-golden-1": {"block": "provenance", "remove": ("host", "gpu.uuid", "gpu.index"), "text": ()},
    "hpcperf-nyx-frozen-reference-1": {"block": "provenance", "remove": ("host", "gpu.uuid", "gpu.index"), "text": ()},
    "hpcperf-branson-reference-1": {"block": "reference_run", "remove": ("host", "allocation", "slurm_job", "log_kept_at"),
                                    "text": ("note",)},
    "hpcperf-remhos-cpu-backend-reference-1": {"block": "reference_run", "remove": ("host", "allocation", "slurm_job", "log_kept_at"),
                                               "text": ("note",)},
}
# the json.dumps formats a reference may have been written with; a file written with sorted keys keeps them sorted
FORMATS = [(i, s, a) for i in (2, 1, 4, None) for s in (True, False) for a in (True, False)]


class Refused(Exception):
    pass


def dumps(doc, fmt):
    indent, sort_keys, ascii_ = fmt
    return json.dumps(doc, indent=indent, sort_keys=sort_keys, ensure_ascii=ascii_) + "\n"


def parse(raw: str, where: str):
    """(doc, rule, fmt): the document, its schema's rule and the exact json.dumps format that reproduces `raw`."""
    try:
        doc = json.loads(raw)
    except ValueError as ex:
        raise Refused(f"{where}: not JSON ({ex})")
    if not isinstance(doc, dict):
        raise Refused(f"{where}: not a JSON object")
    rule = RULES.get(doc.get("schema"))
    if rule is None:
        raise Refused(f"{where}: schema {doc.get('schema')!r} has no scrub rule (known: {', '.join(sorted(RULES))}) -- not touched")
    fmt = next((f for f in FORMATS if dumps(doc, f) == raw), None)
    if fmt is None:
        raise Refused(f"{where}: not the exact json.dumps output of its content (re-serialising would change bytes outside "
                      f"the provenance) -- not touched")
    if not isinstance(doc.get(rule["block"]), dict):
        raise Refused(f"{where}: no provenance block '{rule['block']}' (a mapping)")
    for t in rule["text"]:
        if doc.get(t) is not None and not isinstance(doc[t], str):
            raise Refused(f"{where}: '{t}' is not free text")
    return doc, rule, fmt


def host_names(doc, rule, extra):
    """The host names to redact from free text: the values of the block's `host` fields (qualified and short) + --host."""
    names = set()
    blk = doc.get(rule["block"]) or {}
    for path in rule["remove"]:
        if path.split(".")[-1] != "host":
            continue
        v = get(blk, path)
        if isinstance(v, str) and v.strip():
            names |= {v.strip(), v.strip().split(".")[0]}
    names |= {h.strip() for h in extra or [] if h.strip()}
    return sorted((n for n in names if len(n) >= 3), key=lambda n: (-len(n), n))


def redact(text, names):
    for n in names:
        rx = re.escape(n)
        text = re.sub(rf"\s+(?:on|at)\s+(?:the\s+node\s+)?{rx}\b(?:\.[A-Za-z0-9.-]+)?", "", text)
        text = re.sub(rf"\b{rx}\b(?:\.[A-Za-z0-9.-]+)?", "the reference node", text)
    return text


_MISSING = object()


def get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return _MISSING
        d = d[k]
    return d


def scrub_doc(doc, rule, extra_hosts, where):
    """(new doc, removed paths, redacted text fields). Works on a deep copy; touches only the block and rule['text']."""
    new = copy.deepcopy(doc)
    blk = new[rule["block"]]
    removed = []
    for path in rule["remove"]:
        *parents, leaf = path.split(".")
        cur = blk
        for p in parents:
            if p not in cur:
                cur = None
                break
            if not isinstance(cur[p], dict):
                if cur[p] is None:
                    cur = None
                    break
                raise Refused(f"{where}: {rule['block']}.{'.'.join(parents)} is not a mapping -- not touched")
            cur = cur[p]
        if cur is not None and leaf in cur:
            del cur[leaf]
            removed.append(path)
    names = host_names(doc, rule, extra_hosts)
    redacted = []
    for t in rule["text"]:
        if isinstance(new.get(t), str):
            r = redact(new[t], names)
            if r != new[t]:
                new[t] = r
                redacted.append(t)
    if removed or redacted:
        prev = blk.get("scrubbed")
        if prev is not None and not (isinstance(prev, dict) and prev.get("rule") == SCRUB_RULE):
            raise Refused(f"{where}: {rule['block']}.scrubbed exists but is not a {SCRUB_RULE} record -- not touched")
        rec = {"by": "tools/golden/scrub_provenance.py", "rule": SCRUB_RULE,
               "removed": sorted(set(removed) | set((prev or {}).get("removed") or []))}
        rt = sorted({f"{t}: host name" for t in redacted} | set((prev or {}).get("redacted_text") or []))
        if rt:
            rec["redacted_text"] = rt
        rec["why"] = WHY
        blk.pop("scrubbed", None)
        blk["scrubbed"] = rec
    return new, removed, redacted


def placeholder(doc, rule):
    """The document with the provenance set (block + text fields) replaced by a fixed marker: what must stay identical."""
    d = dict(doc)
    for k in (rule["block"],) + tuple(rule["text"]):
        if k in d:
            d[k] = "<provenance>"
    return d


def outside_problems(old, new, rule, fmt):
    """Every way `new` differs from `old` outside the provenance set (empty list: byte-identical there)."""
    probs = []
    prov = {rule["block"], *rule["text"]}
    if [k for k in old if k not in prov] != [k for k in new if k not in prov]:
        probs.append("the set or order of the keys outside the provenance differs")
    for k in old:
        if k not in prov and (k not in new or json.dumps(old[k]) != json.dumps(new[k])):
            probs.append(f"key '{k}' differs")
    if dumps(placeholder(old, rule), fmt) != dumps(placeholder(new, rule), fmt):
        probs.append("the serialised bytes outside the provenance differ")
    return probs


def scrub_text(raw, where, extra_hosts):
    """(new raw text, removed, redacted) -- the deterministic scrub of one file's content, self-checked."""
    doc, rule, fmt = parse(raw, where)
    new, removed, redacted = scrub_doc(doc, rule, extra_hosts, where)
    out = dumps(new, fmt)
    probs = outside_problems(doc, new, rule, fmt)
    if probs:
        raise Refused(f"{where}: self-check failed ({'; '.join(probs)}) -- not touched")
    return out, removed, redacted


def repo_of(path: Path) -> Path:
    r = subprocess.run(["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if r.returncode != 0:
        raise Refused(f"{path}: not inside a git repository")
    return Path(r.stdout.strip())


def git_blob(path: Path, rev: str) -> str:
    repo = repo_of(path)
    rel = os.path.relpath(path.resolve(), repo.resolve())
    r = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{rel}"], capture_output=True)
    if r.returncode != 0:
        raise Refused(f"{rel}: not in {rev} ({r.stderr.decode(errors='replace').strip()})")
    return r.stdout.decode()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "apply", "verify"))
    ap.add_argument("files", nargs="+")
    ap.add_argument("--host", action="append", default=[], help="another host name to redact from free-text provenance fields")
    ap.add_argument("--against", help="verify: the git revision holding the unscrubbed files (e.g. HEAD)")
    a = ap.parse_args(argv)
    if a.cmd == "verify" and not a.against:
        ap.error("verify needs --against REV")
    rc = 0
    n_changed = n_same = 0
    for f in a.files:
        p = Path(f)
        try:
            if a.cmd == "verify":
                old_raw = git_blob(p, a.against)
                new_raw = p.read_text()
                want, removed, redacted = scrub_text(old_raw, f"{f}@{a.against}", a.host)
                old, rule, fmt = parse(old_raw, f"{f}@{a.against}")
                try:
                    new, _, _ = parse(new_raw, f)
                    probs = outside_problems(old, new, rule, fmt)
                except Refused as ex:
                    probs = [str(ex)]
                if new_raw != want:
                    probs.append("the file is not the deterministic scrub of its content at " + a.against)
                if probs:
                    rc = max(rc, 1)
                    print(f"FAIL {f}: " + "; ".join(probs))
                else:
                    n_changed += old_raw != new_raw
                    n_same += old_raw == new_raw
                    print(f"ok   {f}: outside the provenance byte-identical to {a.against}; removed "
                          f"{', '.join(removed) or 'nothing'}{'; redacted ' + ', '.join(redacted) if redacted else ''} "
                          f"(sha256 {hashlib.sha256(old_raw.encode()).hexdigest()[:12]} -> {hashlib.sha256(new_raw.encode()).hexdigest()[:12]})")
                continue
            raw = p.read_text()
            out, removed, redacted = scrub_text(raw, f, a.host)
            what = ", ".join(removed) + ("; redacted " + ", ".join(redacted) if redacted else "")
            if out == raw:
                n_same += 1
                print(f"unchanged {f}: nothing to scrub")
                continue
            n_changed += 1
            if a.cmd == "apply":
                tmp = p.with_name(p.name + ".scrub-tmp")
                tmp.write_text(out)
                os.replace(tmp, p)
                print(f"scrubbed  {f}: removed {what}")
            else:
                print(f"would scrub {f}: remove {what}")
        except (Refused, OSError, UnicodeDecodeError) as ex:
            rc = max(rc, 2)
            print(f"REFUSED {ex}")
    verb = {"plan": "would change", "apply": "changed", "verify": "verified (changed)"}[a.cmd]
    print(f"scrub_provenance {a.cmd}: {len(a.files)} file(s), {verb} {n_changed}, unchanged {n_same}, exit {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
