#!/usr/bin/env python3
"""Remove machine identity (host name, GPU UUID / index, batch job, private log location) from the provenance of frozen
reference JSONs -- and nothing else.

    scrub_provenance.py plan   [--host NAME ...] FILE...                  what would be removed (nothing is written)
    scrub_provenance.py apply  [--host NAME ...] FILE...                  rewrite in place (deterministic, idempotent)
    scrub_provenance.py verify [--host NAME ...] --against REV FILE...    prove a rewrite against the file at git REV

Which fields go is fixed per schema (RULES); everything a reproduction or a check needs stays: the platform (GPU model,
driver, CUDA Toolkit), git head, binary / tool / log sha256, the repository-relative command and run directories, dates,
and -- untouched, byte for byte -- every key outside the provenance block (workload, spec, outputs with their sha256,
blobs, rules, d0 and tol_effective; Nyx archive and comparison; Level 2 cases / final_step / tolerances / runs /
columns). The block gets `scrubbed` = {by, rule, removed: [field paths], redacted_text: ["field: kind"], why}: WHAT was
removed, never the values. Schemas without a provenance block (hipBone, miniWeather: block None, provenance only in the
top-level `note`) get the record as a top-level `scrubbed` key right after that free-text field.

A file is refused (exit 2, not written) when its schema has no rule, when none of the serialisations below reproduces it
byte for byte (re-serialising it would change bytes outside the provenance), when the provenance block is missing or a
field to remove sits under a non-mapping, when free text still mentions a batch job after the redaction (a form this
tool does not remove cleanly), or when the self-check fails. `apply` on a scrubbed file changes nothing.
`verify` exits 0 only when every FILE is byte-identical to the deterministic scrub of its content at REV, its keys outside
the provenance set are byte-identical to REV's and its provenance lost only the declared fields; 1 otherwise.

Serialisation: each file's writer is detected, never assumed -- json.dumps with indent 2 / 1 / 4 / none, sorted or
insertion-ordered keys, ensure_ascii on / off, the default, compact or legacy separator pair, the file's own trailing
newline (or none), and optionally PENNANT check_xy.py's post-processing that puts every [index, value] sample on one
line. The first combination that reproduces the file exactly is the one the rewrite uses.

Free text (`text`: top-level fields such as the Level 2 `note`; `block_text`: fields inside the block such as NekBone /
PENNANT provenance.binary and .note) loses the host names the file itself records (its removed `host` fields) and those
named with --host: " on <host>" / " at <host>" is dropped, any other mention becomes "the reference node"; a
parenthesised " (Slurm job N)" and a ", Slurm job N" are dropped. Everything else in the text stays verbatim (CPU core
lists, thread counts, dates, run times). A field in `remove_if_abs` (log_kept_at) goes only when it names an absolute
path (/projects/..., /home/..., /tmp/..., ~/...); a location relative to the results directory or the repository stays.
CPU only, python stdlib; the comparison data are never read for anything but the equality checks.
"""
import argparse, copy, hashlib, json, os, re, subprocess, sys
from pathlib import Path

SCRUB_RULE = "hpcperf-provenance-scrub-1"
WHY = ("machine identity (host name, GPU UUID / index, batch job, private log location) is not needed to reproduce or "
       "check this reference; the platform (GPU model, driver, CUDA Toolkit), git head, sha256 values and every "
       "comparison datum are unchanged")
# schema -> the provenance block (None: the schema has none), the fields removed from it (dotted paths inside the block),
# the fields removed only when they name an absolute path, the top-level ("text") and in-block ("block_text") free-text
# provenance fields whose host-name / batch-job mentions are redacted. Nothing outside block + text + record is modified.
RULES = {
    "hpcperf-golden-1": {"block": "provenance", "remove": ("host", "gpu.uuid", "gpu.index"), "text": ()},
    "hpcperf-nyx-frozen-reference-1": {"block": "provenance", "remove": ("host", "gpu.uuid", "gpu.index"), "text": ()},
    "hpcperf-branson-reference-1": {"block": "reference_run", "remove": ("host", "allocation", "slurm_job"),
                                    "remove_if_abs": ("log_kept_at",), "text": ("note",)},
    "hpcperf-remhos-cpu-backend-reference-1": {"block": "reference_run", "remove": ("host", "allocation", "slurm_job"),
                                               "remove_if_abs": ("log_kept_at",), "text": ("note",)},
    # the Level 2 CPU references made before the golden work (2026-09-30, 2026-10-20/21): hipBone and miniWeather keep
    # their provenance in the top-level note only; NekBone and PENNANT in a provenance block with host + free text
    "hpcperf-hipbone-reference-1": {"block": None, "remove": (), "text": ("note",)},
    "hpcperf-miniweather-reference-1": {"block": None, "remove": (), "text": ("note",)},
    "hpcperf-nekbone-cg-reference-1": {"block": "provenance", "remove": ("host",), "text": (), "block_text": ("binary", "note")},
    "hpcperf-pennant-reference-1": {"block": "provenance", "remove": ("host",), "text": (), "block_text": ("binary", "note")},
}
RECORD = "scrubbed"
ABS_PATH = re.compile(r"(?:^|[\s(\[{=:,;'\"])(?:~|/[A-Za-z0-9._-]+)/")
JOB_DROP = (re.compile(r"\s*\(\s*(?:Slurm|SLURM|batch)\s+job\s+\d+\s*\)"), re.compile(r",\s*(?:Slurm|SLURM|batch)\s+job\s+\d+\b"))
JOB_LEFT = re.compile(r"\b(?:Slurm|SLURM|batch)\s+job\s+\d+|SLURM_JOB_ID\s*=?\s*\d+")
# post-processing of a writer after json.dumps: PENNANT check_xy.py puts every [index, value] sample on one line
POST = {None: lambda t: t,
        "pairs-inline": lambda t: re.sub(r'\[\s*(\d+),\s*(-?[0-9.eE+-]+)\s*\]', r'[\1, \2]', t)}
SEPARATORS = (None, (",", ":"), (", ", ": "))


class Refused(Exception):
    pass


def formats(raw):
    """Every serialisation this tool can reproduce, in preference order (the trailing newline is the file's own)."""
    nl = "\n" if raw.endswith("\n") else ""
    for indent in (2, 1, 4, None):
        for sort_keys in (True, False):
            for ascii_ in (True, False):
                for seps in SEPARATORS:
                    for post in POST:
                        yield (indent, sort_keys, ascii_, seps, nl, post)


def dumps(doc, fmt):
    indent, sort_keys, ascii_, seps, nl, post = fmt
    return POST[post](json.dumps(doc, indent=indent, sort_keys=sort_keys, ensure_ascii=ascii_, separators=seps)) + nl


def prov_keys(rule):
    """The top-level keys of the provenance set: the block (or, without one, the record) and the top-level text fields."""
    return ({rule["block"]} if rule["block"] else {RECORD}) | set(rule["text"])


def parse(raw: str, where: str):
    """(doc, rule, fmt): the document, its schema's rule and the exact serialisation that reproduces `raw`."""
    try:
        doc = json.loads(raw)
    except ValueError as ex:
        raise Refused(f"{where}: not JSON ({ex})")
    if not isinstance(doc, dict):
        raise Refused(f"{where}: not a JSON object")
    rule = RULES.get(doc.get("schema"))
    if rule is None:
        raise Refused(f"{where}: schema {doc.get('schema')!r} has no scrub rule (known: {', '.join(sorted(RULES))}) -- not touched")
    fmt = next((f for f in formats(raw) if dumps(doc, f) == raw), None)
    if fmt is None:
        raise Refused(f"{where}: not the exact json.dumps output of its content (re-serialising would change bytes outside "
                      f"the provenance) -- not touched")
    if rule["block"] and not isinstance(doc.get(rule["block"]), dict):
        raise Refused(f"{where}: no provenance block '{rule['block']}' (a mapping)")
    texts = [(doc, t, t) for t in rule["text"]] + [(doc[rule["block"]], t, f"{rule['block']}.{t}") for t in rule.get("block_text", ())]
    for holder, t, label in texts:
        if holder.get(t) is not None and not isinstance(holder[t], str):
            raise Refused(f"{where}: '{label}' is not free text")
    return doc, rule, fmt


def host_names(doc, rule, extra):
    """The host names to redact from free text: the values of the block's `host` fields (qualified and short) + --host."""
    names = set()
    blk = (doc.get(rule["block"]) or {}) if rule["block"] else {}
    for path in rule["remove"]:
        if path.split(".")[-1] != "host":
            continue
        v = get(blk, path)
        if isinstance(v, str) and v.strip():
            names |= {v.strip(), v.strip().split(".")[0]}
    names |= {h.strip() for h in extra or [] if h.strip()}
    return sorted((n for n in names if len(n) >= 3), key=lambda n: (-len(n), n))


def redact(text, names):
    """(text, kinds): the text without its host-name and batch-job mentions, and which of the two kinds it had."""
    kinds = []
    for n in names:
        rx = re.escape(n)
        new = re.sub(rf"\s+(?:on|at)\s+(?:the\s+node\s+)?{rx}\b(?:\.[A-Za-z0-9.-]+)?", "", text)
        new = re.sub(rf"\b{rx}\b(?:\.[A-Za-z0-9.-]+)?", "the reference node", new)
        if new != text and "host name" not in kinds:
            kinds.append("host name")
        text = new
    for rx in JOB_DROP:
        new = rx.sub("", text)
        if new != text and "batch job" not in kinds:
            kinds.append("batch job")
        text = new
    return text, kinds


_MISSING = object()


def get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return _MISSING
        d = d[k]
    return d


def scrub_doc(doc, rule, extra_hosts, where):
    """(new doc, removed paths, redacted "field: kind" list). Works on a deep copy; touches only the block, the text
    fields and the record."""
    new = copy.deepcopy(doc)
    blk = new[rule["block"]] if rule["block"] else None
    removed = []
    conditional = set(rule.get("remove_if_abs", ()))
    for path in tuple(rule["remove"]) + tuple(rule.get("remove_if_abs", ())):
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
            if path in conditional and not (isinstance(cur[leaf], str) and ABS_PATH.search(cur[leaf])):
                continue                    # a location relative to the results directory / the repository stays
            del cur[leaf]
            removed.append(path)
    names = host_names(doc, rule, extra_hosts)
    redacted = []
    texts = [(new, t, t) for t in rule["text"]] + [(blk, t, t) for t in rule.get("block_text", ())]
    for holder, key, label in texts:
        if isinstance(holder.get(key), str):
            r, kinds = redact(holder[key], names)
            if JOB_LEFT.search(r):
                raise Refused(f"{where}: '{label}' mentions a batch job in a form this tool does not remove cleanly -- not touched")
            if r != holder[key]:
                holder[key] = r
                redacted += [f"{label}: {k}" for k in kinds]
    if removed or redacted:
        prev = (blk if blk is not None else new).get(RECORD)
        if prev is not None and not (isinstance(prev, dict) and prev.get("rule") == SCRUB_RULE):
            raise Refused(f"{where}: {rule['block'] or '(top level)'}.{RECORD} exists but is not a {SCRUB_RULE} record -- not touched")
        rec = {"by": "tools/golden/scrub_provenance.py", "rule": SCRUB_RULE,
               "removed": sorted(set(removed) | set((prev or {}).get("removed") or []))}
        rt = sorted(set(redacted) | set((prev or {}).get("redacted_text") or []))
        if rt:
            rec["redacted_text"] = rt
        rec["why"] = WHY
        if blk is not None:
            blk.pop(RECORD, None)
            blk[RECORD] = rec
        else:                               # no block: the record goes right after the (last) top-level text field
            new.pop(RECORD, None)
            anchor = ([t for t in rule["text"] if t in new] or [None])[-1]
            items = list(new.items())
            new.clear()
            for k, v in items:
                new[k] = v
                if k == anchor:
                    new[RECORD] = rec
            new.setdefault(RECORD, rec)
    return new, removed, redacted


def placeholder(doc, rule):
    """The document with the provenance set (block + text fields) replaced by a fixed marker and a top-level record
    (schemas without a block) left out: what must stay identical."""
    prov = prov_keys(rule)
    return {k: ("<provenance>" if k in prov else v) for k, v in doc.items() if rule["block"] or k != RECORD}


def outside_problems(old, new, rule, fmt):
    """Every way `new` differs from `old` outside the provenance set (empty list: byte-identical there)."""
    probs = []
    prov = prov_keys(rule)
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
            what = (", ".join(removed) or "nothing") + ("; redacted " + ", ".join(redacted) if redacted else "")
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
