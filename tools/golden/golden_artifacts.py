#!/usr/bin/env python3
"""golden_artifacts -- reproducible distribution of the golden store (the raw bytes of the frozen references).

    golden_artifacts.py pack   --staging DIR [--lock-out FILE] [--jobs N] [--verify-determinism] [--rev REV]
    golden_artifacts.py fetch  [SELECTION...] [--from DIR]... [--cache-dir DIR] [--offline] [--drop-downloads]
    golden_artifacts.py verify [SELECTION...] [--quick] [--json OUT]          (alias: status)
    golden_artifacts.py lock-check [--rev REV]
    golden_artifacts.py plan   --tag TAG --commit SHA [--staging DIR] [--json OUT]
  common: [--repo DIR] [--lock FILE] [--store DIR]

The committed reference JSONs (level<N>/<benchmark>/reference/**.json in the git tree) are the root of trust:
they name every blob of the content-addressed store ($HPCPERF_GOLDEN_STORE, default <repo>/.golden-store,
layout sha256/<aa>/<sha256>.bin) by sha256 and size (schema hpcperf-golden-1: outputs[].blob/sha256/bytes;
hpcperf-nyx-frozen-reference-1: archive.sha256/size). A reference of any other schema that mentions the store
is an error (an extractor must be added here first).

pack     maintainer side, CPU only. Reads the references of the committed tree (git ls-tree REV, never the
         working tree or the index: an untracked, staged or modified reference is never packed), collects
         exactly the blobs they name (orphan blobs are never packed), groups them per benchmark (parts when a
         bundle could exceed the 2 GiB release-asset limit) and writes one deterministic bundle per group/part
         into the staging directory (outside the worktree; default $HPCPERF_ARTIFACT_STAGING/golden):
             golden-<level>-<benchmark>[-p<i>of<n>]-<members_digest[:16]>.tar.zst
         Format hpcperf-golden-tar-1: POSIX ustar written by this tool, regular-file members
         sha256/<aa>/<sha256>.bin only, sorted by name, mode 0644, uid/gid 0, empty user/group, mtime
         1704067200, no directory entries, two zero blocks, padded to 10240 bytes; compressed by
         `zstd -3 --single-thread` from a stdin stream (version recorded). Every blob is re-hashed while it is
         packed (a corrupted or missing blob aborts the pack), archive blobs (tar) are scanned for credentials
         (a hit aborts) and node-private locations (recorded for the redistribution review), every staged
         bundle is decompressed and re-verified, and the lock tools/golden/golden_artifacts.lock.yaml is
         written (schema hpcperf-golden-lock-1, no timestamp: the same references + store + zstd give the
         same lock bytes). A staged bundle is never overwritten with other bytes.
fetch    user side. Selects the committed references (all, or SELECTION = level1 | level1/adam |
         level1/adam/reference/x.json), skips blobs already present and correct, and for the others obtains
         the bundle of the lock: --from DIR (e.g. the staging) > the content-addressed cache
         ($HPCPERF_ARTIFACT_CACHE, default <repo>/.artifacts, sha256/<bundle sha256>.tar.zst) > the lock's
         published URL / mirrors (tools/hpcperf_source.download_to_cache, the code path of
         tools/prepare_benchmark.sh; --offline forbids it). Size + zstd magic + sha256 of the bundle are verified
         BEFORE extraction; extraction streams through `zstd -d` into a restricted reader (regular files with
         exactly the lock's member names and sizes, nothing else: no path, link, device, pax header, duplicate
         or extra member) into <store>/.incoming/, re-hashes every member against its name, the lock and the
         committed references, re-hashes the compressed stream (the file must not change while it is read),
         and only then renames each blob atomically into place (0444). Idempotent; a corrupted blob in the store
         is replaced by the verified one (REPAIRED); members no committed reference names are not installed.
verify   every blob the (selected) committed references name: present, a regular file, size and sha256
         correct (--quick: size only); without a selection also ORPHAN blobs (in the store, named by no
         committed reference) and STRAY files; lock coverage is reported. Exit 0 only when nothing is missing
         or corrupted (orphans are reported, never deleted).
lock-check  the lock against the committed references: schema, internal consistency (member digests,
         names, sizes), and that the bundles the current references require are exactly the lock's (CURRENT /
         STALE: re-pack after a reference was made or replaced). CPU only, needs no store.
plan     the publication plan of the staged bundles for a release tag (planned URLs, SHA256SUMS lines,
         PLAN/REFUSE per bundle); nothing is uploaded and no network is used. Uploading is the separate,
         maintainer-authorized step (HPCPERF_CONFIRM_UPLOAD=yes), never this tool.

Exit codes: 0 OK, 1 failure/invalid/stale, 2 usage, 4 a needed bundle is unavailable (not staged, not cached,
unpublished, offline, not in the lock), 5 integrity: hash/size mismatch, unsafe or unexpected content.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import hpcperf_source as hs  # noqa: E402  (sha256 / zstd / cache / download helpers of the Level 3 artifacts)
import hpcperf_lock as hl  # noqa: E402  (forbidden-location and floating-URL rules of the source locks)

LOCK_SCHEMA = "hpcperf-golden-lock-1"
LOCK_SCHEMA_VERSION = 1
STAGING_SCHEMA = "hpcperf-golden-staging-1"
PLAN_SCHEMA = "hpcperf-golden-release-plan-1"
GOLDEN_REF_SCHEMA = "hpcperf-golden-1"
NYX_REF_SCHEMA = "hpcperf-nyx-frozen-reference-1"
TAR_FORMAT = "hpcperf-golden-tar-1"
ZSTD_LEVEL = 3
ZSTD_FLAGS = ["--single-thread"]
ASSET_LIMIT = 2 * 1024 ** 3            # GitHub release assets: each file under 2 GiB
TAR_BLOCK = 512
TAR_RECORD = 10240
ZSTD_BLOCK = 128 * 1024
CHUNK = 4 << 20
MTIME = hs.ARCHIVE_MTIME               # 2024-01-01T00:00:00Z, the fixed mtime of every hpcperf archive entry
LOCK_DEFAULT = "tools/golden/golden_artifacts.lock.yaml"
STAGING_SUBDIR = "golden"
PROPOSED_TAG = "golden-hpcperf-g1-rc1"
REF_RX = re.compile(r"^level[0-9]+/[A-Za-z0-9_.+-]+/reference/(?:[^/]+/)*[^/]+\.json$")
MEMBER_RX = re.compile(r"^sha256/([0-9a-f]{2})/([0-9a-f]{64})\.bin$")
SHA_RX = re.compile(r"^[0-9a-f]{64}$")
STORE_HINT_RX = re.compile(r"golden-store|HPCPERF_GOLDEN_STORE|sha256/[0-9a-f]{2}/[0-9a-f]{64}\.bin")
REDISTRIBUTION = ("cleared", "review", "blocked")
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_UNAVAILABLE, EXIT_MISMATCH = 0, 1, 2, 4, 5


class DistError(Exception):
    def __init__(self, msg, code=EXIT_FAIL):
        super().__init__(msg)
        self.code = code


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------- store layout (= hpcperf_golden.py)
def store_root(repo: Path, explicit=None) -> Path:
    """--store > $HPCPERF_GOLDEN_STORE > <repo>/.golden-store (the resolution of hpcperf_golden.store_root)."""
    return Path(explicit or os.environ.get("HPCPERF_GOLDEN_STORE") or (repo / ".golden-store")).absolute()


def member_name(sha):
    return f"sha256/{sha[:2]}/{sha}.bin"


def blob_path(store: Path, sha) -> Path:
    return store / member_name(sha)


def blob_state(store: Path, sha, size, quick=False):
    """OK | MISSING | NOT_REGULAR | CORRUPTED (size or content) for one blob of the store."""
    p = blob_path(store, sha)
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        return "MISSING"
    if not stat.S_ISREG(st.st_mode):
        return "NOT_REGULAR"
    if st.st_size != size:
        return "CORRUPTED"
    if quick:
        return "OK"
    return "OK" if hs.sha256_file(str(p), CHUNK) == sha else "CORRUPTED"


# ----------------------------------------------------------------------------- committed references
def git(repo: Path, *args, binary=False, input=None):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, input=input)
    if r.returncode != 0:
        raise DistError(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout if binary else r.stdout.decode()


def committed_references(repo: Path, rev="HEAD"):
    """{path: {"json_sha256", "doc"}} for every level<N>/<b>/reference/**.json of the committed tree `rev`
    (blob contents from the object store: the working tree and the index are never read)."""
    out = git(repo, "ls-tree", "-r", "-z", "--full-tree", rev, binary=True)
    recs = []
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        _mode, typ, obj = meta.decode().split()
        p = path.decode("utf-8", "surrogateescape")
        if typ == "blob" and REF_RX.match(p):
            recs.append((p, obj))
    refs = {}
    if not recs:
        return refs
    data = git(repo, "cat-file", "--batch", binary=True, input="".join(o + "\n" for _, o in recs).encode())
    pos = 0
    for p, obj in recs:
        nl = data.index(b"\n", pos)
        hdr = data[pos:nl].decode().split()
        if len(hdr) != 3 or hdr[0] != obj or hdr[1] != "blob":
            raise DistError(f"git cat-file: unexpected header {hdr} for {p}")
        n = int(hdr[2])
        body = data[nl + 1:nl + 1 + n]
        pos = nl + 1 + n + 1
        try:
            doc = json.loads(body.decode())
        except (ValueError, UnicodeDecodeError) as ex:
            raise DistError(f"{p} (committed): not JSON ({ex})")
        refs[p] = {"json_sha256": hashlib.sha256(body).hexdigest(), "doc": doc}
    return refs


def worktree_drift(repo: Path):
    """(modified-or-staged, untracked) reference paths of the working tree: they are IGNORED by pack."""
    out = git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all", binary=True).decode("utf-8", "surrogateescape")
    mod, untracked = [], []
    for rec in out.split("\0"):
        if len(rec) < 4:
            continue
        code, p = rec[:2], rec[3:]
        if not REF_RX.match(p):
            continue
        (untracked if code == "??" else mod).append(p)
    return sorted(mod), sorted(untracked)


def _has_key(o, key):
    if isinstance(o, dict):
        return key in o or any(_has_key(v, key) for v in o.values())
    if isinstance(o, list):
        return any(_has_key(v, key) for v in o)
    return False


def _sha_size(path, where, sha, size):
    if not isinstance(sha, str) or not SHA_RX.match(sha):
        raise DistError(f"{path}: {where}: sha256 {sha!r} is not a sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise DistError(f"{path}: {where}: size {size!r} is not a byte count")
    return {"sha256": sha, "size": size, "where": where}


def blobs_of(path, doc):
    """The store blobs a committed reference names: [{sha256, size, where}]."""
    schema = doc.get("schema") if isinstance(doc, dict) else None
    if schema == GOLDEN_REF_SCHEMA:
        out = []
        for i, o in enumerate(doc.get("outputs") or []):
            if not isinstance(o, dict) or o.get("blob") is None:
                continue
            where = f"output {o.get('name', i)!r}"
            b = _sha_size(path, where, o.get("sha256"), o.get("bytes"))
            if o["blob"] != member_name(b["sha256"]):
                raise DistError(f"{path}: {where}: blob {o['blob']!r} is not the store path of its sha256")
            out.append(b)
        return out
    if schema == NYX_REF_SCHEMA:
        a = doc.get("archive") or {}
        return [_sha_size(path, "archive", a.get("sha256"), a.get("size"))]
    text = json.dumps(doc)
    if STORE_HINT_RX.search(text) or _has_key(doc, "blob"):
        raise DistError(f"{path}: schema {schema!r} refers to the golden store, but golden_artifacts.py does not know "
                        f"where its blobs are recorded: add an extractor to blobs_of() before packing")
    return []


def reference_index(refs):
    """(idx, per_ref): idx = {sha: {"size", "refs": [(path, where)]}}, per_ref = {path: sorted blob shas}
    (references naming no blob are not in per_ref)."""
    idx, per_ref = {}, {}
    for p in sorted(refs):
        shas = set()
        for b in blobs_of(p, refs[p]["doc"]):
            e = idx.setdefault(b["sha256"], {"size": b["size"], "refs": []})
            if e["size"] != b["size"]:
                raise DistError(f"blob {b['sha256']}: {p} ({b['where']}) records {b['size']} bytes, "
                                f"{e['refs'][0][0]} records {e['size']}")
            e["refs"].append((p, b["where"]))
            shas.add(b["sha256"])
        if shas:
            per_ref[p] = sorted(shas)
    return idx, per_ref


def group_of(ref_path):
    return "/".join(ref_path.split("/")[:2])


def select_refs(per_ref, refs, selection, repo: Path):
    """The reference paths (with blobs) matched by SELECTION items (level, benchmark dir or reference file)."""
    if not selection:
        return sorted(per_ref)
    chosen = set()
    for item in selection:
        it = os.path.relpath(item, repo) if os.path.isabs(item) else os.path.normpath(item)
        it = it.strip("/")
        hit = [p for p in refs if p == it or p.startswith(it + "/")]
        if not hit:
            raise DistError(f"selection {item!r} matches no committed reference", EXIT_USAGE)
        chosen.update(p for p in hit if p in per_ref)
    return sorted(chosen)


# ----------------------------------------------------------------------------- bundles: partition, names, format
def tar_size(sizes):
    n = sum(TAR_BLOCK + -(-s // TAR_BLOCK) * TAR_BLOCK for s in sizes) + 2 * TAR_BLOCK
    return -(-n // TAR_RECORD) * TAR_RECORD


def compress_bound(n):
    """ZSTD_compressBound(): the largest frame zstd can produce for n input bytes (incompressible data)."""
    return n + (n >> 8) + (((ZSTD_BLOCK - n) >> 11) if n < ZSTD_BLOCK else 0)


def members_digest(group, members):
    """sha256 over the bundle's identity: format, compression, group and the sorted (sha256, size) members."""
    h = hashlib.sha256()
    h.update(f"{TAR_FORMAT}\nzstd -{ZSTD_LEVEL} {' '.join(ZSTD_FLAGS)}\n{group}\n".encode())
    for sha, size in members:
        h.update(f"{sha} {size}\n".encode())
    return h.hexdigest()


def bundle_name(group, part, parts, digest):
    level, bench = group.split("/")
    return f"golden-{level}-{bench}{f'-p{part}of{parts}' if parts > 1 else ''}-{digest[:16]}.tar.zst"


def plan_bundles(idx, per_ref, asset_limit=ASSET_LIMIT):
    """Deterministic bundle layout: one group per benchmark; within a group first-fit decreasing by blob size
    (ties by sha256), a part taking blobs while ZSTD_compressBound(its tar size) stays below the asset limit."""
    groups = {}
    for p, shas in per_ref.items():
        groups.setdefault(group_of(p), set()).update(shas)
    bundles = []
    for g in sorted(groups):
        size = {s: idx[s]["size"] for s in groups[g]}
        bins = []
        for s in sorted(size, key=lambda x: (-size[x], x)):
            if compress_bound(tar_size([size[s]])) >= asset_limit:
                raise DistError(f"blob {s} ({size[s]} bytes) of {g} cannot fit one release asset of {asset_limit} bytes "
                                f"(splitting a blob is not implemented)")
            for b in bins:
                if compress_bound(tar_size([size[x] for x in b] + [size[s]])) < asset_limit:
                    b.append(s)
                    break
            else:
                bins.append([s])
        for i, b in enumerate(bins, 1):
            mem = [(s, size[s]) for s in sorted(b)]
            dig = members_digest(g, mem)
            bundles.append({"name": bundle_name(g, i, len(bins), dig), "group": g, "part": i, "parts": len(bins),
                            "members_digest": dig, "tar_size": tar_size([n for _, n in mem]),
                            "members": [{"sha256": s, "size": n} for s, n in mem]})
    return bundles


def _octal(v, width):
    s = format(v, "o").encode()
    if len(s) > width - 1:
        raise DistError(f"tar header: {v} does not fit {width - 1} octal digits")
    return s.rjust(width - 1, b"0") + b"\0"


def ustar_header(name, size):
    """One POSIX ustar header with the fixed metadata of hpcperf-golden-tar-1."""
    nb = name.encode("ascii")
    if len(nb) >= 100:
        raise DistError(f"tar header: name too long: {name}")
    h = bytearray(TAR_BLOCK)
    h[0:len(nb)] = nb
    h[100:108] = _octal(0o644, 8)
    h[108:116] = _octal(0, 8)
    h[116:124] = _octal(0, 8)
    h[124:136] = _octal(size, 12)
    h[136:148] = _octal(MTIME, 12)
    h[148:156] = b" " * 8
    h[156:157] = b"0"
    h[257:263] = b"ustar\0"
    h[263:265] = b"00"
    h[329:337] = _octal(0, 8)
    h[337:345] = _octal(0, 8)
    h[148:156] = format(sum(h), "06o").encode() + b"\0 "
    return bytes(h)


def zstd_cmd():
    return ["zstd", f"-{ZSTD_LEVEL}", *ZSTD_FLAGS, "-q", "--no-progress", "-c"]


def write_bundle(store: Path, bundle, out_path: Path):
    """Stream the bundle's tar into zstd (stdin) -> out_path. Every blob is checked (regular file, size, sha256)
    while it is read. Returns the tar size."""
    written = 0
    with open(out_path, "wb") as out:
        proc = subprocess.Popen(zstd_cmd(), stdin=subprocess.PIPE, stdout=out)
        try:
            w = proc.stdin
            for m in bundle["members"]:
                sha, size = m["sha256"], m["size"]
                p = blob_path(store, sha)
                try:
                    st = os.lstat(p)
                except FileNotFoundError:
                    raise DistError(f"{bundle['name']}: blob {sha} missing from the store {store}")
                if not stat.S_ISREG(st.st_mode) or st.st_size != size:
                    raise DistError(f"{bundle['name']}: blob {sha} in the store is not a regular file of {size} bytes", EXIT_MISMATCH)
                w.write(ustar_header(member_name(sha), size))
                h = hashlib.sha256()
                n = 0
                with open(p, "rb") as f:
                    for blk in iter(lambda: f.read(CHUNK), b""):
                        h.update(blk)
                        w.write(blk)
                        n += len(blk)
                if n != size or h.hexdigest() != sha:
                    raise DistError(f"{bundle['name']}: blob {sha} in the store does not match its sha256 (corrupted store)", EXIT_MISMATCH)
                pad = (-size) % TAR_BLOCK
                if pad:
                    w.write(b"\0" * pad)
                written += TAR_BLOCK + size + pad
            end = -(-(written + 2 * TAR_BLOCK) // TAR_RECORD) * TAR_RECORD
            w.write(b"\0" * (end - written))
            written = end
            w.close()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        rc = proc.wait()
    if rc != 0:
        raise DistError(f"{bundle['name']}: zstd exited {rc}")
    if written != bundle["tar_size"]:
        raise DistError(f"{bundle['name']}: wrote {written} tar bytes, planned {bundle['tar_size']}")
    return written


class _Limited:
    """A read-only stream view that counts bytes and refuses more than `limit` (decompression bomb guard)."""

    def __init__(self, f, limit):
        self.f, self.limit, self.count = f, limit, 0

    def read(self, n=-1):
        b = self.f.read(n if n is not None and n >= 0 else CHUNK)
        self.count += len(b)
        if self.count > self.limit:
            raise DistError(f"the bundle decompresses to more than the lock's tar size {self.limit}", EXIT_MISMATCH)
        return b


def read_bundle(path: Path, b, out_dir=None):
    """Decompress + read a bundle through the restricted reader and verify every member. With out_dir, members
    are written to out_dir/<sha>.part (fsync'ed). Returns {sha: part path or None}. Any deviation from the lock
    raises DistError(5): a member that is not a regular file, not named sha256/<aa>/<sha256>.bin, not in the
    lock, duplicated, of another size, with other content, pax-extended; a missing member; trailing data; a
    decompressed size other than the lock's tar size; a compressed stream whose sha256 is not the lock's."""
    members = {m["sha256"]: m["size"] for m in b["members"]}
    seen = {}
    hc = hashlib.sha256()
    proc = subprocess.Popen(["zstd", "-d", "-c", "-q", "--no-progress"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    feed_err = []

    def feed():
        try:
            with open(path, "rb") as f:
                for blk in iter(lambda: f.read(CHUNK), b""):
                    hc.update(blk)
                    proc.stdin.write(blk)
        except BaseException as ex:  # BrokenPipe when the reader stopped early: reported through the reader
            feed_err.append(ex)
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
    t = threading.Thread(target=feed, daemon=True)
    t.start()
    rd = _Limited(proc.stdout, b["tar_size"])
    ok = False
    try:
        try:
            _read_members(tarfile.open(fileobj=rd, mode="r|", bufsize=1 << 20), b, members, seen, out_dir)
        except (tarfile.TarError, EOFError) as ex:
            raise DistError(f"{b['name']}: malformed or truncated tar/zstd stream ({ex})", EXIT_MISMATCH)
        rest = rd.read(CHUNK)
        while rest:
            if rest.strip(b"\0"):
                raise DistError(f"{b['name']}: non-zero data after the end of the archive", EXIT_MISMATCH)
            rest = rd.read(CHUNK)
        ok = True
    finally:
        if not ok:
            proc.kill()
        try:
            proc.stdout.close()
        except OSError:
            pass
        rc = proc.wait()
        t.join()
    if rc != 0:
        raise DistError(f"{b['name']}: zstd -d exited {rc} (not a complete zstd stream)", EXIT_MISMATCH)
    if feed_err:
        raise DistError(f"{b['name']}: reading the bundle failed: {feed_err[0]}", EXIT_MISMATCH)
    if rd.count != b["tar_size"]:
        raise DistError(f"{b['name']}: decompressed {rd.count} bytes, the lock records a tar size of {b['tar_size']}", EXIT_MISMATCH)
    if hc.hexdigest() != b["sha256"]:
        raise DistError(f"{b['name']}: the file changed while it was read (sha256 {hc.hexdigest()} != lock)", EXIT_MISMATCH)
    missing = sorted(set(members) - set(seen))
    if missing:
        raise DistError(f"{b['name']}: {len(missing)} member(s) of the lock missing from the bundle: {missing[:3]}", EXIT_MISMATCH)
    return seen


def _read_members(tf, b, members, seen, out_dir):
    """The restricted reader of read_bundle(): every member checked before any byte of it is written."""
    with tf:
        while True:
            m = tf.next()
            if m is None:
                break
            name = m.name
            if m.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
                raise DistError(f"{b['name']}: member {name!r} is not a regular file (type {m.type!r})", EXIT_MISMATCH)
            mm = MEMBER_RX.match(name)
            if not mm or mm.group(1) != mm.group(2)[:2]:
                raise DistError(f"{b['name']}: member {name!r} is not a store path sha256/<aa>/<sha256>.bin", EXIT_MISMATCH)
            if m.pax_headers:
                raise DistError(f"{b['name']}: member {name!r} carries pax headers {sorted(m.pax_headers)}", EXIT_MISMATCH)
            sha = mm.group(2)
            if sha not in members:
                raise DistError(f"{b['name']}: member {name!r} is not in the lock's member list", EXIT_MISMATCH)
            if sha in seen:
                raise DistError(f"{b['name']}: member {name!r} occurs twice", EXIT_MISMATCH)
            if m.size != members[sha]:
                raise DistError(f"{b['name']}: member {name!r} has {m.size} bytes, the lock records {members[sha]}", EXIT_MISMATCH)
            src = tf.extractfile(m)
            h = hashlib.sha256()
            n = 0
            part = (out_dir / f"{sha}.part") if out_dir else None
            out = open(part, "wb") if part else None
            try:
                for blk in iter(lambda: src.read(CHUNK), b""):
                    h.update(blk)
                    n += len(blk)
                    if out:
                        out.write(blk)
                if out:
                    out.flush()
                    os.fsync(out.fileno())
            finally:
                if out:
                    out.close()
            if n != m.size or h.hexdigest() != sha:
                raise DistError(f"{b['name']}: member {name!r}: content sha256 {h.hexdigest()} is not its name", EXIT_MISMATCH)
            seen[sha] = part


# ----------------------------------------------------------------------------- content scan (archive blobs only)
def scan_blob(path: Path):
    """Archive blobs (a tar, e.g. the Nyx plotfile/checkpoint archives) are scanned member by member with the
    credential rules of the Level 3 freeze (a hit refuses the pack) and the node-private-location rule of the
    source locks, text members only (recorded for the redistribution review). Raw numeric arrays are not text
    and are not scanned."""
    with open(path, "rb") as f:
        head = f.read(TAR_BLOCK)
    if len(head) < 263 or head[257:262] != b"ustar":
        return None
    cred, private = [], []
    try:
        with tarfile.open(path, "r:") as tf:
            for m in tf:
                if not m.isreg() or m.size > hs.SCAN_MAX_BYTES:
                    continue
                data = tf.extractfile(m).read()
                for rule, rx in hs.TOKEN_RULES:
                    if rx.search(data):
                        cred.append(f"{m.name}:{rule}")
                # locations only in text members: binary data (no NUL-free text) matches '~/' etc. by chance
                if b"\0" not in data and hl.FORBIDDEN_LOCATION_RX.search(data.decode("latin-1")):
                    private.append(m.name)
    except tarfile.TarError:
        return None                      # "ustar" at offset 257 by chance: a raw array, not an archive
    return {"credential_hits": cred, "node_private_members": private}


# ----------------------------------------------------------------------------- lock
def load_lock(path: Path):
    if not path.is_file():
        raise DistError(f"lock {path} missing (pack writes it)", EXIT_UNAVAILABLE)
    text = path.read_text()
    import yaml
    lock = yaml.safe_load(text) or {}
    return lock, text


def dump_lock_text(lock):
    import yaml
    return yaml.safe_dump(lock, sort_keys=False, default_flow_style=None, width=4096, allow_unicode=False)


def _url_problems(url, name, allow_loopback_http):
    p = []
    ok_scheme = url.startswith("https://") or (allow_loopback_http and re.match(r"^http://127\.0\.0\.1(:\d+)?/", url))
    if not ok_scheme:
        p.append(f"url must be https: {url}")
    if hl.FLOATING_URL_RX.search(url):
        p.append(f"url looks floating (latest/branch), not immutable: {url}")
    if not url.endswith("/" + name):
        p.append(f"url must end with the bundle name: {url}")
    return p


def validate_lock(lock, text=None, allow_loopback_http=False):
    """Problems of a golden lock (empty list = valid): schema, format, per-bundle internal consistency
    (members sorted and unique, members_digest, name, tar size, size bound), publish/redistribution states,
    URLs, the reference map, no node-private location in the text."""
    p = []
    if not isinstance(lock, dict) or lock.get("schema") != LOCK_SCHEMA or lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        return [f"schema must be {LOCK_SCHEMA} version {LOCK_SCHEMA_VERSION}"]
    fmt = lock.get("bundle_format") or {}
    if (fmt.get("tar") or {}).get("format") != TAR_FORMAT:
        p.append(f"bundle_format.tar.format must be {TAR_FORMAT}")
    comp = fmt.get("compression") or {}
    if comp.get("tool") != "zstd" or comp.get("level") != ZSTD_LEVEL or comp.get("flags") != ZSTD_FLAGS:
        p.append(f"bundle_format.compression must be zstd -{ZSTD_LEVEL} {' '.join(ZSTD_FLAGS)}")
    limit = fmt.get("asset_limit_bytes")
    if not isinstance(limit, int) or not 0 < limit <= ASSET_LIMIT:
        p.append(f"bundle_format.asset_limit_bytes must be in (0, {ASSET_LIMIT}]")
        limit = ASSET_LIMIT
    names, members_of = set(), {}
    for i, b in enumerate(lock.get("bundles") or []):
        w = f"bundles[{i}]"
        if not isinstance(b, dict):
            p.append(f"{w}: not a mapping")
            continue
        name, group = b.get("name"), b.get("group")
        if not isinstance(group, str) or not re.fullmatch(r"level[0-9]+/[A-Za-z0-9_.+-]+", group):
            p.append(f"{w}: group {group!r} is not level<N>/<benchmark>")
            continue
        mem = b.get("members") or []
        try:
            pairs = [(m["sha256"], m["size"]) for m in mem]
        except (TypeError, KeyError):
            p.append(f"{w}: members must be {{sha256, size}} mappings")
            continue
        if not pairs or any(not isinstance(s, str) or not SHA_RX.match(s) or not isinstance(n, int) or isinstance(n, bool) or n < 0 for s, n in pairs):
            p.append(f"{w}: members need sha256 + byte size")
            continue
        if [s for s, _ in pairs] != sorted({s for s, _ in pairs}):
            p.append(f"{w}: members must be sorted by sha256 and unique")
        dig = members_digest(group, pairs)
        if b.get("members_digest") != dig:
            p.append(f"{w}: members_digest does not match its member list")
        part, parts = b.get("part"), b.get("parts")
        if not isinstance(part, int) or not isinstance(parts, int) or not 1 <= part <= parts:
            p.append(f"{w}: part/parts invalid")
        elif name != bundle_name(group, part, parts, dig):
            p.append(f"{w}: name {name!r} is not the name of its group/part/members ({bundle_name(group, part, parts, dig)})")
        if name in names:
            p.append(f"{w}: duplicate bundle name {name}")
        names.add(name)
        ts = tar_size([n for _, n in pairs])
        if b.get("tar_size") != ts:
            p.append(f"{w}: tar_size {b.get('tar_size')} != {ts} computed from the members")
        if compress_bound(ts) >= limit:
            p.append(f"{w}: the bundle could exceed the asset limit")
        if not isinstance(b.get("sha256"), str) or not SHA_RX.match(b["sha256"]):
            p.append(f"{w}: sha256 missing")
        if not isinstance(b.get("size"), int) or not 0 < b["size"] < limit:
            p.append(f"{w}: size must be a byte count below the asset limit")
        prim = b.get("primary") or {}
        st, url = prim.get("status"), prim.get("url")
        if st not in ("unpublished", "published"):
            p.append(f"{w}: primary.status {st!r} must be unpublished | published")
        elif st == "published" and not url:
            p.append(f"{w}: primary.url required when published")
        elif st == "unpublished" and url:
            p.append(f"{w}: primary.url must be null while unpublished (no fabricated future URL)")
        for u in ([url] if url else []) + list(b.get("mirrors") or []):
            p += [f"{w}: {x}" for x in _url_problems(str(u), name or "", allow_loopback_http)]
        red = (b.get("redistribution") or {}).get("status")
        if red not in REDISTRIBUTION:
            p.append(f"{w}: redistribution.status {red!r} must be one of {REDISTRIBUTION}")
        members_of[name] = {s for s, _ in pairs}
    refs = lock.get("references")
    if not isinstance(refs, dict):
        p.append("references must map reference paths to their blobs and bundles")
        refs = {}
    for path, r in refs.items():
        if not REF_RX.match(path) or not isinstance(r, dict):
            p.append(f"references: {path!r} is not a reference path")
            continue
        if not SHA_RX.match(str(r.get("json_sha256", ""))):
            p.append(f"references[{path}]: json_sha256 missing")
        bl, bn = r.get("blobs") or [], r.get("bundles") or []
        if any(x not in members_of for x in bn):
            p.append(f"references[{path}]: names a bundle that is not in the lock")
            continue
        covered = set().union(*[members_of[x] for x in bn]) if bn else set()
        if not bl or any(s not in covered for s in bl):
            p.append(f"references[{path}]: a blob is in none of its bundles")
    if text is not None:
        for n, line in enumerate(text.splitlines(), 1):
            if hl.FORBIDDEN_LOCATION_RX.search(line):
                p.append(f"line {n}: node-private location in the lock")
    return p


def build_lock(bundles, refs, per_ref, idx, zstd_version, prev, asset_limit):
    by_name = {b["name"]: b for b in bundles}
    ref_map = {}
    for path in sorted(per_ref):
        g = group_of(path)
        bn = sorted({b["name"] for b in bundles if b["group"] == g and set(per_ref[path]) & {m["sha256"] for m in b["members"]}})
        ref_map[path] = {"json_sha256": refs[path]["json_sha256"], "schema": refs[path]["doc"].get("schema"),
                         "blobs": per_ref[path], "bundles": bn}
    out = []
    for b in bundles:
        old = prev.get(b["name"]) or {}
        same = old.get("sha256") == b["sha256"]
        if old and not same and (old.get("primary") or {}).get("status") == "published":
            raise DistError(f"{b['name']}: the previous lock records it as PUBLISHED with sha256 {old['sha256']}, this pack "
                            f"produced {b['sha256']} (zstd {zstd_version}): a published name never gets other bytes", EXIT_MISMATCH)
        scan = b["content_scan"]
        notes = ["frozen B200 outputs of this project's benchmarks (raw arrays / archives named by committed references)"]
        if scan["node_private_members"]:
            notes.append(f"{len(scan['node_private_members'])} archive member(s) carry node-private build/run locations "
                         f"(paths, host name): review before publication")
        red = (old.get("redistribution") if same and old.get("redistribution") else {"status": "review", "notes": notes})
        out.append({"name": b["name"], "group": b["group"], "part": b["part"], "parts": b["parts"],
                    "sha256": b["sha256"], "size": b["size"], "tar_size": b["tar_size"], "members_digest": b["members_digest"],
                    "members": b["members"], "content_scan": scan, "redistribution": red,
                    "primary": (old.get("primary") if same and old.get("primary") else {"url": None, "status": "unpublished"}),
                    "mirrors": (list(old.get("mirrors") or []) if same else [])})
    distinct = {m["sha256"]: m["size"] for b in bundles for m in b["members"]}
    digest = hashlib.sha256("".join(f"{p} {refs[p]['json_sha256']}\n" for p in sorted(per_ref)).encode()).hexdigest()
    return {
        "schema": LOCK_SCHEMA, "schema_version": LOCK_SCHEMA_VERSION,
        "about": "bundles of the golden store (frozen reference blobs); the committed references are the root of trust, "
                 "this lock gives the bundle identities and locations (tools/golden/DISTRIBUTION.md)",
        "store": {"layout": "sha256/<aa>/<sha256>.bin", "env": "HPCPERF_GOLDEN_STORE", "default": "<repo>/.golden-store"},
        "reference_scan": {"source": "committed tree (git ls-tree), level<N>/<benchmark>/reference/**.json",
                           "schemas_with_blobs": [f"{GOLDEN_REF_SCHEMA}: outputs[].blob, sha256, bytes",
                                                  f"{NYX_REF_SCHEMA}: archive.sha256, size"],
                           "references_with_blobs": len(per_ref), "digest": digest},
        "bundle_format": {
            "naming": "golden-<level>-<benchmark>[-p<i>of<n>]-<members_digest[:16]>.tar.zst",
            "tar": {"format": TAR_FORMAT, "members": "regular files sha256/<aa>/<sha256>.bin only, sorted by name, no directory entries",
                    "header": f"POSIX ustar, mode 0644, uid/gid 0, empty uname/gname, mtime {MTIME}",
                    "end": f"two zero blocks, padded to {TAR_RECORD} bytes"},
            "compression": {"tool": "zstd", "level": ZSTD_LEVEL, "flags": list(ZSTD_FLAGS), "input": "stdin stream", "version": zstd_version},
            "asset_limit_bytes": asset_limit,
            "partition": "per benchmark; first-fit decreasing by blob size (ties by sha256); a part takes blobs while "
                         "ZSTD_compressBound(tar size) < asset_limit_bytes"},
        "cache": {"env": hs.CACHE_ENV, "default": "<repo>/.artifacts", "layout": "sha256/<bundle sha256>.tar.zst"},
        "release": {"provider": "github-release", "proposed_tag": PROPOSED_TAG, "status": "unpublished"},
        "totals": {"bundles": len(out), "distinct_blobs": len(distinct), "blob_bytes": sum(distinct.values()),
                   "member_entries": sum(len(b["members"]) for b in out), "bundle_bytes": sum(b["size"] for b in out)},
        "bundles": out,
        "references": ref_map,
    }


def write_atomic(path: Path, data: bytes, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, mode)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- pack
def _inside(path: Path, root: Path):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def cmd_pack(a):
    repo = Path(a.repo).resolve()
    store = store_root(repo, a.store)
    st_env = os.environ.get(hs.STAGING_ENV)
    if not a.staging and not st_env:
        raise DistError(f"pack: --staging or ${hs.STAGING_ENV} required (maintainer-side staging outside the worktree)", EXIT_USAGE)
    staging = Path(a.staging or os.path.join(st_env, STAGING_SUBDIR)).absolute()
    if _inside(staging, repo) and not a.allow_staging_in_repo:
        raise DistError(f"pack: the staging {staging} is inside the repository; bundles never live in the worktree", EXIT_USAGE)
    if a.asset_limit > ASSET_LIMIT or a.asset_limit <= 0:
        raise DistError(f"--asset-limit must be in (0, {ASSET_LIMIT}]", EXIT_USAGE)
    lock_out = Path(a.lock_out or a.lock or (repo / LOCK_DEFAULT)).absolute()
    t0 = time.monotonic()
    refs = committed_references(repo, a.rev)
    mod, untracked = worktree_drift(repo) if a.rev == "HEAD" else ([], [])
    for p in mod:
        log(f"note: {p} differs from the committed tree: the committed version is packed")
    for p in untracked:
        log(f"note: {p} is untracked: IGNORED (never packed)")
    idx, per_ref = reference_index(refs)
    bundles = plan_bundles(idx, per_ref, a.asset_limit)
    zv = hs.zstd_version()
    log(f"pack: {len(refs)} committed references ({a.rev}), {len(per_ref)} name blobs; {len(idx)} distinct blobs, "
        f"{sum(e['size'] for e in idx.values())} bytes; {len(bundles)} bundles; zstd {zv} -{ZSTD_LEVEL} {' '.join(ZSTD_FLAGS)}; "
        f"store {store}")
    prev = {}
    if lock_out.is_file():
        try:
            pl, _ = load_lock(lock_out)
            if pl.get("schema") == LOCK_SCHEMA:
                prev = {b["name"]: b for b in pl.get("bundles") or [] if isinstance(b, dict) and b.get("name")}
                pzv = ((pl.get("bundle_format") or {}).get("compression") or {}).get("version")
                if pzv and pzv != zv:
                    log(f"WARNING: the previous lock was packed with zstd {pzv}, this run uses {zv}: bundle bytes may differ")
        except Exception as ex:  # an unreadable previous lock carries nothing over
            log(f"note: previous lock {lock_out} not used ({ex})")
    staging.mkdir(parents=True, exist_ok=True)
    partial = staging / ".partial"
    partial.mkdir(exist_ok=True)
    scans = {}
    for sha in sorted(idx):
        if blob_state(store, sha, idx[sha]["size"], quick=True) != "OK":
            continue                     # write_bundle reports it with its bundle
        r = scan_blob(blob_path(store, sha))
        if r:
            scans[sha] = r
    hits = [f"{sha[:12]}:{h}" for sha, r in scans.items() for h in r["credential_hits"]]
    if hits:
        raise DistError(f"pack REFUSED: credential-looking content in archive blobs (rule names only): {hits[:6]}", EXIT_MISMATCH)
    tag = f"{os.getpid()}.{uuid.uuid4().hex[:8]}"

    def one(b):
        tb = time.monotonic()
        tmp = partial / f"{b['name']}.{tag}"
        try:
            write_bundle(store, b, tmp)
            sha, size = hs.sha256_file(str(tmp), CHUNK), tmp.stat().st_size
            if size >= a.asset_limit:
                raise DistError(f"{b['name']}: {size} bytes exceed the asset limit")
            det = None
            if a.verify_determinism:
                tmp2 = partial / f"{b['name']}.{tag}.2"
                try:
                    write_bundle(store, b, tmp2)
                    det = hs.sha256_file(str(tmp2), CHUNK)
                finally:
                    if tmp2.exists():
                        tmp2.unlink()
                if det != sha:
                    raise DistError(f"{b['name']}: two packs gave different bytes ({sha[:16]} / {det[:16]}): not reproducible")
            final = staging / b["name"]
            if final.exists():
                if final.stat().st_size == size and hs.sha256_file(str(final), CHUNK) == sha:
                    state = "UNCHANGED"
                    tmp.unlink()
                else:
                    raise DistError(f"{b['name']}: the staging already holds this name with other bytes; a staged bundle is never "
                                    f"overwritten (zstd version change? move the old file away explicitly)", EXIT_MISMATCH)
            else:
                os.chmod(tmp, 0o444)
                os.replace(tmp, final)
                state = "NEW"
            done = dict(b, sha256=sha, size=size)
            read_bundle(final, done)            # LOCAL_ARTIFACT_VERIFIED: the staged file reads back member by member
            scan = {"archive_blobs": sum(1 for m in b["members"] if m["sha256"] in scans), "credential_hits": 0,
                    "node_private_members": sorted(f"{m['sha256'][:12]}:{x}" for m in b["members"] for x in (scans.get(m["sha256"]) or {}).get("node_private_members", []))}
            return dict(done, content_scan=scan, state=state, seconds=round(time.monotonic() - tb, 2), deterministic=det is not None)
        finally:
            if tmp.exists():
                tmp.unlink()

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = [ex.submit(one, b) for b in bundles]
        results, errors = [], []
        for f in futs:
            try:
                results.append(f.result())
            except DistError as e:
                errors.append(e)
    try:
        partial.rmdir()
    except OSError:
        pass
    if errors:
        for e in errors:
            log(f"REFUSED: {e}")
        raise DistError(f"pack failed for {len(errors)} bundle(s); no lock written", max(e.code for e in errors))
    lock = build_lock(results, refs, per_ref, idx, zv, prev, a.asset_limit)
    text = dump_lock_text(lock)
    probs = validate_lock(lock, text)
    if probs:
        raise DistError("pack produced an invalid lock: " + "; ".join(probs[:5]))
    write_atomic(lock_out, text.encode())
    lock_sha = hashlib.sha256(text.encode()).hexdigest()
    sums = "".join(hs.sha256sums_line(b["sha256"], b["name"]) for b in results)
    write_atomic(staging / "SHA256SUMS", sums.encode())
    meta = {"schema": STAGING_SCHEMA, "lock_sha256": lock_sha, "packed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "zstd_version": zv, "seconds": round(time.monotonic() - t0, 1),
            "bundles": [{"name": b["name"], "group": b["group"], "sha256": b["sha256"], "size": b["size"], "tar_size": b["tar_size"],
                         "members": len(b["members"]), "status": "LOCAL_ARTIFACT_VERIFIED",
                         "remote_status": "REMOTE_ARTIFACT_UNPUBLISHED" if (next(x for x in lock["bundles"] if x["name"] == b["name"])["primary"]["status"] == "unpublished") else "PUBLISHED",
                         "state": b["state"], "pack_seconds": b["seconds"], "determinism_checked": b["deterministic"]} for b in results]}
    write_atomic(staging / "golden_bundles.json", (json.dumps(meta, indent=1) + "\n").encode())
    w = max(len(b["name"]) for b in results) if results else 10
    for b in results:
        log(f"  {b['state']:<9} {b['name']:<{w}} {len(b['members']):>3} blobs {b['tar_size']:>12} -> {b['size']:>11} B  sha256 {b['sha256'][:16]}  {b['seconds']:>6.1f} s")
    tot_u = sum(b["tar_size"] for b in results)
    tot_c = sum(b["size"] for b in results)
    log(f"pack: {len(results)} bundles, {tot_u} tar bytes -> {tot_c} compressed ({100.0 * tot_c / max(tot_u, 1):.1f}%), "
        f"{time.monotonic() - t0:.1f} s; staging {staging}; lock {lock_out} (sha256 {lock_sha[:16]})"
        + ("; determinism verified (two packs per bundle)" if a.verify_determinism else ""))
    return EXIT_OK


# ----------------------------------------------------------------------------- fetch
def obtain_bundle(b, from_dirs, cdir: Path, offline):
    """(path, origin) of a verified copy of bundle b: --from dirs > cache > published URL / mirrors."""
    name, sha, size = b["name"], b["sha256"], b["size"]
    for d in from_dirs:
        p = Path(d) / name
        if p.is_file():
            try:
                hs.verify_archive_file(str(p), sha, size)
            except hs.SourceError as ex:
                raise DistError(f"{name}: {ex}", EXIT_MISMATCH)
            return p, f"local {d}"
    cp = Path(hs.cache_path(str(cdir), sha))
    if cp.is_file():
        try:
            hs.verify_archive_file(str(cp), sha, size)
            return cp, "cache"
        except hs.SourceError as ex:
            raise DistError(f"{name}: cache entry {cp}: {ex}", EXIT_MISMATCH)
    prim = b.get("primary") or {}
    urls = ([prim["url"]] if prim.get("status") == "published" and prim.get("url") else []) + list(b.get("mirrors") or [])
    if not urls:
        where = ", ".join(str(d) for d in from_dirs) or "no --from directory"
        raise DistError(f"{name}: not in {where}, not in the cache, and unpublished (no URL in the lock)", EXIT_UNAVAILABLE)
    if offline:
        raise DistError(f"{name}: not available locally and --offline forbids the download", EXIT_UNAVAILABLE)
    code, last = EXIT_UNAVAILABLE, None
    for u in urls:
        try:
            return Path(hs.download_to_cache(u, str(cdir), sha, size, log=log)), f"download {u}"
        except hs.SourceError as ex:
            code, last = EXIT_MISMATCH, ex
        except OSError as ex:
            last = ex
        log(f"  {name}: {u}: {last}")
    raise DistError(f"{name}: no URL delivered the bundle ({last})", code)


def install_blob(store: Path, sha, part: Path, size):
    final = blob_path(store, sha)
    final.parent.mkdir(parents=True, exist_ok=True)
    state = "INSTALLED"
    if os.path.lexists(final):
        cur = blob_state(store, sha, size)
        if cur == "OK":
            part.unlink()
            return "PRESENT"
        if os.path.isdir(final) and not os.path.islink(final):
            raise DistError(f"{final} is a directory", EXIT_FAIL)
        state = "REPAIRED"
    os.chmod(part, 0o444)
    os.replace(part, final)
    return state


def cmd_fetch(a, allow_loopback_http=False):
    repo = Path(a.repo).resolve()
    store = store_root(repo, a.store)
    lock_path = Path(a.lock or (repo / LOCK_DEFAULT)).absolute()
    lock, text = load_lock(lock_path)
    probs = validate_lock(lock, text, allow_loopback_http)
    if probs:
        raise DistError(f"REFUSED: the lock {lock_path} is invalid: " + "; ".join(probs[:5]), EXIT_MISMATCH)
    rel = os.path.relpath(lock_path, repo)
    if not rel.startswith(".."):
        tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", rel], capture_output=True).returncode == 0
        clean = tracked and subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", rel], capture_output=True).returncode == 0
        if not clean:
            log(f"WARNING: the lock {rel} is {'modified' if tracked else 'untracked'}; blobs are still verified against the committed references")
    refs = committed_references(repo, "HEAD")
    idx, per_ref = reference_index(refs)
    sel = select_refs(per_ref, refs, a.selection, repo)
    needed = sorted({s for p in sel for s in per_ref[p]})
    before = {s: blob_state(store, s, idx[s]["size"]) for s in needed}
    todo = {s for s, st in before.items() if st != "OK"}
    log(f"fetch: {len(sel)} references, {len(needed)} blobs ({sum(idx[s]['size'] for s in needed)} bytes); "
        f"{len(needed) - len(todo)} already in the store {store}"
        + (f", {sum(1 for s in todo if before[s] != 'MISSING')} corrupted" if any(before[s] != 'MISSING' for s in todo) else ""))
    contains = {}
    for b in lock["bundles"]:
        for m in b["members"]:
            contains.setdefault(m["sha256"], []).append(b)
    chosen, not_in_lock = {}, []
    for s in sorted(todo):
        cands = contains.get(s) or []
        if not cands:
            not_in_lock.append(s)
            continue
        if any(c["name"] in chosen for c in cands):
            continue
        groups = {group_of(p) for p, _ in idx[s]["refs"] if p in sel}
        pick = sorted(cands, key=lambda c: (c["group"] not in groups, c["name"]))[0]
        chosen[pick["name"]] = pick
    rc = EXIT_OK
    for s in not_in_lock:
        log(f"NOT_IN_LOCK {s} ({', '.join(p for p, _ in idx[s]['refs'])}): the lock is stale -- re-pack (golden_artifacts.py pack)")
        rc = max(rc, EXIT_UNAVAILABLE)
    cdir = Path(a.cache_dir or hs.cache_dir()).absolute()
    counts = {"INSTALLED": 0, "PRESENT": 0, "REPAIRED": 0, "UNREFERENCED": 0}
    for name in sorted(chosen):
        b = chosen[name]
        tb = time.monotonic()
        incoming = store / ".incoming" / f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        try:
            path, origin = obtain_bundle(b, a.from_dirs or [], cdir, a.offline)
            incoming.mkdir(parents=True)
            parts = read_bundle(path, b, incoming)
            for sha, part in sorted(parts.items()):
                if sha not in idx:
                    part.unlink()
                    counts["UNREFERENCED"] += 1
                    log(f"  {sha}: in the bundle but named by no committed reference: not installed")
                    continue
                if idx[sha]["size"] != os.path.getsize(part):
                    raise DistError(f"{name}: blob {sha} has {os.path.getsize(part)} bytes, the committed reference "
                                    f"{idx[sha]['refs'][0][0]} records {idx[sha]['size']}", EXIT_MISMATCH)
                counts[install_blob(store, sha, part, idx[sha]["size"])] += 1
            log(f"  ok   {name}: {len(parts)} blobs verified ({origin}, {time.monotonic() - tb:.1f} s)")
            if a.drop_downloads and origin.startswith("download") and path.is_file():
                path.unlink()
        except DistError as ex:
            log(f"  FAIL {ex}" if str(ex).startswith(name) else f"  FAIL {name}: {ex}")
            rc = max(rc, ex.code)
        finally:
            shutil.rmtree(incoming, ignore_errors=True)
    try:
        (store / ".incoming").rmdir()
    except OSError:
        pass
    after = {s: blob_state(store, s, idx[s]["size"], quick=True) if s in todo else "OK" for s in needed}
    still = [s for s, st in after.items() if st != "OK"]
    for s in still:
        log(f"MISSING {s} ({', '.join(p for p, _ in idx[s]['refs'] if p in sel)})")
    if still and rc == EXIT_OK:
        rc = EXIT_UNAVAILABLE
    log(f"fetch: {'PASS' if rc == EXIT_OK else 'FAIL'} -- installed {counts['INSTALLED']}, repaired {counts['REPAIRED']}, "
        f"already present {len(needed) - len(todo)}, missing {len(still)}, bundles read {len(chosen)}"
        + (f" ({counts['PRESENT']} blobs were in more than one bundle read)" if counts["PRESENT"] else ""))
    return rc


# ----------------------------------------------------------------------------- verify / status
def cmd_verify(a):
    repo = Path(a.repo).resolve()
    store = store_root(repo, a.store)
    refs = committed_references(repo, "HEAD")
    idx, per_ref = reference_index(refs)
    sel = select_refs(per_ref, refs, a.selection, repo)
    needed = sorted({s for p in sel for s in per_ref[p]})
    state = {s: blob_state(store, s, idx[s]["size"], quick=a.quick) for s in needed}
    bad = {s: st for s, st in state.items() if st != "OK"}
    groups = {}
    for p in sel:
        groups.setdefault(group_of(p), set()).update(per_ref[p])
    log(f"golden store {store}: {len(sel)} committed references name {len(needed)} blobs "
        f"({sum(idx[s]['size'] for s in needed)} bytes){' [--quick: sizes only]' if a.quick else ''}")
    for g in sorted(groups):
        gb = {s: bad[s] for s in groups[g] if s in bad}
        log(f"  {g}: {'OK' if not gb else 'FAIL'} ({len(groups[g])} blobs" + (", " + ", ".join(f"{v} {k[:12]}" for k, v in sorted(gb.items())) if gb else "") + ")")
    for s, st in sorted(bad.items()):
        log(f"{st} {s} ({'; '.join(f'{p} {w}' for p, w in idx[s]['refs'] if p in sel)})")
    orphans, stray = [], []
    if not a.selection and store.is_dir():
        for dp, dns, fns in os.walk(store):
            dns.sort()
            for n in sorted(fns):
                rel = os.path.relpath(os.path.join(dp, n), store)
                mm = MEMBER_RX.match(rel)
                if mm and mm.group(1) == mm.group(2)[:2]:
                    if mm.group(2) not in idx:
                        orphans.append(rel)
                else:
                    stray.append(rel)
        for r in orphans:
            log(f"ORPHAN {r} (in the store, named by no committed reference; never packed, not deleted)")
        for r in stray:
            log(f"STRAY {r} (not a store blob path)")
    cov = None
    lock_path = Path(a.lock or (repo / LOCK_DEFAULT))
    if lock_path.is_file():
        try:
            lock, _ = load_lock(lock_path)
            inlock = {m["sha256"] for b in lock.get("bundles") or [] for m in b.get("members") or []}
            miss = [s for s in needed if s not in inlock]
            cov = {"lock": str(lock_path), "covered": len(needed) - len(miss), "not_in_lock": miss}
            shown = os.path.relpath(lock_path, repo) if _inside(lock_path, repo) else str(lock_path)
            log(f"lock {shown}: covers {len(needed) - len(miss)} of {len(needed)} blobs"
                + (f"; NOT_IN_LOCK: {[s[:12] for s in miss][:6]} (re-pack)" if miss else ""))
        except Exception as ex:
            log(f"lock {lock_path}: unreadable ({ex})")
    else:
        log(f"lock {lock_path}: missing (fetch needs it)")
    verdict = "PASS" if not bad else "FAIL"
    log(f"verify: {verdict} -- ok {len(needed) - len(bad)}, missing {sum(1 for v in bad.values() if v == 'MISSING')}, "
        f"corrupted {sum(1 for v in bad.values() if v != 'MISSING')}, orphan {len(orphans)}, stray {len(stray)}")
    if a.json:
        Path(a.json).write_text(json.dumps({"schema": "hpcperf-golden-store-verify-1", "store": str(store), "verdict": verdict,
                                            "quick": a.quick, "references": len(sel), "blobs": len(needed), "bad": bad,
                                            "orphans": orphans, "stray": stray, "lock_coverage": cov}, indent=1) + "\n")
    return EXIT_OK if not bad else EXIT_FAIL


# ----------------------------------------------------------------------------- lock-check
def lock_currency(lock, refs, idx, per_ref):
    """(problems, notes) of the lock against the committed references."""
    probs, notes = [], []
    want = plan_bundles(idx, per_ref, (lock.get("bundle_format") or {}).get("asset_limit_bytes") or ASSET_LIMIT)
    have = {b["name"]: b for b in lock.get("bundles") or []}
    wn = {b["name"] for b in want}
    for b in want:
        if b["name"] not in have:
            probs.append(f"STALE: bundle {b['name']} ({b['group']}, {len(b['members'])} blobs) is required by the committed references but not in the lock")
    for n in sorted(set(have) - wn):
        probs.append(f"STALE: bundle {n} is in the lock but no longer matches the committed references")
    lrefs = lock.get("references") or {}
    for p in sorted(per_ref):
        r = lrefs.get(p)
        if r is None:
            probs.append(f"STALE: {p} names blobs but is not in the lock")
        elif r.get("blobs") != per_ref[p]:
            probs.append(f"STALE: {p} names other blobs than the lock records")
        elif r.get("json_sha256") != refs[p]["json_sha256"]:
            notes.append(f"metadata drift: {p} changed without a blob change (bundles still valid; re-pack refreshes the lock)")
    for p in sorted(set(lrefs) - set(per_ref)):
        probs.append(f"STALE: {p} is in the lock but no committed reference with blobs has this path")
    return probs, notes


def cmd_lock_check(a):
    repo = Path(a.repo).resolve()
    lock_path = Path(a.lock or (repo / LOCK_DEFAULT)).absolute()
    lock, text = load_lock(lock_path)
    probs = validate_lock(lock, text)
    if probs:
        for x in probs:
            log(f"INVALID {x}")
        log(f"lock-check: FAIL ({lock_path}: invalid)")
        return EXIT_FAIL
    refs = committed_references(repo, a.rev)
    idx, per_ref = reference_index(refs)
    stale, notes = lock_currency(lock, refs, idx, per_ref)
    for x in stale + notes:
        log(x)
    t = lock["totals"]
    log(f"lock-check: {'PASS (CURRENT)' if not stale else 'FAIL (STALE: re-pack)'} -- {t['bundles']} bundles, "
        f"{t['distinct_blobs']} blobs, {t['bundle_bytes']} bytes; {len(per_ref)} committed references with blobs at {a.rev}")
    return EXIT_OK if not stale else EXIT_FAIL


# ----------------------------------------------------------------------------- plan (nothing uploaded)
def cmd_plan(a):
    repo = Path(a.repo).resolve()
    lock_path = Path(a.lock or (repo / LOCK_DEFAULT)).absolute()
    lock, text = load_lock(lock_path)
    probs = validate_lock(lock, text)
    if probs:
        raise DistError("plan: the lock is invalid: " + "; ".join(probs[:5]))
    st_env = os.environ.get(hs.STAGING_ENV)
    if not a.staging and not st_env:
        raise DistError(f"plan: --staging or ${hs.STAGING_ENV} required", EXIT_USAGE)
    staging = Path(a.staging or os.path.join(st_env, STAGING_SUBDIR)).absolute()
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit or ""):
        raise DistError("plan: --commit must be a full 40-hex commit", EXIT_USAGE)
    sys.path.insert(0, str(HERE.parent / "artifacts"))
    import release_plan as rp   # the repository of the Level 3 releases (OWNER_REPO); nothing of it runs
    refs = committed_references(repo, a.commit)
    idx, per_ref = reference_index(refs)
    stale, _ = lock_currency(lock, refs, idx, per_ref)
    assets, refused = [], []
    for b in lock["bundles"]:
        reasons = []
        if stale:
            reasons.append(f"lock STALE at {a.commit[:12]} ({len(stale)} problems)")
        if (b.get("redistribution") or {}).get("status") != "cleared":
            reasons.append(f"redistribution {b.get('redistribution', {}).get('status')}")
        if (b.get("primary") or {}).get("status") == "published":
            reasons.append("already published (immutable)")
        p = staging / b["name"]
        try:
            hs.verify_archive_file(str(p), b["sha256"], b["size"])
        except hs.SourceError as ex:
            reasons.append(str(ex))
        rec = {"filename": b["name"], "group": b["group"], "size_bytes": b["size"], "sha256": b["sha256"], "members": len(b["members"]),
               "planned_url": f"https://github.com/{rp.OWNER_REPO}/releases/download/{a.tag}/{b['name']}",
               "planned_url_status": "PLANNED (not published; the lock keeps url null / unpublished)",
               "sha256sums_line": hs.sha256sums_line(b["sha256"], b["name"]).strip(),
               "decision": "REFUSE" if reasons else "PLAN", "reasons": reasons}
        (refused if reasons else assets).append(rec)
        log(f"{rec['decision']:<7} {b['name']:<60} {b['size']:>11} {b['sha256'][:16]}  {'; '.join(reasons)}")
    plan = {"schema": PLAN_SCHEMA, "tag": a.tag, "target_commit": a.commit, "repository": rp.OWNER_REPO,
            "provider": "GitHub Release assets of this repository", "lock": os.path.relpath(lock_path, repo),
            "lock_sha256": hashlib.sha256(text.encode()).hexdigest(), "assets": assets, "refused": refused,
            "extra_assets": ["SHA256SUMS (the planned bundles)", "golden_artifacts.lock.yaml (the lock at the target commit)"],
            "procedure": ["maintainer authorization (HPCPERF_CONFIRM_UPLOAD=yes) for one run of the upload adapter",
                          "upload, re-download each asset and compare sha256/size",
                          "anonymous clean-clone fetch: golden_artifacts.py fetch with an empty store and cache, then verify",
                          "only then: lock primary.url = asset URL, status published (a separate reviewed commit)"]}
    if a.json:
        Path(a.json).write_text(json.dumps(plan, indent=1) + "\n")
    log(f"plan: {len(assets)} PLAN, {len(refused)} REFUSE for tag {a.tag} (nothing uploaded)")
    return EXIT_OK if not refused else EXIT_FAIL


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)     # accepted before or after the subcommand
    for parser, default in ((ap, None), (common, argparse.SUPPRESS)):
        parser.add_argument("--repo", default=str(HERE.parent.parent) if default is None else default,
                            help="repository root (default: the one of this tool)")
        parser.add_argument("--lock", default=default, help=f"the lock (default <repo>/{LOCK_DEFAULT})")
        parser.add_argument("--store", default=default, help="the golden store (default $HPCPERF_GOLDEN_STORE, else <repo>/.golden-store)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add = sub.add_parser
    sub.add_parser = lambda name, **kw: _add(name, parents=[common], **kw)
    p = sub.add_parser("pack")
    p.add_argument("--staging"); p.add_argument("--lock-out"); p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--rev", default="HEAD"); p.add_argument("--verify-determinism", action="store_true")
    p.add_argument("--asset-limit", type=int, default=ASSET_LIMIT, help=argparse.SUPPRESS)
    p.add_argument("--allow-staging-in-repo", action="store_true", help=argparse.SUPPRESS)
    f = sub.add_parser("fetch")
    f.add_argument("selection", nargs="*"); f.add_argument("--from", dest="from_dirs", action="append")
    f.add_argument("--cache-dir"); f.add_argument("--offline", action="store_true"); f.add_argument("--drop-downloads", action="store_true")
    for n in ("verify", "status"):
        v = sub.add_parser(n)
        v.add_argument("selection", nargs="*"); v.add_argument("--quick", action="store_true"); v.add_argument("--json")
    c = sub.add_parser("lock-check"); c.add_argument("--rev", default="HEAD")
    pl = sub.add_parser("plan"); pl.add_argument("--tag", required=True); pl.add_argument("--commit", required=True)
    pl.add_argument("--staging"); pl.add_argument("--json")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "pack":
            return cmd_pack(a)
        if a.cmd == "fetch":
            return cmd_fetch(a)
        if a.cmd in ("verify", "status"):
            return cmd_verify(a)
        if a.cmd == "lock-check":
            return cmd_lock_check(a)
        return cmd_plan(a)
    except DistError as ex:
        log(f"ERROR: {ex}")
        return ex.code


if __name__ == "__main__":
    sys.exit(main())
