#!/usr/bin/env python3
"""hpcperf_source -- shared library for Level 3 source freezing / materialization / workspace checks.

Everything here is deterministic and side-effect free unless a function says otherwise.

Source-tree identity (source_tree_sha256, algorithm "hpcperf-tree-1"):
    Walk the tree; take every regular file and symlink (directories carry no identity, empty ones are
    ignored); skip the materialization markers and nothing else. Sort the relative paths by their UTF-8
    bytes. For each entry append one line
        "F\\t<relative path>\\t<sha256 of the file content>\\n"     (regular file)
        "L\\t<relative path>\\t<sha256 of the symlink target>\\n"    (symlink; the target string, never followed)
    and hash the concatenation with SHA-256. mtime, uid/gid, mode and the executable bit are NOT part of
    the identity (the executable bit is recorded in SOURCE_MANIFEST.json). Re-freezing the same content
    therefore gives the same source_tree_sha256, while the archive SHA-256 also depends on the archiver.

Deterministic archive ("hpcperf-tar-1"): GNU tar written by Python, entries sorted by path bytes
(directories first as their own entries), mtime fixed to ARCHIVE_MTIME, uid/gid 0, empty uname/gname,
mode 0755 for directories and executables, 0644 otherwise, symlinks kept as symlinks; compressed with
zstd (level 19, single thread) -- the CLI and its version are recorded in the lock file.
"""
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tarfile

TREE_ALGO = "hpcperf-tree-1"
TAR_ALGO = "hpcperf-tar-1"
ARCHIVE_MTIME = 1704067200  # 2024-01-01T00:00:00Z: fixed timestamp of every archive entry
MARKERS = {".hpcperf-materialized", ".hpcperf-materialized.yaml"}
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class SourceError(Exception):
    pass


# ----------------------------------------------------------------------------- hashing
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def walk_entries(root, subdirs=None, skip_markers=True):
    """Yield (relpath, kind, abspath) for every regular file ('F') and symlink ('L') under root
    (or under the given subdirs of root), unsorted. Directories and other file types are not entries;
    a non-regular, non-symlink file (socket, fifo, device) is an error."""
    # the root itself may be reached through a symlink (worktrees share checkouts that way); it is resolved
    # once here -- only symlinks INSIDE the tree are entries and subject to the escape policy
    root = os.path.realpath(root)
    tops = [root] if not subdirs else [os.path.join(root, s) for s in subdirs]
    for top in tops:
        if not os.path.lexists(top):
            continue
        if os.path.islink(top):
            if not subdirs:
                raise SourceError(f"{top}: not a directory")
            # a materialized src/ or deps/ must be a real directory (never a symlink to another tree)
            raise SourceError(f"{top}: is a symlink -- a materialized source tree must be a real directory")
        if not os.path.isdir(top):
            raise SourceError(f"{top}: not a directory")
        for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
            dirnames.sort()
            for name in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
                p = os.path.join(dirpath, name)
                rel = os.path.relpath(p, root)
                if skip_markers and name in MARKERS:
                    continue
                st = os.lstat(p)
                if stat.S_ISLNK(st.st_mode):
                    yield rel, "L", p
                elif stat.S_ISREG(st.st_mode):
                    yield rel, "F", p
                else:
                    raise SourceError(f"{rel}: unsupported file type (mode {oct(st.st_mode)})")
            # symlinked directories were yielded above as 'L' entries; do not descend into them
            dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]


def manifest(root, subdirs=None):
    """Sorted list of entry dicts: path, type (F|L), sha256 (content or link target), size, exec (bool)."""
    out = []
    for rel, kind, p in walk_entries(root, subdirs):
        if kind == "L":
            target = os.readlink(p)
            out.append({"path": rel, "type": "L", "sha256": sha256_bytes(target.encode()), "size": len(target), "exec": False, "target": target})
        else:
            st = os.lstat(p)
            out.append({"path": rel, "type": "F", "sha256": sha256_file(p), "size": st.st_size, "exec": bool(st.st_mode & 0o111)})
    out.sort(key=lambda e: e["path"].encode())
    return out


def tree_hash_from_manifest(entries):
    h = hashlib.sha256()
    for e in entries:
        h.update(f"{e['type']}\t{e['path']}\t{e['sha256']}\n".encode())
    return h.hexdigest()


def tree_hash(root, subdirs=None):
    return tree_hash_from_manifest(manifest(root, subdirs))


# ----------------------------------------------------------------------------- symlink policy
def symlink_escapes(root, rel, target):
    """True if the symlink at root/rel resolves (lexically) outside root or is absolute."""
    if os.path.isabs(target):
        return True
    link_dir = os.path.dirname(os.path.join(root, rel))
    resolved = os.path.normpath(os.path.join(link_dir, target))
    root_abs = os.path.abspath(root)
    return not (resolved == root_abs or resolved.startswith(root_abs + os.sep))


def escaping_symlinks(root, entries):
    return [e for e in entries if e["type"] == "L" and symlink_escapes(root, e["path"], e["target"])]


# ----------------------------------------------------------------------------- secret / artifact scan
NAME_RULES = [
    ("env-dump", re.compile(r"(^|/)(toolchain\.env|[^/]*\.env)$")),
    ("shell-history", re.compile(r"(^|/)\.[a-z_]*history$")),
    ("session-record", re.compile(r"\.jsonl$")),
    ("profiler-report", re.compile(r"\.(nsys-rep|qdrep|ncu-rep|sqlite|sqlite3)$")),
    ("credential-file", re.compile(r"(^|/)(\.netrc|\.git-credentials|\.npmrc|\.pypirc|credentials\.json|id_rsa|id_dsa|id_ecdsa|id_ed25519|[^/]*\.pem|[^/]*\.p12|[^/]*\.pfx|[^/]*\.keytab)$")),
    ("build-output", re.compile(r"(^|/)(CMakeCache\.txt|CMakeFiles|\.ninja_log|\.ninja_deps|config\.status|__pycache__|\.hpcperf-src-stamp|\.hpcperf-stage-done)(/|$)|\.(o|obj|a|so|so\.[0-9.]+|pyc|gch|lo|la|dylib)$")),
]
# Fortran module files (.mod/.smod) are build output only when they are the compiler's gzip-compressed
# binaries; upstream trees legitimately carry text files with that suffix (LAMMPS potential/deck *.mod)
FORTRAN_MOD_SUFFIX = re.compile(r"\.(mod|smod)$")
GZIP_MAGIC = b"\x1f\x8b"
TOKEN_RULES = [
    ("github-token", re.compile(rb"ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}|gho_[A-Za-z0-9]{36}")),
    ("anthropic-key", re.compile(rb"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai-style-key", re.compile(rb"\bsk-[A-Za-z0-9]{40,}\b")),
    ("huggingface-token", re.compile(rb"\bhf_[A-Za-z0-9]{34,}\b")),
    ("aws-access-key", re.compile(rb"\bAKIA[A-Z0-9]{16}\b")),
    ("slack-token", re.compile(rb"\bxox[baprs]-[0-9A-Za-z\-]{10,}\b")),
    ("private-key-block", re.compile(rb"-----BEGIN (RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY")),
    ("secret-name-assignment", re.compile(rb"(?m)^\s*(declare -x |export )?(CLAUDE_API_KEY|CLAUDE_CODE_MESSAGING_TOKEN|DEEPSEEK_API_KEY|GITHUB_TOKEN|OPENAI_API_KEY|ANTHROPIC_API_KEY|HUGGING_FACE[A-Z_]*|HF_TOKEN|AWS_SECRET_ACCESS_KEY)=[\"']?[A-Za-z0-9_\-/+=]{8,}")),
    ("generic-secret-assignment", re.compile(rb"(?m)^\s*(declare -x |export )?[A-Z][A-Z0-9_]*(API_KEY|SECRET_KEY|ACCESS_TOKEN|AUTH_TOKEN|PASSWORD)=[\"']?[A-Za-z0-9_\-/+]{20,}[\"']?\s*$")),
]
SCAN_MAX_BYTES = 256 << 20


def scan_tree(root, entries=None, allow=None):
    """Return a list of hits {path, rule} -- file-name rules and content token rules. `allow` is a list of
    {path, rule, reason} exceptions (exact path + rule). No file content is ever included in a hit."""
    hits = []
    allow = {(a["path"], a["rule"]) for a in (allow or [])}
    entries = entries if entries is not None else manifest(root)
    for e in entries:
        rel = e["path"]
        for rule, rx in NAME_RULES:
            if rx.search(rel) and (rel, rule) not in allow:
                hits.append({"path": rel, "rule": rule})
        if e["type"] != "F" or e["size"] == 0 or e["size"] > SCAN_MAX_BYTES:
            continue
        with open(os.path.join(root, rel), "rb") as f:
            data = f.read()
        if FORTRAN_MOD_SUFFIX.search(rel) and data[:2] == GZIP_MAGIC and (rel, "build-output") not in allow:
            hits.append({"path": rel, "rule": "build-output"})
        for rule, rx in TOKEN_RULES:
            if rx.search(data) and (rel, rule) not in allow:
                hits.append({"path": rel, "rule": rule})
    return hits


# ----------------------------------------------------------------------------- deterministic archive
def write_deterministic_tar(root, out_path, subdirs=("src", "deps")):
    """Write a GNU tar of root/<subdirs> with fixed metadata (see module docstring). Returns entry count."""
    root = os.path.abspath(root)
    paths = []
    for s in subdirs:
        top = os.path.join(root, s)
        if not os.path.isdir(top):
            continue
        for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
            paths.append(os.path.relpath(dirpath, root))
            for n in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
                paths.append(os.path.relpath(os.path.join(dirpath, n), root))
            dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
    paths = sorted(set(paths), key=lambda p: p.encode())
    n = 0
    with tarfile.open(out_path, "w", format=tarfile.GNU_FORMAT) as tf:
        for rel in paths:
            p = os.path.join(root, rel)
            st = os.lstat(p)
            ti = tarfile.TarInfo(rel)
            ti.mtime = ARCHIVE_MTIME; ti.uid = ti.gid = 0; ti.uname = ti.gname = ""
            if stat.S_ISDIR(st.st_mode):
                ti.type = tarfile.DIRTYPE; ti.mode = 0o755; ti.size = 0
                tf.addfile(ti)
            elif stat.S_ISLNK(st.st_mode):
                ti.type = tarfile.SYMTYPE; ti.linkname = os.readlink(p); ti.mode = 0o777; ti.size = 0
                tf.addfile(ti)
            elif stat.S_ISREG(st.st_mode):
                ti.type = tarfile.REGTYPE; ti.mode = 0o755 if st.st_mode & 0o111 else 0o644; ti.size = st.st_size
                with open(p, "rb") as f:
                    tf.addfile(ti, f)
            else:
                raise SourceError(f"{rel}: unsupported file type")
            n += 1
    return n


def zstd_version():
    try:
        out = subprocess.run(["zstd", "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as ex:
        raise SourceError(f"zstd CLI not available: {ex}")
    m = re.search(r"v(\d+\.\d+\.\d+)", out)
    return m.group(1) if m else out.strip()


def zstd_compress(src, dst, level=19):
    subprocess.run(["zstd", "-q", "-f", f"-{level}", "--single-thread", "--no-progress", "-o", dst, src], check=True)


def zstd_decompress_to_tar(src, dst):
    subprocess.run(["zstd", "-q", "-f", "-d", "--no-progress", "-o", dst, src], check=True)


def safe_extract_tar(tar_path, dest, allowed_tops=("src", "deps")):
    """Extract a source bundle: only regular files, directories and symlinks under the allowed top-level
    directories; no absolute paths, no '..', no hard links, no devices. Symlinks are extracted as-is and
    validated afterwards by the caller (escaping_symlinks)."""
    dest = os.path.abspath(dest)
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            name = m.name
            parts = name.split("/")
            if name.startswith("/") or ".." in parts or parts[0] not in allowed_tops:
                raise SourceError(f"archive entry outside the allowed layout: {name!r}")
            if m.isdir():
                os.makedirs(os.path.join(dest, name), exist_ok=True)
            elif m.isreg():
                p = os.path.join(dest, name)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with tf.extractfile(m) as src, open(p, "wb") as out:
                    for b in iter(lambda: src.read(1 << 20), b""):
                        out.write(b)
                os.chmod(p, 0o755 if m.mode & 0o111 else 0o644)
            elif m.issym():
                p = os.path.join(dest, name)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                if os.path.lexists(p):
                    os.remove(p)
                os.symlink(m.linkname, p)
            else:
                raise SourceError(f"archive entry of unsupported type: {name!r}")


# ----------------------------------------------------------------------------- git helpers
def git(args, cwd):
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, check=True).stdout


def git_head(checkout):
    return git(["rev-parse", "HEAD"], checkout).strip()


def git_export_tree(checkout, dest, exclude=None):
    """Write the committed HEAD tree of a git checkout (blobs from the object store, NOT the working tree,
    so in-place modifications never leak) into dest. Gitlinks (submodules) are skipped -- the caller
    exports the submodule checkouts it wants explicitly. `exclude`: list of path prefixes to drop.
    Returns (files, symlinks, skipped_gitlinks)."""
    exclude = [x.rstrip("/") + "/" for x in (exclude or [])]
    listing = subprocess.run(["git", "ls-files", "-s", "-z"], cwd=checkout, capture_output=True, check=True).stdout
    entries = []
    gitlinks = []
    for rec in listing.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, sha, _stage = meta.decode().split()
        rel = path.decode("utf-8", "surrogateescape")
        if mode == "160000":
            gitlinks.append(rel); continue
        if any(rel == x[:-1] or rel.startswith(x) for x in exclude):
            continue
        entries.append((mode, sha, rel))
    proc = subprocess.Popen(["git", "cat-file", "--batch"], cwd=checkout, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    nfiles = nlinks = 0
    for mode, sha, rel in entries:
        proc.stdin.write((sha + "\n").encode()); proc.stdin.flush()
        header = proc.stdout.readline().decode().split()
        if len(header) != 3 or header[0] != sha:
            raise SourceError(f"git cat-file: unexpected header for {rel}: {header}")
        size = int(header[2])
        data = proc.stdout.read(size); proc.stdout.read(1)  # trailing newline
        p = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if mode == "120000":
            if os.path.lexists(p):
                os.remove(p)
            os.symlink(data.decode("utf-8", "surrogateescape"), p); nlinks += 1
        else:
            with open(p, "wb") as f:
                f.write(data)
            os.chmod(p, 0o755 if mode == "100755" else 0o644); nfiles += 1
    proc.stdin.close(); proc.wait()
    if proc.returncode != 0:
        raise SourceError("git cat-file --batch failed")
    return nfiles, nlinks, gitlinks


# ----------------------------------------------------------------------------- LFS pointer
def is_lfs_pointer(path):
    try:
        with open(path, "rb") as f:
            return f.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:
        return False


def lfs_pointer_text(sha256, size):
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {size}\n"


# ----------------------------------------------------------------------------- misc
def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def dump_yaml(obj, path):
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, default_flow_style=False, width=120)


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


if __name__ == "__main__":
    # tiny CLI: tree hash of a directory (optionally restricted to subdirs)
    import argparse
    ap = argparse.ArgumentParser(description="source_tree_sha256 of a directory (algorithm hpcperf-tree-1)")
    ap.add_argument("root"); ap.add_argument("--subdir", action="append", default=None)
    ap.add_argument("--scan", action="store_true", help="also run the secret/artifact scan (paths + rules only)")
    a = ap.parse_args()
    ents = manifest(a.root, a.subdir)
    print(f"source_tree_sha256={tree_hash_from_manifest(ents)} entries={len(ents)} algorithm={TREE_ALGO}")
    if a.scan:
        for h in scan_tree(a.root, ents):
            print(f"HIT {h['rule']}: {h['path']}")
        sys.exit(1 if scan_tree(a.root, ents) else 0)
