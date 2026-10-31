#!/usr/bin/env python3
"""Fixture and tamper helpers for test_golden_artifacts.sh (CPU only, small files, no network but loopback).

    golden_artifacts_fixture.py mkfix <dir>                         fake repository + store; prints shell variables
    golden_artifacts_fixture.py flip <file> [offset]                 flip one byte (default: the middle)
    golden_artifacts_fixture.py lockedit <in> <out> <edit> [args]    edits: bundle-sha <group> | member-sha <group> |
                                                                     url-unpublished <group> | private-path |
                                                                     clear-all | publish <group> <sha|same>
    golden_artifacts_fixture.py craft <kind> <lock-in> <group> <staging-out> <lock-out>
                                                                     a hostile bundle under the group's bundle name
                                                                     + a lock copy whose sha256/size match it
    golden_artifacts_fixture.py serve <lock-in> <staging> <store> <cache> <repo> [tamper]
                                                                     fetch through a loopback http server
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import golden_artifacts as ga  # noqa: E402
import yaml  # noqa: E402


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(store, data):
    s = sha(data)
    p = store / "sha256" / s[:2] / f"{s}.bin"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return s


def golden_ref(bench, inp, outs):
    return {"schema": "hpcperf-golden-1", "benchmark": bench, "level": 1, "input_id": inp, "frozen": True,
            "outputs": [dict(o) for o in outs],
            "provenance": {"store": "blobs of tolerance outputs: $HPCPERF_GOLDEN_STORE (default <repo>/.golden-store)"}}


def out(name, s, n, blob=True, basis="fixture"):
    o = {"name": name, "dtype": "u8", "count": n, "bytes": n, "sha256": s, "rule": {"metric": "max_abs" if blob else "exact"},
         "basis": basis}
    if blob:
        o["blob"] = f"sha256/{s[:2]}/{s}.bin"
    return o


def archive(members):
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for name, data in members:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return bio.getvalue()


def wjson(p, doc):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2) + "\n")


def git(repo, *a):
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *a], check=True, capture_output=True)


def mkfix(d):
    d = Path(d)
    repo, store = d / "repo", d / "store"
    (repo / "level1").mkdir(parents=True)
    (repo / "hpcperf_env.sh").write_text("")
    rnd = lambda n, seed: (hashlib.sha256(seed.encode()).digest() * (n // 32 + 1))[:n] if n else b""
    A, B, C = put(store, rnd(3000, "A") + b"a"), put(store, rnd(70000, "B")), put(store, rnd(10, "C"))
    D, E = put(store, os.urandom(100000)), put(store, b"")
    F = put(store, archive([("plt00010/Header", b"HyperCLaw-V1.1\n"),
                            ("plt00010/job_info", b"inputs file: /projects/somebody/run/inputs\nbuild machine: node001\n")]))
    O, U, M = put(store, rnd(500, "O")), put(store, rnd(600, "U")), put(store, rnd(700, "M"))
    X = sha(rnd(3001, "X"))          # exact output only: never a blob
    wjson(repo / "level1/alpha/reference/in1.json", golden_ref("alpha", "in1", [out("a", A, 3001), out("b", B, 70000), out("c", X, 3001, blob=False)]))
    wjson(repo / "level1/alpha/reference/in2.json", golden_ref("alpha", "in2", [out("c", C, 10), out("a", A, 3001)]))
    wjson(repo / "level1/beta/reference/b1.json", golden_ref("beta", "b1", [out("a", A, 3001), out("d", D, 100000)]))
    wjson(repo / "level2/gamma/reference/sub/g.json", golden_ref("gamma", "g", [out("e", E, 0)]))
    fdata = (store / "sha256" / F[:2] / f"{F}.bin").read_bytes()
    wjson(repo / "level3/nyxlike/reference/c.smoke.json", {"schema": "hpcperf-nyx-frozen-reference-1", "benchmark": "nyxlike", "level": 3,
                                                           "archive": {"sha256": F, "size": len(fdata), "members": []}})
    wjson(repo / "level2/other/reference/x.json", {"schema": "hpcperf-other-reference-1", "value": 1.5})
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "fixture")
    # after the commit: an untracked reference and a working-tree modification -- neither may be packed
    wjson(repo / "level1/delta/reference/u.json", golden_ref("delta", "u", [out("u", U, 600)]))
    wjson(repo / "level1/alpha/reference/in2.json", golden_ref("alpha", "in2", [out("c", C, 10), out("m", M, 700)]))
    for k, v in dict(FR=repo, SRC=store, A=A, B=B, C=C, D=D, E=E, F=F, O=O, U=U, M=M).items():
        print(f"{k}={v}")


def flip(p, off=None):
    p = Path(p)
    os.chmod(p, 0o644)
    b = bytearray(p.read_bytes())
    i = len(b) // 2 if off is None else int(off)
    b[i] ^= 0x01
    p.write_bytes(bytes(b))


def load(p):
    return yaml.safe_load(Path(p).read_text())


def dump(lock, p):
    Path(p).write_text(ga.dump_lock_text(lock))


def bundle_of(lock, group):
    return next(b for b in lock["bundles"] if b["group"] == group)


def lockedit(src, dst, edit, *args):
    lock = load(src)
    if edit == "bundle-sha":
        bundle_of(lock, args[0])["sha256"] = "0" * 64
    elif edit == "member-sha":
        b = bundle_of(lock, args[0])
        b["members"][0]["sha256"] = "f" * 64
    elif edit == "url-unpublished":
        b = bundle_of(lock, args[0])
        b["primary"]["url"] = f"https://example.invalid/releases/download/t/{b['name']}"
    elif edit == "private-path":
        lock["about"] = lock["about"] + " see /projects/x"
    elif edit == "clear-all":
        for b in lock["bundles"]:
            b["redistribution"] = {"status": "cleared", "notes": ["fixture"]}
    elif edit == "publish":
        b = bundle_of(lock, args[0])
        b["primary"] = {"url": f"https://example.invalid/releases/download/t/{b['name']}", "status": "published"}
        if args[1] != "same":
            b["sha256"] = args[1]
    else:
        raise SystemExit(f"unknown edit {edit}")
    dump(lock, dst)


def craft(kind, lock_in, group, staging, lock_out):
    """A hostile bundle under the group's bundle name; the lock copy gets its sha256 + size (a consistent forgery of
    the bundle identity -- the member list, names and digests of the lock stay the real ones)."""
    lock = load(lock_in)
    b = bundle_of(lock, group)
    m0 = b["members"][0]
    good = f"sha256/{m0['sha256'][:2]}/{m0['sha256']}.bin"
    bio = io.BytesIO()
    tf = tarfile.open(fileobj=bio, mode="w", format=tarfile.PAX_FORMAT if kind == "pax" else tarfile.GNU_FORMAT)

    def reg(name, data, pax=None):
        ti = tarfile.TarInfo(name)
        ti.size = len(data)
        if pax:
            ti.pax_headers = pax
        tf.addfile(ti, io.BytesIO(data))
    if kind == "traversal":
        reg("../evil.bin", b"evil")
    elif kind == "absolute":
        reg(os.path.join(os.environ["TMP"], "abs-evil.bin"), b"evil")
    elif kind == "dotdot-inside":
        reg(f"sha256/{m0['sha256'][:2]}/../../../evil.bin", b"evil")
    elif kind == "symlink":
        ti = tarfile.TarInfo(good)
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/passwd"
        tf.addfile(ti)
    elif kind == "hardlink":
        ti = tarfile.TarInfo(good)
        ti.type = tarfile.LNKTYPE
        ti.linkname = "../../evil.bin"
        tf.addfile(ti)
    elif kind == "directory":
        ti = tarfile.TarInfo(f"sha256/{m0['sha256'][:2]}")
        ti.type = tarfile.DIRTYPE
        tf.addfile(ti)
    elif kind == "device":
        ti = tarfile.TarInfo(good)
        ti.type = tarfile.CHRTYPE
        tf.addfile(ti)
    elif kind == "pax":
        reg(good, b"x" * m0["size"], pax={"comment": "hidden"})
    elif kind == "wrong-content":
        reg(good, b"x" * m0["size"])
    elif kind == "extra-member":
        for m in b["members"]:
            src = Path(os.environ["SRC"]) / "sha256" / m["sha256"][:2] / f"{m['sha256']}.bin"
            reg(f"sha256/{m['sha256'][:2]}/{m['sha256']}.bin", src.read_bytes())
        reg(f"sha256/{'ab'}/{'ab' * 32}.bin", b"")
    elif kind == "missing-member":
        m = b["members"][-1]
        src = Path(os.environ["SRC"]) / "sha256" / m["sha256"][:2] / f"{m['sha256']}.bin"
        reg(f"sha256/{m['sha256'][:2]}/{m['sha256']}.bin", src.read_bytes())
    elif kind == "duplicate":
        src = Path(os.environ["SRC"]) / "sha256" / m0["sha256"][:2] / f"{m0['sha256']}.bin"
        reg(good, src.read_bytes())
        reg(good, src.read_bytes())
    elif kind == "size-mismatch":
        reg(good, b"x" * (m0["size"] + 1))
    else:
        raise SystemExit(f"unknown kind {kind}")
    tf.close()
    raw = bio.getvalue()
    comp = subprocess.run(["zstd", "-q", "-3", "-c"], input=raw, capture_output=True, check=True).stdout
    Path(staging).mkdir(parents=True, exist_ok=True)
    (Path(staging) / b["name"]).write_bytes(comp)
    b["sha256"], b["size"] = sha(comp), len(comp)
    dump(lock, lock_out)


def serve(lock_in, staging, store, cache, repo, tamper=None):
    """Publish the staged bundles on a loopback http server (lock primary = http://127.0.0.1:<port>/<name>, status
    published) and fetch through golden_artifacts.cmd_fetch with the loopback exception: the download path is
    hpcperf_source.download_to_cache. With tamper, the server delivers other bytes under every name."""
    import argparse
    import functools
    import http.server
    import shutil
    import socketserver
    import threading
    serve_dir = Path(cache).parent / "served"
    if serve_dir.exists():
        shutil.rmtree(serve_dir)
    shutil.copytree(staging, serve_dir)
    if tamper:
        for p in serve_dir.glob("*.tar.zst"):
            flip(p)
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass
    h = functools.partial(Quiet, directory=str(serve_dir))

    class Q(socketserver.TCPServer):
        allow_reuse_address = True
    srv = Q(("127.0.0.1", 0), h)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    lock = load(lock_in)
    for b in lock["bundles"]:
        b["primary"] = {"url": f"http://127.0.0.1:{port}/{b['name']}", "status": "published"}
    lp = Path(cache).parent / "lock.loopback.yaml"
    dump(lock, lp)
    a = argparse.Namespace(repo=repo, store=store, lock=str(lp), selection=[], from_dirs=None, cache_dir=cache,
                           offline=False, drop_downloads=False)
    try:
        rc = ga.cmd_fetch(a, allow_loopback_http=True)
    except ga.DistError as ex:
        print(f"ERROR: {ex}")
        rc = ex.code
    srv.shutdown()
    print(f"LOOPBACK_LOCK={lp}")
    return rc


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "mkfix":
        mkfix(*args)
    elif cmd == "flip":
        flip(*args)
    elif cmd == "lockedit":
        lockedit(*args)
    elif cmd == "craft":
        craft(*args)
    elif cmd == "serve":
        sys.exit(serve(*args))
    else:
        raise SystemExit(__doc__)
