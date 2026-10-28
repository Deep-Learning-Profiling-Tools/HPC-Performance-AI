#!/usr/bin/env python3
"""Frozen B200 GPU reference for validate.sh check [3] (cross-backend reference) of Nyx.

    nyx_reference.py freeze --case C --mode M --steps S --gpu-run D --cpu-run C --rel-tol T --out reference/<case>.<mode>.json
                            [--diagnostic VAR ...] [--report-dir DIR] [--validate-log FILE] [--replace --reason TEXT]
    nyx_reference.py fetch  --ref reference/<case>.<mode>.json --case C --mode M --steps S --run D --dest DIR
    nyx_reference.py tol    --rel-tol T --d0 D0           (the strict budget, for tests and documentation)

`freeze` (ONE-TIME, offline; validate.sh calls it with HPCPERF_CPU_REFERENCE=1 HPCPERF_NYX_FREEZE_REFERENCE=1, right
after its unchanged check [3] compared THIS 1-GPU run with the live CPU-profile reference and agreed):
  - refuses unless level3/nyx (outside reference/) is unmodified against git HEAD (--allow-dirty overrides and records the
    uncommitted files as provenance.harness_dirty), never from an agent workspace, the GPU run is a 1-rank CUDA run of
    exactly this case/mode/steps with exit code 0, the CPU run is the CPU profile's 1-rank run of the same deck/IC/steps,
    and the [3] reports that validate.sh wrote for this very pair say AGREE / PARTICLES AGREE at the original tolerance;
  - measures d0, the frozen run's deviation from the CPU reference with the comparators' own definitions: per plotfile
    variable fcompare's norm-0 relative error ||G - C||_inf / ||C||_inf (read from the fcompare table of the [3] report)
    and per DM particle component max|G - C| / max|C| (recomputed exactly with nyx_particle_compare.py's own reader);
  - writes a deterministic tar of plt<final>/ (the plotfile without its run log job_info), chk00000/DM/ and
    chk<final>/DM/ (what [3] compares), refuses when a text member it keeps names this node (absolute paths, $HOME,
    the host or user name), puts it into the content-addressed golden store (tools/golden/hpcperf_golden.py
    store-put) and writes the reference JSON (schema hpcperf-nyx-frozen-reference-1) with the provenance, the
    excluded members and the effective tolerances. plt<final>/job_info is AMReX's run log (inputs file, output and
    build directories, compiler paths and flags, build machine with kernel version, build and output dates,
    CPU-hours): no comparison of [3] reads it (fcompare / fextrema read Header + Level_*/Cell_*, the DM count reads
    plt<final>/DM/Header, nyx_particle_compare.py reads chk*/DM), so it is not archived and the archive can be
    published. The provenance keeps what reproduces the reference -- the platform (GPU model, driver, compute
    capability; CUDA Toolkit), git HEAD, the GPU and CPU runs' identities (run id, profile, binary / fingerprint / input
    sha256, repository-relative run directory), the [3] reports' sha256 and the tools' sha256 -- and records no host
    name, GPU UUID / index or absolute location of this machine. A re-freeze that replaces the reference of the same
    GPU run keeps that run's recorded platform (gpu_source says which).
`fetch` (the default [3]): verifies the reference JSON (committed and unmodified -- in an agent workspace byte-identical
  to the canonical repository's committed one; d0 finite in [0, tol); budget as derived), that the run is the same workload (case, mode, steps, upstream
  deck sha256, IC sha256, grid), gets the archive from the store (verified sha256; a missing or corrupted blob is an
  error -- there is no fallback, in particular never a CPU run), extracts it into DIR after checking every member against
  the JSON, and prints shell assignments FROZEN_DIR / TOL_MESH / TOL_PART / ... for validate.sh. Exit 0 ok, 1 error.

Tolerance budget (the derivation is in level3/nyx/README.md, "Frozen B200 reference"): with fcompare -n 0,
rel(A, B) = ||B - A||_inf / ||A||_inf over the cells of a level (normalised by the REFERENCE field's maximum, not per
cell). For a candidate X, the frozen GPU output G and the CPU reference C with d0 = ||G - C|| / ||C||:
  ||X - C|| <= ||X - G|| + ||G - C|| = r ||G|| + d0 ||C||,   ||G|| <= (1 + d0) ||C||
  => rel(C, X) <= r (1 + d0) + d0 <= tol   whenever   r <= (tol - d0) / (1 + d0).
The particle comparison uses the same max-normalised definition, so the same bound applies per component. validate.sh
applies ONE bound per comparison: d0 = the largest d0 over the gated variables (the strictest), inflated by 1e-9
relative because fcompare prints 10 significant digits. Diagnostic variables (I_R) and identically-zero reference
fields (exact rule) are not part of the relative budget.
"""
import argparse, datetime, getpass, hashlib, json, math, os, re, socket, subprocess, sys, tarfile, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent.parent
SCHEMA = "hpcperf-nyx-frozen-reference-1"
PRINT_MARGIN = 1e-9        # fcompare prints rel with 10 significant digits (relative rounding <= 5e-10)
# Files of plt<final>/ that are NOT archived: read by no comparison of [3], and node-private (never published).
EXCLUDED_PLOTFILE_FILES = {
    "job_info": "AMReX/Nyx run log: inputs file, output and build directories, AMReX and compiler paths and flags, build "
                "machine (host name, kernel), build and output dates, CPU-hours; read by no comparison of validate.sh [3]",
}
# Node-private locations in a kept TEXT member refuse the freeze (the rule of tools/hpcperf_lock.py
# FORBIDDEN_LOCATION_RX, plus this node's own names); binary members (NUL bytes) are raw data and not scanned.
PRIVATE_LOCATION_RX = re.compile(rb"(^|[\s\"'=:(])(/tmp/|/home/|/projects/|/scratch/|\$HOME|~/|file://|/lustre/|/gpfs/)")
sys.path.insert(0, str(HERE))


class RefError(Exception):
    pass


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def manifest(run):
    p = Path(run) / "run_manifest.txt"
    if not p.is_file():
        raise RefError(f"{p} missing")
    m = {}
    for line in p.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            m.setdefault(k, v)
    return m


def golden_tool():
    """tools/golden/hpcperf_golden.py of this repository; in an agent workspace (tools/golden is not copied there) the one of
    the canonical repository named by workspace.yaml -- its store is content-addressed and every blob is sha256-verified
    against the (protected) reference JSON, so the workspace cannot substitute data."""
    t = R / "tools" / "golden" / "hpcperf_golden.py"
    if t.is_file():
        return t
    ws = HERE / "workspace.yaml"
    m = re.search(r"^canonical_dir: (.+)$", ws.read_text(), re.M) if ws.is_file() else None
    if m:
        t = Path(m.group(1).strip()).parent.parent / "tools" / "golden" / "hpcperf_golden.py"
        if t.is_file():
            return t
    raise RefError("tools/golden/hpcperf_golden.py not found (neither in this tree nor in the canonical repository of a workspace)")


def tol_effective(tol, d0):
    """(tol - d0) / (1 + d0) with d0 inflated by PRINT_MARGIN; never negative."""
    d = d0 * (1.0 + PRINT_MARGIN)
    return max((tol - d) / (1.0 + d), 0.0)


def git(*a):
    return subprocess.run(["git", "-C", str(R), *a], capture_output=True, text=True).stdout


def gpu_identity():
    """The platform record of the GPU CUDA_VISIBLE_DEVICES names: model name, driver, compute capability. Its index and
    UUID select the device but are not recorded (they identify one machine, not what the reference depends on); a driver
    without the compute_cap query field gives compute_capability null."""
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    for fields in (("index", "uuid", "name", "driver_version", "compute_cap"), ("index", "uuid", "name", "driver_version")):
        try:
            r = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20)
        except Exception:
            return None
        if r.returncode != 0:
            continue
        for line in r.stdout.splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) == len(fields) and dev in (f[0], f[1]):
                return {"name": f[2], "driver": f[3], "compute_capability": f[4] if len(f) > 4 else None}
        return None
    return None


def run_dir_rel(p):
    """A run directory as recorded: repository-relative; outside this repository (a scratch mirror of it) the part from
    the Level 3 run tree build/level3/nyx/ on (l3_rundir's layout), never an absolute location of this machine."""
    p = os.path.abspath(p)
    rel = os.path.relpath(p, R)
    if not rel.startswith(".."):
        return rel
    m = re.search(r"(?:^|/)(build/level3/nyx/.+)$", p)
    return m.group(1) if m else f"<outside the repository>/{os.path.basename(p)}"


def deck_rel(p):
    """The deck as recorded: relative to the frozen source tree level3/nyx/src, symbolic links resolved on both sides (a
    scratch mirror whose src/ links to the materialized tree records the same path); never an absolute location."""
    rel = os.path.relpath(os.path.realpath(p), os.path.realpath(HERE / "src"))
    if not rel.startswith(".."):
        return rel
    m = re.search(r"/level3/nyx/src/(.+)$", os.path.abspath(p))
    return m.group(1) if m else f"<outside the source tree>/{os.path.basename(p)}"


def portable(x):
    """Recorded strings without an absolute location of this machine: the repository and $HOME become <repo> / <home>
    (free text such as --note / --reason, porcelain lines, report result lines)."""
    subs = sorted({(str(R), "<repo>"), (os.path.expanduser("~"), "<home>")}, key=lambda t: (-len(t[0]), t))
    subs = [(a, b) for a, b in subs if os.path.isabs(a) and a.rstrip("/")]
    if isinstance(x, str):
        for a, b in subs:
            x = x.replace(a, b)
        return x
    if isinstance(x, list):
        return [portable(e) for e in x]
    if isinstance(x, dict):
        return {k: portable(e) for k, e in x.items()}
    return x


def members_of(run, steps):
    """The files [3] reads: plt<final>/ (fcompare / fextrema / the DM count) and the DM particle data of chk00000 and
    chk<final> (nyx_particle_compare.py). Sorted relative paths; plt<final>/job_info (EXCLUDED_PLOTFILE_FILES) is left
    out."""
    run = Path(run); fin = f"{steps:05d}"
    roots = [run / f"plt{fin}", run / "chk00000" / "DM", run / f"chk{fin}" / "DM"]
    skip = {f"plt{fin}/{n}" for n in EXCLUDED_PLOTFILE_FILES}
    out = []
    for r in roots:
        if not r.is_dir():
            raise RefError(f"{r} missing")
        for p in sorted(r.rglob("*")):
            if p.is_symlink():
                raise RefError(f"{p}: symbolic link inside the data to freeze")
            if p.is_file() and str(p.relative_to(run)) not in skip:
                out.append(str(p.relative_to(run)))
    return sorted(out)


def excluded_of(run, steps):
    """The plotfile files members_of() leaves out, as recorded in the reference JSON (path + reason, no content)."""
    fin = f"plt{steps:05d}"
    return [{"path": f"{fin}/{n}", "reason": why} for n, why in sorted(EXCLUDED_PLOTFILE_FILES.items())
            if (Path(run) / fin / n).is_file()]


def publishable_scan(run, members):
    """Refuse to archive a text member that names this node: an absolute private location (PRIVATE_LOCATION_RX), the
    repository or run directory, $HOME, the host name (short or qualified) or the user name. The members [3] compares
    (Headers, Cell_H, particle Headers) carry none on dgx003; a future AMReX/Nyx that writes another log into the
    plotfile is caught here instead of being published."""
    names = {str(R), str(Path(run).resolve()), os.path.expanduser("~")}
    host = socket.gethostname()
    names |= {host, host.split(".")[0], socket.getfqdn()}
    try:
        names.add(getpass.getuser())
    except Exception:  # noqa: BLE001 - no user name available: the location rule still applies
        pass
    needles = sorted({n.encode() for n in names if n and len(n) >= 3 and n not in ("/", "localhost")})
    hits = []
    for m in members:
        data = (Path(run) / m).read_bytes()
        if b"\0" in data:
            continue                                   # raw particle / FAB data
        found = [n.decode() for n in needles if n in data]
        if PRIVATE_LOCATION_RX.search(data):
            found.append(PRIVATE_LOCATION_RX.search(data).group(2).decode())
        if found:
            hits.append(f"{m} ({', '.join(found[:3])})")
    if hits:
        raise RefError("refused: archive member(s) name this node (paths / host / user), the archive would not be "
                       f"publishable: {'; '.join(hits[:8])} -- add the file to EXCLUDED_PLOTFILE_FILES if no comparison reads it")


def write_tar(run, members, dst):
    """Deterministic tar: sorted members, mtime 0, uid/gid 0, no names, mode 0644 (directories implicit)."""
    with open(dst, "wb") as fh, tarfile.open(fileobj=fh, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for m in members:
            p = Path(run) / m
            ti = tarfile.TarInfo(m); ti.size = p.stat().st_size; ti.mtime = 0; ti.mode = 0o644
            ti.uid = ti.gid = 0; ti.uname = ti.gname = ""
            with open(p, "rb") as f:
                tf.addfile(ti, f)


def parse_fcompare_report(report, names, nlevels):
    """The fcompare table that nyx_fcompare_check.py embedded in its [3] report -> {name: {level: (abs, rel)}}."""
    import nyx_fcompare_check as fc
    txt = Path(report).read_text()
    m = re.search(r"^## fcompare rc=(\d+)\n(.*?)(?=^    L\d|^RESULT:|\Z)", txt, re.S | re.M)
    if not m:
        raise RefError(f"{report}: no embedded fcompare table")
    try:
        rows = fc.parse_fcompare(m.group(2), names, nlevels)
    except fc.Fail as ex:
        raise RefError(f"{report}: fcompare table does not parse strictly: {ex}")
    out = {}
    for (lev, n), (ab, rel) in rows.items():
        out.setdefault(n, {})[str(lev)] = {"abs": ab, "rel": rel}
    return txt, out


def particle_d0(cpu_run, gpu_run, steps):
    """max|G - C| / max|C| per component, with nyx_particle_compare.py's own reader and identity matching (C first)."""
    import numpy as np
    import nyx_particle_compare as pc
    hc, pcpos, fc_ = pc.keyed_final(str(cpu_run), steps)
    hg, pgpos, fg = pc.keyed_final(str(gpu_run), steps)
    if hc["names"] != hg["names"] or not np.array_equal(pcpos, pgpos):
        raise RefError("CPU and GPU runs: particle components or initial positions differ")
    out = {}
    for j, name in enumerate(hc["names"]):
        d = float(np.abs(fc_[:, j] - fg[:, j]).max()); ref = float(np.abs(fc_[:, j]).max())
        out[name] = {"abs": d, "rel": d / ref if ref > 0 else d}
    return hg["nparticles"], out


def freeze(a):
    run, cpu = Path(a.gpu_run).resolve(), Path(a.cpu_run).resolve()
    steps, fin = a.steps, f"plt{a.steps:05d}"
    out = Path(a.out)
    rel_bench = os.path.relpath(HERE, R)
    dirty = [l for l in git("status", "--porcelain", "--", rel_bench).splitlines()
             if not l[3:].startswith(f"{rel_bench}/reference/")]
    if dirty and not a.allow_dirty:
        raise RefError(f"refused: {rel_bench} has uncommitted changes ({'; '.join(dirty[:5])}) -- a frozen reference is made "
                       f"only from the committed harness and the unmodified frozen source build")
    if "/workspaces/" in str(run) or "/workspaces/" in str(HERE):
        raise RefError("refused: a frozen reference is never made from an agent workspace")
    if out.exists() and not a.replace:
        raise RefError(f"refused: {out} exists (replacing a frozen reference needs --replace --reason)")
    if a.replace and not a.reason:
        raise RefError("--replace needs --reason")
    mg, mc = manifest(run), manifest(cpu)
    want = {"case": a.case, "mode": a.mode, "steps": str(steps), "ranks": "1", "exit_code": "0"}
    for k, v in want.items():
        if mg.get(k) != v: raise RefError(f"GPU run {run}: manifest {k}={mg.get(k)!r}, expected {v!r}")
        if mc.get(k) != v: raise RefError(f"CPU run {cpu}: manifest {k}={mc.get(k)!r}, expected {v!r}")
    if mg.get("backend") != "CUDA": raise RefError(f"GPU run backend {mg.get('backend')!r}: the frozen reference is the CUDA (B200) run")
    if mc.get("backend") != "CPU": raise RefError(f"CPU run backend {mc.get('backend')!r}, expected CPU")
    for k in ("deck_sha256", "ic_sha256", "grid", "max_grid_size", "boxes"):
        if mg.get(k) != mc.get(k): raise RefError(f"GPU and CPU runs differ in {k}: {mg.get(k)!r} vs {mc.get(k)!r}")
    import nyx_fcompare_check as fc
    hdr = fc.read_header(str(run / fin)); names = hdr["names"]; nlev = hdr["finest"] + 1
    label = f"{a.case}.np1.vs-CPU"
    repdir = Path(a.report_dir).resolve() if a.report_dir else run
    rep, prep = repdir / f"fcompare.{label}.txt", repdir / f"particle_compare.{label}.txt"
    for p in (rep, prep):
        if not p.is_file(): raise RefError(f"{p} missing: run validate.sh check [3] (HPCPERF_CPU_REFERENCE=1) on this run first")
    txt, rows = parse_fcompare_report(rep, names, nlev)
    lines = txt.splitlines()
    named = {k: os.path.realpath(l.split("=", 1)[1]) for l in lines for k in ("ref", "test") if l.startswith(k + "=")}
    if named.get("ref") != os.path.realpath(cpu / fin) or named.get("test") != os.path.realpath(run / fin):
        raise RefError(f"{rep} is not the [3] report of this pair (ref={cpu / fin}, test={run / fin})")
    if not any(l.startswith("RESULT: AGREE") for l in lines):
        raise RefError(f"{rep}: [3] did not AGREE -- nothing is frozen from a run that failed its CPU validation")
    m = re.search(r"^rel_tol=([0-9.eE+-]+) ", txt, re.M)
    if not m or float(m.group(1)) != a.rel_tol:
        raise RefError(f"{rep}: compared at rel_tol {m.group(1) if m else '?'}, expected the original {a.rel_tol!r}")
    ptxt = prep.read_text()
    if "PARTICLES AGREE" not in ptxt:
        raise RefError(f"{prep}: particles did not AGREE in [3]")
    # extrema of the CPU reference decide zero-reference fields (as nyx_fcompare_check.py did in [3])
    zero = sorted({m_.group(1) for m_ in (re.match(r"^\s+L\d+ (\S+)\s+ZERO-REFERENCE ", x) for x in lines) if m_})
    gated = [n for n in names if n not in a.diagnostic and n not in zero]
    worst = max(((rows[n][str(l)]["rel"], n) for n in gated for l in range(nlev)), default=(0.0, None))
    for n in gated:
        for l in range(nlev):
            if not rows[n][str(l)]["rel"] <= a.rel_tol:
                raise RefError(f"{rep}: {n}@L{l} rel {rows[n][str(l)]['rel']} > {a.rel_tol} although the report says AGREE")
    _, pd0 = particle_d0(cpu, run, steps)
    plines = [l.strip() for l in open(run / fin / "DM" / "Header") if l.strip()]   # the plotfile's DM count, as validate.sh reads it
    hi_ = 2; hi_ += 1 + int(plines[hi_]); hi_ += 1 + int(plines[hi_]); hi_ += 1
    npart = int(plines[hi_])
    pworst = max(((v["rel"], k) for k, v in pd0.items()), default=(0.0, None))
    if pworst[0] > a.rel_tol:
        raise RefError(f"particle d0 {pworst} exceeds rel_tol {a.rel_tol}")
    members = members_of(run, steps)
    excluded = excluded_of(run, steps)
    publishable_scan(run, members)
    with tempfile.TemporaryDirectory() as td:
        tarp = Path(td) / f"nyx-{a.case}.{a.mode}.tar"
        write_tar(run, members, tarp)
        tsha, tsize = sha256_file(tarp), tarp.stat().st_size
        r = subprocess.run([sys.executable, str(golden_tool()), "store-put", str(tarp)],
                           capture_output=True, text=True)
        put = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        if r.returncode != 0 or put != tsha:
            raise RefError(f"store-put failed (rc={r.returncode}): {r.stdout.strip()} {r.stderr.strip()}")
    gate = lambda d: {"tol": a.rel_tol, "d0_max": d, "rel_tol_effective": tol_effective(a.rel_tol, d)}
    old = json.loads(out.read_text()) if a.replace else None
    # the platform record (GPU model, driver, compute capability): from nvidia-smi now, or -- when this replaces the
    # reference of the SAME GPU run (a re-freeze of an existing run tree, e.g. to change the archived members) -- the one
    # recorded when that run was frozen, so a re-freeze in a shell without (or with another) GPU keeps the run's platform
    gpu, gpu_source = gpu_identity(), "nvidia-smi at freeze time (CUDA_VISIBLE_DEVICES)"
    oprov = (old or {}).get("provenance") or {}
    if old and (oprov.get("gpu_run") or {}).get("run_id") == mg.get("run_id") and isinstance(oprov.get("gpu"), dict):
        og = oprov["gpu"]
        cc = og.get("compute_capability") or ((gpu or {}).get("compute_capability") if (gpu or {}).get("name") == og.get("name") else None)
        gpu = {"name": og.get("name"), "driver": og.get("driver"), "compute_capability": cc}
        gpu_source = ("the replaced reference of the same GPU run (recorded when that run was frozen)"
                      + ("; compute capability from nvidia-smi now (same GPU model)" if cc and not og.get("compute_capability") else ""))
    doc = {
        "schema": SCHEMA, "benchmark": "nyx", "level": 3, "case": a.case, "mode": a.mode, "steps": steps,
        "variant": "heatcool" if a.case == "lya_heatcool" else "adiabatic",
        "workload": {k: mg.get(k) for k in ("case", "mode", "steps", "grid", "max_grid_size", "boxes", "deck_sha256", "ic_sha256")}
                    | {"deck": deck_rel(mg["deck"]) if mg.get("deck") else None, "ranks": 1},
        "archive": {"sha256": tsha, "size": tsize, "format": "tar (GNU, deterministic: sorted, mtime 0, uid/gid 0, mode 0644)",
                    "store": f"$HPCPERF_GOLDEN_STORE/sha256/{tsha[:2]}/{tsha}.bin (default <repo>/.golden-store; never in git)",
                    "members": [{"path": m_, "size": (run / m_).stat().st_size, "sha256": sha256_file(run / m_)} for m_ in members],
                    "excluded": excluded,
                    "dm_particles": npart},
        "comparison": {
            "original_check": "validate.sh [3]: nyx_fcompare_check.py (AMReX fcompare -n 0 --rel_tol T) + nyx_particle_compare.py "
                              "of this run against the CPU-profile build (Nyx_GPU_BACKEND=NONE) of the same sources and deck, 1 rank",
            "metric_mesh": "fcompare -n 0: ||B - A||_inf / ||A||_inf per variable and level, A = the reference (normalised by the "
                           "reference field's maximum over the level, not per cell)",
            "metric_particles": "max|B - A| / max|A| per DM component, A = the reference (nyx_particle_compare.py)",
            "budget": "rel_tol_effective = (tol - d0') / (1 + d0'), d0' = d0_max (1 + 1e-9): triangle inequality in the max norm, "
                      "||X - C|| <= r ||G|| + d0 ||C|| and ||G|| <= (1 + d0) ||C||, so r <= rel_tol_effective keeps X within tol "
                      "of the CPU reference C; one bound per comparison from the largest d0 over the gated variables",
            "mesh": gate(worst[0]) | {"d0_max_variable": worst[1], "d0_vs_cpu_reference": rows, "zero_reference_fields": zero,
                                      "diagnostic_fields": list(a.diagnostic)},
            "particles": gate(pworst[0]) | {"d0_max_component": pworst[1], "d0_vs_cpu_reference": pd0},
        },
        "provenance": {
            "made_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "gpu": gpu, "gpu_source": gpu_source,
            "cuda": (json.load(open("/usr/local/cuda/version.json"))["cuda"]["version"] if os.path.isfile("/usr/local/cuda/version.json") else None),
            "git_head": git("rev-parse", "HEAD").strip(), "harness_dirty": dirty,
            "gpu_run": {"run_id": mg.get("run_id"), "profile": mg.get("profile"), "binary": os.path.basename(mg.get("binary", "")),
                        "binary_sha256": mg.get("binary_sha256"), "fingerprint_sha256": mg.get("fingerprint_sha256"),
                        "input_sha256": mg.get("input_sha256"), "utc": mg.get("utc"), "dir": run_dir_rel(run)},
            "cpu_validation": {"run_id": mc.get("run_id"), "profile": mc.get("profile"), "binary_sha256": mc.get("binary_sha256"),
                               "fingerprint_sha256": mc.get("fingerprint_sha256"), "utc": mc.get("utc"), "dir": run_dir_rel(cpu),
                               "fcompare_report_sha256": sha256_file(rep), "particle_report_sha256": sha256_file(prep),
                               "validate_log_sha256": sha256_file(a.validate_log) if a.validate_log else None,
                               "result": next(l for l in lines if l.startswith("RESULT:"))},
            "source_tree_sha256": (re.search(r"source_tree_sha256: (\S+)", (HERE / "benchmark.yaml").read_text()) or [None, None])[1],
            "tool_sha256": {p: sha256_file(HERE / p) for p in ("nyx_reference.py", "nyx_fcompare_check.py", "nyx_particle_compare.py", "validate.sh", "run.sh")},
            "tool_sha256_at_git_head": {p: hashlib.sha256(subprocess.run(["git", "-C", str(R), "show", f"HEAD:{rel_bench}/{p}"],
                                                                         capture_output=True).stdout).hexdigest()
                                        for p in ("nyx_fcompare_check.py", "nyx_particle_compare.py", "validate.sh", "run.sh")},
            "note": a.note,
        },
    }
    if a.replace:
        doc["provenance"]["replaces"] = {"archive_sha256": old.get("archive", {}).get("sha256"), "made_utc": old.get("provenance", {}).get("made_utc"), "reason": a.reason}
    # no host name, GPU UUID / index or absolute location of this machine in the provenance (free text included)
    doc["provenance"] = portable(doc["provenance"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    print(f"FROZEN {a.case}.{a.mode}: archive {tsha} ({tsize} bytes, {len(members)} files) -> {out}; "
          f"mesh d0 {worst[0]:.3e} ({worst[1]}) -> rel_tol {doc['comparison']['mesh']['rel_tol_effective']:.6e}; "
          f"particles d0 {pworst[0]:.3e} ({pworst[1]}) -> rel_tol {doc['comparison']['particles']['rel_tol_effective']:.6e}")


def reference_committed(rp):
    """A frozen reference counts only when it is the committed, unmodified file (as tools/golden requires):
    'yes' in the canonical repository (git-tracked, no uncommitted change); 'canonical' in an agent workspace (the
    workspace copy is byte-identical to the canonical repository's committed reference named by workspace.yaml);
    'NO' only with HPCPERF_GOLDEN_ALLOW_UNCOMMITTED_REFERENCE=1 (test harnesses; validate.sh prints it loudly)."""
    allow = os.environ.get("HPCPERF_GOLDEN_ALLOW_UNCOMMITTED_REFERENCE", "") not in ("", "0")
    rp = Path(rp).resolve()

    def committed_in(repo, f):
        rel = os.path.relpath(f, repo)
        if rel.startswith(".."):
            return f"{f} is outside the repository {repo}"
        if subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", rel], capture_output=True).returncode != 0:
            return f"{rel} is not committed"
        st = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", rel], capture_output=True, text=True).stdout.strip()
        return f"{rel} has uncommitted changes" if st else None

    ws = HERE / "workspace.yaml"
    m = re.search(r"^canonical_dir: (.+)$", ws.read_text(), re.M) if ws.is_file() else None
    if m:
        can = Path(m.group(1).strip()) / "reference" / rp.name
        why = None if can.is_file() else f"{can} missing"
        why = why or committed_in(can.parent.parent.parent.parent, can.resolve())
        if not why and can.read_bytes() != rp.read_bytes():
            why = f"{rp} differs from the canonical committed {can}"
        verdict = "canonical"
    else:
        why = committed_in(R, rp)
        verdict = "yes"
    if why is None:
        return verdict
    if allow:
        return "NO"
    raise RefError(f"frozen reference not usable: {why} (a frozen reference counts only when committed and unmodified)")


def safe_extract(tarp, dest, members):
    want = {m["path"]: m for m in members}
    with tarfile.open(tarp, "r:") as tf:
        got = tf.getmembers()
        names = [m.name for m in got]
        if sorted(names) != sorted(want) or len(set(names)) != len(names):
            raise RefError(f"archive members differ from the reference JSON ({len(names)} vs {len(want)})")
        for m in got:
            if not m.isreg() or m.name.startswith("/") or ".." in Path(m.name).parts:
                raise RefError(f"unsafe archive member {m.name!r}")
            p = Path(dest) / m.name
            p.parent.mkdir(parents=True, exist_ok=True)
            data = tf.extractfile(m).read()
            if len(data) != want[m.name]["size"] or hashlib.sha256(data).hexdigest() != want[m.name]["sha256"]:
                raise RefError(f"archive member {m.name} does not match the reference JSON")
            p.write_bytes(data)


def fetch(a):
    rp = Path(a.ref)
    if not rp.is_file():
        raise RefError(f"frozen reference {rp} missing")
    try:
        doc = json.loads(rp.read_text())
    except json.JSONDecodeError as ex:
        raise RefError(f"{rp}: not JSON ({ex})")
    if doc.get("schema") != SCHEMA:
        raise RefError(f"{rp}: schema {doc.get('schema')!r}, expected {SCHEMA}")
    committed = reference_committed(rp)
    for k, v in (("case", a.case), ("mode", a.mode), ("steps", a.steps)):
        if doc.get(k) != v:
            raise RefError(f"{rp}: {k}={doc.get(k)!r}, this run has {v!r} -- no frozen reference for this configuration")
    m = manifest(a.run); wl = doc["workload"]
    for k in ("case", "mode", "steps", "grid", "deck_sha256", "ic_sha256"):
        if str(m.get(k)) != str(wl.get(k)):
            raise RefError(f"workload differs from the frozen reference: {k} run={m.get(k)!r} reference={wl.get(k)!r}")
    sha = doc["archive"]["sha256"]
    r = subprocess.run([sys.executable, str(golden_tool()), "store-get", sha], capture_output=True, text=True)
    path = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    if r.returncode != 0 or not os.path.isfile(path):
        raise RefError(f"frozen archive {sha[:16]}... not available: {r.stdout.strip()} {r.stderr.strip()}".strip())
    dest = Path(a.dest)
    if dest.exists() and any(dest.iterdir()):
        raise RefError(f"{dest} is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    safe_extract(path, dest, doc["archive"]["members"])
    c = doc["comparison"]
    for g in ("mesh", "particles"):     # a measured d0 is finite, >= 0 and inside the original tolerance (freeze guarantees it)
        t, d = c[g]["tol"], c[g]["d0_max"]
        if not (isinstance(d, (int, float)) and math.isfinite(d) and 0.0 <= d < t):
            raise RefError(f"{rp}: {g} d0_max {d!r} is not a measured deviation in [0, tol={t!r})")
    for g in ("mesh", "particles"):     # the recorded budget must be the one this tool derives (an edited JSON is refused)
        if abs(c[g]["rel_tol_effective"] - tol_effective(c[g]["tol"], c[g]["d0_max"])) > 1e-15 * c[g]["tol"]:
            raise RefError(f"{rp}: {g} rel_tol_effective is not (tol - d0)/(1 + d0) of its recorded tol/d0")
    if a.expect_tol is not None and (c["mesh"]["tol"] != a.expect_tol or c["particles"]["tol"] != a.expect_tol):
        raise RefError(f"{rp}: original tolerance {c['mesh']['tol']!r}/{c['particles']['tol']!r} differs from validate.sh's {a.expect_tol!r}")
    prov = doc["provenance"]; gpu = prov.get("gpu") or {}
    print(f"FROZEN_DIR={dest}")
    print(f"TOL_MESH={c['mesh']['rel_tol_effective']!r}")
    print(f"TOL_PART={c['particles']['rel_tol_effective']!r}")
    print(f"D0_MESH={c['mesh']['d0_max']!r}")
    print(f"D0_PART={c['particles']['d0_max']!r}")
    print(f"REF_SHA={sha}")
    print(f"REF_COMMITTED={committed}")
    print(f"REF_INFO='{gpu.get('name')} {prov.get('made_utc')} binary {str(prov['gpu_run'].get('binary_sha256'))[:12]}'")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    for k in ("case", "mode", "gpu-run", "cpu-run", "out"):
        f.add_argument(f"--{k}", required=True)
    f.add_argument("--steps", type=int, required=True); f.add_argument("--rel-tol", type=float, required=True)
    f.add_argument("--diagnostic", nargs="*", default=[]); f.add_argument("--validate-log")
    f.add_argument("--note", help="free-text provenance note recorded verbatim")
    f.add_argument("--replace", action="store_true"); f.add_argument("--reason")
    f.add_argument("--report-dir", help="where validate.sh wrote the [3] reports (default: the GPU run directory)")
    f.add_argument("--allow-dirty", action="store_true",
                   help="freeze although level3/nyx has uncommitted changes; the list is recorded as provenance.harness_dirty")
    g = sub.add_parser("fetch")
    for k in ("ref", "case", "mode", "run", "dest"):
        g.add_argument(f"--{k}", required=True)
    g.add_argument("--steps", type=int, required=True); g.add_argument("--expect-tol", type=float)
    t = sub.add_parser("tol"); t.add_argument("--rel-tol", type=float, required=True); t.add_argument("--d0", type=float, required=True)
    a = ap.parse_args()
    try:
        if a.cmd == "freeze":
            freeze(a)
        elif a.cmd == "fetch":
            fetch(a)
        else:
            print(repr(tol_effective(a.rel_tol, a.d0)))
        return 0
    except (RefError, OSError, KeyError, ValueError, SystemExit) as ex:
        print(f"ERROR: nyx_reference {a.cmd}: {ex}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
