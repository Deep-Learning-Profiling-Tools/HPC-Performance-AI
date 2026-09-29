#!/usr/bin/env python3
"""Level 3: the applications' own timers -> the timed region of one run.

    apptimers.py extract <app> <clean-run-dir>     # JSON of the timed region (exit 1 if absent)
    apptimers.py harvest <app> <run-tree> <dest>   # copy the files the extractor reads
    apptimers.py describe [<app>]                  # the definition of each application's region

Level 1 and Level 2 sources carry ROI markers (tools/timing/roi/). Level 3 applications
are full production codes -- too large to mark by hand -- so their measured region is the
one the application times itself. For every application this module fixes, once and
with source citations, WHICH printed timer is the region, what it includes and excludes,
how multi-rank values are reduced and whether the timer waits for the GPU. The region is
chosen by the same rule as the markers (tools/timing/roi/README.md): the time-step /
iteration loop, without start-up, set-up, the first (warm-up) step where the application
charges set-up to it, verification and final output.

Evidence layout of one clean run (tools/timing/lib/engine.sh, Level 3):
    <dir>/run.log          stdout + stderr of level3/<app>/run.sh
    <dir>/app/<run>/...    files harvested from the run directory run.sh created
                           (build/level3/<app>/<profile>/<HPCPERF_L3_RUN_SUBDIR>/<run>/)

extract() returns
    wall_s       the region's time (seconds)            -> roi.wall_s of the record
    steps        time steps / iterations / blocks inside it, or None
    excluded_s   time the application reports INSIDE the loop that the region leaves out
                 (bulk output, checkpointing), or None
    setup_s      the application's own set-up time when it prints one (context), or None
    parts        {name: seconds} the application's own sub-timers of the region (context;
                 see `device_sync` before reading them as GPU time)
    files        the evidence files that were read (relative to <dir>)
and raises TimerMissing when the run does not contain the timer: the measurement then
fails (status app_timer_missing) instead of falling back to the process wall clock.

Source citations are relative to level3/<app>/src/ (the frozen source tree, see the
application's provenance/source.lock.yaml) and were read on 2026-09-29.
"""

import glob
import json
import os
import re
import shutil
import sys

NUM = r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"


class TimerMissing(Exception):
    pass


# ----------------------------------------------------------------- helpers

def _read(path):
    with open(path, errors="replace") as f:
        return f.read()


def _stdout(d):
    p = os.path.join(d, "run.log")
    if not os.path.isfile(p):
        raise TimerMissing("run.log missing")
    return _read(p)


def _find(d, pattern):
    """Harvested files matching pattern (glob relative to <d>/app/<run>/), sorted."""
    return sorted(glob.glob(os.path.join(d, "app", "*", pattern), recursive=True))


def _one(d, pattern):
    hits = _find(d, pattern)
    if not hits:
        raise TimerMissing(f"no harvested file matches {pattern!r}")
    if len(hits) > 1:
        raise TimerMissing(f"{len(hits)} harvested files match {pattern!r}: {[os.path.relpath(h, d) for h in hits]}")
    return hits[0]


def _f(x):
    return float(x)


def _rel(d, paths):
    return [os.path.relpath(p, d) for p in paths]


def _result(wall, steps=None, excluded=None, setup=None, parts=None, files=None):
    if wall is None or not (wall == wall) or wall < 0 or wall == float("inf"):
        raise TimerMissing(f"timer value {wall!r} is not a finite non-negative time")
    return {"wall_s": wall, "steps": steps, "excluded_s": excluded, "setup_s": setup,
            "parts": parts or {}, "files": files or ["run.log"]}


# ----------------------------------------------------------------- LAMMPS / SPARTA

def _loop_block(text, what):
    """The LAST 'Loop time of T on N procs for S steps with P <what>' block and its section table."""
    rx = re.compile(rf"^Loop time of {NUM} on (\d+) procs for (\d+) steps with (\d+) {what}", re.M)
    hits = list(rx.finditer(text))
    if not hits:
        raise TimerMissing("no 'Loop time of ...' line")
    m = hits[-1]
    parts = {}
    tail = text[m.end():].split("\n\n", 3)
    table = "\n".join(tail[:3])
    for row in re.finditer(rf"^(\w[\w ]*?)\s*\|\s*{NUM}?\s*\|\s*{NUM}\s*\|", table, re.M):
        name = row.group(1).strip()
        if name in ("Section",):
            continue
        parts[name.lower()] = _f(row.group(3))            # the avg column
    return _f(m.group(1)), int(m.group(3)), parts


def lammps(d):
    t, steps, parts = _loop_block(_stdout(d), "atoms")
    return _result(t, steps=steps, parts=parts)


def sparta(d):
    t, steps, parts = _loop_block(_stdout(d), "particles")
    return _result(t, steps=steps, parts=parts)


# ----------------------------------------------------------------- WarpX / Nyx (AMReX)

def warpx(d):
    text = _stdout(d)
    ev = re.findall(rf"^Evolve time = {NUM} s; This step = {NUM} s", text, re.M)
    if not ev:
        raise TimerMissing("no 'Evolve time = ...' line (warpx.verbose must be >= 1)")
    steps = re.findall(r"^STEP (\d+) ends\.", text, re.M)
    total = re.findall(rf"^Total Time\s*:\s*{NUM}", text, re.M)
    parts = {"first_step": _f(ev[0][1])}
    if total:
        parts["total_time_incl_init_and_output"] = _f(total[-1])
    return _result(_f(ev[-1][0]), steps=int(steps[-1]) if steps else len(ev), parts=parts)


def nyx(d):
    text = _stdout(d)
    st = re.findall(rf"^\[STEP (\d+)\] Coarse TimeStep time: {NUM}", text, re.M)
    if not st:
        raise TimerMissing("no '[STEP n] Coarse TimeStep time' line (amr.v must be >= 1)")
    steps = [_f(t) for _, t in st]
    io = sum(_f(x) for x in re.findall(rf"^checkPoint\(\) time = {NUM}", text, re.M)) + \
        sum(_f(x) for x in re.findall(rf"^Write plotfile time = {NUM}", text, re.M))
    parts = {"first_step": steps[0]}
    run = re.findall(rf"^Run time = {NUM}", text, re.M)
    if run:
        parts["run_time_incl_init_and_io"] = _f(run[-1])
    return _result(sum(steps), steps=len(steps), excluded=io, parts=parts)


# ----------------------------------------------------------------- nekRS

def nekrs(d):
    text = _stdout(d)
    blocks = list(re.finditer(rf">>> runtime statistics \(step= (\d+)\s+totalElapsed= {NUM}s\):", text))
    if not blocks:
        raise TimerMissing("no '>>> runtime statistics' block")
    b = blocks[-1]
    body = text[b.end():].split("\n\n", 1)[0]
    rows = {}
    for m in re.finditer(rf"^( *)(\S[\w /-]*?)\s+{NUM}s\s", body, re.M):
        name = m.group(2).strip()
        key = name if len(m.group(1)) <= 4 else f"{name} (nested)"
        rows.setdefault(key, _f(m.group(3)))
    if "solve" not in rows:
        raise TimerMissing("runtime statistics block has no 'solve' row")
    ckpt = rows.get("checkpointing", 0.0)
    init = re.findall(rf"^initialization took {NUM} s", text, re.M)
    # min / max are the shortest and longest single step, not parts of the loop
    parts = {k: v for k, v in rows.items() if k not in ("solve", "min", "max") and "(nested)" not in k}
    for k in ("min", "max"):
        if k in rows:
            parts[f"step_{k}"] = rows[k]
    return _result(rows["solve"] - ckpt, steps=int(b.group(1)), excluded=ckpt,
                   setup=_f(init[-1]) if init else None, parts=parts)


# ----------------------------------------------------------------- SPECFEM3D

def specfem3d(d):
    path = _one(d, "**/OUTPUT_FILES/output_solver.txt")
    text = _read(path)
    el = [_f(x) for x in re.findall(rf"^\s*Elapsed time in seconds =\s*{NUM}", text, re.M)]
    done = [(int(a), int(b)) for a, b in re.findall(r"^\s*Time steps done =\s*(\d+)\s+out of\s+(\d+)", text, re.M)]
    if not el or len(el) != len(done):
        raise TimerMissing("output_solver.txt has no matching 'Elapsed time' / 'Time steps done' reports")
    if done[-1][0] != done[-1][1]:
        raise TimerMissing(f"the last stability report is at step {done[-1][0]} of {done[-1][1]} (run incomplete)")
    seis = re.findall(rf"^\s*Writing the seismograms in parallel took\s+{NUM}", text, re.M)
    total = re.findall(rf"^\s*Total elapsed time in seconds =\s*{NUM}", text, re.M)
    parts = {"first_report_at_step_%d" % done[0][0]: el[0]}
    if total:
        parts["total_incl_final_output"] = _f(total[-1])
    return _result(el[-1], steps=done[-1][0], excluded=None,
                   parts=parts | ({"final_seismogram_write": _f(seis[-1])} if seis else {}),
                   files=_rel(d, [path]))


# ----------------------------------------------------------------- ExaCA

def exaca(d):
    text = _stdout(d)
    ca = re.findall(rf"^Time spent performing CA calculations = {NUM} s", text, re.M)
    if not ca:
        raise TimerMissing("no 'Time spent performing CA calculations' line")
    parts = {}
    for name, mx, _mn in re.findall(rf"^Max/min rank time (?:in )?(CA [\w ]+?|exporting data) = {NUM} / {NUM} s", text, re.M):
        parts[name.replace("CA ", "").strip() + " (max rank)"] = _f(mx)
    init = re.findall(rf"^Time spent initializing data = {NUM} s", text, re.M)
    out = re.findall(rf"^Time spent collecting and printing output data = {NUM} s", text, re.M)
    if out:
        parts["final output"] = _f(out[-1])
    return _result(_f(ca[-1]), setup=_f(init[-1]) if init else None, parts=parts)


# ----------------------------------------------------------------- CP2K

def cp2k(d):
    path = _one(d, "*-1.ener")
    rows = []
    for line in _read(path).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        c = line.split()
        rows.append((int(c[0]), _f(c[-1])))              # Step Nr. ... UsedTime[s]
    steps = [(n, t) for n, t in rows if n >= 1]
    if len(steps) < 2:
        raise TimerMissing(f"{os.path.basename(path)} has {len(steps)} MD steps; need >= 2 (step 1 is excluded)")
    first = steps[0][1]
    loop = steps[1:]
    return _result(sum(t for _, t in loop), steps=len(loop), excluded=None,
                   parts={"step_1_incl_initial_force_eval": first,
                          "mean_step": sum(t for _, t in loop) / len(loop)},
                   files=_rel(d, [path]))


# ----------------------------------------------------------------- QMCPACK

def _qmc_stack(text):
    """The 'Stack timer profile' table -> {path: (inclusive, exclusive, calls)}; the path joins the
    names of the enclosing timers with '/' (indentation = nesting, two spaces per level)."""
    at = text.rfind("Stack timer profile")
    if at < 0:
        raise TimerMissing("no 'Stack timer profile' (qmcpack built without ENABLE_TIMERS, or run incomplete)")
    rows, stack = {}, []
    rx = re.compile(rf"^( *)(\S+)\s+{NUM}\s+{NUM}\s+(\d+)\s+{NUM}\s*$")
    for line in text[at:].splitlines()[2:]:
        m = rx.match(line)
        if not m:
            if rows:
                break
            continue
        depth = len(m.group(1)) // 2
        stack = stack[:depth] + [m.group(2)]
        rows["/".join(stack)] = (_f(m.group(3)), _f(m.group(4)), int(m.group(5)))
    if not rows:
        raise TimerMissing("empty 'Stack timer profile'")
    return rows


def qmcpack(d):
    rows = _qmc_stack(_stdout(d))
    prod = [p for p in rows if p.endswith("/DMCBatched::Production")]
    if len(prod) != 1:
        raise TimerMissing(f"{len(prod)} DMCBatched::Production timers (need exactly one DMC section at "
                           f"--enable-timers=medium or finer; timers found: {len(rows)})")
    p = prod[0]
    steps = [v[2] for k, v in rows.items() if k.startswith(p + "/") and k.endswith("/DMCBatched::RunSteps")]
    parts = {}
    run_steps = p + "/DMCBatched::RunSteps"
    for k, v in rows.items():
        # the Production timer's children, and one level further inside RunSteps (which is ~all of it)
        if (k.startswith(p + "/") and k.count("/") == p.count("/") + 1) or \
           (k.startswith(run_steps + "/") and k.count("/") == run_steps.count("/") + 1):
            parts[k.rsplit("/", 1)[1]] = v[0]
    for k, v in rows.items():
        if k.endswith("/VMCBatched::Production"):
            parts["VMCBatched::Production (before DMC)"] = v[0]
    setup = [v[0] for k, v in rows.items() if k.count("/") == 1 and k.endswith("/Startup")]
    return _result(rows[p][0], steps=steps[0] if steps else None, setup=setup[0] if setup else None,
                   parts=parts)


# ----------------------------------------------------------------- DFT-FE

def dftfe(d):
    """MD steps 1..N: 'Time taken for updateAtomPositionsAndMoveMesh' + the step's SCF iterations.
    The SCF block before the first 'MD STEP' banner is the initial ground state (set-up)."""
    text = _stdout(d)
    first = re.search(r"^-+MD STEP", text, re.M)
    if not first:
        raise TimerMissing("no 'MD STEP' banner (not an MD run, or it did not start)")
    scf = [(m.start(), _f(m.group(1))) for m in
           re.finditer(rf"^Wall time for the above scf iteration: {NUM} seconds", text, re.M)]
    if not scf:
        raise TimerMissing("no 'Wall time for the above scf iteration' line (VERBOSITY must be >= 1)")
    upd = [_f(x) for x in re.findall(rf"^Time taken for updateAtomPositionsAndMoveMesh: {NUM}", text, re.M)]
    if not upd:
        raise TimerMissing("no 'Time taken for updateAtomPositionsAndMoveMesh' line (no MD step completed)")
    gs = [t for pos, t in scf if pos < first.start()]
    md = [t for pos, t in scf if pos > first.start()]
    parts = {"md_scf_iterations": sum(md), "md_update_positions_and_move_mesh": sum(upd),
             "initial_ground_state_scf": sum(gs)}
    total = re.findall(rf"Elapsed wall time since start of the program: {NUM} seconds", text)
    if total:
        parts["program_total"] = _f(total[-1])
    return _result(sum(md) + sum(upd), steps=len(upd), parts=parts)


# ----------------------------------------------------------------- registry

class Timer:
    """fn: the extractor; harvest: file patterns it reads from the run directory (besides
    run.log); definition / where / reduction / device_sync: what the region is, the source
    lines that time it, how ranks are combined, whether the device is waited for; caveat:
    a sentence every record of this application carries (None if nothing to add)."""

    def __init__(self, fn, harvest, definition, where, reduction, device_sync, caveat=None):
        self.fn, self.harvest, self.definition = fn, harvest, definition
        self.where, self.reduction, self.device_sync = where, reduction, device_sync
        self.caveat = caveat


TIMERS = {
    "lammps": Timer(
        lammps, ["log.*.lammps"],
        "'Loop time' of the last `run` command: all timesteps, without integrate->setup and the "
        "final statistics; the Section rows (Pair/Neigh/Comm/Output/Modify/Other) are its breakdown.",
        ["src/run.cpp:170-176", "src/finish.cpp:117-120", "src/KOKKOS/verlet_kokkos.cpp:301-526"],
        "average over ranks (sum/nprocs), ranks aligned by barriers at start and end",
        "total: yes (the loop ends with a blocking device->host sync before the stop barrier); "
        "sections: NO -- the timer never fences the device, asynchronous kernel time is charged to "
        "the next section that synchronizes (the pair force lands in Comm under newton on)",
        caveat="LAMMPS's per-section times (parts) are not device-synchronized: with the KOKKOS package "
               "asynchronous kernel time is charged to the next synchronizing section (Comm), so they are "
               "not a GPU breakdown; the loop total is."),
    "sparta": Timer(
        sparta, ["log.*.sparta"],
        "'Loop time' of the last `run` command (the production run; earlier runs of a deck are "
        "warm-up), without setup and final statistics.",
        ["src/run.cpp:226-229", "src/finish.cpp:83-85", "src/KOKKOS/update_kokkos.cpp:454-536"],
        "average over ranks, barriers at start and end",
        "total: yes (the last step's statistics output syncs particles to the host); sections: "
        "probably close (per-step blocking read-backs) but the timer does not fence the device"),
    "warpx": Timer(
        warpx, [],
        "final 'Evolve time': the whole WarpX::Evolve step loop, including in-loop diagnostics; "
        "without InitData and the final diagnostics flush.",
        ["Source/Evolve/WarpXEvolve.cpp:171,372-396", "Source/main.cpp:24-37"],
        "rank 0's clock",
        "yes: TinyProfiler regions synchronize the device (WarpX sets "
        "tiny_profiler.device_synchronize_around_region = amrex_use_gpu, "
        "Source/Initialization/WarpXAMReXInit.cpp:63-80), and every step ends inside them"),
    "nyx": Timer(
        nyx, [],
        "sum of '[STEP n] Coarse TimeStep time' over all steps: computeNewDt + timeStep + "
        "postCoarseTimeStep, without checkpoint/plotfile writes (reported as excluded_s) and "
        "initialization.",
        ["../deps/amrex/Src/Amr/AMReX_Amr.cpp:2154-2207,2335-2343", "Source/Driver/nyx_main.cpp:84,122,163-187"],
        "maximum over ranks per step",
        "step totals: effectively (the step contains stream synchronizations and a ReduceMin in the "
        "dt computation) but the timer itself does not synchronize"),
    "nekrs": Timer(
        nekrs, [],
        "'solve' of the final runtime statistics (elapsedStepSum: every time step between MPI "
        "barriers) minus its 'checkpointing' row (field-file output, reported as excluded_s); "
        "without the initialization (mesh, JIT load, kernel autotuning).",
        ["src/app/nrs/nrs.cpp:748-768", "src/bin/driver.cpp:220-227,256-316", "src/platform/timer.cpp:41-103"],
        "maximum over ranks",
        "total: yes (host clock between barriers, steps end in blocking event syncs); nested rows are "
        "CUDA-event times on the current stream"),
    "specfem3d": Timer(
        specfem3d, ["OUTPUT_FILES/output_solver.txt"],
        "'Elapsed time' of the last stability report (at the final time step): the time loop from "
        "its start after the start-up barrier, without the final seismogram write and the "
        "mesher / database generation.",
        ["src/specfem3D/iterate_time.F90:113,208,222-223,339", "src/specfem3D/check_stability.f90:198-205"],
        "rank 0's clock, read after an MPI max reduction of the field norms",
        "yes: every stability report synchronizes the compute stream and copies the norm back "
        "(src/gpu/check_fields_cuda.cu:440-443)"),
    "exaca": Timer(
        exaca, [],
        "'Time spent performing CA calculations': the whole cellular-automaton loop over layers "
        "(nucleation, steering vector, cell capture, halo exchange, periodic progress checks), "
        "without initialization and the final output.",
        ["src/runCA.hpp:75-201", "src/CAtimers.hpp:13-147"],
        "rank 0's clock, ranks aligned by barriers; sub-timers reported as the maximum rank",
        "yes: every timed phase ends in a Kokkos fence or a blocking reduction/copy"),
    "qmcpack": Timer(
        qmcpack, [],
        "the DMC driver's 'DMCBatched::Production' timer (stack timer profile, --enable-timers=medium): "
        "every DMC block and step after a barrier, including the per-block estimator reduction and "
        "output; without start-up, wavefunction set-up, the VMC section before DMC (reported as a part) "
        "and the driver's own start-up.",
        ["src/QMCDrivers/DMC/DMCBatched.cpp:477-521", "src/QMCDrivers/QMCDriverNew.h:371-384",
         "src/Utilities/TimerManager.cpp:245-296"],
        "rank 0's clock (the stack profile is not reduced over ranks); the region starts after a barrier",
        "the timers read the host clock only; every DMC step ends with host-side branching on the walkers' "
        "energies, so the step loop is synchronous, but nested timers can be charged with asynchronous "
        "offload work of a neighbour",
        caveat="QMCPACK's nested timers (parts) are not device-synchronized; the Production total is."),
    "dftfe": Timer(
        dftfe, [],
        "MD steps 1..N as DFT-FE times them: per step 'Time taken for updateAtomPositionsAndMoveMesh' "
        "(atom update + mesh move + re-initialization) plus the step's SCF iterations ('Wall time for "
        "the above scf iteration'). Not in it: the initial ground state before MD step 0 (set-up, reported "
        "as a part), and the force evaluation and integrator of each step, which DFT-FE does not time "
        "while REPRODUCIBLE OUTPUT = true (the upstream reference deck; the per-MD-step timer is "
        "suppressed). The region is the sum of DFT-FE's own step timers, not one contiguous span.",
        ["src/md/molecularDynamicsClass.cc:1225,1291-1296", "src/dft/dft.cc:2603,3655-3658", "src/main.cc:317-325"],
        "SCF timer: deal.II Timer on the parent communicator (barrier at start); mesh update: rank 0's "
        "clock after an MPI barrier",
        "no explicit device synchronization in either timer; each SCF iteration ends in host-side "
        "reductions of energies/residuals (implicit synchronization, UNVERIFIED)",
        caveat="DFT-FE's timed region leaves out the force evaluation and MD integration of every step: "
               "under REPRODUCIBLE OUTPUT = true (the upstream reference deck, which must stay: it also fixes "
               "numerical settings) DFT-FE does not time them."),
    "cp2k": Timer(
        cp2k, ["*-1.ener"],
        "sum of the per-MD-step times (UsedTime of the .ener file) of steps 2..N; step 1 also "
        "carries the initial force evaluation of step 0 (set-up and SCF from scratch) and is "
        "reported as a part.",
        ["src/motion/md_run.F:320,384,553-555", "src/motion/md_energies.F:407-412,719"],
        "rank 0's clock",
        "the timer does not synchronize; the GPU back-ends (DBM, PW, grid) synchronize their "
        "streams before returning"),
}


# ----------------------------------------------------------------- entry points

def extract(app, d):
    t = TIMERS.get(app)
    if t is None:
        raise TimerMissing(f"no timer definition for {app!r}")
    return t.fn(d)


def describe(app):
    t = TIMERS[app]
    return {"definition": t.definition, "where": t.where, "reduction": t.reduction,
            "device_sync": t.device_sync, "harvest": t.harvest, "caveat": t.caveat}


def harvest(app, tree, dest):
    """Copy run_manifest.txt and the extractor's files of every run directory under <tree>
    (build/level3/<app>/*/<subdir>) into <dest>/<run>/. Returns the copied paths."""
    t = TIMERS.get(app)
    if t is None:
        raise TimerMissing(f"no timer definition for {app!r}")
    copied = []
    for run in sorted(glob.glob(os.path.join(tree, "*"))):
        if not os.path.isdir(run):
            continue
        for pat in ["run_manifest.txt"] + t.harvest:
            for src in glob.glob(os.path.join(run, "**", pat) if "/" not in pat else os.path.join(run, pat),
                                 recursive=True):
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(dest, os.path.basename(run), os.path.relpath(src, run))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
                copied.append(dst)
    return copied


def main(argv):
    if len(argv) >= 3 and argv[0] == "extract":
        try:
            print(json.dumps(extract(argv[1], argv[2]), sort_keys=True))
        except (TimerMissing, OSError, ValueError, IndexError) as exc:
            print(f"apptimers: {argv[1]}: {exc}", file=sys.stderr)
            return 1
        return 0
    if len(argv) == 4 and argv[0] == "harvest":
        try:
            n = harvest(argv[1], argv[2], argv[3])
        except (TimerMissing, OSError) as exc:
            print(f"apptimers: {argv[1]}: {exc}", file=sys.stderr)
            return 1
        print(len(n))
        return 0
    if argv and argv[0] == "describe":
        apps = argv[1:] or sorted(TIMERS)
        print(json.dumps({a: describe(a) for a in apps}, indent=2, sort_keys=True))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
