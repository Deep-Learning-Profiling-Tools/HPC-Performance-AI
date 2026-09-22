"""NVIDIA Nsight Systems adapter: the sqlite export of one profiled run -> canonical model.

The measurement side (tools/timing/lib/collectors.sh) runs

    nsys profile -t cuda,nvtx -s none --cpuctxsw=none --cuda-graph-trace=node ...
    nsys export --type sqlite

and leaves <raw>/prof/trace.sqlite. This module only reads that file with the standard
library, so summarizing does not need nsys installed.

Facts this relies on, checked on nsys 2025.6.3 (conformance probe, 2026-09-22):
  * NVTX push/pop ranges are NVTX_EVENTS rows with eventType 59; the message is in
    `text` or, when registered, in StringIds via `textId`.
  * CUPTI_ACTIVITY_KIND_{KERNEL,MEMCPY,MEMSET} and NVTX_EVENTS share one timeline.
  * The process of an activity row is globalPid >> 24; of an NVTX / runtime row,
    globalTid >> 24. Both give the same key for the same process.
  * MEMCPY.copyKind indexes ENUM_CUDA_MEMCPY_OPER (0 unknown, 1 HtoD, 2 DtoH,
    3 HtoA, 4 AtoH, 5 AtoA, 6 AtoD, 7 DtoA, 8 DtoD, 9 HtoH, 10 PtoP, 11/12/13 unified-memory
    migrations HtoD/DtoH/DtoD). CUDA arrays are device memory, so A counts as D; unified
    migrations are copies in their direction (quicksilver: 83k HtoD + 81k DtoH per run).
"""

import bisect
import os
import re
import sqlite3

from collectors import EXCLUDE_RANGE, ROI_RANGE, Interval, Marker

NAME = "nvidia_nsys"
RUNTIME = "cuda"
VERIFIED = True
CAPABILITIES = frozenset({"compute", "copy_h2d", "copy_d2h", "copy_d2d", "copy_other", "fill"})

TRACE_FILE = "trace.sqlite"
NVTX_PUSHPOP = 59
_COPY_KIND = {1: "copy_h2d", 3: "copy_h2d", 11: "copy_h2d",
              2: "copy_d2h", 4: "copy_d2h", 12: "copy_d2h",
              5: "copy_d2d", 6: "copy_d2d", 7: "copy_d2d", 8: "copy_d2d", 13: "copy_d2d"}
# everything else (0 unknown, 9 host-to-host, 10 peer-to-peer) is copy_other
# host calls that block until the device (or a stream / event) is done
SYNC_API = re.compile(r"^cuda(DeviceSynchronize|StreamSynchronize|EventSynchronize|Memcpy|Memset)(_v\d+)?$")
_MANY_WINDOWS = 64


class Trace:
    def __init__(self, path):
        self.path = path
        # read-only, immutable: never let a summarize write into raw evidence
        self.db = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        self.tables = {r[0] for r in self.db.execute("select name from sqlite_master where type='table'")}
        self._sync_ids = None

    def close(self):
        self.db.close()

    # ------------------------------------------------------------------ metadata
    def info(self):
        out = {"collector": NAME, "trace_file": os.path.basename(self.path),
               "trace_bytes": os.path.getsize(self.path)}
        if "TARGET_INFO_SYSTEM_ENV" in self.tables:
            row = self.db.execute(
                "select value from TARGET_INFO_SYSTEM_ENV where name='DeviceEnvironment'").fetchone()
            if row and row[0]:
                # names only: the values are exactly what must never be copied anywhere
                names = sorted({kv.split("=", 1)[0] for kv in row[0].split(";") if "=" in kv})
                out["recorded_env_names"] = names
        return out

    # ------------------------------------------------------------------ markers
    def markers(self):
        if "NVTX_EVENTS" not in self.tables:
            return []
        q = ("select coalesce(n.text, s.value), n.start, n.end, n.globalTid >> 24 "
             "from NVTX_EVENTS n left join StringIds s on n.textId = s.id "
             "where n.eventType = ? and n.end is not null")
        out = []
        for text, start, end, proc in self.db.execute(q, (NVTX_PUSHPOP,)):
            if text == ROI_RANGE:
                out.append(Marker(proc, "roi", start, end))
            elif text == EXCLUDE_RANGE:
                out.append(Marker(proc, "exclude", start, end))
        return out

    # ------------------------------------------------------------------ activity
    def intervals(self):
        parts = []
        if "CUPTI_ACTIVITY_KIND_KERNEL" in self.tables:
            parts.append("select start, end, 0, demangledName, globalPid >> 24, 0 "
                         "from CUPTI_ACTIVITY_KIND_KERNEL")
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in self.tables:
            parts.append("select start, end, 1, copyKind, globalPid >> 24, bytes "
                         "from CUPTI_ACTIVITY_KIND_MEMCPY")
        if "CUPTI_ACTIVITY_KIND_MEMSET" in self.tables:
            parts.append("select start, end, 2, 0, globalPid >> 24, bytes "
                         "from CUPTI_ACTIVITY_KIND_MEMSET")
        if not parts:
            return
        q = " union all ".join(parts) + " order by 1"
        for start, end, table, sub, proc, nbytes in self.db.execute(q):
            if table == 0:
                yield Interval(start, end, "compute", ("k", sub), proc, 0)
            elif table == 1:
                cat = _COPY_KIND.get(sub, "copy_other")
                yield Interval(start, end, cat, ("c", sub), proc, nbytes or 0)
            else:
                yield Interval(start, end, "fill", ("f", 0), proc, nbytes or 0)

    def op_names(self, keys):
        out = {}
        kernel_ids = [k[1] for k in keys if k[0] == "k"]
        names = {}
        for i in range(0, len(kernel_ids), 500):
            chunk = kernel_ids[i:i + 500]
            q = f"select id, value from StringIds where id in ({','.join('?' * len(chunk))})"
            names.update(dict(self.db.execute(q, chunk)))
        copy_names = {}
        if "ENUM_CUDA_MEMCPY_OPER" in self.tables:
            copy_names = {r[0]: r[2] for r in self.db.execute("select * from ENUM_CUDA_MEMCPY_OPER")}
        for k in keys:
            if k[0] == "k":
                out[k] = names.get(k[1], f"kernel#{k[1]}")
            elif k[0] == "c":
                out[k] = f"[memcpy {copy_names.get(k[1], k[1])}]"
            else:
                out[k] = "[memset]"
        return out

    # ------------------------------------------------------------------ runtime API
    def _sync_name_ids(self):
        if self._sync_ids is None:
            ids = []
            if "CUPTI_ACTIVITY_KIND_RUNTIME" in self.tables:
                q = "select distinct r.nameId, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId = s.id"
                ids = [i for i, v in self.db.execute(q) if SYNC_API.match(v or "")]
            self._sync_ids = ids
        return self._sync_ids

    def runtime_calls(self, windows=None):
        zero = {"calls": 0, "time_ns": 0, "sync_calls": 0, "sync_ns": 0}
        if "CUPTI_ACTIVITY_KIND_RUNTIME" not in self.tables:
            return None
        sync = self._sync_name_ids()
        in_sync = f"nameId in ({','.join(str(i) for i in sync)})" if sync else "0"
        agg = (f"select count(*), total(end - start), total({in_sync}), "
               f"total(case when {in_sync} then end - start else 0 end) from CUPTI_ACTIVITY_KIND_RUNTIME")
        acc = dict(zero)

        def add(row):
            acc["calls"] += int(row[0] or 0)
            acc["time_ns"] += int(row[1] or 0)
            acc["sync_calls"] += int(row[2] or 0)
            acc["sync_ns"] += int(row[3] or 0)

        if windows is None:
            add(self.db.execute(agg).fetchone())
            return acc
        n_windows = sum(len(w) for w in windows.values())
        if n_windows <= _MANY_WINDOWS:
            for proc, wins in windows.items():
                for ws, we in wins:
                    add(self.db.execute(agg + " where (globalTid >> 24) = ? and start >= ? and end <= ?",
                                        (proc, ws, we)).fetchone())
            return acc
        # many windows: one unordered pass, bisect each call into its process's windows
        starts = {p: [w[0] for w in ws] for p, ws in windows.items()}
        sync_set = set(sync)
        for start, end, name_id, proc in self.db.execute(
                "select start, end, nameId, globalTid >> 24 from CUPTI_ACTIVITY_KIND_RUNTIME"):
            wins = windows.get(proc)
            if not wins:
                continue
            i = bisect.bisect_right(starts[proc], start) - 1
            if i >= 0 and end <= wins[i][1]:
                acc["calls"] += 1
                acc["time_ns"] += end - start
                if name_id in sync_set:
                    acc["sync_calls"] += 1
                    acc["sync_ns"] += end - start
        return acc


def open(raw_dir):  # noqa: A001 -- the collector interface name
    path = os.path.join(raw_dir, TRACE_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path}: no nsys sqlite export (profiled run failed or export skipped)")
    return Trace(path)
