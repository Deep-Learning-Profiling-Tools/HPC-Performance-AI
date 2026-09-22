"""AMD adapter (rocprofiler-sdk / rocprofv3) -- INTERFACE ONLY, UNVERIFIED.

Not implemented: there is no ROCm on the node this suite was built on, so nothing here
could be run, and an adapter that has never produced a trace must not produce numbers.
What an implementation has to do, and has to prove with the conformance probe
(tools/timing/probes/conformance/probe.cu compiled with hipcc) before VERIFIED = True:

  measurement (tools/timing/lib/collectors.sh, collector "amd_rocprofv3"):
      rocprofv3 --kernel-trace --memory-copy-trace --marker-trace --hip-runtime-trace
                --output-format <a machine-readable format> -d <raw>/prof -- <command>
      (flag names as documented for rocprofiler-sdk; confirm on the target ROCm version)

  mapping to the canonical model (collectors/__init__.py):
      kernel dispatches                       -> "compute"
      memory copies by direction              -> "copy_h2d" / "copy_d2h" / "copy_d2d" / "copy_other"
      ROCTX push/pop "hpcperf:roi" / ":exclude" -> Marker(kind="roi" / "exclude")
      HIP runtime API calls                   -> runtime_calls(); sync = hipDeviceSynchronize,
                                                 hipStreamSynchronize, hipEventSynchronize,
                                                 hipMemcpy, hipMemset
      process key                             -> identical for markers and activity of one process

  facts that must be checked, not assumed:
      * markers and device timestamps are on one timeline (probe: 10 / 20 / 1 split exact);
      * whether the tool records the process environment in its output (nsys does: the
        probe plants a sentinel and greps the raw output for it);
      * which categories are observable -> CAPABILITIES (e.g. an APU with unified memory
        has no H2D copies: that category is null there, never 0).
"""

NAME = "amd_rocprofv3"
RUNTIME = "hip"
VERIFIED = False
CAPABILITIES = frozenset({"compute", "copy_h2d", "copy_d2h", "copy_d2d", "copy_other", "fill"})


def open(raw_dir):  # noqa: A001
    raise NotImplementedError(
        "collector amd_rocprofv3 is interface-only and UNVERIFIED (no ROCm on the node it was "
        "written on); implement it and pass tools/timing/probes/conformance before using it")
