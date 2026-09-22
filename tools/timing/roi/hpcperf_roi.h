/*
 * hpcperf_roi.h -- region-of-interest (ROI) markers for tools/timing.
 *
 * Marks the part of a benchmark that is "the computation": after set-up and
 * warm-up, before result checking and output.
 *
 *   HPCPERF_ROI_BEGIN();            ... the computation ...          HPCPERF_ROI_END();
 *   HPCPERF_ROI_EXCLUDE_BEGIN();    ... a check inside the ROI ...   HPCPERF_ROI_EXCLUDE_END();
 *   HPCPERF_ROI_EXCLUDE_BEGIN_SYNC()  an exclude that follows asynchronous device work
 *   HPCPERF_ROI_BEGIN_SYNC() / HPCPERF_ROI_END_SYNC()   the same, after a device-wide
 *                                   synchronize (only when measuring; see below)
 *
 * OFF BY DEFAULT. Unless a profiler is attached or HPCPERF_ROI_LOG is set, every
 * marker is a few-nanosecond no-op and the program behaves exactly as before: no
 * extra synchronization, no file, no output. ctest and validate.sh never set either.
 *
 * One set of markers, two consumers:
 *   HPCPERF_ROI_LOG=<prefix>   clean timing without a profiler. CLOCK_MONOTONIC and
 *                              CLOCK_REALTIME timestamps of each ROI entry are buffered
 *                              in memory; excludes are summed per entry, not logged one
 *                              by one (an exclude inside a 100k-step loop costs no buffer).
 *                              The buffer is appended to <prefix>.<pid> at exit, or after
 *                              an ROI ends once it is half full -- never inside an ROI.
 *   a profiler                 the markers are also emitted as named ranges
 *                              "hpcperf:roi" / "hpcperf:exclude" through the backend's
 *                              annotation API, so device activity can be clipped to them.
 *
 * Annotation backend (compile time):
 *   default                    NVTX v3, vendored header-only copy (NVIDIA Nsight Systems)
 *   HPCPERF_ROI_ROCTX          ROCTX resolved with dlopen at first use (AMD rocprof); no
 *                              ROCm header or link dependency. Selected automatically for
 *                              __HIP_PLATFORM_AMD__. UNVERIFIED: no ROCm on the node it was
 *                              written on. If no ROCTX library is found the markers still
 *                              do clean timing; the profiler just sees no range.
 *   HPCPERF_ROI_NO_ANNOTATION  clean timing only
 *
 * Rules the call sites follow (tools/timing/roi/README.md has the full placement rule):
 *   - one outermost ROI per computation phase; nesting is allowed and only the
 *     outermost BEGIN/END count; the ROI may be entered several times (all entries sum);
 *   - BEGIN and END on the thread that drives the device (NVTX push/pop are per thread);
 *   - EXCLUDE only inside a ROI; it is ignored outside one.
 * Linking: glibc >= 2.34 has dlopen in libc; older systems need -ldl (NVTX needs it too).
 */
#ifndef HPCPERF_ROI_H
#define HPCPERF_ROI_H

/* Strict ISO C (-std=c99/c11) hides POSIX. Ask for it only in that mode: in the gnu
 * modes and in C++ it is already visible, and defining _POSIX_C_SOURCE there would
 * switch _DEFAULT_SOURCE off and could break the application's own code (M_PI, ...).
 * It takes effect only if no system header was included before this one. */
#if !defined(__cplusplus) && defined(__STRICT_ANSI__) && !defined(_POSIX_C_SOURCE) && \
    !defined(_GNU_SOURCE) && !defined(_XOPEN_SOURCE) && !defined(_DEFAULT_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if !defined(CLOCK_MONOTONIC)
#error "hpcperf_roi.h needs POSIX clock_gettime: include it before other system headers, or build with a gnu C standard / -D_GNU_SOURCE"
#endif

#if !defined(HPCPERF_ROI_NO_ANNOTATION) && !defined(HPCPERF_ROI_ROCTX) && defined(__HIP_PLATFORM_AMD__)
#define HPCPERF_ROI_ROCTX 1
#endif
#if !defined(HPCPERF_ROI_NO_ANNOTATION) && !defined(HPCPERF_ROI_ROCTX)
#include "hpcperf_vendor/nvtx3/nvToolsExt.h"
#define HPCPERF_ROI_NVTX 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define HPCPERF_ROI_LOG_VERSION 2
#define HPCPERF_ROI_MAX_EVENTS 8192

/* One state per process. A weak definition in every translation unit that includes
 * this header collapses to a single object at link time, so BEGIN and END may sit in
 * different files. */
typedef struct {
    int initialized, enabled, measuring, depth, xdepth, header_written;
    unsigned n, overflow, unmatched, xcount;
    long long xstart, xsum;
    char kind[HPCPERF_ROI_MAX_EVENTS];
    long long mono[HPCPERF_ROI_MAX_EVENTS];
    long long real[HPCPERF_ROI_MAX_EVENTS];
    char path[4096];
    int roctx_tried;
    int (*roctx_push)(const char*);
    int (*roctx_pop)(void);
} hpcperf_roi_state_t;

__attribute__((weak)) hpcperf_roi_state_t hpcperf_roi_state;

static inline long long hpcperf_roi__ns(clockid_t c) {
    struct timespec t;
    clock_gettime(c, &t);
    return (long long)t.tv_sec * 1000000000LL + (long long)t.tv_nsec;
}

static inline void hpcperf_roi__json_str(FILE* f, const char* s, size_t n) {
    size_t i;
    fputc('"', f);
    for (i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') { fputc('\\', f); fputc(c, f); }
        else if (c < 0x20) fprintf(f, "\\u%04x", c);
        else fputc(c, f);
    }
    fputc('"', f);
}

static inline void hpcperf_roi__write_header(FILE* f) {
    static const char* rank_vars[] = {"OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "SLURM_PROCID", 0};
    const char* rank = "-";
    char buf[65536];
    int i;
    ssize_t len;
    FILE* cf;
    for (i = 0; rank_vars[i]; i++) {
        const char* v = getenv(rank_vars[i]);
        if (v && *v) { rank = v; break; }
    }
    fprintf(f, "# hpcperf-roi-log %d\n", HPCPERF_ROI_LOG_VERSION);
    fprintf(f, "pid %ld\nrank %s\nclock CLOCK_MONOTONIC CLOCK_REALTIME\n", (long)getpid(), rank);
    if (gethostname(buf, sizeof buf) == 0) { buf[sizeof buf - 1] = 0; fprintf(f, "host %s\n", buf); }
    len = readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (len > 0) { buf[len] = 0; fprintf(f, "exe %s\n", buf); }
    if (getcwd(buf, sizeof buf)) fprintf(f, "cwd %s\n", buf);
    cf = fopen("/proc/self/cmdline", "rb");
    if (cf) {
        size_t got = fread(buf, 1, sizeof buf, cf), start = 0, k;
        int first = 1;
        fclose(cf);
        fputs("argv [", f);
        for (k = 0; k < got; k++) {
            if (buf[k] == 0) {
                if (!first) fputs(", ", f);
                hpcperf_roi__json_str(f, buf + start, k - start);
                first = 0;
                start = k + 1;
            }
        }
        fputs("]\n", f);
    }
}

static inline void hpcperf_roi__flush(void) {
    hpcperf_roi_state_t* s = &hpcperf_roi_state;
    FILE* f;
    unsigned i;
    if (!s->enabled) return;
    if (s->n == 0 && s->header_written && !s->overflow && !s->unmatched) return;
    f = fopen(s->path, "a");
    if (!f) return;
    if (!s->header_written) { hpcperf_roi__write_header(f); s->header_written = 1; }
    /* B/E/U: <monotonic ns> <realtime ns>;  x: <excluded ns summed> <number of excludes> */
    for (i = 0; i < s->n; i++) fprintf(f, "%c %lld %lld\n", s->kind[i], s->mono[i], s->real[i]);
    if (s->overflow) fprintf(f, "overflow %u\n", s->overflow);
    if (s->unmatched) fprintf(f, "unmatched_end %u\n", s->unmatched);
    fclose(f);
    s->n = 0;
    s->overflow = 0;
    s->unmatched = 0;
}

static void hpcperf_roi__atexit(void) {
    hpcperf_roi_state_t* s = &hpcperf_roi_state;
    if (s->enabled && s->depth > 0) {            /* an ROI that never ended: say so */
        if (s->n < HPCPERF_ROI_MAX_EVENTS) {
            s->kind[s->n] = 'U';
            s->mono[s->n] = hpcperf_roi__ns(CLOCK_MONOTONIC);
            s->real[s->n] = hpcperf_roi__ns(CLOCK_REALTIME);
            s->n++;
        }
    }
    hpcperf_roi__flush();
}

static inline void hpcperf_roi__init(void) {
    hpcperf_roi_state_t* s = &hpcperf_roi_state;
    const char* p;
    if (s->initialized) return;
    s->initialized = 1;
    p = getenv("HPCPERF_ROI_LOG");
    if (p && *p) {
        snprintf(s->path, sizeof s->path, "%s.%ld", p, (long)getpid());
        s->enabled = 1;
        atexit(hpcperf_roi__atexit);
    }
    /* "measuring": clean timing, or a profiler injected into this process. Only then
     * do the _SYNC variants synchronize, so the default path gains no synchronization. */
    s->measuring = s->enabled || getenv("NVTX_INJECTION64_PATH") != 0 ||
                   getenv("ROCP_TOOL_LIBRARIES") != 0 || getenv("ROCPROFILER_REGISTER_LIBRARY") != 0;
}

static inline void hpcperf_roi__record_values(char k, long long a, long long b) {
    hpcperf_roi_state_t* s = &hpcperf_roi_state;
    if (!s->enabled) return;
    if (s->n >= HPCPERF_ROI_MAX_EVENTS) { s->overflow++; return; }
    s->kind[s->n] = k;
    s->mono[s->n] = a;
    s->real[s->n] = b;
    s->n++;
}

static inline void hpcperf_roi__record(char k) {
    if (!hpcperf_roi_state.enabled) return;
    hpcperf_roi__record_values(k, hpcperf_roi__ns(CLOCK_MONOTONIC), hpcperf_roi__ns(CLOCK_REALTIME));
}

#if defined(HPCPERF_ROI_ROCTX)
static inline void hpcperf_roi__roctx_load(void) {
    static const char* libs[] = {"librocprofiler-sdk-roctx.so.1", "librocprofiler-sdk-roctx.so",
                                 "libroctx64.so.4", "libroctx64.so", 0};
    hpcperf_roi_state_t* s = &hpcperf_roi_state;
    int i;
    if (s->roctx_tried) return;
    s->roctx_tried = 1;
    for (i = 0; libs[i]; i++) {
        void* h = dlopen(libs[i], RTLD_LAZY | RTLD_LOCAL);
        if (!h) continue;
        *(void**)(&s->roctx_push) = dlsym(h, "roctxRangePushA");
        *(void**)(&s->roctx_pop) = dlsym(h, "roctxRangePop");
        if (s->roctx_push && s->roctx_pop) return;
        s->roctx_push = 0;
        s->roctx_pop = 0;
    }
}
#endif

static inline void hpcperf_roi__annotate_push(const char* name) {
#if defined(HPCPERF_ROI_NVTX)
    nvtxRangePushA(name);
#elif defined(HPCPERF_ROI_ROCTX)
    hpcperf_roi__roctx_load();
    if (hpcperf_roi_state.roctx_push) hpcperf_roi_state.roctx_push(name);
#else
    (void)name;
#endif
}

static inline void hpcperf_roi__annotate_pop(void) {
#if defined(HPCPERF_ROI_NVTX)
    nvtxRangePop();
#elif defined(HPCPERF_ROI_ROCTX)
    if (hpcperf_roi_state.roctx_pop) hpcperf_roi_state.roctx_pop();
#endif
}

/* Device-wide synchronize without any GPU runtime header or link dependency: use the
 * runtime the process has ALREADY loaded (RTLD_NOLOAD never loads one). CUDA goes
 * through the driver API so a statically linked cudart works too. */
static inline void hpcperf_roi_device_sync(void) {
    static const char* hip_libs[] = {"libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.7", 0};
    void* h;
    int i;
    hpcperf_roi__init();
    if (!hpcperf_roi_state.measuring) return;
    h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (h) {
        int (*f)(void);
        *(void**)(&f) = dlsym(h, "cuCtxSynchronize");
        if (f) f();
        dlclose(h);
        return;
    }
    for (i = 0; hip_libs[i]; i++) {
        h = dlopen(hip_libs[i], RTLD_LAZY | RTLD_NOLOAD);
        if (!h) continue;
        {
            int (*f)(void);
            *(void**)(&f) = dlsym(h, "hipDeviceSynchronize");
            if (f) f();
        }
        dlclose(h);
        return;
    }
}

static inline void hpcperf_roi_begin(void) {
    hpcperf_roi_state_t* s = &hpcperf_roi_state;
    hpcperf_roi__init();
    if (s->depth++ > 0) return;                  /* nested: only the outermost counts */
    hpcperf_roi__annotate_push("hpcperf:roi");
    hpcperf_roi__record('B');                    /* after the push: its cost is outside */
}

static inline void hpcperf_roi_end(void) {
    hpcperf_roi_state_t* s = &hpcperf_roi_state;
    hpcperf_roi__init();
    if (s->depth <= 0) { s->unmatched++; return; }
    if (--s->depth > 0) return;
    if (s->xdepth > 0) {                         /* an exclude left open: close it here */
        s->xdepth = 0;
        hpcperf_roi__annotate_pop();
        if (s->enabled) { s->xsum += hpcperf_roi__ns(CLOCK_MONOTONIC) - s->xstart; s->xcount++; }
    }
    hpcperf_roi__record('E');                    /* before the pop and the flush */
    if (s->xcount) hpcperf_roi__record_values('x', s->xsum, (long long)s->xcount);
    s->xsum = 0;
    s->xcount = 0;
    hpcperf_roi__annotate_pop();
    if (s->n >= HPCPERF_ROI_MAX_EVENTS / 2) hpcperf_roi__flush();   /* outside any ROI */
}

static inline void hpcperf_roi_exclude_begin(void) {
    hpcperf_roi_state_t* s = &hpcperf_roi_state;
    hpcperf_roi__init();
    if (s->depth <= 0) return;                   /* excludes only mean something inside */
    if (s->xdepth++ > 0) return;
    if (s->enabled) s->xstart = hpcperf_roi__ns(CLOCK_MONOTONIC);
    hpcperf_roi__annotate_push("hpcperf:exclude");
}

static inline void hpcperf_roi_exclude_end(void) {
    hpcperf_roi_state_t* s = &hpcperf_roi_state;
    hpcperf_roi__init();
    if (s->depth <= 0 || s->xdepth <= 0) return;
    if (--s->xdepth > 0) return;
    hpcperf_roi__annotate_pop();
    if (s->enabled) { s->xsum += hpcperf_roi__ns(CLOCK_MONOTONIC) - s->xstart; s->xcount++; }
}

#ifdef __cplusplus
}
#endif

#define HPCPERF_ROI_BEGIN()         hpcperf_roi_begin()
#define HPCPERF_ROI_END()           hpcperf_roi_end()
#define HPCPERF_ROI_BEGIN_SYNC()    do { hpcperf_roi_device_sync(); hpcperf_roi_begin(); } while (0)
#define HPCPERF_ROI_END_SYNC()      do { hpcperf_roi_device_sync(); hpcperf_roi_end(); } while (0)
#define HPCPERF_ROI_EXCLUDE_BEGIN() hpcperf_roi_exclude_begin()
/* Use before an exclude that follows asynchronous device work (e.g. a snapshot after a
 * step): otherwise the tail of that work would fall inside the excluded window. */
#define HPCPERF_ROI_EXCLUDE_BEGIN_SYNC() do { hpcperf_roi_device_sync(); hpcperf_roi_exclude_begin(); } while (0)
#define HPCPERF_ROI_EXCLUDE_END()   hpcperf_roi_exclude_end()

#endif /* HPCPERF_ROI_H */
