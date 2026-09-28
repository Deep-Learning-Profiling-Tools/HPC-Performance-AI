/* tools/timing/probes/bindprobe.c -- where did the measured process actually run?
 *
 * A tiny LD_PRELOAD library, injected by tools/timing/lib/engine.sh into every run it
 * starts. It records, for the benchmark process itself (not the launching shell), the
 * CPU set and NUMA memory set it was allowed to use, the CPU each of its threads last
 * ran on, the /dev/nvidia<N> devices it held open and the placement-related
 * environment (CUDA_VISIBLE_DEVICES, OMP_*, MPI local rank). Everything is read at
 * process start (constructor, before main) and at exit (destructor, after the
 * application's own atexit handlers): nothing runs inside the ROI, no thread is
 * created, no signal is used. The application binary is not changed, so the record's
 * exe sha256 stays what it was.
 *
 * Output: <HPCPERF_BIND_LOG>.<pid>, one file per process that either held a GPU
 * device open or wrote an ROI log (<HPCPERF_ROI_LOG>.<pid>); shells, launchers and
 * helpers write nothing. Lines are "key value"; see tools/timing/SCHEMA.md
 * ("placement"). The file is diagnostic context, never a metric.
 *
 * Build (the engine does this once per raw root):
 *   cc -O2 -shared -fPIC -o bindprobe.so bindprobe.c
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>

#define BP_VERSION 1
#define BP_SMALL 256

static char bp_cpus0[BP_SMALL], bp_mems0[BP_SMALL];
static int bp_cpu0 = -1;
static long long bp_t0 = 0;
static int bp_armed = 0;

static long long bp_now(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) return 0;
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* value of "Key:\t..." in a /proc status file, without the newline; "" if absent */
static void bp_status_field(const char* path, const char* key, char* out, size_t n) {
    FILE* f = fopen(path, "r");
    char line[BP_SMALL * 4];
    size_t k = strlen(key);
    out[0] = 0;
    if (!f) return;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, key, k) == 0 && line[k] == ':') {
            char* v = line + k + 1;
            while (*v == ' ' || *v == '\t') v++;
            v[strcspn(v, "\n")] = 0;
            snprintf(out, n, "%s", v);
            break;
        }
    }
    fclose(f);
}

/* field 39 of /proc/<..>/stat (the CPU the task last ran on), -1 if unreadable */
static int bp_last_cpu(const char* path) {
    FILE* f = fopen(path, "r");
    char buf[2048];
    char* p;
    int i, cpu = -1;
    if (!f) return -1;
    if (!fgets(buf, sizeof buf, f)) { fclose(f); return -1; }
    fclose(f);
    p = strrchr(buf, ')');                 /* comm may contain spaces and parentheses */
    if (!p) return -1;
    p++;
    for (i = 3; i < 39 && p; i++) {        /* fields 3.. follow the closing parenthesis */
        while (*p == ' ') p++;
        p = strchr(p, ' ');
    }
    if (p) cpu = (int)strtol(p, NULL, 10);
    return cpu;
}

static void bp_init(void) __attribute__((constructor));
static void bp_init(void) {
    const char* log = getenv("HPCPERF_BIND_LOG");
    if (!log || !*log) return;
    bp_armed = 1;
    bp_t0 = bp_now();
    bp_cpu0 = sched_getcpu();
    bp_status_field("/proc/self/status", "Cpus_allowed_list", bp_cpus0, sizeof bp_cpus0);
    bp_status_field("/proc/self/status", "Mems_allowed_list", bp_mems0, sizeof bp_mems0);
}

static void bp_fini(void) __attribute__((destructor));
static void bp_fini(void) {
    const char* log = getenv("HPCPERF_BIND_LOG");
    const char* roi = getenv("HPCPERF_ROI_LOG");
    static const char* const envs[] = {
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HPCPERF_GPUS", "OMP_NUM_THREADS",
        "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY",
        "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "OMPI_COMM_WORLD_SIZE",
        "PMIX_RANK", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_JOB_ID", "SLURM_JOB_GPUS", 0 };
    char path[4096], buf[4096], val[BP_SMALL];
    int minors[64], nminor = 0, has_roi = 0, i;
    DIR* d;
    struct dirent* e;
    FILE* out;
    if (!bp_armed || !log || !*log) return;

    /* GPU devices this process holds open: /dev/nvidia<N> (not nvidiactl / nvidia-uvm) */
    d = opendir("/proc/self/fd");
    if (d) {
        while ((e = readdir(d))) {
            ssize_t n;
            int minor;
            if (e->d_name[0] == '.') continue;
            snprintf(path, sizeof path, "/proc/self/fd/%s", e->d_name);
            n = readlink(path, buf, sizeof buf - 1);
            if (n <= 0) continue;
            buf[n] = 0;
            if (strncmp(buf, "/dev/nvidia", 11) != 0) continue;
            if (buf[11] < '0' || buf[11] > '9') continue;
            minor = (int)strtol(buf + 11, NULL, 10);
            for (i = 0; i < nminor; i++) if (minors[i] == minor) break;
            if (i == nminor && nminor < 64) minors[nminor++] = minor;
        }
        closedir(d);
    }
    if (roi && *roi) {
        snprintf(path, sizeof path, "%s.%ld", roi, (long)getpid());
        has_roi = access(path, F_OK) == 0;
    }
    if (nminor == 0 && !has_roi) return;      /* a shell, a launcher, a helper: not measured */

    snprintf(path, sizeof path, "%s.%ld", log, (long)getpid());
    out = fopen(path, "w");
    if (!out) return;
    fprintf(out, "# hpcperf-bind-log %d\n", BP_VERSION);
    fprintf(out, "pid %ld\nppid %ld\n", (long)getpid(), (long)getppid());
    {
        ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
        if (n > 0) { buf[n] = 0; fprintf(out, "exe %s\n", buf); }
    }
    if (gethostname(buf, sizeof buf) == 0) { buf[sizeof buf - 1] = 0; fprintf(out, "host %s\n", buf); }
    fprintf(out, "t_start_ns %lld\nt_end_ns %lld\n", bp_t0, bp_now());
    fprintf(out, "roi_log %d\n", has_roi);
    fprintf(out, "cpu_start %d\ncpu_end %d\n", bp_cpu0, sched_getcpu());
    fprintf(out, "cpus_allowed_start %s\nmems_allowed_start %s\n", bp_cpus0, bp_mems0);
    bp_status_field("/proc/self/status", "Cpus_allowed_list", val, sizeof val);
    fprintf(out, "cpus_allowed_end %s\n", val);
    bp_status_field("/proc/self/status", "Mems_allowed_list", val, sizeof val);
    fprintf(out, "mems_allowed_end %s\n", val);
    bp_status_field("/proc/self/status", "Threads", val, sizeof val);
    fprintf(out, "threads %s\n", val);
    bp_status_field("/proc/self/status", "voluntary_ctxt_switches", val, sizeof val);
    fprintf(out, "voluntary_ctxt_switches %s\n", val);
    bp_status_field("/proc/self/status", "nonvoluntary_ctxt_switches", val, sizeof val);
    fprintf(out, "nonvoluntary_ctxt_switches %s\n", val);

    /* every thread still alive at exit: its own CPU set and the CPU it last ran on */
    d = opendir("/proc/self/task");
    if (d) {
        while ((e = readdir(d))) {
            char comm[64], cpus[BP_SMALL];
            if (e->d_name[0] == '.') continue;
            snprintf(path, sizeof path, "/proc/self/task/%s/status", e->d_name);
            bp_status_field(path, "Cpus_allowed_list", cpus, sizeof cpus);
            bp_status_field(path, "Name", comm, sizeof comm);
            snprintf(path, sizeof path, "/proc/self/task/%s/stat", e->d_name);
            fprintf(out, "task %s cpus %s last_cpu %d name %s\n", e->d_name, cpus, bp_last_cpu(path), comm);
        }
        closedir(d);
    }

    /* /dev/nvidia<N> -> PCI bus id, from the driver's own table */
    for (i = 0; i < nminor; i++) {
        const char* bus = "unknown";
        char busbuf[64] = "";
        DIR* g = opendir("/proc/driver/nvidia/gpus");
        if (g) {
            while ((e = readdir(g))) {
                char mv[BP_SMALL];
                if (e->d_name[0] == '.') continue;
                snprintf(path, sizeof path, "/proc/driver/nvidia/gpus/%s/information", e->d_name);
                bp_status_field(path, "Device Minor", mv, sizeof mv);
                if (mv[0] && strtol(mv, NULL, 10) == minors[i]) {
                    snprintf(busbuf, sizeof busbuf, "%s", e->d_name);
                    bus = busbuf;
                    break;
                }
            }
            closedir(g);
        }
        fprintf(out, "gpu minor %d bus %s\n", minors[i], bus);
    }
    for (i = 0; envs[i]; i++) {
        const char* v = getenv(envs[i]);
        if (v) fprintf(out, "env %s=%s\n", envs[i], v);
    }
    fclose(out);
}
