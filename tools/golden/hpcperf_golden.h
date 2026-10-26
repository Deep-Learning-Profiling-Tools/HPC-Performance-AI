/*
 * hpcperf_golden.h -- the CPU-reference gate and golden-output capture for the benchmarks.
 *
 * Correctness of a benchmark run is judged against a FROZEN golden reference: the outputs
 * of the original, unmodified build on the reference GPU (NVIDIA B200), captured once by
 * `tools/golden/hpcperf_golden.py make` (which is also the only place the benchmark's
 * own CPU reference still runs, to validate that capture) and compared by
 * `hpcperf_golden.py check`. See tools/golden/README.md.
 *
 *   hpcperf_cpu_reference_enabled()
 *       1 only when HPCPERF_CPU_REFERENCE is set to a value not starting with '0' (an
 *       explicit, offline validation request) and HPCPERF_SKIP_VERIFY is not set. The
 *       DEFAULT is 0: a plain run, ctest, tools/inputs and tools/timing never execute the
 *       host-side reference computation, and nothing falls back to it.
 *
 *   hpcperf_golden_output(name, data, count, dtype)
 *       one host buffer of `count` elements of `dtype` ("f32" "f64" "i8" "u8" "i16" "u16"
 *       "i32" "u32" "i64" "u64"). A no-op unless HPCPERF_GOLDEN_DUMP names a directory;
 *       then the bytes go to <dir>/<name>.bin and one JSON line (name, dtype, count, bytes,
 *       sha256, data) is appended to <dir>/manifest.jsonl. HPCPERF_GOLDEN_MODE=hash records
 *       the sha256 only (outputs compared exactly that are too large to write out).
 *   hpcperf_golden_begin / _append / _end
 *       the same for an output assembled from several pieces (rows, frames, variants).
 *
 * Call sites capture outputs AFTER the timed region (never inside the ROI) and after the
 * final device-to-host copy. Names are [A-Za-z0-9._-], unique within one run. An I/O
 * failure while capturing ends the process with exit code 97, so a check never reads a
 * partial capture. Header-only C99/C++, no dependencies beyond libc and POSIX mkdir.
 */
#ifndef HPCPERF_GOLDEN_H
#define HPCPERF_GOLDEN_H

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define HPCPERF_GOLDEN_FN static __attribute__((unused))
#define HPCPERF_GOLDEN_DATA static __attribute__((unused))
#else
#define HPCPERF_GOLDEN_FN static
#define HPCPERF_GOLDEN_DATA static
#endif

/* ------------------------------------------------------------------ CPU reference gate */
HPCPERF_GOLDEN_FN int hpcperf_golden_flag_(const char* name) {
  const char* e = getenv(name);
  return e != NULL && *e != '\0' && *e != '0';
}

HPCPERF_GOLDEN_FN int hpcperf_cpu_reference_enabled(void) {
  if (hpcperf_golden_flag_("HPCPERF_SKIP_VERIFY")) return 0;
  return hpcperf_golden_flag_("HPCPERF_CPU_REFERENCE");
}

/* ------------------------------------------------------------------ SHA-256 (FIPS 180-4) */
typedef struct {
  uint32_t h[8];
  uint64_t bits;
  unsigned char buf[64];
  size_t used;
} hpcperf_sha256_ctx;

HPCPERF_GOLDEN_DATA const uint32_t hpcperf_sha256_k_[64] = {
  0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
  0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
  0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
  0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
  0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
  0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
  0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
  0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

#define HPCPERF_SHA_ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

HPCPERF_GOLDEN_FN void hpcperf_sha256_block_(hpcperf_sha256_ctx* c, const unsigned char* p) {
  uint32_t w[64], a, b, d, e, f, g, h, cc, t1, t2;
  int i;
  for (i = 0; i < 16; i++)
    w[i] = ((uint32_t)p[4 * i] << 24) | ((uint32_t)p[4 * i + 1] << 16) | ((uint32_t)p[4 * i + 2] << 8) | (uint32_t)p[4 * i + 3];
  for (i = 16; i < 64; i++) {
    uint32_t s0 = HPCPERF_SHA_ROR(w[i - 15], 7) ^ HPCPERF_SHA_ROR(w[i - 15], 18) ^ (w[i - 15] >> 3);
    uint32_t s1 = HPCPERF_SHA_ROR(w[i - 2], 17) ^ HPCPERF_SHA_ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  a = c->h[0]; b = c->h[1]; cc = c->h[2]; d = c->h[3]; e = c->h[4]; f = c->h[5]; g = c->h[6]; h = c->h[7];
  for (i = 0; i < 64; i++) {
    t1 = h + (HPCPERF_SHA_ROR(e, 6) ^ HPCPERF_SHA_ROR(e, 11) ^ HPCPERF_SHA_ROR(e, 25)) + ((e & f) ^ (~e & g)) + hpcperf_sha256_k_[i] + w[i];
    t2 = (HPCPERF_SHA_ROR(a, 2) ^ HPCPERF_SHA_ROR(a, 13) ^ HPCPERF_SHA_ROR(a, 22)) + ((a & b) ^ (a & cc) ^ (b & cc));
    h = g; g = f; f = e; e = d + t1; d = cc; cc = b; b = a; a = t1 + t2;
  }
  c->h[0] += a; c->h[1] += b; c->h[2] += cc; c->h[3] += d; c->h[4] += e; c->h[5] += f; c->h[6] += g; c->h[7] += h;
}

HPCPERF_GOLDEN_FN void hpcperf_sha256_init(hpcperf_sha256_ctx* c) {
  static const uint32_t iv[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  memcpy(c->h, iv, sizeof(iv));
  c->bits = 0; c->used = 0;
}

HPCPERF_GOLDEN_FN void hpcperf_sha256_update(hpcperf_sha256_ctx* c, const void* data, size_t len) {
  const unsigned char* p = (const unsigned char*)data;
  c->bits += (uint64_t)len * 8u;
  if (c->used) {
    size_t take = 64 - c->used < len ? 64 - c->used : len;
    memcpy(c->buf + c->used, p, take);
    c->used += take; p += take; len -= take;
    if (c->used < 64) return;
    hpcperf_sha256_block_(c, c->buf); c->used = 0;
  }
  while (len >= 64) { hpcperf_sha256_block_(c, p); p += 64; len -= 64; }
  if (len) { memcpy(c->buf, p, len); c->used = len; }
}

HPCPERF_GOLDEN_FN void hpcperf_sha256_hex(hpcperf_sha256_ctx* c, char out[65]) {
  static const char hx[] = "0123456789abcdef";
  uint64_t bits = c->bits;
  unsigned char pad = 0x80, zero = 0, len[8];
  int i;
  hpcperf_sha256_update(c, &pad, 1);
  while (c->used != 56) hpcperf_sha256_update(c, &zero, 1);
  for (i = 0; i < 8; i++) len[i] = (unsigned char)(bits >> (56 - 8 * i));
  hpcperf_sha256_update(c, len, 8);
  for (i = 0; i < 32; i++) {
    unsigned char byte = (unsigned char)(c->h[i / 4] >> (24 - 8 * (i % 4)));
    out[2 * i] = hx[byte >> 4]; out[2 * i + 1] = hx[byte & 15];
  }
  out[64] = '\0';
}

/* ------------------------------------------------------------------ golden output capture */
typedef struct {
  int active, data;
  FILE* f;
  hpcperf_sha256_ctx sha;
  size_t esize;
  unsigned long long count;
  char dir[1024], name[128], dtype[8];
} hpcperf_golden_stream;

HPCPERF_GOLDEN_FN void hpcperf_golden_die_(const char* what, const char* name) {
  fprintf(stderr, "HPCPERF_GOLDEN ERROR: %s (%s): %s\n", what, name, errno ? strerror(errno) : "-");
  fflush(stderr);
  exit(97);
}

HPCPERF_GOLDEN_FN size_t hpcperf_golden_dtype_size_(const char* t) {
  if (!strcmp(t, "f32") || !strcmp(t, "i32") || !strcmp(t, "u32")) return 4;
  if (!strcmp(t, "f64") || !strcmp(t, "i64") || !strcmp(t, "u64")) return 8;
  if (!strcmp(t, "i16") || !strcmp(t, "u16")) return 2;
  if (!strcmp(t, "i8") || !strcmp(t, "u8")) return 1;
  return 0;
}

HPCPERF_GOLDEN_FN int hpcperf_golden_active(void) {
  const char* d = getenv("HPCPERF_GOLDEN_DUMP");
  return d != NULL && *d != '\0';
}

HPCPERF_GOLDEN_FN void hpcperf_golden_begin(hpcperf_golden_stream* s, const char* name, const char* dtype) {
  const char* dir = getenv("HPCPERF_GOLDEN_DUMP");
  const char* mode = getenv("HPCPERF_GOLDEN_MODE");
  const char* q;
  memset(s, 0, sizeof(*s));
  if (dir == NULL || *dir == '\0') return;
  errno = 0;
  if (strlen(name) == 0 || strlen(name) >= sizeof(s->name) || strlen(dir) >= sizeof(s->dir) - sizeof(s->name) - 8)
    hpcperf_golden_die_("output name or dump directory too long", name);
  for (q = name; *q; q++)
    if (!((*q >= 'a' && *q <= 'z') || (*q >= 'A' && *q <= 'Z') || (*q >= '0' && *q <= '9') || *q == '.' || *q == '_' || *q == '-'))
      hpcperf_golden_die_("output name must be [A-Za-z0-9._-]", name);
  s->esize = hpcperf_golden_dtype_size_(dtype);
  if (s->esize == 0) hpcperf_golden_die_("unknown dtype", dtype);
  strcpy(s->dir, dir); strcpy(s->name, name); strcpy(s->dtype, dtype);
  if (mkdir(dir, 0777) != 0 && errno != EEXIST) hpcperf_golden_die_("cannot create the dump directory", dir);
  errno = 0;
  s->data = !(mode != NULL && strcmp(mode, "hash") == 0);
  if (s->data) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    s->f = fopen(path, "wb");
    if (s->f == NULL) hpcperf_golden_die_("cannot open the output file", path);
  }
  hpcperf_sha256_init(&s->sha);
  s->active = 1;
}

HPCPERF_GOLDEN_FN void hpcperf_golden_append(hpcperf_golden_stream* s, const void* data, size_t count) {
  size_t bytes;
  if (!s->active || count == 0) return;
  bytes = count * s->esize;
  hpcperf_sha256_update(&s->sha, data, bytes);
  if (s->data && fwrite(data, 1, bytes, s->f) != bytes) hpcperf_golden_die_("short write", s->name);
  s->count += count;
}

HPCPERF_GOLDEN_FN void hpcperf_golden_end(hpcperf_golden_stream* s) {
  char hex[65], path[1200];
  FILE* m;
  if (!s->active) return;
  hpcperf_sha256_hex(&s->sha, hex);
  errno = 0;
  if (s->data && fclose(s->f) != 0) hpcperf_golden_die_("close failed", s->name);
  snprintf(path, sizeof(path), "%s/manifest.jsonl", s->dir);
  m = fopen(path, "a");
  if (m == NULL) hpcperf_golden_die_("cannot open the manifest", path);
  fprintf(m, "{\"name\": \"%s\", \"dtype\": \"%s\", \"count\": %llu, \"bytes\": %llu, \"sha256\": \"%s\", \"data\": %s}\n",
          s->name, s->dtype, s->count, s->count * (unsigned long long)s->esize, hex, s->data ? "true" : "false");
  if (fclose(m) != 0) hpcperf_golden_die_("manifest close failed", path);
  s->active = 0;
}

HPCPERF_GOLDEN_FN void hpcperf_golden_output(const char* name, const void* data, size_t count, const char* dtype) {
  hpcperf_golden_stream s;
  hpcperf_golden_begin(&s, name, dtype);
  hpcperf_golden_append(&s, data, count);
  hpcperf_golden_end(&s);
}

#ifdef __cplusplus
}
#endif

#endif /* HPCPERF_GOLDEN_H */
