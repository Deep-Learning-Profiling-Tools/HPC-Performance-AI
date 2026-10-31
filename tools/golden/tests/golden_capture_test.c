/* Test driver for hpcperf_golden.h (tools/golden/tests/run_all.sh): reads stdin, captures it as
 * output "blob" (u8) through the streaming API in pieces of argv[1] bytes, and also as one call. */
#include "hpcperf_golden.h"
int main(int argc, char** argv) {
  size_t piece = argc > 1 ? (size_t)atol(argv[1]) : 7, n = 0, cap = 1 << 20, i;
  unsigned char* buf = (unsigned char*)malloc(cap);
  int c;
  hpcperf_golden_stream s;
  while ((c = getchar()) != EOF) { if (n == cap) buf = (unsigned char*)realloc(buf, cap *= 2); buf[n++] = (unsigned char)c; }
  hpcperf_golden_begin(&s, "blob.streamed", "u8");
  for (i = 0; i < n; i += piece) hpcperf_golden_append(&s, buf + i, n - i < piece ? n - i : piece);
  hpcperf_golden_end(&s);
  hpcperf_golden_output("blob.whole", buf, n, "u8");
  hpcperf_golden_output("blob.f32", buf, n / 4, "f32");
  printf("cpu_reference_enabled=%d\n", hpcperf_cpu_reference_enabled());
  free(buf);
  return 0;
}
