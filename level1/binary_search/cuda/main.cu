#include "hpcperf_roi.h"
#include "hpcperf_golden.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <iostream>
#include <string>
#include <cuda.h>

#ifndef Real_t 
#define Real_t float
#endif

template <typename T>
__global__ void
kernel_BS (const T* __restrict__ acc_a,
           const T* __restrict__ acc_z,
            size_t* __restrict__ acc_r,
           const size_t zSize,
           const size_t n)
{ 
  size_t i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= zSize) return;
  T z = acc_z[i];
  size_t low = 0;
  size_t high = n;
  while (high - low > 1) {
    size_t mid = low + (high - low)/2;
    if (z < acc_a[mid])
      high = mid;
    else
      low = mid;
  }
  acc_r[i] = low;
}

template <typename T>
__global__ void
kernel_BS2 (const T* __restrict__ acc_a,
            const T* __restrict__ acc_z,
             size_t* __restrict__ acc_r,
            const size_t zSize,
            const size_t n)
{
  size_t i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= zSize) return;
  unsigned  nbits = 0;
  while (n >> nbits) nbits++;
  size_t k = 1ULL << (nbits - 1);
  T z = acc_z[i];
  size_t idx = (acc_a[k] <= z) ? k : 0;
  while (k >>= 1) {
    size_t r = idx | k;
    if (r < n && z >= acc_a[r]) { 
      idx = r;
    }
  }
  acc_r[i] = idx;
}

template <typename T>
__global__ void
kernel_BS3 (const T* __restrict__ acc_a,
            const T* __restrict__ acc_z,
             size_t* __restrict__ acc_r,
           const size_t zSize,
            const size_t n)
{
  size_t i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= zSize) return;
  unsigned nbits = 0;
  while (n >> nbits) nbits++;
  size_t k = 1ULL << (nbits - 1);
  T z = acc_z[i];
  size_t idx = (acc_a[k] <= z) ? k : 0;
  while (k >>= 1) {
    size_t r = idx | k;
    size_t w = r < n ? r : n; 
    if (z >= acc_a[w]) { 
      idx = r;
    }
  }
  acc_r[i] = idx;
}

template <typename T>
__global__ void
kernel_BS4 (const T* __restrict__ acc_a,
            const T* __restrict__ acc_z,
             size_t* __restrict__ acc_r,
            const size_t zSize,
            const size_t n)
{
  __shared__  size_t k;

  size_t i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= zSize) return;
  size_t lid = threadIdx.x; 

  if (lid == 0) {
    unsigned nbits = 0;
    while (n >> nbits) nbits++;
    k = 1ULL << (nbits - 1);
  }
  __syncthreads();

  size_t p = k;
  T z = acc_z[i];
  size_t idx = (acc_a[p] <= z) ? p : 0;
  while (p >>= 1) {
    size_t r = idx | p;
    size_t w = r < n ? r : n;
    if (z >= acc_a[w]) { 
      idx = r;
    }
  }
  acc_r[i] = idx;
}

template <typename T>
void bs ( const size_t aSize,
    const size_t zSize,
    const T *d_a,  // N+1
    const T *d_z,  // T
    size_t *d_r,   // T
    const size_t n,
    const int repeat )
{
  dim3 grids ((zSize + 255) / 256);
  dim3 blocks (256);
  auto start = std::chrono::steady_clock::now();
  HPCPERF_ROI_BEGIN_SYNC();  // tools/timing ROI: this timed GPU region

  for (int i = 0; i < repeat; i++)
    kernel_BS<<<grids, blocks>>>(d_a, d_z, d_r, zSize, n);

  cudaDeviceSynchronize();
  HPCPERF_ROI_END_SYNC();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time (bs1) " << (time * 1e-9f) / repeat << " (s)\n";
}

template <typename T>
void bs2 ( const size_t aSize,
    const size_t zSize,
    const T *d_a,  // N+1
    const T *d_z,  // T
    size_t *d_r,   // T
    const size_t n,
    const int repeat )
{
  dim3 grids ((zSize + 255) / 256);
  dim3 blocks (256);
  auto start = std::chrono::steady_clock::now();
  HPCPERF_ROI_BEGIN_SYNC();  // tools/timing ROI: this timed GPU region

  for (int i = 0; i < repeat; i++)
    kernel_BS2<<<grids, blocks>>>(d_a, d_z, d_r, zSize, n);

  cudaDeviceSynchronize();
  HPCPERF_ROI_END_SYNC();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time (bs2) " << (time * 1e-9f) / repeat << " (s)\n";
}

template <typename T>
void bs3 ( const size_t aSize,
    const size_t zSize,
    const T *d_a,  // N+1
    const T *d_z,  // T
    size_t *d_r,   // T
    const size_t n,
    const int repeat )
{
  dim3 grids ((zSize + 255) / 256);
  dim3 blocks (256);
  auto start = std::chrono::steady_clock::now();
  HPCPERF_ROI_BEGIN_SYNC();  // tools/timing ROI: this timed GPU region

  for (int i = 0; i < repeat; i++)
    kernel_BS3<<<grids, blocks>>>(d_a, d_z, d_r, zSize, n);

  cudaDeviceSynchronize();
  HPCPERF_ROI_END_SYNC();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time (bs3) " << (time * 1e-9f) / repeat << " (s)\n";
}

template <typename T>
void bs4 ( const size_t aSize,
    const size_t zSize,
    const T *d_a,  // N+1
    const T *d_z,  // T
    size_t *d_r,   // T
    const size_t n,
    const int repeat )
{
  dim3 grids ((zSize + 255) / 256);
  dim3 blocks (256);
  auto start = std::chrono::steady_clock::now();
  HPCPERF_ROI_BEGIN_SYNC();  // tools/timing ROI: this timed GPU region

  for (int i = 0; i < repeat; i++)
    kernel_BS4<<<grids, blocks>>>(d_a, d_z, d_r, zSize, n);

  cudaDeviceSynchronize();
  HPCPERF_ROI_END_SYNC();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time (bs4) " << (time * 1e-9f) / repeat << " (s)\n";
}

// HPC-Performance-AI correctness switch. verify() is upstream's own check of every search result: the
// property a[r] <= z < a[r+1] (r+1 < aSize), which determines r uniquely for the non-decreasing a[] (also
// where float rounding makes a[] repeat values); a host-side pass over the GPU output (~0.1 s per variant
// at the default input), not a CPU recomputation -- a property check like lud's and graph_coloring's.
// Upstream compiled it (and the copy-back of the results) only with -DDEBUG (Makefile VERIFY=yes), so the
// default build checked nothing. It now runs BY DEFAULT, after each variant's timed region / ROI, next to
// the comparison of the captured results of bs1..bs4 (hpcperf_golden_output, HPCPERF_GOLDEN_DUMP) with
// the frozen B200 golden reference (tools/golden), and a failure ends the run with exit code 1.
// HPCPERF_SKIP_VERIFY=1 (tools/timing/measure_level1.sh sets it) turns it off for timing; a -DDEBUG build
// runs it regardless, as upstream. No kernel, tolerance or algorithm is touched; the data initialization only by
// the query-range correction in main() (the queries no r brackets, README "Input n134217728").
static bool hpcperf_skip_verify() {
#ifdef DEBUG
  return false;
#else
  const char* e = getenv("HPCPERF_SKIP_VERIFY");
  return e != nullptr && *e != '\0' && *e != '0';
#endif
}

// returns whether every result satisfied the property (upstream: void, `break` at the first failure)
bool verify(Real_t *a, Real_t *z, size_t *r, size_t aSize, size_t zSize, std::string msg)
{
  for (size_t i = 0; i < zSize; ++i)
  {
    // check result
    // [HPC-Performance-AI] r[i] < aSize - 1 instead of upstream's r[i]+1 < aSize: the same for every r except
    // SIZE_MAX -- the poison a variant leaves where it wrote no result (hpcperf_poison_r) -- for which r+1 wraps to
    // 0 and upstream then read a[SIZE_MAX], i.e. before the allocation (ASan: heap-buffer-overflow)
    if (!(r[i] < aSize - 1 && a[r[i]] <= z[i] && z[i] < a[r[i] + 1]))
    {
      std::cout << msg << ": incorrect result:" << std::endl;
      std::cout << "index = " << i << " r[index] = " << r[i] << std::endl;
      // [HPC-Performance-AI] upstream printed a[r[i]], a[r[i]+1] also out of range, and with cout's 6 significant
      // digits, which above 10^6 shows a violated bracket as satisfied (1.34218e+08 <= 1.34218e+08 < 1.34218e+08)
      char vals[192];
      if (r[i] < aSize - 1)
        snprintf(vals, sizeof vals, "%.1f <= %.1f < %.1f", (double)a[r[i]], (double)z[i], (double)a[r[i] + 1]);
      else
        snprintf(vals, sizeof vals, "r[index] is not below aSize - 1 = %zu (z[index] = %.1f)", aSize - 1, (double)z[i]);
      std::cout << vals << std::endl;
      // [HPC-Performance-AI] diagnostic only: such a query has no valid r at all (the query generator in main()
      // excludes it since 2026-10-28; upstream's generator produced 6 of them at 2^27 elements)
      if (!(z[i] < a[aSize - 1])) {
        char why[256];
        snprintf(why, sizeof why, "z[index] = %.1f is not below a[aSize-1] = %.1f: no r with r+1 < aSize satisfies "
                 "a[r] <= z < a[r+1] (README \"Input n134217728\")", (double)z[i], (double)a[aSize - 1]);
        std::cout << why << std::endl;
      }
      return false;
    }
    // clear result
    r[i] = 0xFFFFFFFF;
  }
  return true;
}

// tools/golden: poison the result buffer that bs1..bs4 share -- d_r and the host copy r that is captured (all-ones
// bytes, SIZE_MAX in every size_t: never a valid index) -- before each variant's timed region, outside the ROI and
// the chrono timer in bs*(). Every kernel writes r[i] for all zSize queries, so a correct variant overwrites the
// poison; a variant that does not run leaves SIZE_MAX in its capture instead of the previous variant's identical
// result.
static void hpcperf_poison_r(size_t *d_r, size_t *r, size_t zSize)
{
  cudaMemset(d_r, 0xFF, sizeof(size_t)*zSize);
  memset(r, 0xFF, sizeof(size_t)*zSize);
  cudaDeviceSynchronize();
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::cout << "Usage ./main <number of elements> <repeat>\n";
    return 1;
  }

  size_t numElem = atol(argv[1]);
  uint repeat = atoi(argv[2]);
  // [HPC-Performance-AI] at least 2 elements: N = numElem-1 is the modulus of the query generator (rand() % 0, a
  // SIGFPE, for 1) and the kernels' n; 0 wrapped N to SIZE_MAX with zSize = 0 queries and a vacuous verify() PASS
  if (numElem < 2) {
    std::cout << "number of elements must be at least 2\n";
    return 1;
  }

  srand(2);
  size_t aSize = numElem;
  size_t zSize = 2*aSize;
  Real_t *a = NULL;
  Real_t *z = NULL;
  size_t *r = NULL;
  posix_memalign((void**)&a, 1024, aSize * sizeof(Real_t));
  posix_memalign((void**)&z, 1024, zSize * sizeof(Real_t));
  posix_memalign((void**)&r, 1024, zSize * sizeof(size_t));

  size_t N = aSize-1;

  // strictly ascending
  for (size_t i = 0; i < aSize; i++) a[i] = i;

  // lower = 0, upper = n-1
  for (size_t i = 0; i < zSize; i++) z[i] = rand() % N;

  // [HPC-Performance-AI] query-range correction (README "Input n134217728"). Upstream's intent above -- every query
  // integer at most n-1, i.e. strictly below a[N] = a[aSize-1], so that verify()'s a[r] <= z < a[r+1], r+1 < aSize
  // has exactly one solution -- holds only in exact arithmetic. With Real_t = float, a[i] = i and the query integer
  // are rounded to float above 2^24, and when aSize-2 and aSize-1 round to the same float a query integer can round
  // to a[N] itself, which no r brackets: at 2^27 elements the integers 2^27-4 .. 2^27-2 round to 2^27 = a[N], 6 of
  // the 268435456 queries for this rand() stream. Such a query (and only such a query) is moved to the largest
  // Real_t below a[N] -- the smallest change that gives it its one valid answer (2^27-8 at 2^27 elements, the value
  // these integers round down to; r = N-4). The rand() stream, a[] and every other query are unchanged: at 2^23
  // and 2^25 elements no query reaches a[N] and the input is byte-identical to upstream's; with -DReal_t=double a
  // no-op. A pure insertion after upstream's generator; outside every timed region / ROI.
  for (size_t i = 0; i < zSize; i++)
    if (!(z[i] < a[N])) z[i] = std::nextafter(a[N], (Real_t)0);

  Real_t* d_a;
  Real_t* d_z;
  size_t *d_r;
  cudaMalloc((void**)&d_a, sizeof(Real_t)*aSize);
  cudaMalloc((void**)&d_z, sizeof(Real_t)*zSize);
  cudaMalloc((void**)&d_r, sizeof(size_t)*zSize);
  cudaMemcpy(d_a, a, sizeof(Real_t)*aSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, z, sizeof(Real_t)*zSize, cudaMemcpyHostToDevice);

  static_assert(sizeof(size_t) == 8, "the results are captured as u64");
  const bool verify_results = !hpcperf_skip_verify();
  bool verify_ok = true;

  hpcperf_poison_r(d_r, r, zSize);
  bs(aSize, zSize, d_a, d_z, d_r, N, repeat);

  // the result back to the host after the timed region / ROI (a D2H copy only; upstream did it only
  // under -DDEBUG), captured for tools/golden before verify() clears it
  cudaMemcpy(r, d_r, sizeof(size_t)*zSize, cudaMemcpyDeviceToHost);
  hpcperf_golden_output("r.bs1", r, zSize, "u64");
  if (verify_results) verify_ok &= verify(a, z, r, aSize, zSize, "bs");

  hpcperf_poison_r(d_r, r, zSize);
  bs2(aSize, zSize, d_a, d_z, d_r, N, repeat);

  cudaMemcpy(r, d_r, sizeof(size_t)*zSize, cudaMemcpyDeviceToHost);
  hpcperf_golden_output("r.bs2", r, zSize, "u64");
  if (verify_results) verify_ok &= verify(a, z, r, aSize, zSize, "bs2");

  hpcperf_poison_r(d_r, r, zSize);
  bs3(aSize, zSize, d_a, d_z, d_r, N, repeat);

  cudaMemcpy(r, d_r, sizeof(size_t)*zSize, cudaMemcpyDeviceToHost);
  hpcperf_golden_output("r.bs3", r, zSize, "u64");
  if (verify_results) verify_ok &= verify(a, z, r, aSize, zSize, "bs3");

  hpcperf_poison_r(d_r, r, zSize);
  bs4(aSize, zSize, d_a, d_z, d_r, N, repeat);

  cudaMemcpy(r, d_r, sizeof(size_t)*zSize, cudaMemcpyDeviceToHost);
  hpcperf_golden_output("r.bs4", r, zSize, "u64");
  if (verify_results) verify_ok &= verify(a, z, r, aSize, zSize, "bs4");

  if (!verify_results)
    // HPCPERF_SKIP_VERIFY=1 (timing): no host-side check; correctness = the captured results against the frozen
    // golden reference (tools/golden), a separate step
    std::cout << "SKIP_VERIFY" << std::endl;
  else if (verify_ok)
    std::cout << "PASS: verify: a[r] <= z < a[r+1] for all " << zSize << " queries of bs1, bs2, bs3, bs4" << std::endl;
  else
    std::cout << "FAIL: verify: incorrect result (see above)" << std::endl;

  cudaFree(d_a);
  cudaFree(d_z);
  cudaFree(d_r);
  free(a);
  free(z);
  free(r);
  // [HPC-Performance-AI] a verify() failure fails the run (upstream: exit 0 either way)
  return (verify_results && !verify_ok) ? 1 : 0;
}
