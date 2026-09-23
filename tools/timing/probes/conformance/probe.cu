// probe.cu -- conformance probe for a tools/timing backend (marker + collector + adapter).
//
// A new platform is "supported" only when measuring this program reproduces
// expected.json exactly. Phases, in order:
//   set-up   cudaMalloc, H2D copy, memset                          -> before the ROI
//   warm-up  10 launches of work()                                 -> before the ROI
//   ROI      20 launches of work(), 1 device-to-device copy,
//            EXCLUDE { 5 launches of check() }                     -> 20 compute + 1 d2d
//   after    D2H copy, CPU checksum, 1 launch of work()            -> after the ROI
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "gpu_compat.h"
#include "hpcperf_roi.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 2; } } while (0)

__global__ void work(float* x, int n, int reps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; for (int r = 0; r < reps; ++r) v = v * 0.99999f + 1.0e-3f; x[i] = v; }
}
__global__ void check(const float* x, int n, unsigned* bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && !(x[i] == x[i])) atomicAdd(bad, 1u);
}

int main() {
    const int n = 1 << 25, reps = 256, threads = 256, blocks = (n + threads - 1) / threads;
    std::vector<float> host(n, 1.0f);
    float *a = nullptr, *b = nullptr;
    unsigned* bad = nullptr;
    CK(cudaMalloc(&a, n * sizeof(float)));
    CK(cudaMalloc(&b, n * sizeof(float)));
    CK(cudaMalloc(&bad, sizeof(unsigned)));
    CK(cudaMemcpy(a, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(bad, 0, sizeof(unsigned)));
    for (int i = 0; i < 10; ++i) work<<<blocks, threads>>>(a, n, reps);           // warm-up
    CK(cudaDeviceSynchronize());

    HPCPERF_ROI_BEGIN();
    for (int i = 0; i < 20; ++i) work<<<blocks, threads>>>(a, n, reps);           // the computation
    CK(cudaMemcpy(b, a, n * sizeof(float), cudaMemcpyDeviceToDevice));
    CK(cudaDeviceSynchronize());
    HPCPERF_ROI_EXCLUDE_BEGIN();
    for (int i = 0; i < 5; ++i) check<<<blocks, threads>>>(b, n, bad);            // a check inside
    CK(cudaDeviceSynchronize());
    HPCPERF_ROI_EXCLUDE_END();
    HPCPERF_ROI_END();

    CK(cudaMemcpy(host.data(), b, n * sizeof(float), cudaMemcpyDeviceToHost));    // after
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += host[i];
    work<<<blocks, threads>>>(a, n, reps);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    std::printf("probe checksum %.6e\n", sum);
    CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(bad));
    return 0;
}
