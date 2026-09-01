//
// SpGEMM (C = A x A) benchmark, standalone port of the Kokkos Kernels KKMEM
// algorithm (upstream --algorithm KKMEM, the hash-based native GPU path),
// BSD-3-Clause; commit 23a699f3c5bd662bf2ed52116e56533ecc3ddae0.
//
// What is ported verbatim (structure and semantics):
//  - Numeric phase: PortableNumericCHASH GPUTag operator from
//    sparse/impl/KokkosSparse_spgemm_impl_kkmem.hpp -- one row per team
//    thread (warp), two-level hash accumulator: per-thread shared-memory
//    hashmap (linked-list chaining, bitwiseAnd hash) with overflow into a
//    global-memory hashmap whose key/value storage is the C row itself.
//  - HashmapAccumulator::vector_atomic_insert_into_hash_mergeAdd and
//    ..._TrackHashes from common/src/KokkosKernels_HashmapAccumulator.hpp.
//  - UniformMemoryPool (ManyThread2OneChunk): atomic-CAS chunk locks with
//    linear probing, from KokkosKernels_Uniform_Initialized_MemoryPool.hpp.
//  - All launch/shmem heuristics with upstream defaults: shmem budget 16384 B,
//    unit_memory = 20 B, thread_memory = shmem/team_size, shared hash size =
//    largest pow2 <= key budget, key-size resize formula, suggested vector
//    size from avg B-row nnz (=32 here), team size 256/vector (=8),
//    team_work_size = team size, min_hash_size = pow4 >= max C row nnz,
//    chunk = 2*min_hash_size + max_nnz ints, ideal chunks = concurrency /
//    vector_size capped by free memory (compute_num_pool_chunks).
//
// Documented deviations (see README):
//  - Upstream default '--algorithm KK' may select other variants (dense
//    accumulators, cuckoo hashes) via runtime heuristics; this port pins the
//    KKMEM variant, which is an explicit upstream option.
//  - The symbolic phase (C row sizes) is computed exactly on the host instead
//    of porting the ~3.4k-line compression-based GPU symbolic; only the
//    numeric phase is timed and reported.
//  - Input: the upstream perf_test requires a matrix file; the input here is
//    generated with the upstream generator kk_sparseMatrix_generate
//    (glibc srand(13721)), 10000x10000, 30 nnz/row, values uniform(-50,50)
//    from a fixed-seed xorshift64* (Kokkos pool not reproduced bit-exactly).
//
// Validation (added): C recomputed on the host with a dense accumulator;
// per-row entries sorted and compared (cols exact, values rel tol 1e-10).
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>

using Scalar  = double;
using Ordinal = int;      // nnz_lno_t
using Offset  = size_t;   // size_type

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemset hipMemset
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetLastError hipGetLastError
#define cudaMemGetInfo hipMemGetInfo
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define SYNCWARP()
#define SHFL_PTR(p) (p)  /* HIP: lockstep wavefront, lane 0 value broadcast via shared not needed */
#else
#include <cuda_runtime.h>
#define SYNCWARP() __syncwarp()
#endif

#define GPU_CHECK(call)                                                  \
  do {                                                                   \
    cudaError_t err_ = (call);                                           \
    if (err_ != cudaSuccess) {                                           \
      fprintf(stderr, "GPU error %s at %s:%d\n",                         \
              cudaGetErrorString(err_), __FILE__, __LINE__);             \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

// ---------------------------------------------------------------------------
// Input generation (see header comment)
// ---------------------------------------------------------------------------
static void kk_sparseMatrix_generate(Ordinal nrows, Ordinal ncols, Offset& nnz,
                                     Ordinal bandwidth, std::vector<Scalar>& values,
                                     std::vector<Offset>& rowPtr, std::vector<Ordinal>& colInd)
{
  rowPtr.assign(nrows + 1, 0);
  Ordinal elements_per_row = nrows ? static_cast<Ordinal>(nnz / nrows) : 0;
  srand(13721);
  for (int row = 0; row < nrows; row++) {
    (void)rand();  // row_size_variance = 0 still consumes one rand() per row
    rowPtr[row + 1] = rowPtr[row] + elements_per_row;
  }
  nnz = rowPtr[nrows];
  values.resize(nnz);
  colInd.resize(nnz);
  for (Ordinal row = 0; row < nrows; row++) {
    for (Offset k = rowPtr[row]; k < rowPtr[row + 1]; ++k) {
      while (true) {
        Ordinal pos = (1.0 * rand() / RAND_MAX - 0.5) * bandwidth + row;
        while (pos < 0) pos += ncols;
        while (pos >= ncols) pos -= ncols;
        bool dup = false;
        for (Offset j = rowPtr[row]; j < k; j++)
          if (colInd[j] == pos) { dup = true; break; }
        if (!dup) { colInd[k] = pos; break; }
      }
    }
  }
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (Offset k = 0; k < nnz; k++) {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    uint64_t r = s * 0x2545F4914F6CDD1Dull;
    values[k] = -50.0 + 100.0 * ((r >> 11) * (1.0 / 9007199254740992.0));
  }
}

// ---------------------------------------------------------------------------
// Numeric kernel: port of PortableNumericCHASH::operator()(GPUTag)
// blockDim.x = vector_size (warp lanes), blockDim.y = team_size.
// Per-thread (per-warp here, since team threads map to warps) shared layout:
//   [used_hash_sizes(2) | globally_used_hash_count(2) | begins(shmem_hash) |
//    nexts(shmem_key) | keys(shmem_key) | pad | vals(shmem_key) ]
// ---------------------------------------------------------------------------
extern __shared__ char spgemm_shared[];

__global__ void kkmem_numeric(
    Ordinal numrows,
    const Offset* __restrict__ row_mapA, const Ordinal* __restrict__ entriesA,
    const Scalar* __restrict__ valuesA,
    const Offset* __restrict__ row_mapB, const Ordinal* __restrict__ entriesB,
    const Scalar* __restrict__ valuesB,
    const Offset* __restrict__ rowmapC, Ordinal* __restrict__ pEntriesC,
    Scalar* __restrict__ pvaluesC,
    const Ordinal team_work_size, const int thread_memory,
    const Ordinal thread_shmem_hash_size, const Ordinal thread_shmem_key_size,
    const Ordinal pow2_hash_size,
    int* pool_data, int* pool_locks, const Ordinal pool_num_chunks_mod,
    const size_t pool_chunk_size)
{
  const Ordinal pow2_hash_func = pow2_hash_size - 1;
  const Ordinal thread_shared_memory_hash_func = thread_shmem_hash_size - 1;

  Ordinal team_row_begin = blockIdx.x * team_work_size;
  const Ordinal team_row_end = min(team_row_begin + team_work_size, numrows);

  char* all_shared_memory = spgemm_shared + (size_t)thread_memory * threadIdx.y;
  volatile Ordinal* used_hash_sizes = (volatile Ordinal*)(all_shared_memory);
  all_shared_memory += sizeof(Ordinal) * 2;
  Ordinal* globally_used_hash_count = (Ordinal*)(all_shared_memory);
  all_shared_memory += sizeof(Ordinal) * 2;
  Ordinal* begins = (Ordinal*)(all_shared_memory);
  all_shared_memory += sizeof(Ordinal) * thread_shmem_hash_size;
  Ordinal* nexts = (Ordinal*)(all_shared_memory);
  all_shared_memory += sizeof(Ordinal) * thread_shmem_key_size;
  Ordinal* keys = (Ordinal*)(all_shared_memory);
  all_shared_memory += sizeof(Ordinal) * thread_shmem_key_size;
  Scalar* vals = (Scalar*)((((uintptr_t)all_shared_memory) + alignof(Scalar) - 1) & ~(uintptr_t)(alignof(Scalar) - 1));

  // TeamThreadRange over the team's rows
  for (Ordinal row_index = team_row_begin + threadIdx.y; row_index < team_row_end; row_index += blockDim.y) {
    const Offset c_row_begin = rowmapC[row_index];
    const Offset c_row_end   = rowmapC[row_index + 1];
    const Ordinal global_memory_hash_size = Ordinal(c_row_end - c_row_begin);

    bool is_global_alloced = false;
    Ordinal* globally_used_hash_indices = NULL;
    Ordinal* hm2_begins = NULL;
    Ordinal* hm2_nexts  = NULL;

    if (global_memory_hash_size > thread_shmem_key_size) {
      // Kokkos::single(PerThread) + broadcast: lane 0 allocates a pool chunk
      unsigned long long ptrval = 0;
      while (ptrval == 0) {
        if (threadIdx.x == 0) {
          // UniformMemoryPool::get_arbitrary_free_chunk (ManyThread2OneChunk)
          size_t chunk_index = ((size_t)row_index) & (size_t)pool_num_chunks_mod;
          size_t num_try = 0;
          const size_t max_tries = (size_t)pool_num_chunks_mod + 1;
          int* got = NULL;
          while (true) {
            if (atomicCAS(pool_locks + chunk_index, 0, 1) == 0) {
              got = pool_data + chunk_index * pool_chunk_size;
              break;
            }
            chunk_index = (chunk_index + 1) & (size_t)pool_num_chunks_mod;
            if (++num_try > max_tries) { got = NULL; break; }
          }
          ptrval = (unsigned long long)(uintptr_t)got;
        }
#if !defined(__HIPCC__)
        ptrval = __shfl_sync(0xffffffffu, ptrval, 0);
#endif
      }
      Ordinal* tmp = (Ordinal*)(uintptr_t)ptrval;
      is_global_alloced = true;
      globally_used_hash_indices = tmp;
      tmp += pow2_hash_size;
      hm2_begins = tmp;
      tmp += pow2_hash_size;
      hm2_nexts = tmp;
    }
    Ordinal* hm2_keys = pEntriesC + c_row_begin;
    Scalar* hm2_values = pvaluesC + c_row_begin;

    // ThreadVectorRange: init shared hash begins
    for (Ordinal i = threadIdx.x; i < thread_shmem_hash_size; i += blockDim.x) begins[i] = -1;
    if (threadIdx.x == 0) {
      used_hash_sizes[0] = 0;
      used_hash_sizes[1] = 0;
      globally_used_hash_count[0] = 0;
    }
    SYNCWARP();

    const Offset col_begin = row_mapA[row_index];
    const Ordinal left_work = Ordinal(row_mapA[row_index + 1] - col_begin);
    Ordinal ii = left_work;
    while (ii-- > 0) {
      Offset a_col = col_begin + ii;
      Ordinal rowB = entriesA[a_col];
      Scalar valA  = valuesA[a_col];
      Offset rowBegin = row_mapB[rowB];
      Ordinal left_work_ = Ordinal(row_mapB[rowB + 1] - rowBegin);
      // ThreadVectorRange over B row: vector lanes insert concurrently
      for (Ordinal i = threadIdx.x; i < left_work_; i += blockDim.x) {
        const Offset adjind = i + rowBegin;
        Ordinal b_col_ind = entriesB[adjind];
        Scalar b_val = valuesB[adjind] * valA;
        // --- vector_atomic_insert_into_hash_mergeAdd (shared-level) ---
        int num_unsuccess = 0;
        {
          Ordinal hash = b_col_ind & thread_shared_memory_hash_func;
          bool done = false;
          for (Ordinal j = begins[hash]; j != -1; j = nexts[j]) {
            if (keys[j] == b_col_ind) { vals[j] += b_val; done = true; break; }
          }
          if (!done) {
            if (used_hash_sizes[0] >= thread_shmem_key_size) {
              num_unsuccess = 1;
            } else {
              Ordinal my_write_index = atomicAdd((Ordinal*)used_hash_sizes, 1);
              if (my_write_index >= thread_shmem_key_size) {
                num_unsuccess = 1;
              } else {
                keys[my_write_index] = b_col_ind;
                vals[my_write_index] = b_val;
                nexts[my_write_index] = begins[hash];  // CUDA independent-threads pre-link
                Ordinal hashbeginning = atomicExch(begins + hash, my_write_index);
                nexts[my_write_index] = hashbeginning;
              }
            }
          }
        }
        // --- overflow: ..._mergeAdd_TrackHashes (global-level) ---
        if (num_unsuccess) {
          Ordinal hash = b_col_ind & pow2_hash_func;
          bool done = false;
          for (Ordinal j = hm2_begins[hash]; j != -1; j = hm2_nexts[j]) {
            if (hm2_keys[j] == b_col_ind) { hm2_values[j] += b_val; done = true; break; }
          }
          if (!done) {
            Ordinal my_write_index = atomicAdd((Ordinal*)(used_hash_sizes + 1), 1);
            if (my_write_index < global_memory_hash_size) {
              hm2_keys[my_write_index] = b_col_ind;
              hm2_values[my_write_index] = b_val;
              hm2_nexts[my_write_index] = hm2_begins[hash];
              Ordinal hashbeginning = atomicExch(hm2_begins + hash, my_write_index);
              if (hashbeginning == -1) {
                globally_used_hash_indices[atomicAdd(globally_used_hash_count, 1)] = hash;
              }
              hm2_nexts[my_write_index] = hashbeginning;
            }
          }
        }
      }
      SYNCWARP();
    }

    if (is_global_alloced) {
      Ordinal dirty_hashes = globally_used_hash_count[0];
      for (Ordinal i = threadIdx.x; i < dirty_hashes; i += blockDim.x) {
        hm2_begins[globally_used_hash_indices[i]] = -1;
      }
      SYNCWARP();
      if (threadIdx.x == 0) {
        // release_arbitrary_chunk
        size_t alloc_index = (globally_used_hash_indices - pool_data) / pool_chunk_size;
        pool_locks[alloc_index] = 0;
      }
    }
    if (threadIdx.x == 0) {
      if (used_hash_sizes[0] > thread_shmem_key_size) used_hash_sizes[0] = thread_shmem_key_size;
    }
    SYNCWARP();
    Ordinal num_elements  = used_hash_sizes[0];
    Ordinal written_index = used_hash_sizes[1];
    for (Ordinal i = threadIdx.x; i < num_elements; i += blockDim.x) {
      pEntriesC[c_row_begin + written_index + i] = keys[i];
      pvaluesC[c_row_begin + written_index + i]  = vals[i];
    }
    SYNCWARP();
  }
}

int main(int argc, char** argv)
{
  Ordinal m = 10000, n = 10000;
  int nnzPerRow = 30;
  int repeat = 1;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--m") && i + 1 < argc) { m = atoi(argv[++i]); n = m; }
    if (!strcmp(argv[i], "--nnz") && i + 1 < argc) nnzPerRow = atoi(argv[++i]);
    if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
  }

  std::vector<Scalar> Aval; std::vector<Offset> Arow; std::vector<Ordinal> Acol;
  Offset nnzA = (Offset)m * nnzPerRow;
  kk_sparseMatrix_generate(m, n, nnzA, (n + 3) / 3, Aval, Arow, Acol);
  // sort rows for determinism (matrix-market inputs are typically sorted)
  for (Ordinal i = 0; i < m; i++) {
    std::vector<std::pair<Ordinal, Scalar>> tmp;
    for (Offset k = Arow[i]; k < Arow[i + 1]; k++) tmp.push_back({Acol[k], Aval[k]});
    std::sort(tmp.begin(), tmp.end());
    Offset k = Arow[i];
    for (auto& p : tmp) { Acol[k] = p.first; Aval[k] = p.second; k++; }
  }
  printf("A: %dx%d, %zu entries; computing C = A * A\n", m, n, (size_t)nnzA);

  // ---- symbolic (host, exact; substitution documented in the header) ----
  std::vector<Offset> Crow(m + 1, 0);
  {
    std::vector<char> marker(n, 0);
    std::vector<Ordinal> touched;
    for (Ordinal i = 0; i < m; i++) {
      touched.clear();
      for (Offset ka = Arow[i]; ka < Arow[i + 1]; ka++) {
        Ordinal j = Acol[ka];
        for (Offset kb = Arow[j]; kb < Arow[j + 1]; kb++) {
          Ordinal c = Acol[kb];
          if (!marker[c]) { marker[c] = 1; touched.push_back(c); }
        }
      }
      Crow[i + 1] = Crow[i] + touched.size();
      for (Ordinal c : touched) marker[c] = 0;
    }
  }
  const Offset nnzC = Crow[m];
  Ordinal max_nnz = 0;
  for (Ordinal i = 0; i < m; i++) max_nnz = std::max<Ordinal>(max_nnz, Crow[i + 1] - Crow[i]);
  printf("C: %zu entries, max row %d\n", (size_t)nnzC, max_nnz);

  // ---- upstream launch configuration (defaults; see header) ----
  const size_t shmem_budget = 16384;                       // handle default
  int avg_b_nnz = nnzPerRow;                               // B = A
  int vector_size;                                         // kk_get_suggested_vector_size (CUDA)
  { int v = avg_b_nnz; if (v < 3) vector_size = 2; else if (v <= 6) vector_size = 4;
    else if (v <= 12) vector_size = 8; else if (v <= 24) vector_size = 16;
    else if (v <= 48) vector_size = 32; else vector_size = 32; }
  const int team_size = 256 / vector_size;
  const Ordinal team_work_size = team_size;                // GPU: get_team_work_size = team_size
  const size_t shared_memory_size = shmem_budget / 8 * 8;
  const int unit_memory = sizeof(Ordinal) * 2 + sizeof(Ordinal) + sizeof(Scalar);  // 20
  const int thread_memory = (shared_memory_size / 8 / team_size) * 8;
  constexpr size_t scalarAlignPad = (alignof(Scalar) > alignof(Ordinal)) ? (alignof(Scalar) - alignof(Ordinal)) : 0;
  Ordinal thread_shmem_key_size = ((thread_memory - sizeof(Ordinal) * 4 - scalarAlignPad) / unit_memory);
  Ordinal thread_shmem_hash_size = 1;
  while (thread_shmem_hash_size * 2 <= thread_shmem_key_size) thread_shmem_hash_size *= 2;
  thread_shmem_key_size = thread_shmem_key_size +
      ((thread_shmem_key_size - thread_shmem_hash_size) * sizeof(Ordinal)) / (sizeof(Ordinal) * 2 + sizeof(Scalar));
  thread_shmem_key_size = (thread_shmem_key_size >> 1) << 1;

  Ordinal min_hash_size = 1;
  Ordinal tmp_max_nnz = max_nnz;  // min_hash_size_scale default = 1
  while (tmp_max_nnz > min_hash_size) min_hash_size *= 4;
  size_t chunksize = (size_t)min_hash_size + (size_t)min_hash_size + (size_t)max_nnz;

  cudaDeviceProp prop;
  GPU_CHECK(cudaGetDeviceProperties(&prop, 0));
  size_t concurrency = (size_t)prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
  size_t ideal_num_chunks = concurrency / vector_size;
  size_t free_byte, total_byte;
  GPU_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
  size_t num_chunks = ideal_num_chunks;
  if (ideal_num_chunks * chunksize * sizeof(Ordinal) > free_byte / 2)
    num_chunks = (free_byte / 2) / (chunksize * sizeof(Ordinal));
  size_t po2_num_chunks = 1;                               // largest pow2 < num_chunks (upstream)
  while (po2_num_chunks * 2 < num_chunks) po2_num_chunks *= 2;
  num_chunks = po2_num_chunks;
  printf("config: vector_size=%d team_size=%d shmem_hash=%d shmem_keys=%d "
         "min_hash_size=%d chunksize=%zu num_chunks=%zu\n",
         vector_size, team_size, thread_shmem_hash_size, thread_shmem_key_size,
         min_hash_size, chunksize, num_chunks);

  // ---- device data ----
  Offset *dArow, *dCrow; Ordinal *dAcol, *dCcol; Scalar *dAval, *dCval;
  int *d_pool, *d_locks;
  GPU_CHECK(cudaMalloc(&dArow, (m + 1) * sizeof(Offset)));
  GPU_CHECK(cudaMalloc(&dAcol, nnzA * sizeof(Ordinal)));
  GPU_CHECK(cudaMalloc(&dAval, nnzA * sizeof(Scalar)));
  GPU_CHECK(cudaMalloc(&dCrow, (m + 1) * sizeof(Offset)));
  GPU_CHECK(cudaMalloc(&dCcol, nnzC * sizeof(Ordinal)));
  GPU_CHECK(cudaMalloc(&dCval, nnzC * sizeof(Scalar)));
  GPU_CHECK(cudaMalloc(&d_pool, num_chunks * chunksize * sizeof(int)));
  GPU_CHECK(cudaMalloc(&d_locks, num_chunks * sizeof(int)));
  GPU_CHECK(cudaMemcpy(dArow, Arow.data(), (m + 1) * sizeof(Offset), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dAcol, Acol.data(), nnzA * sizeof(Ordinal), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dAval, Aval.data(), nnzA * sizeof(Scalar), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dCrow, Crow.data(), (m + 1) * sizeof(Offset), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemset(d_locks, 0, num_chunks * sizeof(int)));
  GPU_CHECK(cudaMemset(d_pool, 0xFF, num_chunks * chunksize * sizeof(int)));  // init -1 (hash begins)

  // ---- numeric (timed, like upstream numericTime) ----
  dim3 block(vector_size, team_size);
  int league = (m + team_work_size - 1) / team_work_size;
  double numeric_time = 0;
  for (int rep = 0; rep < repeat; rep++) {
    auto t0 = std::chrono::steady_clock::now();
    kkmem_numeric<<<league, block, shared_memory_size>>>(
        m, dArow, dAcol, dAval, dArow, dAcol, dAval, dCrow, dCcol, dCval,
        team_work_size, thread_memory, thread_shmem_hash_size, thread_shmem_key_size,
        min_hash_size, d_pool, d_locks, (Ordinal)(num_chunks - 1), chunksize);
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
    numeric_time += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  printf("mm_time: %f (numeric phase, %d repeats)\n", numeric_time / repeat, repeat);

  // ---- validation (added): host dense-accumulator reference ----
  std::vector<Ordinal> Ccol(nnzC); std::vector<Scalar> Cval(nnzC);
  GPU_CHECK(cudaMemcpy(Ccol.data(), dCcol, nnzC * sizeof(Ordinal), cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaMemcpy(Cval.data(), dCval, nnzC * sizeof(Scalar), cudaMemcpyDeviceToHost));
  bool ok = true;
  {
    std::vector<Scalar> acc(n, 0.0);
    std::vector<char> marker(n, 0);
    std::vector<Ordinal> touched;
    std::vector<std::pair<Ordinal, Scalar>> gpu_row;
    for (Ordinal i = 0; i < m && ok; i++) {
      touched.clear();
      for (Offset ka = Arow[i]; ka < Arow[i + 1]; ka++) {
        Ordinal j = Acol[ka]; Scalar va = Aval[ka];
        for (Offset kb = Arow[j]; kb < Arow[j + 1]; kb++) {
          Ordinal c = Acol[kb];
          if (!marker[c]) { marker[c] = 1; touched.push_back(c); acc[c] = 0.0; }
          acc[c] += va * Aval[kb];
        }
      }
      std::sort(touched.begin(), touched.end());
      // gather + sort the GPU row
      gpu_row.clear();
      for (Offset k = Crow[i]; k < Crow[i + 1]; k++) gpu_row.push_back({Ccol[k], Cval[k]});
      std::sort(gpu_row.begin(), gpu_row.end());
      if ((Offset)touched.size() != Crow[i + 1] - Crow[i]) {
        printf("row %d: size mismatch gpu %zu vs ref %zu\n", i,
               (size_t)(Crow[i + 1] - Crow[i]), touched.size());
        ok = false;
      }
      for (size_t k = 0; k < touched.size() && ok; k++) {
        if (gpu_row[k].first != touched[k]) {
          printf("row %d entry %zu: col %d vs ref %d\n", i, k, gpu_row[k].first, touched[k]);
          ok = false;
        } else if (std::fabs(gpu_row[k].second - acc[touched[k]]) >
                   1e-10 * (1.0 + std::fabs(acc[touched[k]]))) {
          printf("row %d col %d: val %.17g vs ref %.17g\n", i, touched[k],
                 gpu_row[k].second, acc[touched[k]]);
          ok = false;
        }
      }
      for (Ordinal c : touched) marker[c] = 0;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
