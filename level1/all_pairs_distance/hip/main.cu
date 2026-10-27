/* This code is provided as supplementary material for the book
   chapter "Exploiting graphics processing units for computational
   biology and bioinformatics," by Payne, Sinnott-Armstrong, and
   Moore, to appear in "The Handbook of Research on Computational and
   Systems Biology: Interdisciplinary applications," by IGI Global.
 */

#include "hpcperf_roi.h"
#include "hpcperf_golden.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

#define INSTANCES 224   /* # of instances */
#define ATTRIBUTES 4096 /* # of attributes */
#define THREADS 128    /* # of threads per block */

/* CPU implementation */
void CPU(int * data, int * distance) {
  /* compare all pairs of instances, accessing the attributes in
     row-major order */
#pragma omp parallel for collapse(2)
  for (int i = 0; i < INSTANCES; i++) {
    for (int j = 0; j < INSTANCES; j++) {
      for (int k = 0; k < ATTRIBUTES; k++) {
        distance[i + INSTANCES * j] +=
          (data[i * ATTRIBUTES + k] != data[j * ATTRIBUTES + k]);
      }
    }
  }
}


/*  coalesced GPU implementation of the all-pairs kernel using
    character data types and registers */
__global__ void k1 (const char *data, int *distance) {
  int idx = threadIdx.x;
  int gx = blockIdx.x;
  int gy = blockIdx.y;

  for(int i = 4*idx; i < ATTRIBUTES; i+=THREADS*4) {
    char4 j = *(char4 *)(data + i + ATTRIBUTES*gx);
    char4 k = *(char4 *)(data + i + ATTRIBUTES*gy);

    /* use a local variable (stored in register) to hold intermediate
       values. This reduces writes to global memory */
    char count = 0;

    if(j.x ^ k.x)
      count++;
    if(j.y ^ k.y)
      count++;
    if(j.z ^ k.z)
      count++;
    if(j.w ^ k.w)
      count++;

    /* atomic write to global memory */
    atomicAdd(distance + INSTANCES*gx + gy, count);
  }
}

/*  coalesced GPU implementation of the all-pairs kernel using
    character data types, registers, and shared memory */
__global__ void k2 (const char *data, int *distance) {
  int idx = threadIdx.x;
  int gx = blockIdx.x;
  int gy = blockIdx.y;

  /* Shared memory is the other major memory (other than registers and
     global). It is used to store values between multiple threads. In
     particular, the shared memory access is defined by the __shared__
     attribute and it is a special area of memory on the GPU
     itself. Because the memory is on the chip, it is a lot faster
     than global memory. Multiple threads can still access it, though,
     provided they are in the same block.
   */
  __shared__ int dist[THREADS];

  /* each thread initializes its own location of the shared array */
  dist[idx] = 0;

  /* At this point, the threads must be synchronized to ensure that
     the shared array is fully initialized. */
  __syncthreads();

  for(int i = idx*4; i < ATTRIBUTES; i+=THREADS*4) {
    char4 j = *(char4 *)(data + i + ATTRIBUTES*gx);
    char4 k = *(char4 *)(data + i + ATTRIBUTES*gy);
    char count = 0;

    if(j.x ^ k.x)
      count++;
    if(j.y ^ k.y)
      count++;
    if(j.z ^ k.z)
      count++;
    if(j.w ^ k.w)
      count++;

    /* Increment shared array */
    dist[idx] += count;
  }

  /* Synchronize threads to make sure all have completed their updates
     of the shared array. Since the distances for each thread are read
     by thread 0 below, this must be ensured. Above, it was not
     necessary because each thread was accessing its own memory
   */
  __syncthreads();

  /* Perform balanced tree reduction across the shared memory */
  for (int stride = THREADS/2; stride > 0; stride /= 2) {
    if (idx < stride) {
      dist[idx] += dist[idx + stride];
    }
    __syncthreads();
  }

  if(idx == 0) {
    /* Thread 0 will then write the output to global memory. Note that
       this does not need to be performed atomically, because only one
       thread per block is writing to global memory, and each block
       corresponds to a unique memory address.
     */
    distance[INSTANCES*gy + gx] = dist[0];
  }
}

/*  coalesced GPU implementation of the all-pairs kernel using
    character data types, registers, and CUB block reduction */
__global__ void k3 (const char *data, int *distance) {
  int idx = threadIdx.x;
  int gx = blockIdx.x;
  int gy = blockIdx.y;

  typedef hipcub::BlockReduce<int, THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int dist = 0;

  for(int i = idx*4; i < ATTRIBUTES; i+=THREADS*4) {
    char4 j = *(char4 *)(data + i + ATTRIBUTES*gx);
    char4 k = *(char4 *)(data + i + ATTRIBUTES*gy);
    char count = 0;

    if(j.x ^ k.x)
      count++;
    if(j.y ^ k.y)
      count++;
    if(j.z ^ k.z)
      count++;
    if(j.w ^ k.w)
      count++;

    dist += count;
  }

  int sum = BlockReduce(temp_storage).Sum(dist);

  if(idx == 0) {
    distance[INSTANCES*gy + gx] = sum;
  }
}

// tools/golden: poison the output that k1, k2 and k3 share -- distance_device and the host copy gpu_distance that
// is captured (all-ones bytes, -1 in every int) -- before each kernel's timed region, outside the ROI and the chrono
// timer. Every timed iteration clears distance_device and recomputes it, so a correct timed region overwrites the
// poison; a kernel that does not run leaves -1 in its capture instead of the previous kernel's identical matrix.
static void hpcperf_poison_distance(int *d, int *h, size_t bytes) {
  hipMemset(d, 0xFF, bytes);
  memset(h, 0xFF, bytes);
  hipDeviceSynchronize();
}

int main(int argc, char **argv) {

  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }

  const int iterations = atoi(argv[1]);

  /* host data */
  int *data;
  char *data_char;
  int *cpu_distance;
  int *gpu_distance;

  /* device data */
  char *data_char_device;
  int *distance_device;

  /* block and grid dimensions */
  dim3 dimBlock;
  dim3 dimGrid;

  /* verification result */
  int status;

  /* seed RNG */
  srand(2);

  /* allocate host memory */
  data = (int *)malloc(INSTANCES * ATTRIBUTES * sizeof(int));
  data_char = (char *)malloc(INSTANCES * ATTRIBUTES * sizeof(char));
  cpu_distance = (int *)malloc(INSTANCES * INSTANCES * sizeof(int));
  gpu_distance = (int *)malloc(INSTANCES * INSTANCES * sizeof(int));

  /* randomly initialize host data */
#pragma omp parallel for collapse(2)
  for (int i = 0; i < ATTRIBUTES; i++) {
    for (int j = 0; j < INSTANCES; j++) {
      data[i + ATTRIBUTES * j] = data_char[i + ATTRIBUTES * j] = random() % 3;
    }
  }

  /* allocate GPU memory */
  hipMalloc((void **)&data_char_device,
      INSTANCES * ATTRIBUTES * sizeof(char));

  hipMalloc((void **)&distance_device,
      INSTANCES * INSTANCES * sizeof(int));

  hipMemcpy(data_char_device, data_char,
      INSTANCES * ATTRIBUTES * sizeof(char),
      hipMemcpyHostToDevice);

  /* specify grid and block dimensions */
  dimBlock.x = THREADS;
  dimBlock.y = 1;
  dimGrid.x = INSTANCES;
  dimGrid.y = INSTANCES;


  /* CPU reference for the three memcmp checks below; runs only with HPCPERF_CPU_REFERENCE=1 (golden make).
     Default: skipped; correctness = the captured outputs against the frozen golden reference (tools/golden) */
  auto start = std::chrono::steady_clock::now();
  auto end = start;
  double elapsedTime = 0;
  if (hpcperf_cpu_reference_enabled()) {
    bzero(cpu_distance,INSTANCES*INSTANCES*sizeof(int));
    CPU(data, cpu_distance);
    end = std::chrono::steady_clock::now();
    elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("CPU time: %f (us)\n", elapsedTime);
  }

  hpcperf_poison_distance(distance_device, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
  start = std::chrono::steady_clock::now();
  HPCPERF_ROI_BEGIN_SYNC();  // tools/timing ROI: this timed GPU region
  for (int n = 0; n < iterations; n++) {
    /* register GPU kernel */
    hipMemset(distance_device, 0, INSTANCES * INSTANCES * sizeof(int));
    k1<<<dimGrid,dimBlock>>>(data_char_device, distance_device);
  }
  hipDeviceSynchronize();
  HPCPERF_ROI_END_SYNC();
  end = std::chrono::steady_clock::now();
  elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  hipMemcpy(gpu_distance, distance_device,
             INSTANCES * INSTANCES * sizeof(int), hipMemcpyDeviceToHost);
  hpcperf_golden_output("distance.k1", gpu_distance, INSTANCES * INSTANCES, "i32");   // tools/golden: the checked output of k1

  printf("Average kernel execution time: %f (us)\n", elapsedTime / iterations);
  if (!hpcperf_cpu_reference_enabled()) {
    printf("SKIP_VERIFY\n");
  } else {
    status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
    printf("%s\n", status ? "FAIL" : "PASS");
  }

  hpcperf_poison_distance(distance_device, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
  start = std::chrono::steady_clock::now();
  HPCPERF_ROI_BEGIN_SYNC();  // tools/timing ROI: this timed GPU region
  for (int n = 0; n < iterations; n++) {
    /* shared memory GPU kernel */
    hipMemset(distance_device, 0, INSTANCES * INSTANCES * sizeof(int));
    k2<<<dimGrid,dimBlock>>>(data_char_device, distance_device);
  }
  hipDeviceSynchronize();
  HPCPERF_ROI_END_SYNC();
  end = std::chrono::steady_clock::now();
  elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  hipMemcpy(gpu_distance, distance_device,
             INSTANCES * INSTANCES * sizeof(int), hipMemcpyDeviceToHost);
  hpcperf_golden_output("distance.k2", gpu_distance, INSTANCES * INSTANCES, "i32");   // tools/golden: the checked output of k2

  printf("Average kernel execution time: %f (us)\n", elapsedTime / iterations);
  if (!hpcperf_cpu_reference_enabled()) {
    printf("SKIP_VERIFY\n");
  } else {
    status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
    printf("%s\n", status ? "FAIL" : "PASS");
  }

  hpcperf_poison_distance(distance_device, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
  start = std::chrono::steady_clock::now();
  HPCPERF_ROI_BEGIN_SYNC();  // tools/timing ROI: this timed GPU region
  for (int n = 0; n < iterations; n++) {
    /* shared memory GPU kernel */
    hipMemset(distance_device, 0, INSTANCES * INSTANCES * sizeof(int));
    k3<<<dimGrid,dimBlock>>>(data_char_device, distance_device);
  }
  hipDeviceSynchronize();
  HPCPERF_ROI_END_SYNC();
  end = std::chrono::steady_clock::now();
  elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  hipMemcpy(gpu_distance, distance_device,
             INSTANCES * INSTANCES * sizeof(int), hipMemcpyDeviceToHost);
  hpcperf_golden_output("distance.k3", gpu_distance, INSTANCES * INSTANCES, "i32");   // tools/golden: the checked output of k3

  printf("Average kernel execution time: %f (us)\n", elapsedTime / iterations);
  if (!hpcperf_cpu_reference_enabled()) {
    printf("SKIP_VERIFY\n");
  } else {
    status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
    printf("%s\n", status ? "FAIL" : "PASS");
  }

  free(cpu_distance);
  free(gpu_distance);
  free(data);
  free(data_char);
  hipFree(data_char_device);
  hipFree(distance_device);

  return hpcperf_cpu_reference_enabled() ? status : 0;  /* exit code is the verify result */
}
