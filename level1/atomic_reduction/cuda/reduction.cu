/*
   Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

#include "hpcperf_roi.h"
#include "hpcperf_golden.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include "kernels.h"

// HPC-Performance-AI correctness switch. The host-side CPU reference below runs ONLY on an
// explicit offline request (HPCPERF_CPU_REFERENCE=1; tools/golden/hpcperf_golden.py make uses it
// once to validate the frozen golden reference); by default it is skipped and correctness is
// judged by comparing the captured output (hpcperf_golden_output, HPCPERF_GOLDEN_DUMP) with the
// frozen B200 golden reference (tools/golden). No kernel, data initialization, tolerance or
// algorithm is touched.
static bool hpcperf_skip_verify() {
  return !hpcperf_cpu_reference_enabled();
}


int main(int argc, char** argv)
{
  int arrayLength = 52428800;
  int block_sizes[] = {128, 256, 512, 1024};
  int N = 100;

  if (argc == 3) {
    arrayLength=atoi(argv[1]);
    N=atoi(argv[2]);
  }

  std::cout << "Array size: " << arrayLength*sizeof(int)/1024.0/1024.0 << " MB"<<std::endl;
  std::cout << "Repeat the kernel execution: " << N << " times" << std::endl;

  int* array=(int*)malloc(arrayLength*sizeof(int));
  int checksum =0;
  for(int i=0;i<arrayLength;i++) {
    array[i]=rand()%2;
    checksum+=array[i];
  }

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  size_t size=sizeof(int)*arrayLength;

  // Get device properties
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  std::cout << "Device name: " << props.name << std::endl;

  int *in, *out;
  cudaMalloc(&in,size);
  cudaMalloc(&out,sizeof(int));

  cudaMemcpy(in,array,arrayLength*sizeof(int),cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // warmup
  for(int i=0;i<N;i++) {
    cudaMemset(out,0,sizeof(int));
    atomic_reduction<<< dim3(2048), dim3(256) >>>(in,out,arrayLength);
  }
  cudaDeviceSynchronize();

  float GB=(float)arrayLength*sizeof(int)*N;
  int sum;
  double times;

#define benchmark(kernel_name, grid_size) \
    t1 = std::chrono::high_resolution_clock::now(); \
    HPCPERF_ROI_BEGIN_SYNC(); \
    for(int i=0;i<N;i++) { \
      cudaMemset(out,0,sizeof(int)); \
      kernel_name<<< dim3(grid_size), dim3(block_size) >>>(in,out,arrayLength); \
    } \
    cudaDeviceSynchronize(); \
    HPCPERF_ROI_END_SYNC(); \
    t2 = std::chrono::high_resolution_clock::now(); \
    times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count(); \
    std::cout << "Thread block size: " <<  block_size << ", "; \
    std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl; \
    cudaMemcpy(&sum,out,sizeof(int),cudaMemcpyDeviceToHost); \
    { /* tools/golden: the checked output (one int per kernel variant and block size) */ \
      char hpcperf_name[96]; \
      snprintf(hpcperf_name, sizeof(hpcperf_name), "sum.%s.b%d", #kernel_name, block_size); \
      hpcperf_golden_output(hpcperf_name, &sum, 1, "i32"); \
    } \
    if(hpcperf_skip_verify()) \
      std::cout<<"SKIP_VERIFY"<<std::endl<<std::endl; \
    else if(sum==checksum) \
      std::cout<<"VERIFICATION: PASS"<<std::endl<<std::endl; \
    else \
      std::cout<<"VERIFICATION: FAIL!!"<<std::endl<<std::endl; \

  for (size_t k = 0; k < sizeof(block_sizes) / sizeof(int); k++) {
    int block_size = block_sizes[k];
    int blocks=std::min((arrayLength+block_size-1)/block_size,2048);
    benchmark(atomic_reduction, blocks);
    benchmark(atomic_reduction_v2, blocks/2);
    benchmark(atomic_reduction_v4, blocks/4);
    benchmark(atomic_reduction_v8, blocks/8);
    benchmark(atomic_reduction_v16, blocks/16);
  }

  cudaFree(in);
  cudaFree(out);
  free(array);
  return 0;
}
