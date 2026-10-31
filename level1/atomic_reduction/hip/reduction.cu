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
#include <hip/hip_runtime.h>
#include "kernels.h"

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
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, 0);
  std::cout << "Device name: " << props.name << std::endl;

  int *in, *out;
  hipMalloc(&in,size);
  hipMalloc(&out,sizeof(int));

  hipMemcpy(in,array,arrayLength*sizeof(int),hipMemcpyHostToDevice);
  hipDeviceSynchronize();

  // warmup
  for(int i=0;i<N;i++) {
    hipMemset(out,0,sizeof(int));
    atomic_reduction<<< dim3(2048), dim3(256) >>>(in,out,arrayLength);
  }
  hipDeviceSynchronize();

  float GB=(float)arrayLength*sizeof(int)*N;
  int sum;
  double times;

// tools/golden: before each variant's timed region (outside the ROI and the chrono timer) the output `out` that all
// 20 variants share and its captured host copy `sum` are poisoned (all-ones bytes, -1). Every timed iteration clears
// `out` and recomputes it, so a correct variant overwrites the poison; a variant that does not run leaves -1 in its
// capture instead of the previous variant's identical sum.
#define benchmark(kernel_name, grid_size) \
    hipMemset(out, 0xFF, sizeof(int)); \
    sum = -1; \
    hipDeviceSynchronize(); \
    t1 = std::chrono::high_resolution_clock::now(); \
    HPCPERF_ROI_BEGIN_SYNC(); \
    for(int i=0;i<N;i++) { \
      hipMemset(out,0,sizeof(int)); \
      kernel_name<<< dim3(grid_size), dim3(block_size) >>>(in,out,arrayLength); \
    } \
    hipDeviceSynchronize(); \
    HPCPERF_ROI_END_SYNC(); \
    t2 = std::chrono::high_resolution_clock::now(); \
    times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count(); \
    std::cout << "Thread block size: " <<  block_size << ", "; \
    std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl; \
    hipMemcpy(&sum,out,sizeof(int),hipMemcpyDeviceToHost); \
    { /* tools/golden: the checked output (one int per kernel variant and block size) */ \
      char hpcperf_name[96]; \
      snprintf(hpcperf_name, sizeof(hpcperf_name), "sum.%s.b%d", #kernel_name, block_size); \
      hpcperf_golden_output(hpcperf_name, &sum, 1, "i32"); \
    } \
    if(!hpcperf_cpu_reference_enabled()) /* default: no host check; golden comparison (tools/golden) */ \
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

  hipFree(in);
  hipFree(out);
  free(array);
  return 0;
}
