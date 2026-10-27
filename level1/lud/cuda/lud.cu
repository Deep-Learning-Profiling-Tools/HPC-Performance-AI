/*
 * =====================================================================================
 *
 *       Filename:  lud.cu
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include "hpcperf_roi.h"
#include "hpcperf_golden.h"
#include <cuda.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 16
#endif

static int do_verify = 0;

/* HPC-Performance-AI correctness switches. -v requests upstream's verification; by default it is
   deferred: the factor matrix is captured after the timed region (hpcperf_golden_output "lu", a
   no-op unless HPCPERF_GOLDEN_DUMP is set) and level1/lud/lud_residual_check.py applies upstream's
   criterion (every |A - L*U| element <= 1e-4) to it after the run. Upstream's own host check
   (lud_verify: a single-thread float32 O(n^3) recomposition, ~15 min at -s 8192) runs only on an
   explicit offline request, HPCPERF_CPU_REFERENCE=1 (hpcperf_cpu_reference_enabled, tools/golden).
   HPCPERF_SKIP_VERIFY=1 (tools/timing) still forces it off. Kernels, matrix and timers unchanged. */
static int hpcperf_skip_verify(void) {
  const char* e = getenv("HPCPERF_SKIP_VERIFY");
  return e != NULL && *e != '\0' && *e != '0';
}

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {0,0,0,0}
};

extern void
lud_cuda(float *d_m, int matrix_dim);


int
main ( int argc, char *argv[] )
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

  int matrix_dim = 32; /* default matrix_dim */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;
  float *m, *d_m, *mm;
  stopwatch sw;

  while ((opt = getopt_long(argc, argv, "::vs:i:", 
                            long_options, &option_index)) != -1 ) {
    switch(opt){
    case 'i':
      input_file = optarg;
      break;
    case 'v':
      do_verify = 1;
      break;
    case 's':
      matrix_dim = atoi(optarg);
      printf("Generate input matrix internally, size =%d\n", matrix_dim);
      // fprintf(stderr, "Currently not supported, use -i instead\n");
      // fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
      // exit(EXIT_FAILURE);
      break;
    case '?':
      fprintf(stderr, "invalid option\n");
      break;
    case ':':
      fprintf(stderr, "missing argument\n");
      break;
    default:
      fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
	      argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  
  if ( (optind < argc) || (optind == 1)) {
    fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  /* measurement mode: drop the host-side L*U verification, keep everything else */
  if (hpcperf_skip_verify()) { do_verify = 0; printf("SKIP_VERIFY\n"); }
  /* -v: upstream's host recomposition only offline (HPCPERF_CPU_REFERENCE=1); by default the check is
     deferred to the post-run residual check of the captured factors (see the switch comment above) */
  if (do_verify && !hpcperf_cpu_reference_enabled()) {
    do_verify = 0;
    printf("VERIFY_DEFERRED: the L*U residual of the captured factors is checked after the run "
           "(level1/lud/lud_residual_check.py); HPCPERF_CPU_REFERENCE=1 runs upstream's host check\n");
  }

  if (input_file) {
    printf("Reading matrix from file %s\n", input_file);
    ret = create_matrix_from_file(&m, input_file, &matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix from file %s\n", input_file);
      exit(EXIT_FAILURE);
    }
  } 
  else if (matrix_dim) {
    printf("Creating matrix internally size=%d\n", matrix_dim);
    ret = create_matrix(&m, matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
      exit(EXIT_FAILURE);
    }
  }


  else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  }

  if (do_verify){
    printf("Before LUD\n");
    // print_matrix(m, matrix_dim);
    matrix_duplicate(m, &mm, matrix_dim);
  }

  cudaMalloc((void**)&d_m, 
             matrix_dim*matrix_dim*sizeof(float));

  /* beginning of timing point */
  stopwatch_start(&sw);
  cudaMemcpy(d_m, m, matrix_dim*matrix_dim*sizeof(float), 
	     cudaMemcpyHostToDevice);

  HPCPERF_ROI_BEGIN_SYNC();  // tools/timing ROI: lud_cuda (the blocked decomposition), not the copies inside the upstream stopwatch
  lud_cuda(d_m, matrix_dim);

  HPCPERF_ROI_END_SYNC();
  cudaMemcpy(m, d_m, matrix_dim*matrix_dim*sizeof(float), 
	     cudaMemcpyDeviceToHost);

  /* end of timing point */
  stopwatch_stop(&sw);
  printf("Time consumed(ms): %lf\n", 1000*get_interval_by_sec(&sw));

  cudaFree(d_m);

  /* tools/golden capture of the result: the packed factor matrix (unit-lower L strictly below the
     diagonal, U on and above it), after the timed region and the final device-to-host copy; a no-op
     unless HPCPERF_GOLDEN_DUMP is set. Checked by lud_residual_check.py (A - L*U), not by a golden. */
  hpcperf_golden_output("lu", m, (size_t)matrix_dim * (size_t)matrix_dim, "f32");

  if (do_verify){
    printf("After LUD\n");
    // print_matrix(m, matrix_dim);
    printf(">>>Verify<<<<\n");
    lud_verify(mm, m, matrix_dim); 
    free(mm);
  }

  free(m);

  return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
