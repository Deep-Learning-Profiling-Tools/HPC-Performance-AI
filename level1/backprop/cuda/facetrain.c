

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"
#include "hpcperf_golden.h"

/* HPC-Performance-AI correctness switch. The host-side CPU reference below runs ONLY on an
   explicit offline request (HPCPERF_CPU_REFERENCE=1; tools/golden/hpcperf_golden.py make uses it
   once to validate the frozen golden reference); by default it is skipped and correctness is
   judged by comparing the captured outputs (hpcperf_golden_output, HPCPERF_GOLDEN_DUMP) with the
   frozen B200 golden reference (tools/golden). No kernel, data initialization, tolerance or
   algorithm is touched. */
static int hpcperf_skip_verify(void) {
  return !hpcperf_cpu_reference_enabled();
}

/* tools/golden: capture a weight matrix stored as row pointers, rows 0..m x columns 0..n
   (the index ranges the check below compares), streamed row by row */
static void hpcperf_golden_rows(const char *name, float **w, int m, int n)
{
  hpcperf_golden_stream s;
  int i;
  hpcperf_golden_begin(&s, name, "f32");
  for (i = 0; i <= m; i++) hpcperf_golden_append(&s, w[i], (size_t)(n + 1));
  hpcperf_golden_end(&s);
}


extern char *strcpy();
extern void exit();

int layer_size = 0;

/* [HPC-Performance-AI] standalone correctness check: duplicate the
 * identically-initialized network and train it with the upstream CPU
 * implementation (bpnn_train from backprop.c), then compare weights and
 * errors against the CUDA training step. */
static BPNN *clone_net(BPNN *net)
{
  BPNN *c = bpnn_create(net->input_n, net->hidden_n, net->output_n);
  int i, j;
  for (i = 0; i <= net->input_n; i++)
    for (j = 0; j <= net->hidden_n; j++) {
      c->input_weights[i][j] = net->input_weights[i][j];
      c->input_prev_weights[i][j] = net->input_prev_weights[i][j];
    }
  for (i = 0; i <= net->hidden_n; i++)
    for (j = 0; j <= net->output_n; j++) {
      c->hidden_weights[i][j] = net->hidden_weights[i][j];
      c->hidden_prev_weights[i][j] = net->hidden_prev_weights[i][j];
    }
  for (i = 0; i <= net->input_n; i++) c->input_units[i] = net->input_units[i];
  for (i = 0; i <= net->output_n; i++) c->target[i] = net->target[i];
  return c;
}

static float rel_diff(float a, float b)
{
  float d = fabsf(a - b);
  return d / (1.0f + fabsf(b));
}

backprop_face()
{
  BPNN *net, *ref;
  int i, j;
  float out_err, hid_err;
  float out_err_ref, hid_err_ref;
  float max_err = 0.0f, e;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)

  printf("Input layer size : %d\n", layer_size);
  load(net);
  ref = hpcperf_skip_verify() ? NULL : clone_net(net);   /* identical starting state for the CPU reference */
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_cuda(net, &out_err, &hid_err);
  printf("Training done\n");
  /* tools/golden: the checked outputs -- the trained weights (the GPU-adjusted input weights were
     copied back into net by bpnn_train_cuda, after its ROI). out_err / hid_err are NOT captured:
     bpnn_train_cuda never writes *eo / *eh, so they are uninitialized (see README) */
  hpcperf_golden_rows("input_weights", net->input_weights, net->input_n, net->hidden_n);
  hpcperf_golden_rows("hidden_weights", net->hidden_weights, net->hidden_n, net->output_n);

  if (hpcperf_skip_verify()) {
    /* default: the CPU reference training step is skipped; correctness = the captured outputs
       against the frozen golden reference (tools/golden) */
    printf("SKIP_VERIFY\n");
  } else {
    /* CPU reference training step (upstream bpnn_train) and comparison */
    bpnn_train(ref, &out_err_ref, &hid_err_ref);
    /* the CPU reference (golden make only) */
    hpcperf_golden_rows("input_weights.cpu", ref->input_weights, ref->input_n, ref->hidden_n);
    hpcperf_golden_rows("hidden_weights.cpu", ref->hidden_weights, ref->hidden_n, ref->output_n);
    max_err = rel_diff(out_err, out_err_ref);
    e = rel_diff(hid_err, hid_err_ref); if (e > max_err) max_err = e;
    for (i = 0; i <= net->input_n; i++)
      for (j = 0; j <= net->hidden_n; j++) {
        e = rel_diff(net->input_weights[i][j], ref->input_weights[i][j]);
        if (e > max_err) max_err = e;
      }
    for (i = 0; i <= net->hidden_n; i++)
      for (j = 0; j <= net->output_n; j++) {
        e = rel_diff(net->hidden_weights[i][j], ref->hidden_weights[i][j]);
        if (e > max_err) max_err = e;
      }
    printf("Output error: %f (CPU reference %f)\n", out_err, out_err_ref);
    printf("Hidden error: %f (CPU reference %f)\n", hid_err, hid_err_ref);
    printf("Max relative difference vs CPU reference: %e\n", max_err);
    printf("%s\n", (max_err <= 1e-3f) ? "PASS" : "FAIL");
    bpnn_free(ref);
  }
  bpnn_free(net);
  if (!hpcperf_skip_verify() && max_err > 1e-3f) exit(1);
}

int setup(argc, argv)
int argc;
char *argv[];
{
	
  int seed;

  if (argc!=2){
  fprintf(stderr, "usage: backprop <num of input elements>\n");
  exit(0);
  }
  layer_size = atoi(argv[1]);
  if (layer_size%16!=0){
  fprintf(stderr, "The number of input points must be divided by 16\n");
  exit(0);
  }
  

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
