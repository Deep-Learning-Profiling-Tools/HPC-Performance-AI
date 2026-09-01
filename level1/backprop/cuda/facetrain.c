

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"

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
  ref = clone_net(net);   /* identical starting state for the CPU reference */
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_cuda(net, &out_err, &hid_err);
  printf("Training done\n");

  /* CPU reference training step (upstream bpnn_train) and comparison */
  bpnn_train(ref, &out_err_ref, &hid_err_ref);
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
  bpnn_free(net);
  if (max_err > 1e-3f) exit(1);
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
