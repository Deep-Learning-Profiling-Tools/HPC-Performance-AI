/* CUDA-aware MPI runtime smoke test for the HPC-Performance-AI Level 2 stack.
 *
 * Confirms the *runtime* capability, not just that Open MPI was built with it:
 *   1. MPIX_Query_cuda_support() -- the compiled/requested capability flag.
 *   2. An actual device-buffer MPI exchange (ring Sendrecv) plus an
 *      Allreduce over GPU memory, both numerically checked on the host.
 * Every rank binds to a GPU (local rank within CUDA_VISIBLE_DEVICES), so with
 * >=2 ranks the device-to-device MPI path is genuinely exercised.
 *
 * Exit 0 only if MPIX_Query_cuda_support()==1 AND both checks pass on all
 * ranks; non-zero otherwise. Rank 0 prints a one-line verdict.
 */
#include <mpi.h>
#include <mpi-ext.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define N (1<<20)
#define CHK(x) do{ cudaError_t e=(x); if(e){fprintf(stderr,"[rank %d] CUDA error %s at %d\n",rank,cudaGetErrorString(e),__LINE__); MPI_Abort(MPI_COMM_WORLD,2);} }while(0)
int main(int argc,char**argv){
  MPI_Init(&argc,&argv);
  int rank,size,ndev; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
  int cuda_aware = MPIX_Query_cuda_support();
  CHK(cudaGetDeviceCount(&ndev));
  int lrank_env; { const char*e=getenv("OMPI_COMM_WORLD_LOCAL_RANK");
                   if(!e)e=getenv("SLURM_LOCALID"); lrank_env=e?atoi(e):rank; }
  CHK(cudaSetDevice(lrank_env % (ndev>0?ndev:1)));
  char bus[64]; CHK(cudaDeviceGetPCIBusId(bus,64,lrank_env%(ndev>0?ndev:1)));
  double *dsend,*drecv,*dsum,*h=malloc(N*sizeof(double));
  CHK(cudaMalloc(&dsend,N*sizeof(double))); CHK(cudaMalloc(&drecv,N*sizeof(double))); CHK(cudaMalloc(&dsum,N*sizeof(double)));
  for(long i=0;i<N;i++) h[i]=rank+1.0;
  CHK(cudaMemcpy(dsend,h,N*sizeof(double),cudaMemcpyHostToDevice));
  int to=(rank+1)%size, from=(rank+size-1)%size;
  MPI_Sendrecv(dsend,N,MPI_DOUBLE,to,0,drecv,N,MPI_DOUBLE,from,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  CHK(cudaMemcpy(h,drecv,N*sizeof(double),cudaMemcpyDeviceToHost));
  int ring_ok=1; for(long i=0;i<N;i++) if(h[i]!=from+1.0){ring_ok=0;break;}
  MPI_Allreduce(dsend,dsum,N,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  CHK(cudaMemcpy(h,dsum,N*sizeof(double),cudaMemcpyDeviceToHost));
  double want=size*(size+1.0)/2.0; int red_ok=1;
  for(long i=0;i<N;i++) if(h[i]!=want){red_ok=0;break;}
  int bad=(!cuda_aware||!ring_ok||!red_ok), anybad=0;
  MPI_Reduce(&bad,&anybad,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
  if(size==1){ /* single rank: still report the query + self loopback */
    printf("mpi_cuda_check: 1 rank, MPIX_Query_cuda_support=%d (device exchange needs >=2 ranks)\n",cuda_aware);
  }
  if(rank==0 && size>1){
    printf("mpi_cuda_check: %d ranks, MPIX_Query_cuda_support=%d, device ring=%s, device Allreduce=%s -> %s\n",
           size,cuda_aware,ring_ok?"OK":"FAIL",red_ok?"OK":"FAIL",anybad?"FAIL":"PASS");
  }
  printf("  rank %d/%d -> GPU %d (PCI %s)\n",rank,size,lrank_env%(ndev>0?ndev:1),bus);
  fflush(stdout);
  MPI_Finalize();
  if(size==1) return cuda_aware?0:1;
  return anybad?1:0;
}
