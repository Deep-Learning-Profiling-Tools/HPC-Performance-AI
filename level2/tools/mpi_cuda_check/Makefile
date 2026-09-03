# Build the CUDA-aware MPI smoke test. Needs mpicc (Open MPI) and CUDA.
# CUDA_HOME defaults to /usr/local/cuda; override on the command line.
CUDA_HOME ?= /usr/local/cuda
mpi_cuda_check: mpi_cuda_check.c
	mpicc -O2 -I$(CUDA_HOME)/include $< -o $@ -L$(CUDA_HOME)/lib64 -lcudart
clean:
	rm -f mpi_cuda_check
