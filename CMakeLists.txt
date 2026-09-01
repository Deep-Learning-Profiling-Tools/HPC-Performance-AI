cmake_minimum_required(VERSION 3.24)
project(HPC-Performance-AI LANGUAGES NONE)

# Aggregate build for all extracted Level 1 benchmarks.
# Each benchmark remains independently configurable via
#   cmake -S level1/<benchmark> -B build/<benchmark>/cuda -DBACKEND=CUDA
add_subdirectory(level1)
