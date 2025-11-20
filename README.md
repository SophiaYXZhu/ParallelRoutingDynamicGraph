# ParallelRoutingDynamicGraph

## Project Overview
Please refer to this doc for what this project is about: https://docs.google.com/document/d/1eNZh5MjM0KwcqvB14qrfGHoRefIECy8Bp2h_8brXHOQ/

## How to run
Run `make` to compile the dynamicgraph.o library and all test cases in the `./test` directory.

To run k-core test cases with `k = K` and `NUM_PROC` number of threads, run `./testcase -k K -n NUM_PROC`. For example, `./tests/graph_size/kcore/4096_all -k 10000 -n 8` finds all subgraphs with degree at least 10000 in a fully-connected graph with 4096 vertices using 8 threads in parallel.

To run multiple packages routing simulation with `NUM_PACK` number of packages with at most `NUM_T` number of discrete timesteps and `NUM_PROC` number of threads, run `./testcase -p NUM_PACK -s NUM_T -n NUM_PROC`. For example, `./tests/graph_size/routing/4096_all -p 10000 -s 100000 -n 8` searches for an approximate for the optimal routing strategy of 10000 packages to be delivered from random source vertices to destination vertices in a fully-connected graph with 4096 vertices using 8 threads in parallel, simulating for at most 100000 timesteps.

Run `make clean` to clean up all compiled files.

## Test Cases
All test cases we provided can be found in the `./tests` subdirectory. Note that some test cases tests k-core, while some others tests multiple packages routing, and they should be ran using different flags.
