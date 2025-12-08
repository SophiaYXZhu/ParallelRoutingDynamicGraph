# ParallelRoutingDynamicGraph

## Summary
We implemented a high-performance parallel dynamic graph library on shared-memory multicore CPUs using OpenMP. Our system supports low-latency updates, including edge and vertex insertions and deletions, using a consistent snapshot mechanism and executes two computationally intensive analytics: discrete-time multi-package routing with edge capacity constraints and k-core decomposition. We developed two distinct parallel routing strategies: Edge Parallel and Graph Partition, to explore the tradeoffs between fine-grained synchronization and cache locality. We evaluated our system on both the GHC and PSC machines, demonstrating the scalability limits of each approach under varying graph densities and congestion levels, ultimately showing that atomics-based edge parallelism outperforms spatial partitioning for fine-grained routing workloads.

## Background
Many real-world systems such as communication networks, distributed services, and social platforms can be modeled as large, sparse, and continuously updating graphs. These graphs undergo frequent structural updates, including edge and vertex insertions and deletions, yet applications often require running computationally expensive analytics on a consistent snapshot of the graph. Two important analytics in this setting are k-core decomposition, which identifies structurally dense regions of a graph, and least time-consuming multiple packages routing strategy, where many independent packages simultaneously travel from their source to destination vertices respectively subject to the edge capacity constraint that only one package can travel through an edge at one timestep. Both computations are expensive on large graphs and can be optimized with parallelism.

The underlying data structure is a sparse graph represented by adjacency lists. The graph is mutable, supporting atomic additions and removals of vertices and edges. We aim to provide an interface that implements such dynamic graphs and the two key analytics, which includes the following operations:

- `add_edge`(u, v) adds the undirected edge (u, v) to the dynamic graph.
- `remove_edge`(u, v) removes the undirected edge (u, v) to the dynamic graph.
- `add_vertex`(v) adds the vertex v to the dynamic graph.
- `remove_vertex`(v) removes the vertex v to the dynamic graph.
- `min_cost_routing`([(s1, d1), (s2, d2), …, (sn, dn)]) computes a list of paths from each source vertex si to their corresponding destination vertex di through the graph such that, if we route a single package from each source vertex to each destination vertex, the total routing process would take a close-to-minimal amount of timesteps. Each edge has a capacity of 1 package per timestep. In each timestep, a package queries the graph to find a neighbor that reduces its distance to the destination, guided by a BFS heuristic. It then attempts to move to that neighbor.
- `k_core`(k) returns a list of subgraphs (subsets of vertices) such that each vertex has degree at least k in the subgraph on a snapshot of the current graph. 

In both of the two key analytics, the workloads are not SIMD amenable  due to irregular memory access patterns and control flow divergence. Traversing an adjacency list involves accessing the indices that are data dependent and non-contiguous. This requires expensive gather operations rather than efficient vector loads. In addition, the routing logic is highly dependent on the branch, including checking whether a neighbor is active, edge capacity, and comparing distances. These conditional branches cause serialization in SIMD lanes, negating the throughput benefits of vector instructions. However, we eventually made optimizations to data layouts that allow the compiler to auto-vectorize linear scans, such as the initial degree computation in k-core and the check for completion in routing. We deliberately chose a multicore CPU platform over a GPU because CPUs deal with irregularity and divergence more efficiently than SIMD architectures.

## Running Experiments
Run `make` to compile the dynamicgraph.o library and all test cases in the `./test` directory.

To run multi-packages routing simulation with `NUM_PACK` number of packages with at most `NUM_T` number of discrete timesteps and `NUM_PROC` threads, run `./testcase -p NUM_PACK -s NUM_T -n NUM_PROC [-m MODE]`, where `MODE` selects the routing implementation: `o` for the original package-parallel version (baseline), `p` for the graph partition scheme, and `e` for the edge parallel scheme. We will introduce what both parallel schemes do in our Approach section. For example, `./tests/routing/4096_all -p 10000 -s 100000 -n 8 -m o` searches for an approximate optimal routing strategy of 10000 packages to be delivered from random source vertices to destination vertices in a fully-connected graph with 4096 vertices using 8 threads in parallel, simulating for at most 100000 timesteps.

To run k-core test cases with `k = K` and `NUM_PROC` number of threads, run `./testcase -k K -n NUM_PROC`. For example, `./tests/kcore/4096_all -k 10000 -n 8` finds all subgraphs with degree at least 10000 in a fully-connected graph with 4096 vertices using 8 threads in parallel.

Run `make clean` to clean up all compiled files.

## Test Cases
All test cases we provided can be found in the `./tests` subdirectory. Note that some test cases tests k-core, while some others tests multiple packages routing, and they should be ran using different flags.

### Multi-package Routing Test Cases
Some important test cases for multi-package routing simulation are described below:
- long_complex_chain: A linear chain of 32768 vertices with random forward shortcut edges with 50% probability.
large_imbalance: Consists of two massive dense cores connected by bridges, with a single periphery node handling half the traffic to force contention, totalling 4097 vertices.
- large_imbalance_peripheral: In addition to the two dense inner cores in large_imbalance, the graph is surrounded by a large, sparse outer shell of 4095 vertices that funnels traffic into the center, totalling 8192 vertices.
- 8192_all: A fully connected graph with 8192 vertices.
- 2048_sparse: A randomized graph with 2048 vertices and a 33% edge probability, representing a moderately sparse graph density.
- 8192_highway_barbell: Consists of two disjoint cliques of 4096 vertices linked by 4096 bridge edges in between.

### K-Core Test Cases
Some important test cases for k-core are described below:
- 4096_all: A fully connected graph with 4096 vertices.
- huge_dense: A massive random graph with 32768 vertices and a 67% edge probability.
- large_imbalance_peripheral: Consists of two massive dense cores connected by bridges, surrounded by a large, sparse outer shell of 4095 vertices that funnels traffic into the center, totalling 8192 vertices.

## Approach
### Graph Operations
For add_vertex, remove_vertex, add_edge, and remove_edge operations, we maintain the dynamic graph data structure with private variables such as its adjacency matrix, a snapshot of its adjacency matrix, its list of active vertices, and the list of its edge capacities. 

To support consistent parallel analytics, we implemented the snapshot mechanism. The snapshot mechanism allows us to generate a read-only view of the graph’s adjacency list at a specific timestamp. This snapshot serves as the input for algorithms, ensuring our analytics run in a stable state even if the underlying graph topology is being modified between simulation steps.

### Multi-Package Routing
#### Baseline
Our baseline approach implements the core routing logic using a centralized contention resolution mechanism. The algorithm begins by pre-computing a distance field for every distinct destination vertex using reverse Breadth-First Search (BFS). This step is parallelized across destinations, producing a read-only lookup table where dist(v, d) is the shortest hop distance from vertex v to destination d.

During the routing simulation, we parallelize the workload across packages. In each timestep, threads process a static subset of packages concurrently. Each package independently scans its current neighbors and greedily selects a next hop that strictly reduces the distance to its destination. To enforce edge capacity constraints, which allow at most one package per directed edge per timestep, after all packages propose their moves, a single global hashmap on the main thread resolves conflicts. If multiple packages attempt to reserve the same edge, we permit one move and force the others to stall. Once conflicts are resolved, package positions and global arrival counters are updated in parallel. 

#### Edge Parallel Scheme
To eliminate the bottleneck of the centralized hashmap found in the baseline, the Edge Parallel scheme distributes synchronization across the graph structure itself. Instead of a single global hashmap, we map each directed edge to a unique synchronization primitive, allowing fine-grained, localized contention resolution.
Every undirected edge in the graph is represented by two slots in a flat, global array of atomic integers, corresponding to the two possible directions of travel. The routing simulation proceeds in two parallel phases per timestep:
- Decision Phase: Threads iterate over their assigned static chunk of packages. For each active package, the thread scans the adjacency list of the package’s current vertex and queries the precomputed BFS distance matrix dist to select the neighbor that maximally reduces the distance to the destination.
- Reservation Phase: Threads attempt to reserve the edge corresponding to the chosen move. Each thread computes the unique slot index for the edge that connects the current vertex to its selected next hop, and performs an atomic Compare-And-Swap (CAS) operation. If CAS succeeds, the package claims the edge for the current timestep. If CAS fails, another package has already claimed the edge, forcing the current package to stall.

<div align="center"> 
  ![Edge Parallel](https://github.com/SophiaYXZhu/ParallelRoutingDynamicGraph/blob/main/images/EdgeParallel.png) 
</div>

These two phases are followed by a cleanup and commit phase. This design transforms the contention logic from a centralized serial queue into thousands of independent micro-contests distributed across memory, significantly reducing allocator pressure and serialization.

By maintaining package states, including positions, destinations, and status, in separate contiguous vectors, we maximize cache line utilization. Such data layout is optimal for the Edge Parallel access pattern, which requires linear scans over the package list to check completion status. In addition, we used `uint8_t` vectors instead of bool vectors for tracking package completion and movement permissions. The bit-packed bool vectors introduced severe false sharing and read-modify-write overheads. By using unsigned integers, the CPU now loads 64 distinct completion flags per cache line, enabling threads to efficiently skip completed packages without cache pollution.

Finally, to avoid an O(|E|) scan to reset locks at the end of each timestep, threads record successfully acquired locks in a local buffer. We concatenate these buffers and perform a parallel reset only on the slots that were actually modified in parallel, ensuring the overhead scales with the number of moving packages rather than the graph size. Such optimizations make the Edge Parallel scheme a highly regular, cache-friendly two-phase loop where the only synchronization is per-edge CAS and a small amount of work to recycle the few edge slots used in a timestep.

#### Graph Partition Scheme
To address the potential scalability limits of atomic contention, we implemented a Graph Partition scheme that adopts a message-passing model within shared memory. In this approach, we invert the ownership model: instead of threads sharing the entire graph and contending for edges, each thread is assigned exclusive ownership of a disjoint subset of vertices and the packages currently residing on them.

The graph vertices are statically partitioned among T threads using either a cyclic assignment. The simulation proceeds in three distinct phases per timestep:
- Local Routing Phase: Threads iterate exclusively over their local package lists. For each package, the thread computes the next hop using the precomputed BFS distances in dist. Edge capacity is enforced using a thread-local hashmap. Because a thread owns the source vertex of any outgoing edge in its partition, it can arbitrate conflicts locally on a first-come-first-served basis without any atomic operations or cache coherency traffic.
- Exchange Phase: If a package moves to a vertex owned by a different thread, the thread enqueues the package index into an outgoing mailbox. A global barrier separates the calculation and communication phases, ensuring that all writes to the mailboxes are visible before reading.
- Merge Phase: Threads scan the mailboxes addressed to them, retrieving incoming packages and merging them with the packages that remained within the partition to rebuild the local queue for the next timestep.

<div align="center"> ![Graph Partition](https://github.com/SophiaYXZhu/ParallelRoutingDynamicGraph/blob/main/images/GraphPartition.png) </div>

To eliminate allocation overhead during the communication phase, we pre-allocate a fixed T × T matrix of vectors for the mailboxes. Each thread appends indices of packages that need to move to another partition into its row, before each thread concatenates the columns addressed to it into its local package queue. These vectors are reused across timesteps, providing a stable memory footprint for inter-thread communication. Because each vector row is owned by exactly one thread and each column is read only after a barrier, all pushes and reads are free of contentions. Hence, in this partition scheme, edge conflicts are resolved locally, cross-thread traffic is batched and predictable, and cache locality inside each partition is improved significantly.

We reorganized the data layout to suit the random access patterns inherent in partitioned routing. Unlike the linear scans in the Edge Parallel scheme, partitioned routing involves accessing packages in a scattered order based on their location. To minimize cache misses, we grouped all per-package data (position, destination, BFS row index, and status) into a single 16-byte aligned PackageState structure. Each package now carries its current vertex, destination vertex, precomputed row index into the BFS distance table, and a done flag together in the same struct. When a thread processes package i, it issues exactly one cacheline fetch to get everything it needs packed in the struct instead of three separate loads from different vectors, which ensures that a single cache line fetch retrieves the complete context for a package.

Anecdotally, such cache and data layout optimizations largely  boosted the performance of the large_imbalance test case. In particular, using 8 threads on GHC machines, the local queue cache optimization decreased runtime from 219.36 to 62.50 seconds, and using a padded 16-bytes aligned struct further decreased the runtime to merely 41.01 seconds.
We explored two partitioning strategies of vertices to threads to balance locality and load distribution:
- Cyclic Partitioning (v `mod T`): Randomizes vertex assignment to distribute load more evenly across the threads, but destroys spatial locality by scattering contiguous vertices around the threads.
- Block Partitioning (Contiguous): Preserves spatial locality due to contiguous cache access patterns within a thread, but creates high load imbalance if a dense cluster of active packages falls under a single partition.

The two partition strategies produced similar performances, which suggests that the cost of synchronization dominates the execution time regardless of partition quality, which is proved to be true in the Results section. Therefore, we believe there is no significant difference in the trade-off between cache locality and load-balancing.

### K-Core
We implemented the k-core decomposition using a vertex peeling algorithm, which operates by iteratively removing vertices with a degree less than k and updating the degrees of their neighbors until no such vertices remain. Our parallel implementation optimizes this process by eliminating centralized contention during the management of the frontier, or the set of vertices to be removed in the current iteration. Just like in multi-package routing, to ensure efficient concurrent updates, we utilize a vector of unsigned integer flags to track vertex liveness as opposed to a bool vector.

The execution begins with a parallel scan of the graph to compute initial degrees and mark all active vertices. The peeling process then proceeds in parallelized, iterative rounds. In each round, threads first process the current frontier in parallel. We employ a semi-static schedule to effectively load-balance the workload, adapting to the skewed degree distributions common in real-world graphs. For each vertex in the frontier, we mark it as dead and traverse its adjacency list. Then, for every neighbor of a removed vertex that is still alive, we perform an atomic capture to decrement its degree safely. If the degree transitions exactly from k to k-1, the neighbor has just violated the k-core condition and is identified as a candidate for the next frontier.

A critical optimization in our approach is the management of these new candidates. Instead of pushing them to a shared global vector, which would require a lock per insertion, each thread appends candidates to a private local buffer stored in L1 cache. After all threads finish their share of the current frontier, we flatten their buffers into a new global frontier. This gives us a lock-free way of building successive frontiers while ensuring each neighbor only enters the frontier once.

At the end of a round, we merge these private buffers into the global frontier array using a two-step parallel process. First, we compute the starting write position for each thread by calculating the prefix sum of the buffer sizes. If Thread 0 has `N0` items, Thread 1 knows to begin writing at index `N0`, reserving a non-overlapping segment of the global array for each thread. Once these offsets are known, all threads copy their data into their reserved segments simultaneously, avoiding the serialization of a shared atomic counter and allowing memory bandwidth to be fully utilized during the merge.

<div align="center"> ![K-Core](https://github.com/SophiaYXZhu/ParallelRoutingDynamicGraph/blob/main/images/KCore.png) </div>

## Results
Please refer to our final report for experiment results and benchmarks at https://docs.google.com/document/d/1oxg-tKCexYw8OshllP46ZXYg4PcXCuVVvzN8hx8QYuI/. 

In general, we observed good speedup on both GHC and PSC machines up to 8 threads, and concluded the Edge Parallel scheme to be generally more useful than the Graph Partition scheme, espcially in test cases like large_imbalance. We discovered less than theoretical speedup (8x) at around 4.5x to 7.8x speedup on different test cases due to unparallelizable work, such as the precomputation of BFS must happen before simulation. In particular, barriers and implicit waiting are a dominant factor in achieving less than theoretical speedup. For multi-package routing, we also discovered that sensitivity to graph density is higher than sensitivty to the number of packages. 

## References
- Batagelj, V., & Zaversnik, M. (2003). An O(m) Algorithm for Cores Decomposition of Networks. arXiv preprint cs/0310049.
- Valiant, L. G. (1990). A Bridging Model for Parallel Computation. Communications of the ACM. (The foundational reference for the Bulk Synchronous Parallel (BSP) model used in our Partitioned Routing scheme).
- Dagum, L., & Menon, R. (1998). OpenMP: An Industry Standard API for Shared-Memory Programming. IEEE Computational Science and Engineering. (The parallel programming model used for implementation).
- 15-418/618 Course Staff. (2025). Parallel Computer Architecture and Programming. Carnegie Mellon University.

## Documents
Proposal: https://docs.google.com/document/d/1eNZh5MjM0KwcqvB14qrfGHoRefIECy8Bp2h_8brXHOQ/

Milestone Report: https://docs.google.com/document/d/1fJGb3wdToGd9lI4-tq156Pgj-uxjkK_7oeaf3CegNPs/

Final Report: https://docs.google.com/document/d/1oxg-tKCexYw8OshllP46ZXYg4PcXCuVVvzN8hx8QYuI/
