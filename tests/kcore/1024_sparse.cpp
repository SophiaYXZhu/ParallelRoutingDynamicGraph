#include "dynamicgraph.h"

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include <omp.h>

using std::cout;
using std::cerr;
using std::endl;

struct Config {
    int num_threads = -1;
    int k = 1000;
};

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i=1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--threads") && i + 1 < argc)
            cfg.num_threads = std::stoi(argv[++i]);
        else if ((arg == "-k" || arg == "--kcore") && i + 1 < argc)
            cfg.k = std::stoi(argv[++i]);
    }
    return cfg;
}

bool check_k_core_correctness(const DynamicGraph& g, int k, const std::vector<int>& core_vertices) {
    if (core_vertices.empty())
        return true;

    std::unordered_set<int> core_set;
    core_set.reserve(2*core_vertices.size());
    for (int v : core_vertices)
        core_set.insert(v);

    for (int v : core_vertices) {
        auto neigh = g.neighbors(v);
        int count = 0;
        for (int u : neigh) {
            if (core_set.count(u) != 0)
                count++;
        }
        if (count < k)
            return false;
    }
    return true;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    if (cfg.num_threads > 0) {
        omp_set_num_threads(cfg.num_threads);
    }
    int used_threads = omp_get_max_threads();

    const int N = 1024;
    const double p = 1.0 / 3.0;   // edge probability

    cout << "k-core test on 1024-vertex sparse graph (p = " << p << ")\n";
    cout << "Using " << used_threads << " thread(s), k = " << cfg.k << ".\n";

    // build random graph
    DynamicGraph g(N);
    std::mt19937 rng(12345);
    std::bernoulli_distribution bern(p);

    std::size_t edge_count = 0;
    for (int u=0; u < N; u++) {
        for (int v=u+1; v < N; v++) {
            if (bern(rng)) {
                g.add_edge(u, v);
                edge_count++;
            }
        }
    }

    cout << "Graph generated: N = " << N << ", M ≈ " << edge_count << " edges.\n";

    g.snapshot();

    auto start = std::chrono::steady_clock::now();
    std::vector<int> kcore_vertices = g.k_core(cfg.k);
    auto end = std::chrono::steady_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    bool ok = check_k_core_correctness(g, cfg.k, kcore_vertices);

    cout << "k_core(" << cfg.k << "): time = " << ms << " ms, "
         << "size = " << kcore_vertices.size() << ", "
         << "correct = " << (ok ? "YES" : "NO") << "\n";

    return ok ? 0 : 1;
}
