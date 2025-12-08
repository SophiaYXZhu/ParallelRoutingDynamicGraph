#include "dynamicgraph.h"

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

using std::cout;
using std::cerr;
using std::endl;

struct Config {
    int  num_threads = 1;
    int  max_steps   = 64;
    int  num_pairs   = 256;
    char mode        = 'o';   // 'o' = original, 'p' = partitioned, 'e' = edge-parallel
};

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--threads") && i + 1 < argc)
            cfg.num_threads = std::stoi(argv[++i]);
        else if ((arg == "-s" || arg == "--steps") && i + 1 < argc)
            cfg.max_steps = std::stoi(argv[++i]);
        else if ((arg == "-p" || arg == "--pairs") && i + 1 < argc)
            cfg.num_pairs = std::stoi(argv[++i]);
        else if ((arg == "-m" || arg == "--mode") && i + 1 < argc)
            cfg.mode = argv[++i][0];
    }
    return cfg;
}

bool edge_exists(const DynamicGraph& g, int u, int v) {
    if (u == v) return true;
    auto neigh = g.neighbors(u);
    for (int x : neigh)
        if (x == v)
            return true;
    return false;
}

// Basic structural check: paths follow edges / stay-put and aren't too long.
bool check_routing_correctness(const DynamicGraph& g,
                               const std::vector<std::pair<int,int>>& pairs,
                               const std::vector<std::vector<int>>& paths,
                               int max_steps) {
    if (paths.size() != pairs.size())
        return false;

    int P = static_cast<int>(pairs.size());
    for (int i = 0; i < P; i++) {
        int s = pairs[i].first;
        int d = pairs[i].second;
        const auto& path = paths[i];

        if (path.empty())
            continue;

        if (path.front() != s) {
            cerr << "Package " << i << ": path does not start at source.\n";
            return false;
        }

        if (static_cast<int>(path.size()) > max_steps + 1) {
            cerr << "Package " << i << ": path too long.\n";
            return false;
        }

        for (std::size_t j = 1; j < path.size(); j++) {
            int u = path[j - 1];
            int v = path[j];
            if (!edge_exists(g, u, v)) {
                cerr << "Package " << i << ": invalid step " << u << " -> " << v << "\n";
                return false;
            }
        }

        (void)d;
    }
    return true;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    if (cfg.num_threads > 0)
        omp_set_num_threads(cfg.num_threads);
    int used_threads = omp_get_max_threads();

    const int N = 8192;
    const int CORE1_START = 0;
    const int CORE1_END   = 4096;
    const int CORE2_START = 4096;
    const int CORE2_END   = 8192;

    cout << "Routing test on large mixed core-periphery graph\n";
    cout << "Using " << used_threads << " thread(s), max_steps = " << cfg.max_steps
         << ", num_pairs = " << cfg.num_pairs << ".\n";

    if (cfg.mode == 'p')
        cout << "Mode: partitioned min_cost_routing_partitioned\n";
    else if (cfg.mode == 'e')
        cout << "Mode: edge-parallel min_cost_routing_edge_parallel\n";
    else
        cout << "Mode: original min_cost_routing\n";

    DynamicGraph g(N);
    std::mt19937 rng(12345);

    std::size_t edge_count = 0;

    // Core 1
    for (int u = CORE1_START; u < CORE1_END; u++) {
        for (int v = u + 1; v < CORE1_END; v++) {
            g.add_edge(u, v);
            edge_count++;
        }
    }

    // Core 2
    for (int u = CORE2_START; u < CORE2_END; u++) {
        for (int v = u + 1; v < CORE2_END; v++) {
            g.add_edge(u, v);
            edge_count++;
        }
    }

    int bridge_width = 4096; 
    for (int i = 0; i < bridge_width; i++) {
        int u = CORE1_START + i;
        int v = CORE2_START + i;
        g.add_edge(u, v); 
        edge_count++;
    }

    // remaining vertex connects to a single random vertex in core 1 or core 2
    std::uniform_int_distribution<int> dist_core(CORE1_START, CORE2_END - 1);

    cout << "Graph generated: N = " << N << ", M ≈ " << edge_count
         << " edges (two fully connected cores + 4096 bridge).\n";

    std::vector<std::pair<int,int>> pairs;
    pairs.reserve(cfg.num_pairs);

    std::uniform_int_distribution<int> dist_any(0, N - 1);

    for (int i = 0; i < cfg.num_pairs; ++i) {
        int s = dist_any(rng);
        int d = dist_any(rng);
        while (d == s)
            d = dist_any(rng);
        pairs.emplace_back(s, d);
    }    

    g.snapshot();

    auto start = std::chrono::steady_clock::now();
    std::vector<int> arrival_time(cfg.num_pairs, -1);

    std::vector<std::vector<int>> paths;
    if (cfg.mode == 'p') {
        paths = g.min_cost_routing_partitioned(pairs, &arrival_time, cfg.max_steps);
    } else if (cfg.mode == 'e') {
        paths = g.min_cost_routing_edge_parallel(pairs, &arrival_time, cfg.max_steps);
    } else {
        paths = g.min_cost_routing(pairs, &arrival_time, cfg.max_steps);
    }

    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    bool ok = check_routing_correctness(g, pairs, paths, cfg.max_steps);

    int delivered   = 0;
    int makespan_ts = 0;   // max arrival_time among delivered packages

    for (std::size_t i = 0; i < paths.size(); i++) {
        if (!paths[i].empty() && paths[i].back() == pairs[i].second) {
            delivered++;
            if (arrival_time[i] > makespan_ts)
                makespan_ts = arrival_time[i];
        }
    }

    cout << "routing: time = " << ms << " ms, "
         << "correct = " << (ok ? "YES" : "NO") << "\n";
    cout << "  delivered " << delivered << " / " << paths.size() << " packages\n";
    cout << "  total timesteps is " << makespan_ts << "\n";

    return ok ? 0 : 1;
}
