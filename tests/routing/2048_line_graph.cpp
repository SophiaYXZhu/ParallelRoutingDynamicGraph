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
    int num_threads = 1;
    int max_steps = 64;
    int num_pairs = 256;
    char mode = 'o';
};

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i=1; i < argc; i++) {
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
    if (u == v) 
        return true;
    auto neigh = g.neighbors(u);
    for (int x : neigh)
        if (x == v) 
            return true;
    return false;
}


bool check_routing_correctness(const DynamicGraph& g, const std::vector<std::pair<int,int>>& pairs, const std::vector<std::vector<int>>& paths, int max_steps) {
    if (paths.size() != pairs.size()) 
        return false;

    int P = static_cast<int>(pairs.size());
    for (int i=0; i < P; i++) {
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

        for (std::size_t j=1; j < path.size(); j++) {
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

    const int N = 2048;
    const int P = cfg.num_pairs;
    const int THEORETICAL_OPT = N + P - 2;

    cout << "Routing test on 2048-vertex line graph\n";
    cout << "Using " << used_threads << " threads, max_steps = " << cfg.max_steps << ", num_pairs = " << P << ".\n";
    if (cfg.mode == 'p')
        cout << "Mode: partitioned min_cost_routing_partitioned\n";
    else if (cfg.mode == 'e')
        cout << "Mode: edge-parallel min_cost_routing_edge_parallel\n";
    else
        cout << "Mode: original min_cost_routing\n";
        
    cout << "Theoretical optimal makespan is N + P - 2 = " << THEORETICAL_OPT << " time steps\n";

    if (cfg.max_steps < THEORETICAL_OPT)
        cout << "Warning: max_steps < theoretical optimum. Some packages cannot possibly be delivered in simulation.\n";

    DynamicGraph g(N);
    for (int u=0; u < N-1; u++)
        g.add_edge(u, u+1);

    // all packages from 0 to N-1
    std::vector<std::pair<int,int>> pairs;
    pairs.reserve(P);
    for (int i=0; i < P; i++)
        pairs.emplace_back(0, N-1);
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

    int delivered = 0;
    int makespan = -1;
    for (std::size_t i=0; i < paths.size(); i++) {
        const auto& path = paths[i];
        if (!path.empty() && path.back() == pairs[i].second) {
            delivered++;
            int cost_i = arrival_time[i];
            if (cost_i > makespan) 
                makespan = cost_i;
        }
    }

    cout << "routing: time = " << ms << " ms, " << "correct = " << (ok ? "YES" : "NO") << "\n";
    cout << "  delivered " << delivered << " / " << paths.size() << " packages\n";

    if (delivered > 0) {
        cout << "  actual routing cost (makespan over delivered packages) = " << makespan << " time steps\n";
        cout << "  theoretical optimal makespan (capacity-1 edges) = " << THEORETICAL_OPT << " time steps\n";

        if (makespan >= 0)
            cout << "  cost ratio (actual / optimal) = " << static_cast<double>(makespan) / THEORETICAL_OPT << "\n";
    } 
    else
        cout << "  routing cost: no packages delivered (makespan undefined)\n";

    return ok ? 0 : 1;
}
