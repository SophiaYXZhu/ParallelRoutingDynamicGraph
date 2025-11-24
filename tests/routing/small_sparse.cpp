#include "dynamicgraph.h"

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <queue>
#include <unordered_set>
#include <cstdint>

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


// compute exact bound
using State = std::vector<int>;

struct StateHasher {
    std::size_t operator()(State const& s) const noexcept {
        std::size_t h = 0;
        for (int v : s) {
            h = h * 1315423911u + std::hash<int>()(v);
        }
        return h;
    }
};

static inline std::uint64_t pack_edge_key(int u, int v) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32) | static_cast<std::uint32_t>(v);
}

static void enumerate_moves(int idx,
                            const State& cur,
                            State& next,
                            const std::vector<std::pair<int,int>>& pairs,
                            const DynamicGraph& g,
                            std::unordered_set<std::uint64_t>& used_edges,
                            std::vector<State>& out_states) {
    int P = static_cast<int>(cur.size());
    if (idx == P) {
        out_states.push_back(next);
        return;
    }

    int u = cur[idx];
    int dest = pairs[idx].second;

    if (u == dest) {
        next[idx] = u;
        enumerate_moves(idx + 1, cur, next, pairs, g, used_edges, out_states);
        return;
    }

    next[idx] = u;
    enumerate_moves(idx+1, cur, next, pairs, g, used_edges, out_states);

    auto neigh = g.neighbors(u);
    for (int v : neigh) {
        if (v == u) 
            continue;

        std::uint64_t key = pack_edge_key(u, v);
        if (used_edges.find(key) != used_edges.end())
            continue; 

        used_edges.insert(key);
        next[idx] = v;
        enumerate_moves(idx+1, cur, next, pairs, g, used_edges, out_states);
        used_edges.erase(key);
    }
}

int compute_true_optimal_makespan_exact(const DynamicGraph& g, const std::vector<std::pair<int,int>>& pairs, int max_steps_limit) {
    int P = static_cast<int>(pairs.size());
    if (P == 0) 
        return 0;

    State start(P);
    for (int i=0; i < P; i++)
        start[i] = pairs[i].first;

    auto is_goal = [&](const State& s) {
        for (int i=0; i < P; i++) {
            if (s[i] != pairs[i].second) 
                return false;
        }
        return true;
    };

    if (is_goal(start)) 
        return 0;

    std::unordered_set<State, StateHasher> visited;
    std::queue<std::pair<State,int>> q; // (state, time)

    visited.insert(start);
    q.emplace(start, 0);

    while (!q.empty()) {
        auto [cur, t] = q.front();
        q.pop();

        if (t >= max_steps_limit)
            continue;

        State next(P);
        std::unordered_set<std::uint64_t> used_edges;
        std::vector<State> next_states;

        enumerate_moves(0, cur, next, pairs, g, used_edges, next_states);

        for (auto& ns : next_states) {
            if (visited.find(ns) != visited.end()) 
                continue;
            int nt = t+1;
            if (is_goal(ns))
                return nt;
            visited.insert(ns);
            q.emplace(std::move(ns), nt);
        }
    }
    return -1;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    if (cfg.num_threads > 0)
        omp_set_num_threads(cfg.num_threads);
    int used_threads = omp_get_max_threads();

    const int N = 10;
    const double p = 1.0 / 3.0;

    cout << "Routing test on 2048-vertex sparse graph (p = " << p << ")\n";
    cout << "Using " << used_threads << " thread(s), max_steps = " << cfg.max_steps << ", num_pairs = " << cfg.num_pairs << ".\n";

    if (cfg.mode == 'p')
        cout << "Mode: partitioned min_cost_routing_partitioned\n";
    else if (cfg.mode == 'e')
        cout << "Mode: edge-parallel min_cost_routing_edge_parallel\n";
    else
        cout << "Mode: original min_cost_routing\n";
    // build random graph
    DynamicGraph g(N);
    std::mt19937 rng(23345);
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

    // random (s, d) pairs
    std::vector<std::pair<int,int>> pairs;
    pairs.reserve(cfg.num_pairs);

    std::uniform_int_distribution<int> dist_v(0, N-1);
    for (int i=0; i < cfg.num_pairs; i++) {
        int s = dist_v(rng);
        int d = dist_v(rng);
        if (s == d)
            d = (d+1) % N;
        pairs.emplace_back(s, d);
    }

    int theoretical_lb = -1;
    if (cfg.num_pairs <= 5 && N <= 10) {
        theoretical_lb = compute_true_optimal_makespan_exact(g, pairs, cfg.max_steps);
        cout << "theoretical optimal lower bound in " << theoretical_lb << " time steps\n";
    }
    else
        cout << "theoretical exact lower bound skipped because instance is too large for brute-force search\n";

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
        if (!paths[i].empty() && paths[i].back() == pairs[i].second &&
            arrival_time[i] >= 0) {
            delivered++;
            if (arrival_time[i] > makespan)
                makespan = arrival_time[i];
        }
    }

    cout << "routing: time = " << ms << " ms, " << "correct = " << (ok ? "YES" : "NO") << "\n";
    cout << "  delivered " << delivered << " / " << paths.size() << " packages\n";
    cout << "  actual routing makespan = " << makespan << " time steps\n";

    if (theoretical_lb >= 0 && delivered == cfg.num_pairs)
        cout << "  cost ratio (actual / exact optimal) = " << static_cast<double>(makespan) / theoretical_lb << "\n";

    return ok ? 0 : 1;
}
