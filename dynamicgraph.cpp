#include "dynamicgraph.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include <omp.h>


DynamicGraph::DynamicGraph(std::size_t num_vertices)
    : adj_(num_vertices),
      snapshot_adj_(),
      active_(num_vertices, true) {}

static inline void 
check_vertex(const DynamicGraph::Vertex v, std::size_t n) {
    assert(v >= 0 && static_cast<std::size_t>(v) < n && "Vertex ID out of range.");
}

// Dynamic Updates

void DynamicGraph::add_vertex(Vertex v) {
    if (v < 0) 
        return;
    std::size_t id = static_cast<std::size_t>(v);
    if (id >= adj_.size()) {
        adj_.resize(id+1);
        snapshot_adj_.resize(id+1);
        active_.resize(id + 1, false);
    }
    active_[id] = true;
}

void DynamicGraph::remove_vertex(Vertex v) {
    check_vertex(v, adj_.size());
    active_[v] = false;
}

void DynamicGraph::add_edge(Vertex u, Vertex v, int capacity) {
    std::size_t n = adj_.size();
    check_vertex(u, n);
    check_vertex(v, n);
    if (!active_[u] || !active_[v]) return;

    EdgeId id_uv = static_cast<EdgeId>(edge_capacity_.size());
    edge_capacity_.push_back(capacity);

    Edge e1{v, id_uv};
    Edge e2{u, id_uv};

    adj_[u].push_back(e1);
    adj_[v].push_back(e2);
}

void DynamicGraph::remove_edge(Vertex u, Vertex v) {
    std::size_t n = adj_.size();
    check_vertex(u, n);
    check_vertex(v, n);

    auto &nu = adj_[u];
    auto &nv = adj_[v];

    nu.erase(
        std::remove_if(nu.begin(), nu.end(), [v](const Edge& e) {
            return e.to == v;
        }), nu.end()
    );
    nv.erase(
        std::remove_if(nv.begin(), nv.end(), [u](const Edge& e) {
            return e.to == u;
        }), nv.end()
    );
}


std::vector<DynamicGraph::Vertex> DynamicGraph::neighbors(Vertex v) const {
    const auto& G = snapshot_adj_.empty() ? adj_ : snapshot_adj_;
    check_vertex(v, G.size());

    std::vector<Vertex> out;
    out.reserve(G[v].size());
    for (const Edge& e : G[v]) {
        if (active_[e.to])
            out.push_back(e.to);
    }
    return out;
}


// Snapshots
void DynamicGraph::snapshot() {
    snapshot_adj_ = adj_;

    for (std::size_t v=0; v < snapshot_adj_.size(); v++) {
        if (!active_[v])
            snapshot_adj_[v].clear();
    }
}

static inline std::uint64_t pack_edge_key(int u, int v) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32) | static_cast<std::uint32_t>(v);
}

// Multiple Packages Routing Simulation (parallel via dynamic scheduling)
std::vector<std::vector<DynamicGraph::Vertex>>
DynamicGraph::min_cost_routing(const std::vector<std::pair<Vertex, Vertex>>& pairs, std::vector<int>* arrival_time, int max_steps) const 
{
    const auto &G = snapshot_adj_.empty() ? adj_ : snapshot_adj_;
    std::size_t n = G.size();
    const int INF = std::numeric_limits<int>::max() / 4;

    int P = static_cast<int>(pairs.size());
    if (P == 0)
        return {};

    // group by destination and build dist_to_dest[d][v] via BFS
    std::unordered_map<Vertex, int> dest_index;
    std::vector<Vertex> dest_list;
    dest_list.reserve(P);

    for (const auto &p : pairs) {
        Vertex d = p.second;
        if (d < 0 || static_cast<std::size_t>(d) >= n || !active_[d])
            continue;
        if (dest_index.find(d) == dest_index.end()) {
            int idx = static_cast<int>(dest_list.size());
            dest_index[d] = idx;
            dest_list.push_back(d);
        }
    }

    int D = static_cast<int>(dest_list.size());
    std::vector<std::vector<int>> dist_to_dest(D, std::vector<int>(n, INF));

    // parallel over destinations as each BFS is independent
    #pragma omp parallel for schedule(dynamic)
    for (int k=0; k < D; k++) {
        Vertex dest = dest_list[k];
        auto &dist = dist_to_dest[k];
        std::queue<Vertex> q;

        dist[dest] = 0;
        q.push(dest);

        while (!q.empty()) {
            Vertex u = q.front();
            q.pop();

            int du = dist[u];
            for (const Edge &e : G[u]) {
                Vertex v = e.to;
                if (!active_[v]) 
                    continue;
                if (dist[v] == INF) {
                    dist[v] = du+1;
                    q.push(v);
                }
            }
        }
    }

    auto get_dist = [&](Vertex v, Vertex d) -> int {
        auto it = dest_index.find(d);
        if (it == dest_index.end())
            return INF;
        int idx = it->second;
        return dist_to_dest[idx][v];
    };

    // initialize package state
    std::vector<Vertex> pos(P); // current position
    std::vector<Vertex> dst(P); // destination
    std::vector<bool> done(P, false);
    std::vector<std::vector<Vertex>> paths(P);

    for (int i=0; i < P; i++) {
        Vertex s = pairs[i].first;
        Vertex d = pairs[i].second;

        if (s < 0 || d < 0 || static_cast<std::size_t>(s) >= n || static_cast<std::size_t>(d) >= n || !active_[s] || !active_[d]) {
            done[i] = true;
            (*arrival_time)[i] = -1;
            continue;
        }

        pos[i] = s;
        dst[i] = d;
        paths[i].push_back(s);
        if (s == d) {
            done[i] = true;
            (*arrival_time)[i] = 0;
        }
    }

    int undelivered = 0;
    for (int i=0; i < P; i++) {
        if (!done[i]) 
            undelivered++;
    }

    if (undelivered == 0)
        return paths;

    std::vector<Vertex> next_pos(P);

    // simulation
    for (int t=0; t < max_steps && undelivered > 0; t++) {
        std::vector<Vertex> desired_next(P, -1);

        // each package chooses neighbor that decreases dist to dest in parallel
        #pragma omp parallel for schedule(static)
        for (int i=0; i < P; i++) {
            if (done[i]) {
                desired_next[i] = pos[i];
                continue;
            }

            Vertex u = pos[i];
            Vertex d = dst[i];

            int best_dist = get_dist(u, d);
            Vertex best_v = u;

            if (best_dist == INF) {
                desired_next[i] = u;
                continue;
            }

            for (const Edge &e : G[u]) {
                Vertex v = e.to;
                if (!active_[v]) 
                    continue;
                int dv = get_dist(v, d);
                if (dv < best_dist) {
                    best_dist = dv;
                    best_v = v;
                }
            }
            desired_next[i] = best_v;
        }

        // edge capacity constraint
        std::unordered_map<std::uint64_t, int> edge_owner;
        edge_owner.reserve(2*P);
        std::vector<bool> can_move(P, false);

        for (int i=0; i < P; i++) {
            if (done[i]) 
                continue;

            Vertex u = pos[i];
            Vertex v = desired_next[i];

            if (v == u || v < 0) 
                continue;

            std::uint64_t key = pack_edge_key(u, v);
            auto it = edge_owner.find(key);
            if (it == edge_owner.end()) {
                edge_owner.emplace(key, i);
                can_move[i] = true;
            } 
            else
                can_move[i] = false;
        }

        // commit moves and update state in parallel
        int new_undelivered = 0;
        int max_threads = omp_get_max_threads();
        std::vector<int> local_counts(max_threads, 0);

        #pragma omp parallel for schedule(static)
        for (int i=0; i < P; i++) {
            int tid = omp_get_thread_num();

            if (done[i]) {
                next_pos[i] = pos[i];
                continue;
            }

            if (can_move[i]) {
                next_pos[i] = desired_next[i];
                if (paths[i].empty() || paths[i].back() != next_pos[i])
                    paths[i].push_back(next_pos[i]);
            } 
            else
                next_pos[i] = pos[i];

            if (next_pos[i] == dst[i]) {
                done[i] = true;
                (*arrival_time)[i] = t+1;
            }

            if (!done[i])
                local_counts[tid]++;
        }

        for (int t=0; t < max_threads; t++)
            new_undelivered += local_counts[t];

        undelivered = new_undelivered;
        pos.swap(next_pos);
    }
    return paths;
}

// K-cores (parallel with static scheduling)
std::vector<int> DynamicGraph::k_core(int k) const {
    const auto &G = snapshot_adj_.empty() ? adj_ : snapshot_adj_;
    std::size_t n = G.size();
    std::vector<int> degree(n, 0);
    std::vector<bool> alive(n, false);

    // degree initialization
    #pragma omp parallel for
    for (long long v=0; v < (long long)n; v++) {
        if (!active_[v]) {
            degree[v] = 0;
            alive[v] = false;
            continue;
        }

        alive[v] = true;
        int d = 0;
        for (const Edge &e : G[v]) {
            Vertex u = e.to;
            if (active_[u])
                d++;
        }
        degree[v] = d;
    }

    // frontier initialization
    std::vector<Vertex> frontier;
    frontier.reserve(n);

    for (Vertex v = 0; v < (Vertex)n; v++) {
        if (alive[v] && degree[v] < k)
            frontier.push_back(v);
    }

    // iteratively decrement frontier degrees
    while (!frontier.empty()) {
        std::vector<Vertex> next_frontier;
        next_frontier.reserve(frontier.size());

        #pragma omp parallel
        {
            std::vector<Vertex> local_next;

            #pragma omp for nowait
            for (long long i=0; i < (long long)frontier.size(); i++) {
                Vertex v = frontier[i];
                if (!alive[v]) 
                    continue;
                alive[v] = false;

                // decrement degree of neighbors
                for (const Edge &e : G[v]) {
                    Vertex u = e.to;
                    if (!alive[u]) 
                        continue;

                    int newdeg;
                    #pragma omp atomic capture
                    newdeg = --degree[u];

                    if (newdeg == k-1)
                        local_next.push_back(u);
                }
            }

            #pragma omp critical
            next_frontier.insert(next_frontier.end(), local_next.begin(), local_next.end());
        }

        frontier.swap(next_frontier);
    }

    // collect remaining vertices
    std::vector<int> result;
    for (Vertex v=0; v < (Vertex)n; v++) {
        if (alive[v]) 
            result.push_back(v);
    }

    return result;
}
