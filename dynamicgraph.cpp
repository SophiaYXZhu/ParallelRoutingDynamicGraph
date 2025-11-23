#include "dynamicgraph.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>
#include <atomic>

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

std::vector<std::vector<DynamicGraph::Vertex>>
DynamicGraph::min_cost_routing_partitioned( const std::vector<std::pair<Vertex, Vertex>>& pairs, std::vector<int>* arrival_time, int max_steps) const
{
    const auto &G = snapshot_adj_.empty() ? adj_ : snapshot_adj_;
    std::size_t n = G.size();
    const int INF = std::numeric_limits<int>::max() / 4;

    int P = static_cast<int>(pairs.size());
    if (P == 0)
        return {};

    //group by destination and build dist_to_dest[d][v] via BFS
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

    // Parallel over distinct destinations
    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < D; k++) {
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
                    dist[v] = du + 1;
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

    //initialize package state
    std::vector<Vertex> pos(P);
    std::vector<Vertex> dst(P);
    std::vector<bool> done(P, false);
    std::vector<std::vector<Vertex>> paths(P);

    for (int i = 0; i < P; i++) {
        Vertex s = pairs[i].first;
        Vertex d = pairs[i].second;

        if (s < 0 || d < 0 || static_cast<std::size_t>(s) >= n || static_cast<std::size_t>(d) >= n || !active_[s] || !active_[d]) {
            done[i] = true;
            if (arrival_time)
                (*arrival_time)[i] = -1;
            continue;
        }

        pos[i] = s;
        dst[i] = d;
        paths[i].push_back(s);

        if (s == d) {
            done[i] = true;
            if (arrival_time)
                (*arrival_time)[i] = 0;
        }
    }

    int undelivered = 0;
    for (int i = 0; i < P; i++) {
        if (!done[i])
            undelivered++;
    }

    if (undelivered == 0)
        return paths;

    std::vector<Vertex> next_pos(P);

    //partition vertices into subgraphs
    int T = omp_get_max_threads();
    if (T <= 0) T = 1;

    std::vector<Vertex> part_start(T + 1);
    for (int t = 0; t <= T; t++) {
        part_start[t] = static_cast<Vertex>((n * 1LL * t) / T);
    }

    auto vertex_partition = [&](Vertex v) -> int {
        // binary search over part_start to find partition owning v
        int lo = 0, hi = T;
        while (lo + 1 < hi) {
            int mid = (lo + hi) / 2;
            if (v < part_start[mid]) hi = mid;
            else lo = mid;
        }
        return lo;
    };

    // Assign initial packets to partitions by their starting positions
    std::vector<std::vector<int>> local_pkgs(T);
    for (int i = 0; i < P; i++) {
        if (done[i]) continue;
        int p_id = vertex_partition(pos[i]);
        local_pkgs[p_id].push_back(i);
    }

    std::vector<std::vector<std::vector<int>>> thread_outgoing(
        T, std::vector<std::vector<int>>(T)
    );

    //simulation loop
    for (int t_step = 0; t_step < max_steps && undelivered > 0; t_step++) {

        // Clear outgoing buffers
        for (int tid = 0; tid < T; tid++) {
            for (int p = 0; p < T; p++) {
                thread_outgoing[tid][p].clear();
            }
        }

        // Parallel region: each thread works on its own partition
        #pragma omp parallel num_threads(T)
        {
            int tid = omp_get_thread_num();
            auto &pkgs = local_pkgs[tid];

            // Local edge-capacity map for this partition
            std::unordered_map<std::uint64_t, int> edge_owner;
            edge_owner.reserve(2 * pkgs.size());

            std::vector<bool>   local_can_move(pkgs.size(), false);
            std::vector<Vertex> desired_next_local(pkgs.size(), -1);

            //decide desired next hop for local packages
            for (std::size_t idx = 0; idx < pkgs.size(); idx++) {
                int i = pkgs[idx];
                if (done[i]) {
                    desired_next_local[idx] = pos[i];
                    continue;
                }

                Vertex u = pos[i];
                Vertex d = dst[i];

                int best_dist = get_dist(u, d);
                Vertex best_v = u;

                if (best_dist == INF) {
                    desired_next_local[idx] = u;
                    continue;
                }

                for (const Edge &e : G[u]) {
                    Vertex v = e.to;
                    if (!active_[v]) continue;
                    int dv = get_dist(v, d);
                    if (dv < best_dist) {
                        best_dist = dv;
                        best_v = v;
                    }
                }

                desired_next_local[idx] = best_v;
            }

            // capacity resolution for edges whose source is in this partition
            for (std::size_t idx = 0; idx < pkgs.size(); idx++) {
                int i = pkgs[idx];
                if (done[i]) continue;

                Vertex u = pos[i];
                Vertex v = desired_next_local[idx];

                if (v == u || v < 0) continue;

                std::uint64_t key = pack_edge_key(u, v);
                auto it = edge_owner.find(key);
                if (it == edge_owner.end()) {
                    edge_owner.emplace(key, i);
                    local_can_move[idx] = true;
                } else {
                    local_can_move[idx] = false;
                }
            }

            // commit move, update arrival times, and enqueue packets for next partitions
            for (std::size_t idx = 0; idx < pkgs.size(); idx++) {
                int i = pkgs[idx];

                if (done[i]) {
                    next_pos[i] = pos[i];
                    continue;
                }

                if (local_can_move[idx]) {
                    next_pos[i] = desired_next_local[idx];
                    if (paths[i].empty() || paths[i].back() != next_pos[i]) {
                        paths[i].push_back(next_pos[i]);
                    }
                } else {
                    next_pos[i] = pos[i];
                }

                if (next_pos[i] == dst[i]) {
                    done[i] = true;
                    if (arrival_time)
                        (*arrival_time)[i] = t_step + 1;
                }

                // If still not done, assign to the partition that owns its new vertex
                if (!done[i]) {
                    int next_part = vertex_partition(next_pos[i]);
                    thread_outgoing[tid][next_part].push_back(i);
                }
            }
        } // end parallel region

        // Swap positions for next timestep
        pos.swap(next_pos);

        // Rebuild local_pkgs from thread_outgoing
        for (int p = 0; p < T; p++) {
            local_pkgs[p].clear();
        }
        for (int tid = 0; tid < T; tid++) {
            for (int p = 0; p < T; p++) {
                auto &buf = thread_outgoing[tid][p];
                if (!buf.empty()) {
                    local_pkgs[p].insert(local_pkgs[p].end(), buf.begin(), buf.end());
                }
            }
        }

        // Recompute undelivered
        undelivered = 0;
        for (int i = 0; i < P; i++) {
            if (!done[i]) undelivered++;
        }
    }

    return paths;
}

std::vector<std::vector<DynamicGraph::Vertex>>
DynamicGraph::min_cost_routing_edge_parallel(const std::vector<std::pair<Vertex, Vertex>>& pairs, std::vector<int>* arrival_time, int max_steps) const 
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

    // per-directed-edge owner slots, 2 slots per undirected edge
    std::size_t E = edge_capacity_.size();
    std::vector<std::atomic<int>> edge_owner(2 * E);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)edge_owner.size(); i++) {
        edge_owner[i].store(-1, std::memory_order_relaxed);
    }

    auto directed_slot = [](EdgeId eid, bool forward) -> int {
        return static_cast<int>(2 * static_cast<std::size_t>(eid) + (forward ? 0 : 1));
    };


    // simulation
    for (int t=0; t < max_steps && undelivered > 0; t++) {
        std::vector<Vertex> desired_next(P, -1);
        std::vector<EdgeId> desired_eid(P, -1);

        // each package chooses neighbor that decreases dist to dest in parallel
        #pragma omp parallel for schedule(static)
        for (int i=0; i < P; i++) {
            if (done[i]) {
                desired_next[i] = pos[i];
                desired_eid[i]  = -1;
                continue;
            }

            Vertex u = pos[i];
            Vertex d = dst[i];

            int best_dist = get_dist(u, d);
            Vertex best_v = u;
            EdgeId best_edge = -1;

            if (best_dist != INF) {
                for (const Edge &e : G[u]) {
                    Vertex v = e.to;
                    if (!active_[v]) continue;
                    int dv = get_dist(v, d);
                    if (dv < best_dist) {
                        best_dist = dv;
                        best_v = v;
                        best_edge = e.id;
                    }
                }
            }

            desired_next[i] = best_v;
            if (best_v != u && best_edge >= 0)
                desired_eid[i] = best_edge;
            else
                desired_eid[i] = -1;
        }

        // edge capacity constraint
        std::vector<char> can_move(P, 0);
        int max_threads = omp_get_max_threads();
        std::vector<std::vector<int>> used_slots_local(max_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto &local_used = used_slots_local[tid];

            #pragma omp for schedule(static)
            for (int i = 0; i < P; i++) {
                if (done[i])
                    continue;

                EdgeId eid = desired_eid[i];
                if (eid < 0)
                    continue;

                Vertex u = pos[i];
                Vertex v = desired_next[i];

                bool forward = (u < v);
                int slot = directed_slot(eid, forward);

                int expected = -1;
                if (edge_owner[slot].compare_exchange_strong(
                        expected, i, std::memory_order_relaxed)) {
                    can_move[i] = 1;
                    local_used.push_back(slot);
                } else {
                    can_move[i] = 0;
                }
            }
        }

        // Reset only the slots that were used this timestep
        std::vector<int> used_slots;
        used_slots.reserve(P);
        for (int tid = 0; tid < max_threads; tid++) {
            auto &vec = used_slots_local[tid];
            used_slots.insert(used_slots.end(), vec.begin(), vec.end());
        }

        #pragma omp parallel for schedule(static)
        for (long long k = 0; k < (long long)used_slots.size(); k++) {
            int slot = used_slots[k];
            edge_owner[slot].store(-1, std::memory_order_relaxed);
        }

        // commit moves and update state in parallel
        int new_undelivered = 0;
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
