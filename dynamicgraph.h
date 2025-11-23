#include <cstddef>
#include <utility>
#include <vector> 

class DynamicGraph {
    public:
        using Vertex = int;
        using EdgeId = int;
    
        explicit DynamicGraph(std::size_t num_vertices);
    
        void add_edge(Vertex u, Vertex v, int capacity = 1);
        void remove_edge(Vertex u, Vertex v);
        void add_vertex(Vertex v);
        void remove_vertex(Vertex v);
        std::vector<Vertex> neighbors(Vertex v) const;
    
        void snapshot();
    
        std::vector<std::vector<Vertex>>
        min_cost_routing(const std::vector<std::pair<Vertex, Vertex>>& pairs, std::vector<int>* arrival_time, int max_steps = 5) const;

        std::vector<std::vector<Vertex>> 
        min_cost_routing_partitioned(const std::vector<std::pair<Vertex, Vertex>>& pairs, std::vector<int>* arrival_time, int max_steps = 5) const;

        std::vector<std::vector<Vertex>> 
        min_cost_routing_edge_parallel(const std::vector<std::pair<Vertex, Vertex>>& pairs, std::vector<int>* arrival_time, int max_steps = 5) const;

        std::vector<int> k_core(int k = 1) const;
    
    private:
        struct Edge {
            Vertex to;
            EdgeId id;
        };
    
        std::vector<std::vector<Edge>> adj_;
        std::vector<std::vector<Edge>> snapshot_adj_; // snapshot adjacency
        std::vector<bool> active_;
    
        std::vector<int> edge_capacity_; 
    };
    