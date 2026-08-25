import numpy as np
import networkx as nx

class GraphManager:
    def __init__(self, problem):
        self.problem = problem
        self.graph = problem.graph
        self.alpha = problem.alpha
        self.beta = problem.beta
        self.num_nodes = len(self.graph.nodes)
        
        self.gold_map = nx.get_node_attributes(self.graph, 'gold')
        self.positions = np.array([self.graph.nodes[i]['pos'] for i in range(self.num_nodes)])
        self._path_cache = {}
        
        # Pre-compute shortest paths from the base (node 0)
        self.base_dists, self.base_paths = nx.single_source_dijkstra(
            self.graph, source=0, weight='dist'
        )

        avg_dist = sum(self.base_dists.values()) / max(1, self.num_nodes - 1)
        avg_gold = sum(self.gold_map.values()) / max(1, self.num_nodes - 1)

        # When beta <= 1, there is no exponential penalty for heavy loads
        if self.beta <= 1.0:
            self.global_capacity = float('inf')
        else:
            # Estimate break-even capacity where returning to base is cheaper than carrying extra load
            num = 2.0 * avg_dist
            den = (self.beta - 1.0) * ((self.alpha * avg_dist) ** self.beta)
            if den > 0:
                math_cap = (num / den) ** (1.0 / self.beta)
                # Ensure capacity is at least half the average city gold to avoid micro-trips
                self.global_capacity = max(avg_gold * 0.5, math_cap)
            else:
                self.global_capacity = float('inf')
        

    def get_nearest_neighbors(self, node_id, k=15):
        """Returns the 'k' geometrically closest nodes, excluding base and self."""
        target_pos = self.positions[node_id]
        # Compute Euclidean distances to all nodes in the graph
        distances = np.linalg.norm(self.positions - target_pos, axis=1)

        # Order nodes by distance, from the closest to the farthest
        closest_indices = np.argsort(distances)
        
        neighbors = []
        for idx in closest_indices:
            if idx != node_id and idx != 0:
                neighbors.append(idx)
            if len(neighbors) == k:
                break
        return neighbors

    def get_shortest_path(self, u, v):
        """Lazy A* pathfinding with caching to avoid redundant calculations."""
        # If the start and end nodes are the same, return immediately
        if u == v: 
            return [u], 0.0
        # If either node is the base, return precomputed paths
        if u == 0: 
            return self.base_paths[v], self.base_dists[v]
        if v == 0: 
            return self.base_paths[u][::-1], self.base_dists[u]
        # If the path has been computed before, return it from the cache
        if (u, v) in self._path_cache: 
            return self._path_cache[(u, v)]

        def heuristic(n1, n2):
            return np.linalg.norm(self.positions[n1] - self.positions[n2])
            
        try:
            # Use A* to find the shortest path between nodes u and v
            path = nx.astar_path(self.graph, u, v, heuristic=heuristic, weight='dist')
            # Calculate the total distance of the path
            dist = nx.path_weight(self.graph, path, weight='dist')
        except nx.NetworkXNoPath:
            path, dist = [], float('inf')
            
        self._path_cache[(u, v)] = (path, dist)
        self._path_cache[(v, u)] = (path[::-1], dist)
        return path, dist

    def calculate_path_cost(self, path, current_load):
        """Wrapper for the problem's cost function, ensuring valid paths."""
        if not path or len(path) < 2:
            return 0.0
        
        return sum(
            self.problem.cost([u, v], current_load)
            for u, v in zip(path, path[1:])
        )