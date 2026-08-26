import random

class Ant:
    def __init__(self, graph_manager, aco_alpha=1.0, aco_beta=2.0):
        self.gm = graph_manager
        self.aco_alpha = aco_alpha
        self.aco_beta = aco_beta
        
        self.current_node = 0
        self.current_load = 0.0
        self.total_cost = 0.0
        self.tour = [] 
        self.decision_path = [0]
        
        # Local copy of the gold map for partial pickups
        self.local_gold = self.gm.gold_map.copy()
        
        # Set of unvisited nodes with remaining gold, excluding the base
        self.unvisited = {n for n, g in self.local_gold.items() if n != 0 and g > 0}
        
        # Dynamic capacity to prevent exponential cost explosion on high beta
        self.capacity = self.gm.global_capacity

    def _select_target(self, candidates, heuristic_values, pheromones):
        """Min-max normalize the heuristic values and pick a target via roulette-wheel selection."""
        eta_raw = [heuristic_values[c] for c in candidates]
        eta_min, eta_max = min(eta_raw), max(eta_raw)
        eta_range = eta_max - eta_min
        eta_norm = {
            c: (0.1 + 0.9 * (heuristic_values[c] - eta_min) / eta_range) if eta_range > 1e-9 else 1.0
            for c in candidates
        }

        probabilities = [
            # tau^alpha * eta^beta
            (pheromones.get((self.current_node, c), 1.0) ** self.aco_alpha) * (eta_norm[c] ** self.aco_beta)
            for c in candidates
        ]
        prob_sum = sum(probabilities)
        if prob_sum == 0:
            return random.choice(candidates)
        norm_probs = [p / prob_sum for p in probabilities]
        return random.choices(candidates, weights=norm_probs, k=1)[0]


    def step(self, pheromones):
        """Perform a single step of the ant's movement"""
        # 1. Check available capacity and decide whether to return to base
        available_capacity = self.capacity - self.current_load
        
        if available_capacity <= 1e-4 and self.current_node != 0:
            path, _ = self.gm.get_shortest_path(self.current_node, 0)
            self.total_cost += self.gm.calculate_path_cost(path, self.current_load)
            
            for node in path[1:-1]:
                self.tour.append((node, 0))
            self.tour.append((0, 0))
            
            self.current_load = 0.0
            self.current_node = 0
            self.decision_path.append(0)
            return

        # 2. Evaluate candidate nodes to visit next
        candidates = []
        heuristic_values = {}

        # Cost to return immediately to base (only meaningful when not already there)
        path_u_0, cost_return_now = None, 0.0
        if self.current_node != 0:
            path_u_0, _ = self.gm.get_shortest_path(self.current_node, 0)
            cost_return_now = self.gm.calculate_path_cost(path_u_0, self.current_load)

        neighbors = self.gm.get_nearest_neighbors(self.current_node, k=15)

        # Evaluate each neighbor for potential visit
        for v in neighbors:
            if v not in self.unvisited:
                continue
            gold_to_take = min(self.local_gold[v], available_capacity)

            # If at base, evaluate direct trip to v, with the classic heuristic of gold/distance
            if self.current_node == 0:
                _, dist_0_v = self.gm.get_shortest_path(0, v)
                heuristic_values[v] = gold_to_take / (dist_0_v + 1e-6)
                candidates.append(v)
            else:
                # Evaluate if chaining the trip is cheaper than splitting it via base
                path_u_v, _ = self.gm.get_shortest_path(self.current_node, v)
                path_v_0, _ = self.gm.get_shortest_path(v, 0)
                _, dist_0_v = self.gm.get_shortest_path(0, v)

                # Cost of chaining: current -> v -> base
                cost_chain = self.gm.calculate_path_cost(path_u_v, self.current_load) + \
                             self.gm.calculate_path_cost(path_v_0, self.current_load + gold_to_take)

                # Cost of splitting: current -> base, then base -> v -> base   
                cost_split = cost_return_now + dist_0_v + \
                             self.gm.calculate_path_cost(path_v_0, gold_to_take)

                savings = cost_split - cost_chain
                # Savings > 0 means visiting 'v' now is cheaper than an extra dedicated trip from base
                if savings > 0:
                    heuristic_values[v] = savings
                    candidates.append(v)

        # 3. Target choice: base and non-base cases 
        if self.current_node == 0:
            # Consider a few unvisited nodes to avoid deadlock
            if not candidates:  
                for v in list(self.unvisited)[:5]:
                    candidates.append(v)
                    _, dist_0_v = self.gm.get_shortest_path(0, v)
                    heuristic_values[v] = min(self.local_gold[v], available_capacity) / (dist_0_v + 1e-6)
            chosen_target = self._select_target(candidates, heuristic_values, pheromones)
        else:
            if not candidates:
                chosen_target = 0
            else:
                # Return to base is always a candidate
                candidates.append(0)
                # Heuristic for returning to base is based on the current load and the cost to return
                beta_p, alpha_p = self.gm.beta, self.gm.alpha
                if self.current_load > 0 and beta_p > 1:
                    # Higher current_load increases the urgency to return to base and unloa
                    marginal_rate = beta_p * alpha_p * self.current_load * (alpha_p * self.current_load) ** (beta_p - 1)
                else:
                    marginal_rate = 0.0
                heuristic_values[0] = max(marginal_rate, 1e-6)
                chosen_target = self._select_target(candidates, heuristic_values, pheromones)

        # 4. Move execution
        if chosen_target == 0:
            path, edge_cost = path_u_0, cost_return_now
        else:
            path, _ = self.gm.get_shortest_path(self.current_node, chosen_target)
            edge_cost = self.gm.calculate_path_cost(path, self.current_load)
        self.total_cost += edge_cost

        for node in path[1:-1]:
            self.tour.append((node, 0))

        if chosen_target == 0:
            self.tour.append((0, 0))
            self.current_load = 0.0
        else:
            # Pick up gold at the chosen target, respecting available capacity
            gold_collected = min(self.local_gold[chosen_target], available_capacity)
            self.tour.append((chosen_target, gold_collected))
            self.current_load += gold_collected

            # Update remaining gold at the target and remove it from unvisited if empty
            self.local_gold[chosen_target] -= gold_collected
            if self.local_gold[chosen_target] < 1e-4:
                self.unvisited.remove(chosen_target)

        self.current_node = chosen_target
        self.decision_path.append(chosen_target)
        
    def run_tour(self, pheromones):
        """Execute a complete tour until all nodes are visited"""
        while self.unvisited:
            self.step(pheromones)

        # Ensure the ant returns to base at the end of its tour
        if self.current_node != 0:
            path, _ = self.gm.get_shortest_path(self.current_node, 0)
            self.total_cost += self.gm.calculate_path_cost(path, self.current_load)
            for node in path[1:]:
                self.tour.append((node, 0))
            self.decision_path.append(0) 


class TTP_ACO:
    def __init__(
        self,
        graph_manager,
        num_ants=15,
        max_iter=20,
        rho=0.15,
        aco_alpha=1.0,      
        aco_beta=2.0,
        elite_k=3,          
        tau_min=0.1,        
        tau_max=10.0,
    ):
        self.gm = graph_manager
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.rho = rho
        self.aco_alpha = aco_alpha
        self.aco_beta = aco_beta
        self.elite_k = elite_k
        self.tau_min = tau_min
        self.tau_max = tau_max

        self.pheromones = {}
        self.best_tour = None
        self.best_cost = float('inf')

    def optimize(self):
        """Run the Ant Colony Optimization algorithm to find the best tour"""
        for _ in range(self.max_iter):
            ants = [
                Ant(self.gm, aco_alpha=self.aco_alpha, aco_beta=self.aco_beta)
                for _ in range(self.num_ants)
            ]
            for ant in ants:
                ant.run_tour(self.pheromones)

            ranked = sorted(ants, key=lambda a: a.total_cost)
            iteration_best_ant = ranked[0]

            if iteration_best_ant.total_cost < self.best_cost:
                self.best_cost = iteration_best_ant.total_cost
                self.best_tour = iteration_best_ant.tour

            avg_cost = sum(a.total_cost for a in ants) / len(ants)

            # Evaporation, clipped to [tau_min, tau_max]
            for edge in self.pheromones:
                self.pheromones[edge] = max(
                    self.tau_min, self.pheromones[edge] * (1.0 - self.rho)
                )

            # Multi-ant deposit: top-k ants reinforce, weighted by rank so the
            # best still contributes most but a single lucky/unlucky ant can't
            # dominate the whole pheromone map on its own.
            elite = ranked[: self.elite_k]
            for rank, ant in enumerate(elite):
                deposit = (avg_cost / (ant.total_cost + 1e-9)) / (rank + 1)
                nodes = ant.decision_path
                for i in range(len(nodes) - 1):
                    u, v = nodes[i], nodes[i + 1]
                    if u != v:
                        new_val = self.pheromones.get((u, v), 1.0) + deposit
                        self.pheromones[(u, v)] = min(self.tau_max, new_val)

        return self.best_tour