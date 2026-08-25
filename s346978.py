from src.graph_manager import GraphManager
from src.ant import TTP_ACO
import math 

def solution(problem):
    """
    Solves the Traveling Thief Problem using Ant Colony Optimization.
    Dynamically adapts parameters based on the graph size to prevent timeouts.
    """
    gm = GraphManager(problem)

    # Dynamically compute ants and iterations based on the number of nodes to stay within a computational budget
    computational_budget = 100_000_000 / max(1, gm.num_nodes ** 2)
    ideal_scale = math.sqrt(computational_budget)
    ants = max(1, min(25, int(ideal_scale)))
    iters = max(1, min(30, int(ideal_scale)))

    aco = TTP_ACO(gm, num_ants=ants, max_iter=iters)
    best_tour = aco.optimize()
    
    # Ensure the tour ends at the base (0, 0) for correct cost evaluation
    if best_tour and best_tour[0][0] == 0:
        best_tour.pop(0)

    if not best_tour or best_tour[-1][0] != 0:
        best_tour.append((0, 0))
      
    return best_tour