# ⁠Computational Intelligence - Project Work
## Authors

**This project has been done in collaboration with Riccardo Vaccari (s348856).**

- Luca Lodesani (s346978)
- Riccardo Vaccari (s348856)

## Overview

This project solves a variant of the Traveling Thief Problem for the Computational Intelligence course at Politecnico di Torino. 
A thief must traverse a graph of cities, collecting gold at each city while minimizing travel distance. 
The thief starts and ends at the origin city (0, 0) and may return to unload gold during the route.

## Structure

- `Problem.py` - Problem generator (provided); defines the graph with cities, edges, distances, and gold values.
- `s346978.py` - Main solution entry point implementing the `solution(p: Problem)` function.
- `src/` - Additional source code for the solution.
- `base_requirements.txt` - Required Python dependencies.

## Solution Format

The solution returns an optimal path as:

 ⁠python
[(c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]


where `ci` is the city visited and `gi` is the gold collected at that city.
