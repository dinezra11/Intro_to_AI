import heapq
import numpy as np

# Admissible optimistic heuristic for Part 2
# state: search state object (created and managed by the search agents of part 2)
# env: reference to environment
def heuristic(state, env):

    if all(count == 0 for count in state.remaining_people):    # if there are no people to rescue in any vertex - return 0 and agent will know that reached goal state
        return 0.0

    best = float('inf')

    for v, count in enumerate(state.remaining_people):
        if count > 0:     # there are people left at vertex v
            d = env.optimistic_dist[state.position][v]
            best = min(best, d)

    return best

# "optimistic" distances ignoring flooding & kit issues, computed once in the env init
def precompute_distances(weights):
    n = weights.shape[0]
    dist_all = np.full((n, n), np.inf, dtype=float)

    for s in range(n):            # for each source vertex s
        # run Dijkstra from s over the optimistic graph
        dist = np.full(n, np.inf, dtype=float)
        dist[s] = 0
        visited = np.zeros(n, dtype=bool)
        pq = [(0, s)]             # (distance, vertex)

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue

            # neighbors: any v with weights[u, v] != -1
            for v in range(n):
                w = weights[u, v]
                if w == -1:
                    continue

                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))

        dist_all[s, :] = dist

    return dist_all
