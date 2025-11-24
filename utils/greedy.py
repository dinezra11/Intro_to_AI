import heapq
import numpy as np


def reconstruct_path(parent, start, goal):
    path = []
    cur = goal
    while cur != -1:
        path.append(cur)
        if cur == start:
            break
        cur = parent[cur]
    return path[::-1]


def dijkstra(start, W, targets):
    """
    Undirected graph.
    W: numpy array shape (n, n)
       W[i, j] = weight >= 0, or -1 if no edge
    targets: list of target node indices
    flooded: list of flooded vertices
    """
    W = np.asarray(W)
    n = W.shape[0]

    targets = set(targets)

    dist = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    dist[start] = 0

    pq = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)

        # Greedy: first popped target = closest target
        if u in targets:
            return float(d), reconstruct_path(parent, start, u)

        if d > dist[u]:
            continue

        # Vectorized neighbor access:
        # valid neighbors have W[u] != -1
        neighbors = np.where(W[u] != -1)[0]

        for v in neighbors:
            w = W[u, v]
            nd = d + w

            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    return np.inf, []
