from collections import deque

UNKNOWN, FLOODED, CLEAR = 0, 1, 2


class BeliefMDP:
    def __init__(self, n, edges, start, target):
        self.n = n
        self.edges = edges
        self.start = start
        self.target = target

        # Every edge with nonzero flooding probability is uncertain
        self.uncertain_edges = [i for i, e in enumerate(edges) if e[3] > 0.0]
        self.edge_index = {edge_id: idx for idx, edge_id in enumerate(self.uncertain_edges)}

        # adjacency: vertex -> list of edge indices
        self.adj = {v: [] for v in range(n)}
        for i, (u, v, *_rest) in enumerate(edges):
            self.adj[u].append(i)
            self.adj[v].append(i)

    def is_terminal(self, belief):
        pos, _ = belief
        return pos == self.target

    def start_belief(self):
        return (self.start, tuple(UNKNOWN for _ in self.uncertain_edges))

    def legal_actions(self, belief):
        pos, knowledge = belief
        acts = []
        for ei in self.adj[pos]:
            u, v, w, p = self.edges[ei]
            if p == 0.0:
                acts.append(ei)
                continue

            kidx = self.edge_index[ei]
            if knowledge[kidx] == FLOODED:
                continue  # known blocked
            acts.append(ei)
        return acts

    def transitions(self, belief, action_edge):
        """
        Returns list of (prob, next_belief, cost)
        """
        pos, knowledge = belief
        u, v, w, p = self.edges[action_edge]
        nxt = v if pos == u else u

        # deterministic edge
        if p == 0.0:
            return [(1.0, (nxt, knowledge), w)]

        kidx = self.edge_index[action_edge]
        status = knowledge[kidx]

        # already known
        if status == FLOODED:
            return []  # illegal
        if status == CLEAR:
            return [(1.0, (nxt, knowledge), w)]

        # unknown edge: two outcomes
        k_f = list(knowledge)
        k_f[kidx] = FLOODED
        k_c = list(knowledge)
        k_c[kidx] = CLEAR

        # If flooded: we learn it and stay in place (cannot traverse)
        # If clear: we learn it and move
        return [
            (p, (pos, tuple(k_f)), w),
            (1.0 - p, (nxt, tuple(k_c)), w),
        ]

    def reachable_beliefs(self):
        """
        BFS over belief space from start belief using all actions/outcomes.
        Much smaller than |V|*3^k in many graphs.
        """
        start = self.start_belief()
        q = deque([start])
        seen = {start}

        while q:
            b = q.popleft()
            if self.is_terminal(b):
                continue
            for a in self.legal_actions(b):
                for _prob, nb, _cost in self.transitions(b, a):
                    if nb not in seen:
                        seen.add(nb)
                        q.append(nb)

        return list(seen)
