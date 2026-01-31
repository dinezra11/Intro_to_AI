from itertools import product

UNKNOWN, FLOODED, CLEAR = 0, 1, 2

class BeliefMDP:
    def __init__(self, n, edges, start, target):
        self.n = n
        self.edges = edges
        self.start = start
        self.target = target

        self.uncertain_edges = [i for i, e in enumerate(edges) if e[3] > 0]
        self.edge_index = {i: idx for idx, i in enumerate(self.uncertain_edges)}

    def all_beliefs(self):
        beliefs = []
        for pos in range(self.n):
            for states in product([UNKNOWN, FLOODED, CLEAR], repeat=len(self.uncertain_edges)):
                beliefs.append((pos, states))
        return beliefs

    def is_terminal(self, belief):
        pos, _ = belief
        return pos == self.target
