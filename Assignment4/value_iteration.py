import math

def value_iteration(mdp, gamma=1.0, eps=1e-6):
    beliefs = mdp.all_beliefs()
    V = {b: 0.0 for b in beliefs}
    policy = {}

    while True:
        delta = 0
        for b in beliefs:
            if mdp.is_terminal(b):
                continue

            pos, knowledge = b
            best = math.inf
            best_action = None

            for i, (u, v, w, p) in enumerate(mdp.edges):
                if pos not in (u, v):
                    continue

                nxt = v if pos == u else u

                if p == 0:
                    cost = w + V[(nxt, knowledge)]
                else:
                    idx = mdp.edge_index[i]
                    k = knowledge[idx]

                    if k == 1:
                        continue
                    elif k == 2:
                        cost = w + V[(nxt, knowledge)]
                    else:
                        k_f = list(knowledge)
                        k_f[idx] = 1
                        k_c = list(knowledge)
                        k_c[idx] = 2

                        cost = (
                            w +
                            p * V[(pos, tuple(k_f))] +
                            (1 - p) * V[(nxt, tuple(k_c))]
                        )

                if cost < best:
                    best = cost
                    best_action = i

            if best < math.inf:
                delta = max(delta, abs(V[b] - best))
                V[b] = best
                policy[b] = best_action

        if delta < eps:
            break

    return V, policy
