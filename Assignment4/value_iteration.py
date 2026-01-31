import math


def value_iteration(mdp, gamma=0.95, eps=1e-6, max_iters=500):
    beliefs = mdp.reachable_beliefs()  # IMPORTANT: reachable only
    V = {b: 0.0 for b in beliefs}
    policy = {}

    for _it in range(max_iters):
        delta = 0.0

        for b in beliefs:
            if mdp.is_terminal(b):
                V[b] = 0.0
                policy.pop(b, None)
                continue

            best = math.inf
            best_action = None

            for a in mdp.legal_actions(b):
                trans = mdp.transitions(b, a)
                if not trans:
                    continue

                q = 0.0
                for prob, nb, cost in trans:
                    # discounted expected cost-to-go
                    q += prob * (cost + gamma * V[nb])

                if q < best:
                    best = q
                    best_action = a

            if best < math.inf:
                delta = max(delta, abs(V[b] - best))
                V[b] = best
                policy[b] = best_action

        if delta < eps:
            break

    return V, policy
