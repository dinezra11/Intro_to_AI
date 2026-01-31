import random

def simulate(mdp, policy, trials=1):
    for t in range(trials):
        flooded = {}
        for i, (_, _, _, p) in enumerate(mdp.edges):
            flooded[i] = random.random() < p

        pos = mdp.start
        knowledge = tuple(0 for _ in mdp.uncertain_edges)

        print(f"\nSimulation {t+1}")
        print("Flooded edges:", flooded)

        while pos != mdp.target:
            b = (pos, knowledge)
            if b not in policy:
                print("No action available")
                break

            e = policy[b]
            u, v, w, _ = mdp.edges[e]
            nxt = v if pos == u else u

            if flooded[e]:
                idx = mdp.edge_index[e]
                knowledge = list(knowledge)
                knowledge[idx] = 1
                knowledge = tuple(knowledge)
                print(f"Edge {e} flooded, stay at {pos}")
            else:
                if e in mdp.edge_index:
                    idx = mdp.edge_index[e]
                    knowledge = list(knowledge)
                    knowledge[idx] = 2
                    knowledge = tuple(knowledge)
                pos = nxt
                print(f"Move to {pos}")

        print("Reached target")
