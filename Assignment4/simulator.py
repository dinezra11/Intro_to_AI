import random

UNKNOWN, FLOODED, CLEAR = 0, 1, 2


def belief_to_string(mdp, belief):
    pos, knowledge = belief
    s = f"Position: {pos}\nBelief about edges:\n"

    uncertain_map = {
        edge_idx: k
        for edge_idx, k in zip(mdp.uncertain_edges, knowledge)
    }

    for i in range(len(mdp.edges)):
        if i in uncertain_map:
            k = uncertain_map[i]
            state = (
                "UNKNOWN" if k == UNKNOWN else
                "FLOODED" if k == FLOODED else
                "CLEAR"
            )
        else:
            state = "DETERMINISTIC (assumed clear)"

        s += f"  Edge {i}: {state}\n"

    return s



def simulate(mdp, policy, trials=1):
    for t in range(trials):
        # Sample true flooding configuration
        flooded = {}
        for i, (_, _, _, p) in enumerate(mdp.edges):
            flooded[i] = random.random() < p

        pos = mdp.start
        knowledge = tuple(UNKNOWN for _ in mdp.uncertain_edges)

        print("\n" + "=" * 40)
        print(f"Simulation {t + 1}")
        print("True flooded edges (hidden from agent):")
        for i, v in flooded.items():
            print(f"  Edge {i}: {'FLOODED' if v else 'CLEAR'}")

        step = 0
        while pos != mdp.target:
            print("\n" + "-" * 30)
            print(f"Step {step}")
            belief = (pos, knowledge)

            # Print belief state
            print(belief_to_string(mdp, belief))

            if belief not in policy:
                print("No available action — stopping.")
                break

            e = policy[belief]
            u, v, w, _ = mdp.edges[e]
            nxt = v if pos == u else u

            print(f"Chosen action: traverse edge {e} ({u} ↔ {v})")

            if flooded[e]:
                print(f"Observation: Edge {e} is FLOODED → stay at {pos}")
                idx = mdp.edge_index[e]
                knowledge = list(knowledge)
                knowledge[idx] = FLOODED
                knowledge = tuple(knowledge)
            else:
                print(f"Observation: Edge {e} is CLEAR → move to {nxt}")
                if e in mdp.edge_index:
                    idx = mdp.edge_index[e]
                    knowledge = list(knowledge)
                    knowledge[idx] = CLEAR
                    knowledge = tuple(knowledge)
                pos = nxt

            step += 1

        print("\nReached target vertex:", pos)
        print("=" * 40)
