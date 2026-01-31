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
            state = "clear (certain)"

        s += f"  Edge {i}: {state}\n"

    return s



def simulate(mdp, policy, trials=1, max_steps=50):
    for t in range(trials):
        # Sample true flooding configuration
        flooded = {}
        for i, (_, _, _, p) in enumerate(mdp.edges):
            flooded[i] = random.random() < p

        pos = mdp.start
        knowledge = tuple(UNKNOWN for _ in mdp.uncertain_edges)

        total_cost = 0.0   # ✅ NEW: accumulated cost

        print("\n" + "=" * 50)
        print(f"Simulation {t + 1}")
        print("True flooded edges (hidden from agent):")

        for i, v in flooded.items():
            print(f"  Edge {i}: {'FLOODED' if v else 'CLEAR'}")

        step = 0

        while pos != mdp.target and step < max_steps:

            print("\n" + "-" * 30)
            print(f"Step {step}")
            print(f"Total cost so far: {round(total_cost, 3)}")  # ✅ PRINT COST

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
            print(f"Edge cost: {w}")

            # Cost is paid regardless of outcome
            total_cost += w   # ✅ UPDATE COST

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

        if pos == mdp.target:
            print("\nReached target vertex:", pos)
            print(f"Final total cost: {round(total_cost, 3)}")
        else:
            print("\nDidn't reach goal within", max_steps, "steps.")
            print("Current vertex:", pos)
            print("Final belief state:")
            print(belief_to_string(mdp, (pos, knowledge)))
            print(f"Total cost so far: {round(total_cost, 3)}")
        print("=" * 50)
