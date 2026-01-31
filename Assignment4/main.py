from parser import parse_config
from belief_mdp import BeliefMDP
from value_iteration import value_iteration
from simulator import simulate

def main():
    n, edges, start, target = parse_config("environment_mdp_config.yaml")
    mdp = BeliefMDP(n, edges, start, target)

    V, policy = value_iteration(mdp)

    print("\nBelief State Values:")
    for b, v in V.items():
        print(b, "->", round(v, 3))

    simulate(mdp, policy, trials=1)

if __name__ == "__main__":
    main()
