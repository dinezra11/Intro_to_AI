## Domain-specific BN construction
from itertools import product
from bn import Variable, BayesianNetwork


def build_bn(n_vertices, edges, P1, weather_prior):
    bn = BayesianNetwork()

    # Weather node
    weather = Variable(
        "W",
        ["mild", "stormy", "extreme"],
        [],
        {(): weather_prior}
    )
    bn.add(weather)

    # Flooding nodes
    for i, edge in enumerate(edges):
        p = edge["p_mild"]

        cpt = {
            ("mild",): {
                True: p,
                False: 1 - p
            },
            ("stormy",): {
                True: min(1.0, 2 * p),
                False: 1 - min(1.0, 2 * p)
            },
            ("extreme",): {
                True: min(1.0, 3 * p),
                False: 1 - min(1.0, 3 * p)
            }
        }

        flood = Variable(
            f"F{i}",
            [True, False],
            ["W"],
            cpt
        )
        bn.add(flood)

    # Incident edges per vertex
    incident = {v: [] for v in range(n_vertices)}
    for i, e in enumerate(edges):
        incident[e["from"]].append((i, e["weight"]))
        incident[e["to"]].append((i, e["weight"]))

    # Evacuee nodes
    for v in range(n_vertices):
        parents = [f"F{i}" for i, _ in incident[v]]
        cpt = {}

        for combo in product([True, False], repeat=len(parents)):
            prob_not = 1.0
            for (edge_idx, weight), flooded in zip(incident[v], combo):
                if flooded:
                    qi = min(1.0, P1 / weight)
                    prob_not *= (1 - qi)

            cpt[combo] = {
                True: 1 - prob_not,
                False: prob_not
            }

        evac = Variable(
            f"Ev{v}",
            [True, False],
            parents,
            cpt
        )
        bn.add(evac)

    return bn
