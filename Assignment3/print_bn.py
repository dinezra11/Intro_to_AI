## This file is responsible only for formatting output.
from itertools import product


def print_bn(bn, edges, n_vertices, P1):
    print("\nWEATHER:")
    weather = bn.get("W")
    for w in weather.domain:
        print(f"  P({w}) = {weather.cpt[()][w]}")

    # ----------------------------
    # Edge flooding CPTs
    # ----------------------------
    for i, edge in enumerate(edges):
        print(f"\nEDGE {i}:")
        flood = bn.get(f"F{i}")

        for w in ["mild", "stormy", "extreme"]:
            p = flood.cpt[(w,)][True]
            print(f"  P(flooded|{w}) = {round(p, 4)}")

    # ----------------------------
    # Vertex evacuee CPTs
    # ----------------------------
    for v in range(n_vertices):
        evac = bn.get(f"Ev{v}")
        parents = evac.parents

        print(f"\nVERTEX {v}:")

        if not parents:
            # No incident edges → evacuees impossible
            print("  P(Evacuees) = 0")
            continue

        for combo in product([True, False], repeat=len(parents)):
            label_parts = []
            for p, val in zip(parents, combo):
                edge_idx = p[1:]  # F0 → 0
                label_parts.append(
                    f"{'flooded' if val else 'not flooded'} {edge_idx}"
                )

            label = ", ".join(label_parts)
            prob = evac.cpt[combo][True]
            print(f"  P(Evacuees|{label}) = {round(prob, 4)}")
