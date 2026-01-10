## Input file parser
import yaml


def parse_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Number of vertices
    n_vertices = data["vertices"]["N"]

    # Uncertainty parameters
    P1 = float(data["uncertainty"]["P1"])
    weather_prior = {
        "mild": float(data["uncertainty"]["weather_prior"]["mild"]),
        "stormy": float(data["uncertainty"]["weather_prior"]["stormy"]),
        "extreme": float(data["uncertainty"]["weather_prior"]["extreme"]),
    }

    # Parse edges
    edges = []
    for edge_str in data["edges"]:
        tokens = edge_str.split(",")

        u = int(tokens[0])
        v = int(tokens[1])
        weight = int(tokens[2])

        flooded_observed = None
        if len(tokens) >= 4 and tokens[3] == "F":
            flooded_observed = True

        p_mild = 0.0
        if len(tokens) >= 5 and tokens[4] != "":
            p_mild = float(tokens[4])

        edges.append({
            "from": u,
            "to": v,
            "weight": weight,
            "p_mild": p_mild,
            "flooded_observed": flooded_observed
        })

    return n_vertices, edges, P1, weather_prior
