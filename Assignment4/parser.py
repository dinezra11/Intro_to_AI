import yaml

def parse_config(path):
    with open(path) as f:
        data = yaml.safe_load(f)

    n = data["vertices"]["N"]

    edges = []
    for e in data["edges"]:
        parts = e.split(',')
        u, v = int(parts[0]), int(parts[1])
        w = float(parts[2])
        p = float(parts[4]) if len(parts) > 4 and parts[4] else 0.0
        edges.append((u, v, w, p))

    start = data["start"]
    target = data["target"]

    return n, edges, start, target
