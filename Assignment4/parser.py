import yaml


def parse_config(path):
    with open(path) as f:
        data = yaml.safe_load(f)

    n = data["vertices"]["N"]

    edges = []
    for e in data["edges"]:
        parts = [x.strip() for x in e.split(",")]
        if len(parts) < 4:
            raise ValueError(f"Edge must be 'u,v,w,p' (or legacy 5-field). Got: {e}")

        u, v = int(parts[0]), int(parts[1])
        w = float(parts[2])

        # NEW: accept both formats:
        #  - Assignment 4: u,v,w,p  -> p at index 3
        #  - Legacy:       u,v,w,?,p -> p at index 4
        if len(parts) >= 5 and parts[4] != "":
            p = float(parts[4])
        else:
            p = float(parts[3])

        edges.append((u, v, w, p))

    start = int(data["start"])
    target = int(data["target"])

    return n, edges, start, target
