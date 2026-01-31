## Exact inference by enumeration
def enumerate_all(vars_order, bn, assignment):
    if not vars_order:
        return 1.0

    Y = vars_order[0]
    var = bn.get(Y)

    if Y in assignment:
        return (
            var.prob(assignment[Y], assignment)
            * enumerate_all(vars_order[1:], bn, assignment)
        )

    total = 0.0
    for y in var.domain:
        assignment[Y] = y
        total += (
            var.prob(y, assignment)
            * enumerate_all(vars_order[1:], bn, assignment)
        )
        del assignment[Y]

    return total


def query(bn, var_name, evidence):
    # -----------------------------------
    # CASE 1: Query variable is observed
    # -----------------------------------
    if var_name in evidence:
        val = evidence[var_name]
        return {
            v: 1.0 if v == val else 0.0
            for v in bn.get(var_name).domain
        }

    # -----------------------------------
    # CASE 2: Normal inference
    # -----------------------------------
    dist = {}
    for val in bn.get(var_name).domain:
        extended_evidence = dict(evidence)
        extended_evidence[var_name] = val
        dist[val] = enumerate_all(bn.order, bn, extended_evidence)

    dist = {k: round(v, 3) for k, v in dist.items()}

    norm = sum(dist.values())
    for k in dist:
        dist[k] /= norm

    return dist

