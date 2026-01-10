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
    dist = {}
    for val in bn.get(var_name).domain:
        evidence[var_name] = val
        dist[val] = enumerate_all(bn.order, bn, dict(evidence))
        del evidence[var_name]

    norm = sum(dist.values())
    for k in dist:
        dist[k] /= norm

    return dist
