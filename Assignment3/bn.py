## Bayesian Network Core
class Variable:
    def __init__(self, name, domain, parents, cpt):
        self.name = name
        self.domain = domain
        self.parents = parents
        self.cpt = cpt

    def prob(self, value, assignment):
        key = tuple(assignment[p] for p in self.parents)
        return self.cpt[key][value]


class BayesianNetwork:
    def __init__(self):
        self.variables = {}
        self.order = []

    def add(self, var):
        self.variables[var.name] = var
        self.order.append(var.name)

    def get(self, name):
        return self.variables[name]
