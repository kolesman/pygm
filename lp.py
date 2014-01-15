import gurobipy as grb

from itertools import product, chain

import numpy as np


def solveLPonLocalPolytope(g):

    m = grb.Model()

    variables = {factor.members: np.array([m.addVar(0.0, 1.0) for dummy in range(np.prod(factor.values.shape))]).reshape(factor.values.shape)
                 for factor in g.factors}

    m.update()

    # objective

    objective = grb.quicksum([c * x
                              for factor in g.factors
                              for c, x in zip(factor.values.ravel(), variables[factor.members].ravel())])

    m.setObjective(objective)

    # constraints

    unary_constraints = [grb.quicksum([var for var in variables[factor.members]]) == 1
                         for factor in g.factors if len(factor.members) == 1]
    binary_constraints1 = [grb.quicksum([var for var in variables_node]) == variables[(factor.members[0], )][i]
                           for factor in g.factors
                           for i, variables_node in enumerate(variables[factor.members])
                           if len(factor.members) == 2]
    binary_constraints2 = [grb.quicksum([var for var in variables_node]) == variables[(factor.members[1], )][i]
                           for factor in g.factors
                           for i, variables_node in enumerate(variables[factor.members].T)
                           if len(factor.members) == 2]

    [m.addConstr(constraint) for constraint in chain(unary_constraints, binary_constraints1, binary_constraints2)]

    m.optimize()

    return {members: np.array([var.x for var in vars_.ravel()]).reshape(vars_.shape) for members, vars_ in variables.items()}
