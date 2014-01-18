import gurobipy as grb

from itertools import product, chain

import numpy as np

from collections import defaultdict


eps = 1.0e-9


def solveLPonLocalPolytope(g):

    m = grb.Model()

    variables = {factor.members: np.array([m.addVar(0.0, 1.0)
                 for dummy in range(np.prod(factor.values.shape))]).reshape(factor.values.shape)
                 for factor in g.factors}

    m.update()

    # objective

    objective = grb.quicksum([float(c) * x
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


def projectionOnDualOptimal(g, primal_optimal, point):

    m = grb.Model()

    var_dual = {factor.members: np.array([m.addVar(-grb.GRB.INFINITY, grb.GRB.INFINITY, name="lambda-%s-%i" % (str(factor.members), dummy))
                for dummy in range(np.prod(factor.values.shape))]).reshape(factor.values.shape)
                for factor in g.factors}

    var_norm = {factor.members: m.addVar(-grb.GRB.INFINITY, grb.GRB.INFINITY, name="u%i" % (factor.members[0], ))
                for factor in g.factors if len(factor.members) == 1}

    var_agree = {factor.members:
                 np.array([m.addVar(-grb.GRB.INFINITY, grb.GRB.INFINITY, name="u-%s-%i" % (str(factor.members), dummy)) for dummy in range(factor.values.shape[0])])
                 for factor in g.factors if len(factor.members) == 2}

    var_agree_t = {(factor.members[1], factor.members[0]):
                   np.array([m.addVar(-grb.GRB.INFINITY, grb.GRB.INFINITY, name="ur-%s-%i" % (str(factor.members), dummy)) for dummy in range(factor.values.shape[1])])
                   for factor in g.factors if len(factor.members) == 2}

    var_agree = dict(var_agree.items() + var_agree_t.items())

    m.update()

    # aux variables

    edge_dict = defaultdict(list)

    for factor in g.factors:
        if len(factor.members) == 2:
            edge_dict[factor.members[0]].append(factor.members)
            edge_dict[factor.members[1]].append((factor.members[1], factor.members[0]))

    # objective

    terms = []
    for members, variables in var_dual.items():

        variables = variables.ravel()
        values = point[members].ravel()

        for v, val in zip(variables, values):
            terms.append((v - float(val)) * (v - float(val)))

    objective = grb.quicksum(terms)

    m.setObjective(objective)
    print(objective)

    # constraints

    constr_list = []

    for factor in g.factors:
        if len(factor.members) == 1:
            for l in range(factor.values.shape[0]):

                constr = var_dual[factor.members][l] + var_norm[factor.members] +\
                    grb.quicksum([var_agree[members][l] for members in edge_dict[factor.members[0]]])

                if primal_optimal[factor.members][l] > eps:
                    constr = (constr == -float(factor.values[l]))
                else:
                    constr = (constr >= -float(factor.values[l]))

                constr_list.append(constr)
                print(constr)

        if len(factor.members) == 2:
            for l1 in range(factor.values.shape[0]):
                for l2 in range(factor.values.shape[1]):

                    constr = var_dual[factor.members][l1][l2] - var_agree[factor.members][l1] -\
                        var_agree[(factor.members[1], factor.members[0])][l2]

                    if primal_optimal[factor.members][l1][l2] > eps:
                        constr = (constr == -float(factor.values[l1][l2]))
                    else:
                        constr = (constr >= -float(factor.values[l1][l2]))

                    constr_list.append(constr)
                    print(constr)

    [m.addConstr(constr) for constr in constr_list]

    m.optimize()

    return {members: np.array([var.x for var in vars_.ravel()]).reshape(vars_.shape) for members, vars_ in var_dual.items()}
