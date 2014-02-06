import gurobipy as grb

from itertools import product, chain

import numpy as np

from collections import defaultdict

import utils


eps = 1.0e-6


def constructLPonLocalPolytope(g):

    m = grb.Model()

    variables = {factor.members: np.array([m.addVar(lb=0.0, ub=1.0, name="order" + str(len(factor.members)))
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

    m._variables = variables

    return m


def solveLPonLocalPolytope(g):

    m = constructLPonLocalPolytope(g)

    m.optimize()

    variables = m._variables

    return m.objval, {members: np.array([var.x for var in vars_.ravel()]).reshape(vars_.shape) for members, vars_ in variables.items()}


def solveLPonLocalPolytopeAll(g, max_solutions=10):

    solutions = []

    m = constructLPonLocalPolytope(g, relax=False)
    m.setParam('OutputFlag', False)
    variables = m._variables

    node_orders = np.zeros(g.n_vars).astype('int')

    for members in variables.keys():
        if len(members) == 2:
            node_orders[members[0]] += 1
            node_orders[members[1]] += 1

    m.optimize()
    assert(np.all([(v.x <= eps) or (v.x >= 1.0 - eps) for v in m.getVars()]))

    energy = m.objVal
    solutions.append(utils.getSolutionFromLPModel(m))

    while True:

        cut_binary = grb.quicksum([var for var in m.getVars() if var.x > (1 - eps) and var.varName == 'order2'])
        cut_unary = grb.quicksum([var * (1 - int(node_orders[members[0]])) for members, var_list in variables.items() if len(members) == 1
                                  for var in var_list if var.x > (1 - eps)])

        m.addConstr((cut_binary + cut_unary) <= 0)

        m.optimize()
        assert(np.all([(v.x <= eps) or (v.x >= 1 - eps) for v in m.getVars()]))

        energy_current = m.objVal

        if (energy_current - energy) / np.abs(energy) > 0.001:
            break

        solutions.append(utils.getSolutionFromLPModel(m))

        max_solutions -= 1

        if max_solutions <= 1:
            break

    return solutions


def findSteepestGradient(g, tree_solutions, relax=True):

    n_trees = len(g.tree_decomposition)

    m = grb.Model()
    m.setParam('OutputFlag', False)
    coeffs = [[m.addVar(lb=0.0, ub=1.0) for solution in tree_solution] for tree_solution in tree_solutions]

    m.update()

    if relax:
        m.relax()

    # Constraints
    [m.addConstr(grb.quicksum(linear_comb) == 1.0) for linear_comb in coeffs]

    # Subgradient

    update = defaultdict(int)

    for i, solution_list in enumerate(tree_solutions):
        for j, tree_solution in enumerate(solution_list):
            for k, label in enumerate(tree_solution):
                if (j, ) in g.tree_decomposition[i].map_members_index:
                    update[(i, (k, ), (label, ))] += coeffs[i][j] * 1.0

                    for t in range(n_trees):
                        update[(t, (k, ), (label, ))] -= coeffs[i][j] / n_trees

    edges = set([factor.members for factor in g.factors if len(factor.members) == 2])

    for i, solution_list in enumerate(tree_solutions):
        for j, tree_solution in enumerate(solution_list):
            for u, label0 in enumerate(tree_solution):
                for v, label1 in enumerate(tree_solution):
                    if (u, v) in edges:
                        n_dual = np.sum(g.tree_decomposition_edge_mask[(u, v)])
                        if g.tree_decomposition_edge_mask[(u, v)][i]:
                            update[(i, (u, v), (label0, label1))] += coeffs[i][j] * 1.0

                            for t in range(n_trees):
                                if g.tree_decomposition_edge_mask[(u, v)][t]:
                                    update[(t, (u, v), (label0, label1))] -= coeffs[i][j] / float(n_dual)

    # Objective
    obj = grb.quicksum([expr * expr for members, expr in update.items() if len(members) == 1])
    m.setObjective(obj)

    ######

    m.optimize()

    for key, value in update.items():
        update[key] = value.getValue()

    return update


def addSubproblemConstraints(m, alpha, g, primal_optimal, grad, point):

    var_dual = {factor.members: np.array([m.addVar(-grb.GRB.INFINITY, grb.GRB.INFINITY)
                for dummy in range(np.prod(factor.values.shape))]).reshape(factor.values.shape)
                for factor in g.factors}

    var_norm = {factor.members: m.addVar(-grb.GRB.INFINITY, grb.GRB.INFINITY)
                for factor in g.factors if len(factor.members) == 1}

    var_agree = {factor.members:
                 np.array([m.addVar(-grb.GRB.INFINITY, grb.GRB.INFINITY) for dummy in range(factor.values.shape[0])])
                 for factor in g.factors if len(factor.members) == 2}

    var_agree_t = {(factor.members[1], factor.members[0]):
                   np.array([m.addVar(-grb.GRB.INFINITY, grb.GRB.INFINITY) for dummy in range(factor.values.shape[1])])
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

        if len(variables.shape) == 1:
            for l in range(variables.shape[0]):

                terms.append((variables[l] - (point[(members, (l,))] + alpha * grad[(members, (l,))])) *
                            (variables[l] - (point[(members, (l,))] + alpha * grad[(members, (l,))])))

        if len(variables.shape) == 2:
            for l1 in range(variables.shape[0]):
                for l2 in range(variables.shape[1]):
                    terms.append((variables[l1][l2] - (point[(members, (l1, l2))] + alpha * grad[(members, (l1, l2))])) *
                                 (variables[l1][l2] - (point[(members, (l1, l2))] + alpha * grad[(members, (l1, l2))])))

    objective = grb.quicksum(terms)

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
                #print(constr)

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
                    #print(constr)

    [m.addConstr(constr) for constr in constr_list]

    return var_dual, objective


def optimalStepDD(g, optimal_primal, point, grad, prev_model=None, return_projection=False):

    if prev_model is None:

        m = grb.Model()
        m.setParam('barconvtol', 1.0e-2)
        m.setParam('psdtol', 1.0)

        alpha = m.addVar(name='alpha')

        objective = 0

        duals = []

        for i, tree in enumerate(g.tree_decomposition):

            point_i = {(key[1], key[2]): float(value) for key, value in point.items() if key[0] == i}
            grad_i = {(key[1], key[2]): float(value) for key, value in grad.items() if key[0] == i}

            dpoint_i = defaultdict(int)
            dpoint_i.update(point_i)
            dgrad_i = defaultdict(int)
            dgrad_i.update(grad_i)

            var_dual, obj = addSubproblemConstraints(m, alpha, tree, optimal_primal, dgrad_i, dpoint_i)

            duals.append(var_dual)

            objective += obj

        consistency_constraints = defaultdict(int)

        for dual_vars in duals:
            for key, var in dual_vars.items():

                if len(key) == 1:
                    for l in range(var.shape[0]):
                        consistency_constraints[(key, (l, ))] += var[l]

                if len(key) == 2:
                    for l1 in range(var.shape[0]):
                        for l2 in range(var.shape[1]):
                            consistency_constraints[(key, (l1, l2))] += var[l1][l2]

        for constr in consistency_constraints.values():
            m.addConstr(constr == 0)
            #print(constr == 0)

        #m.setObjective(objective)
        m.setObjective(1)

        m._duals = duals
        m._alpha = alpha

    else:

        m = prev_model
        duals = m._duals
        alpha = m._alpha

        terms = []
        for i, var_dual in enumerate(duals):

            pointi = {(key[1], key[2]): float(value) for key, value in point.items() if key[0] == i}
            gradi = {(key[1], key[2]): float(value) for key, value in grad.items() if key[0] == i}

            dpointi = defaultdict(int)
            dpointi.update(pointi)
            dgradi = defaultdict(int)
            dgradi.update(gradi)

            for members, variables in var_dual.items():

                if len(variables.shape) == 1:
                    for l in range(variables.shape[0]):

                        terms.append((variables[l] - (dpointi[(members, (l,))] + alpha * dgradi[(members, (l,))])) *
                                    (variables[l] - (dpointi[(members, (l,))] + alpha * dgradi[(members, (l,))])))

                if len(variables.shape) == 2:
                    for l1 in range(variables.shape[0]):
                        for l2 in range(variables.shape[1]):
                            terms.append((variables[l1][l2] - (dpointi[(members, (l1, l2))] + alpha * dgradi[(members, (l1, l2))])) *
                                        (variables[l1][l2] - (dpointi[(members, (l1, l2))] + alpha * dgradi[(members, (l1, l2))])))

        objective = grb.quicksum(terms)
        m.setObjective(objective)

    m.optimize()

    if return_projection:
        return utils.getUpdateFromProjectionModel(m)
    else:
        return m, alpha.x

    #m.optimize()

    #unary_proj = {(members, (label, )): vars_[label].x for members, vars_ in var_dual.items()
    #              for label in range(vars_.shape[0]) if len(members) == 1}
    #binary_proj = {(members, (label0, label1)): vars_[label0][label1].x for members, vars_ in var_dual.items() if len(members) == 2
    #               for label0 in range(vars_.shape[0]) for label1 in range(vars_.shape[1])}

    #proj = dict(unary_proj.items() + binary_proj.items())

    #return proj
