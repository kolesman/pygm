import pygm.utils as utils

import gurobipy as grb

from itertools import product, chain
from collections import defaultdict

import numpy as np

import multiprocessing


def constructLP(gm):

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


#def addSubproblemConstraints(m, alpha, g, primal_optimal, grad, point):
#
#    var_dual = {factor.members: np.array([m.addVar(-grb.GRB.INFINITY, grb.GRB.INFINITY)
#                for dummy in range(np.prod(factor.values.shape))]).reshape(factor.values.shape)
#                for factor in g.factors}
#
#    var_norm = {factor.members: m.addVar(-grb.GRB.INFINITY, grb.GRB.INFINITY)
#                for factor in g.factors if len(factor.members) == 1}
#
#    var_agree = {factor.members:
#                 np.array([m.addVar(-grb.GRB.INFINITY, grb.GRB.INFINITY) for dummy in range(factor.values.shape[0])])
#                 for factor in g.factors if len(factor.members) == 2}
#
#    var_agree_t = {(factor.members[1], factor.members[0]):
#                   np.array([m.addVar(-grb.GRB.INFINITY, grb.GRB.INFINITY) for dummy in range(factor.values.shape[1])])
#                   for factor in g.factors if len(factor.members) == 2}
#
#    var_agree = dict(var_agree.items() + var_agree_t.items())
#
#    m.update()
#
#    # aux variables
#
#    edge_dict = defaultdict(list)
#
#    for factor in g.factors:
#        if len(factor.members) == 2:
#            edge_dict[factor.members[0]].append(factor.members)
#            edge_dict[factor.members[1]].append((factor.members[1], factor.members[0]))
#
#    # objective
#
#    terms = []
#    for members, variables in var_dual.items():
#
#        if len(variables.shape) == 1:
#            for l in range(variables.shape[0]):
#
#                terms.append((variables[l] - (point[(members, (l,))] + alpha * grad[(members, (l,))])) *
#                            (variables[l] - (point[(members, (l,))] + alpha * grad[(members, (l,))])))
#
#        if len(variables.shape) == 2:
#            for l1 in range(variables.shape[0]):
#                for l2 in range(variables.shape[1]):
#                    terms.append((variables[l1][l2] - (point[(members, (l1, l2))] + alpha * grad[(members, (l1, l2))])) *
#                                 (variables[l1][l2] - (point[(members, (l1, l2))] + alpha * grad[(members, (l1, l2))])))
#
#    objective = grb.quicksum(terms)
#
#    # constraints
#
#    constr_list = []
#
#    for factor in g.factors:
#        if len(factor.members) == 1:
#            for l in range(factor.values.shape[0]):
#
#                constr = var_dual[factor.members][l] + var_norm[factor.members] +\
#                    grb.quicksum([var_agree[members][l] for members in edge_dict[factor.members[0]]])
#
#                if primal_optimal[factor.members][l] > eps:
#                    constr = (constr == -float(factor.values[l]))
#                else:
#                    constr = (constr >= -float(factor.values[l]))
#
#                constr_list.append(constr)
#                #print(constr)
#
#        if len(factor.members) == 2:
#            for l1 in range(factor.values.shape[0]):
#                for l2 in range(factor.values.shape[1]):
#
#                    constr = var_dual[factor.members][l1][l2] - var_agree[factor.members][l1] -\
#                        var_agree[(factor.members[1], factor.members[0])][l2]
#
#                    if primal_optimal[factor.members][l1][l2] > eps:
#                        constr = (constr == -float(factor.values[l1][l2]))
#                    else:
#                        constr = (constr >= -float(factor.values[l1][l2]))
#
#                    constr_list.append(constr)
#                    #print(constr)
#
#    [m.addConstr(constr) for constr in constr_list]
#
#    return var_dual, objective
#
#
#def optimalStepDD(g, optimal_primal, point, grad):
#
#    m = grb.Model()
#    m.setParam('barconvtol', 1.0e-2)
#    m.setParam('psdtol', 1.0)
#
#    alpha = m.addVar(name='alpha')
#
#    objective = 0
#
#    duals = []
#
#    for i, tree in enumerate(g.tree_decomposition):
#
#        point_i = {(key[1], key[2]): float(value) for key, value in point.items() if key[0] == i}
#        grad_i = {(key[1], key[2]): float(value) for key, value in grad.items() if key[0] == i}
#
#        dpoint_i = defaultdict(int)
#        dpoint_i.update(point_i)
#        dgrad_i = defaultdict(int)
#        dgrad_i.update(grad_i)
#
#        var_dual, obj = addSubproblemConstraints(m, alpha, tree, optimal_primal, dgrad_i, dpoint_i)
#
#        duals.append(var_dual)
#
#        objective += obj
#
#    consistency_constraints = defaultdict(int)
#
#    for dual_vars in duals:
#        for key, var in dual_vars.items():
#
#            if len(key) == 1:
#                for l in range(var.shape[0]):
#                    consistency_constraints[(key, (l, ))] += var[l]
#
#            if len(key) == 2:
#                for l1 in range(var.shape[0]):
#                    for l2 in range(var.shape[1]):
#                        consistency_constraints[(key, (l1, l2))] += var[l1][l2]
#
#    for constr in consistency_constraints.values():
#        m.addConstr(constr == 0)
#
#    m.setObjective(objective)
#
#    m._duals = duals
#    m._alpha = alpha
#
#    m.optimize()
#
#    return alpha.x, utils.getUpdateFromProjectionModel(m)
#
#
#def prepareGMs(gms, n_cpus=1):
#
#    import pygm.sgd as sgd
#
#    pool = multiprocessing.Pool(n_cpus)
#    lazy_optimal_primals = []
#    for gm in gms:
#        res = pool.apply_async(solveLPonLocalPolytope, [gm])
#        lazy_optimal_primals.append(res)
#
#    pool.close()
#    optimal_primals = [solution.get()[1] for solution in lazy_optimal_primals]
#
#    pool = multiprocessing.Pool(n_cpus)
#    lazy_subgradients = []
#    for gm in gms:
#        res = pool.apply_async(sgd.computeSubgradient, [gm])
#        lazy_subgradients.append(res)
#
#    pool.close()
#    subgradients = [res.get()[0] for res in lazy_subgradients]
#
#    pool = multiprocessing.Pool(n_cpus)
#    lazy_optimal_parameters = []
#    for gm, optimal_primal, subgr in zip(gms, optimal_primals, subgradients):
#        res = pool.apply_async(optimalStepDD, [gm, optimal_primal, {}, subgr])
#        lazy_optimal_parameters.append(res)
#
#    pool.close()
#    optimal_parameters = [res.get()[1] for res in lazy_optimal_parameters]
#
#    for gm, solution in zip(gms, optimal_parameters):
#        gm.optimal_parameters = solution
#
#    return gms
