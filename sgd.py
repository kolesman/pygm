import numpy as np

import sys
import time

from collections import Counter

from itertools import product


MAXPRIMALS = 1


def computeSubgradient(g):

    subgradient = []

    subtrees = g.tree_decomposition

    tree_solutions = np.array([subtree.getMapState('DynamicProgramming', {}) for subtree in subtrees])

    energy = sum([subtree.Energy(solution) for subtree, solution in zip(subtrees, tree_solutions)])

    #primal_solution = [Counter(preds).most_common()[0][0] for preds in tree_solutions.T]
    #primal_energy = g.Energy(primal_solution)
    primal_energy = np.max([g.Energy(primal_solution) for primal_solution in primalSolutions(tree_solutions)])

    for subtree, solution in zip(subtrees, tree_solutions):

        for factor in subtree.factors:

            if len(factor.members) == 1:
                i = factor.members[0]
                for u in range(len(factor.values)):
                    shift = (solution[i] == u) - np.average(tree_solutions[:, i] == u)
                    subgradient.append(shift)
                    #factor.values[u] += alpha * shift

            if len(factor.members) == 2:
                i, j = factor.members
                edge_mask = g.tree_decomposition_edge_mask[(i, j)]
                if sum(edge_mask) > 1:
                    u_len, v_len = factor.values.shape
                    for u in range(u_len):
                        for v in range(v_len):
                            shift = ((solution[i] == u) * (solution[j] == v)) -\
                                np.average((tree_solutions[:, i] == u)[edge_mask] * (tree_solutions[:, j] == v)[edge_mask])
                            subgradient.append(shift)
                            #factor.values[u][v] += alpha * shift

    return np.array(subgradient), energy, primal_energy


def update(g, subgradient, alpha):

    subtrees = g.tree_decomposition
    iii = 0

    for subtree in subtrees:

        for factor in subtree.factors:

            if len(factor.members) == 1:
                i = factor.members[0]
                for u in range(len(factor.values)):
                    factor.values[u] += alpha * subgradient[iii]
                    iii += 1

            if len(factor.members) == 2:
                i, j = factor.members
                edge_mask = g.tree_decomposition_edge_mask[(i, j)]
                if sum(edge_mask) > 1:
                    u_len, v_len = factor.values.shape
                    for u in range(u_len):
                        for v in range(v_len):
                            factor.values[u][v] += alpha * subgradient[iii]
                            iii += 1

    return 0


def primalSolutions(solutions):
    primals = product(*[np.unique(solution) for solution in solutions.T])
    return [primals.next() for dummy in range(MAXPRIMALS)]


def sgd_expstep(g, maxiter=300, verbose=False, make_log=None, optimal_parameters=None, godmode=None):

    subgradient, energy, primal_energy = computeSubgradient(g)

    best_primal = primal_energy
    best_dual = energy

    parameters = np.zeros(len(subgradient))

    i = 0
    log = []
    while i < maxiter:

        sn = np.linalg.norm(subgradient)

        step = 1.0 / ((1 + i ** 0.5) * sn)
        i += 1

        if not godmode is None:
            step = np.dot(subgradient, godmode - parameters) / sn ** 2

        if make_log:
            log_line = [best_dual, energy, sn, step]

            if not optimal_parameters is None:
                optimal_step = np.dot(subgradient, optimal_parameters) / sn ** 2
                log_line.append(optimal_step)

            log.append(log_line)

        update(g, subgradient, step)

        parameters += step * subgradient

        subgradient, energy, primal_energy = computeSubgradient(g)

        if verbose:
            print(best_primal, best_dual, energy, step)

        best_dual = max(best_dual, energy)
        best_primal = min(best_primal, primal_energy)

    if make_log:
        return parameters, np.array(log)

    return parameters
