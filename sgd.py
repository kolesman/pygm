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


def sgd(g, maxiter=300, step_rule=None, verbose=False, make_log=None, use_optimal_solution=False):

    best_primal = 2 ** 32
    best_dual = -2 ** 32

    subgradient, energy, primal_energy = computeSubgradient(g)
    parameters = np.zeros(len(subgradient))
    #energy_prev = energy

    gn2 = np.sum([tree.n_factors for tree in g.tree_decomposition])

    scope = {}

    if step_rule[0] == 'step_adaptive':
        scope['delta'] = step_rule[1]['delta']
        scope['energy_rec'] = energy
        scope['sigma'] = 0

    optimal_solution = None
    if step_rule[0] == 'step_god' or use_optimal_solution:
        optimal_solution = g.optimal_parameters

    i = 0
    log = []

    while i < maxiter:

        subgradient, energy, primal_energy = computeSubgradient(g)

        best_dual = max(best_dual, energy)
        best_primal = min(best_primal, primal_energy)

        sn = np.linalg.norm(subgradient)

        if sn < 1.0e-9:
            break

        def step_constant(r0=0.01):
            return r0

        def step_array(a=np.ones(maxiter)):
            return a[i]

        def step_power(r0=1.0, alpha=0.5):
            return r0 / ((1 + i ** alpha))

        def step_power_norm(r0=1.0, alpha=0.5):
            return r0 / ((1 + i ** alpha) * np.sqrt(gn2))

        def step_god():
            return np.dot(subgradient, optimal_solution - parameters) / sn ** 2

        def step_adaptive(B=1.0, gamma=1.0, **kwarg):

            update = 0
            if energy > scope['energy_rec'] + scope['delta'] / 2:
                scope['sigma'] = 0
                update = 1
            elif scope['sigma'] > B / np.sqrt(i + 1):
                scope['delta'] = scope['delta'] / 2
                scope['sigma'] = 0
                update = 1

            if update:
                scope['energy_rec'] = best_dual

            approx = scope['energy_rec'] + scope['delta']

            return gamma * (approx - energy) / sn ** 2

        step = eval(step_rule[0])(**step_rule[1])

        if verbose:
            print(best_primal, best_dual, energy, step)

        if 'sigma' in scope:
            scope['sigma'] += step * sn

        i += 1

        if make_log:
            log_line = [best_primal, best_dual, energy, sn, step]

            if use_optimal_solution:
                optimal_step = np.dot(subgradient, optimal_solution - parameters) / sn ** 2
                log_line.append(optimal_step)

            log.append(log_line)

        update(g, subgradient, step)
        parameters += step * subgradient

    if make_log:
        return parameters, np.array(log)

    return parameters
