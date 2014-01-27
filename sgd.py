import numpy as np

import sys
import time

from collections import Counter
from collections import defaultdict

from itertools import product

import cPickle

from compiler.ast import flatten

import utils

import multiprocessing

from copy import deepcopy

import lp


def computeSubgradient(g, parallel=0):

    subtrees = g.tree_decomposition

    tree_solutions = None
    if parallel:
        lazy_solutions = []
        pool = multiprocessing.Pool(parallel)
        for subtree in subtrees:
            lazy_solution = pool.apply_async(subtree, ['getMapState', 'DynamicProgramming', {}])
            lazy_solutions.append(lazy_solution)
        pool.close()
        pool.join()
        tree_solutions = np.array([lazy_solution.get() for lazy_solution in lazy_solutions])
    else:
        tree_solutions = np.array([tree.getMapState('DynamicProgramming', {}) for tree in subtrees])

    energy = sum([subtree.Energy(solution) for subtree, solution in zip(subtrees, tree_solutions)])

    primal_solution = [Counter(preds).most_common()[0][0] for preds in tree_solutions.T]
    primal_energy = g.Energy(primal_solution)

    update = defaultdict(int)
    n_trees = len(subtrees)

    for i, solution in enumerate(tree_solutions):
        for j, label in enumerate(solution):
            if (j, ) in subtrees[i].map_members_index:
                update[(i, (j, ), (label, ))] += 1.0

                for t in range(n_trees):
                    if (j, ) in subtrees[t].map_members_index:
                        update[(t, (j, ), (label, ))] -= 1.0 / n_trees

    edges = set([factor.members for factor in g.factors if len(factor.members) == 2])

    for i, solution in enumerate(tree_solutions):
        for u, label0 in enumerate(solution):
            for v, label1 in enumerate(solution):
                if (u, v) in edges:
                    n_dual = np.sum(g.tree_decomposition_edge_mask[(u, v)])
                    if g.tree_decomposition_edge_mask[(u, v)][i]:
                        update[(i, (u, v), (label0, label1))] += 1.0

                        for t in range(n_trees):
                            if g.tree_decomposition_edge_mask[(u, v)][t]:
                                update[(t, (u, v), (label0, label1))] -= 1.0 / n_dual

    update = dict([(key, value) for key, value in update.items() if abs(value) > 1.0e-9])

    return update, energy, primal_energy


def updateDDParams(g, update, alpha):

    for key, value in update.items():
        i = key[0]
        f = g.tree_decomposition[i].map_members_index[key[1]]
        a = key[2]
        g.tree_decomposition[i].factors[f].values[a] += alpha * value

    return 0


def updateParams(g, update, alpha):

    for key, value in update.items():
        f = g.map_members_index[key[0]]
        a = key[1]
        g.factors[f].values[a] += alpha * value

    return 0


def sgd(g, maxiter=300, step_rule=None, verbose=False, make_log=False):

    g = deepcopy(g)

    best_primal = 2 ** 32
    best_dual = -2 ** 32

    update, energy, primal_energy = computeSubgradient(g)

    parameters = {}
    scope = {}
    g_copy = None

    if step_rule[0] == 'step_adaptive':
        scope['delta'] = (primal_energy - energy) / 3.0
        scope['energy_rec'] = energy
        scope['without_imp'] = 0
        scope['max_without_imp'] = 10

    optimal_solution = g.optimal_parameters if hasattr(g, 'optimal_parameters') else None

    if step_rule[0] == "step_supergod":
        scope['prev_model'] = None
        optimal_primal = lp.solveLPonLocalPolytope(g)
        g_copy = deepcopy(g)

    i = 0
    log = []

    while i < maxiter:

        update, energy, primal_energy = computeSubgradient(g)

        best_dual = max(best_dual, energy)
        best_primal = min(best_primal, primal_energy)

        sn = np.linalg.norm(update.values())

        if sn < 1.0e-9:
            break

        def step_constant(r0=0.01):
            return r0

        def step_array(a):
            return a[i]

        def step_power(r0=1.0, alpha=0.5):
            return r0 / ((1 + i ** alpha))

        def step_power_norm(r0=1.0, alpha=0.5):
            return r0 / ((1 + i ** alpha) * sn)

        def step_god(mode='objective'):
                return utils.dictDot(update, utils.dictDiff(optimal_solution, parameters)) / sn ** 2

        def step_supergod():
            m, step = lp.optimalStepDD(g_copy, optimal_primal, parameters, update, prev_model=scope['prev_model'])
            scope['prev_model'] = m
            return step

        def step_adaptive(gamma=0.1, **kwarg):

            update = 0
            if energy > scope['energy_rec'] + scope['delta'] / 5:
                update = 1
            elif scope['without_imp'] > scope['max_without_imp']:
                scope['delta'] = scope['delta'] * 0.5
                update = 1

            scope['without_imp'] += 1

            if update:
                scope['energy_rec'] = best_dual
                scope['without_imp'] = 0

            approx = scope['energy_rec'] + scope['delta']

            return gamma * ((approx - energy) / sn ** 2)

        step = eval(step_rule[0])(**step_rule[1])
        if verbose:
            print(i, "%.4f" % best_primal, "%.4f" % best_dual, "%.4f" % energy, "%.4f" % step)

        if make_log:

            distance = np.linalg.norm(utils.dictDiff(optimal_solution, parameters).values())

            log_line = [best_primal, best_dual, energy, sn, distance, step]
            log.append(log_line)

        updateDDParams(g, update, step)
        for j in range(len(g.tree_decomposition)):
            parameters = updateDDParams(g, update, step)

        i += 1

    if make_log:
        return parameters, np.array(log)

    return parameters
