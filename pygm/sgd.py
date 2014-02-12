import numpy as np

import sys
import time

from collections import Counter
from collections import defaultdict

from itertools import product

import cPickle

from compiler.ast import flatten

import pygm.utils as utils

import multiprocessing

from copy import deepcopy

import pygm.lp as lp


def computeSubgradient(g, process_pool=None):

    subtrees = g.tree_decomposition

    tree_solutions = None
    if process_pool:
        lazy_solutions = []
        for subtree in subtrees:
            lazy_solution = process_pool.apply_async(subtree, ['getMapState', 'DynamicProgramming', {}])
            lazy_solutions.append(lazy_solution)
        tree_solutions = np.array([lazy_solution.get() for lazy_solution in lazy_solutions])
    else:
        tree_solutions = np.array([tree.getMapState('DynamicProgramming', {}) for tree in g.tree_decomposition])

    energy = sum([subtree.Energy(solution) for subtree, solution in zip(subtrees, tree_solutions)])

    primal_solution = [Counter(preds).most_common()[0][0] for preds in tree_solutions.T]
    primal_energy = g.Energy(primal_solution)

    update = defaultdict(float)
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


def sgd(g, maxiter=300, step_rule=None, verbose=False, make_log=False, heavy_ball=False, parallel=0):

    g = deepcopy(g)

    process_pool = utils.MyPool(parallel) if parallel else None

    best_primal = 2 ** 32
    best_dual = -2 ** 32

    update, energy, primal_energy = computeSubgradient(g, process_pool=process_pool)

    optimal_objective, optimal_primal = lp.solveLPonLocalPolytope(g)

    current_dual_vector = {}

    scope = {}

    if step_rule[0] == 'step_adaptive':
        scope['delta'] = (primal_energy - energy) / 2
        scope['energy_rec'] = energy
        scope['without_imp'] = 0
        scope['max_without_imp'] = 10

    optimal_solution = g.optimal_parameters if hasattr(g, 'optimal_parameters') else None

    i = 0
    log = []

    update = {}

    while i < maxiter:

        update_new, energy, primal_energy = computeSubgradient(g, process_pool=process_pool)

        if heavy_ball and i > 0:
            beta = max(0.0, (-1.5 * utils.dictDot(update, update_new)) / (utils.dictDot(update, update)))
        else:
            beta = 0.0

        update = utils.dictSum(update_new, utils.dictMult(update, beta))

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

        def step_god():
            return utils.dictDot(update, utils.dictDiff(optimal_solution, current_dual_vector)) / sn ** 2

        def step_godobjective():
            return (optimal_objective - energy) / sn ** 2

        def step_supergod():
            step, projection = lp.optimalStepDD(g, optimal_primal, current_dual_vector, update)
            return step

        def step_adaptive(gamma=1.0, **kwarg):

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

            distance = np.linalg.norm(utils.dictDiff(optimal_solution, current_dual_vector).values())

            gn = np.sum([tree.n_factors for tree in g.tree_decomposition])
            theoretical_imp = (optimal_objective - energy) ** 2 / gn

            log_line = [best_primal, best_dual, energy, sn, distance, step, theoretical_imp, optimal_objective]
            log.append(log_line)

        updateDDParams(g, update, step)
        current_dual_vector = utils.dictSum(current_dual_vector, utils.dictMult(update, step))

        i += 1

    if make_log:
        return current_dual_vector, np.array(log)

    return current_dual_vector
