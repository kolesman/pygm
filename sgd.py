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


MAXPRIMALS = 1
MAXTHREADS = 4


def computeSubgradient(g):

    subtrees = g.tree_decomposition

    lazy_solutions = []
    pool = multiprocessing.Pool(MAXTHREADS)
    for subtree in subtrees:
        lazy_solution = pool.apply_async(subtree, ['getMapState', 'DynamicProgramming', {}])
        lazy_solutions.append(lazy_solution)

    pool.close()
    pool.join()

    tree_solutions = np.array([lazy_solution.get() for lazy_solution in lazy_solutions])

    #tree_solutions = np.array([tree.getMapState('DynamicProgramming', {}) for tree in subtrees])

    energy = sum([subtree.Energy(solution) for subtree, solution in zip(subtrees, tree_solutions)])

    primal_solution = [Counter(preds).most_common()[0][0] for preds in tree_solutions.T]
    primal_energy = g.Energy(primal_solution)

    update = defaultdict(int)
    n_trees = len(subtrees)

    for i, solution in enumerate(tree_solutions):
        for j, label in enumerate(solution):
            if (j, ) in subtrees[i].map_members_index:
                update[(i, subtrees[i].map_members_index[(j, )], (label, ))] += 1.0

                for t in range(n_trees):
                    if (j, ) in subtrees[t].map_members_index:
                        update[(t, subtrees[t].map_members_index[(j, )], (label, ))] -= 1.0 / n_trees

    edges = set([factor.members for factor in g.factors if len(factor.members) == 2])

    for i, solution in enumerate(tree_solutions):
        for u, label0 in enumerate(solution):
            for v, label1 in enumerate(solution):
                if (u, v) in edges:
                    n_dual = np.sum(g.tree_decomposition_edge_mask[(u, v)])
                    if g.tree_decomposition_edge_mask[(u, v)][i]:
                        update[(i, subtrees[i].map_members_index[(u, v)], (label0, label1))] += 1.0

                        for t in range(n_trees):
                            if g.tree_decomposition_edge_mask[(u, v)][t]:
                                update[(t, subtrees[t].map_members_index[(u, v)], (label0, label1))] -= 1.0 / n_dual

    update = dict([(key, value) for key, value in update.items() if abs(value) > 1.0e-9])

    return update, energy, primal_energy


def updateParams(g, update, alpha):

    for key, value in update.items():
        i = key[0]
        f = key[1]
        a = key[2]
        g.tree_decomposition[i].factors[f].values[a] += alpha * value

    return 0


def primalSolutions(solutions):
    primals = product(*[np.unique(solution) for solution in solutions.T])
    return [primals.next() for dummy in range(MAXPRIMALS)]


def sgd(g, maxiter=300, step_rule=None, verbose=False, make_log=None, use_optimal_solution=False):

    best_primal = 2 ** 32
    best_dual = -2 ** 32

    update, energy, primal_energy = computeSubgradient(g)

    parameters = dict()

    gn2 = np.sum([tree.n_factors for tree in g.tree_decomposition])

    scope = {}

    if step_rule[0] == 'step_adaptive':
        scope['delta'] = primal_energy - energy
        scope['energy_rec'] = energy
        scope['B'] = 0.1 * g.average_element * g.n_values
        scope['without_imp'] = 0
        scope['max_without_imp'] = 10
        scope['sigma'] = 0

    if step_rule[0] == 'step_array':
        step_rule[1]['a'] = cPickle.load(open(step_rule[1]['a']))

    optimal_solution = None
    if step_rule[0] == 'step_god' or use_optimal_solution:
        optimal_solution = g.optimal_parameters
        gg = deepcopy(g)
        updateParams(gg, g.optimal_parameters, 1.0)
        _, optimal_energy, _ = computeSubgradient(gg)

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

        def step_array(a=np.ones(maxiter)):
            return a[i]

        def step_power(r0=1.0, alpha=0.5):
            return r0 / ((1 + i ** alpha))

        def step_power_norm(r0=1.0, alpha=0.5):
            return r0 / ((1 + i ** alpha) * np.sqrt(gn2))

        def step_god(mode='objective', gamma=1.0):
            if mode == "objective":
                print(gamma * ((optimal_energy - energy) / sn ** 2))
                return gamma * ((optimal_energy - energy) / sn ** 2)
            else:
                return utils.dictDot(update, utils.dictDiff(optimal_solution, parameters)) / sn ** 2

        def step_adaptive(gamma=1.0, **kwarg):

            update = 0
            if energy > scope['energy_rec'] + scope['delta'] / 5:
                scope['sigma'] = 0
                scope['without_imp'] = 0
                update = 1
            #elif scope['sigma'] > scope['B'] * ((primal_energy - energy) / first_gap):
            elif scope['without_imp'] > scope['max_without_imp']:
                scope['delta'] = scope['delta'] * 0.75
                scope['sigma'] = 0
                scope['without_imp'] = 0
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
            scope['without_imp'] += 1

        i += 1

        if make_log:
            log_line = [best_primal, best_dual, energy, sn, step]

            if use_optimal_solution:
                optimal_step = utils.dictDot(update, utils.dictDiff(optimal_solution, parameters)) / sn ** 2
                log_line.append(optimal_step)

                distance = np.linalg.norm(utils.dictDiff(optimal_solution, parameters).values())
                log_line.append(distance)

            log.append(log_line)

        updateParams(g, update, step)
        parameters = utils.dictSum(parameters, utils.dictMult(update, step))

    if make_log:
        return parameters, np.array(log)

    return parameters
