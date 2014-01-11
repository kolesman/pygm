import numpy as np

import sys
import time

from collections import Counter
from collections import defaultdict

from itertools import product

import cPickle

from compiler.ast import flatten


MAXPRIMALS = 1


def computeSubgradient(g):

    subtrees = g.tree_decomposition

    tree_solutions = np.array([subtree.getMapState('DynamicProgramming', {}) for subtree in subtrees])

    energy = sum([subtree.Energy(solution) for subtree, solution in zip(subtrees, tree_solutions)])

    primal_solution = [Counter(preds).most_common()[0][0] for preds in tree_solutions.T]
    primal_energy = g.Energy(primal_solution)

    update = defaultdict(int)
    n_trees = len(subtrees)

    for i, solution in enumerate(tree_solutions):
        for j, label in enumerate(solution):
            update[(i, subtrees[i].map_members_index[(j, )], (label, ))] += 1.0

            for t in range(n_trees):
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

    #parameters = np.zeros(len(subgradient))
    #energy_prev = energy

    gn2 = np.sum([tree.n_factors for tree in g.tree_decomposition])

    scope = {}

    if step_rule[0] == 'step_adaptive':
        scope['delta'] = step_rule[1]['delta']
        scope['energy_rec'] = energy
        scope['sigma'] = 0

    if step_rule[0] == 'step_array':
        step_rule[1]['a'] = cPickle.load(open(step_rule[1]['a']))

    optimal_solution = None
    if step_rule[0] == 'step_god' or use_optimal_solution:
        optimal_solution = g.optimal_parameters

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

        def step_god():
            raise NotImplemented
            return 0
            #return np.dot(subgradient, optimal_solution - parameters) / sn ** 2

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
                raise NotImplemented
                #optimal_step = np.dot(update, optimal_solution - parameters) / sn ** 2
                #log_line.append(optimal_step)

            log.append(log_line)

        updateParams(g, update, step)
        #parameters += step * subgradient

    if make_log:
        raise NotImplemented
        return 0, np.array(log)

    return 0
