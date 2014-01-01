import numpy as np

import sys
import time

from collections import Counter


def subgradient(g, alpha):

    subtrees = g.tree_decomposition

    tree_solutions = np.array([subtree.getMapState('DynamicProgramming', {}) for subtree in subtrees])

    energy = sum([subtree.Energy(solution) for subtree, solution in zip(subtrees, tree_solutions)])

    primal_solution = [Counter(preds).most_common()[0][0] for preds in tree_solutions.T]
    primal_energy = g.Energy(primal_solution)

    for subtree, solution in zip(subtrees, tree_solutions):

        for factor in subtree.factors:

            if len(factor.members) == 1:
                i = factor.members[0]
                for u in range(len(factor.values)):
                    shift = (solution[i] == u) - np.average(tree_solutions[:, i] == u)
                    factor.values[u] += alpha * shift

            if len(factor.members) == 2:
                i, j = factor.members
                edge_mask = g.tree_decomposition_edge_mask[(i, j)]
                if sum(edge_mask) > 1:
                    u_len, v_len = factor.values.shape
                    for u in range(u_len):
                        for v in range(v_len):
                            shift = ((solution[i] == u) * (solution[j] == v)) -\
                                np.average((tree_solutions[:, i] == u)[edge_mask] * (tree_solutions[:, j] == v)[edge_mask])
                            factor.values[u][v] += alpha * shift

    return primal_energy, energy


def sgd_stepfunction(g, f, eps=1.0e-6):

    primal_energy, energy = subgradient(g, f(0))

    i = 1
    while True:
        primal_energy, energy_new = subgradient(g, f(i))
        print(primal_energy, energy_new, f(i))

        if abs(energy - energy_new) < eps:
            break

        i += 1

        energy = energy_new

    return energy_new
