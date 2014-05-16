import numpy as np

from collections import namedtuple

import pygm


def remapFactor(factor, remapping):
    members = tuple(remapping[member] for member in factor.members)

    perm = np.argsort(members)

    new_members = tuple(np.array(members)[perm])
    new_values = np.transpose(factor.values, perm)

    return pygm.PGM.Factor(new_members, new_values)


def fixNumeration(factors):

    new_factors = []

    variables = set([member for factor in factors for member in factor.members])
    member_rmap = dict(enumerate(variables))
    member_map = dict((v, k) for k, v in member_rmap.items())
    for factor in factors:
        new_factor = remapFactor(factor, member_map)
        new_factors.append(new_factor)

    return new_factors, member_map


def generateRandomTree(n, expectation_children=1):

    free_nodes = range(n)[::-1]
    parent_queue = [free_nodes.pop()]

    edges = []
    while free_nodes:
        parent = parent_queue.pop()
        n_children = np.random.poisson(expectation_children) + 1
        for dummy in range(n_children):
            try:
                child = free_nodes.pop()
                parent_queue.insert(0, child)
                edges.append((parent, child) if parent < child else (child, parent))
            except IndexError:
                pass

    return edges


def decomposeOnTrees(edges):

    edges = [{'members': edge, 'weight': 0} for edge in edges]

    subtrees = []

    while not np.all([edge['weight'] for edge in edges]):

        edges = sorted(edges, key=lambda x: x['weight'])
        current_nodes = set()
        current_subtree = []

        for edge in edges:
            if not(edge['members'][0] in current_nodes and edge['members'][1] in current_nodes):
                current_subtree.append(edge['members'])
                current_nodes.add(edge['members'][0])
                current_nodes.add(edge['members'][1])
                edge['weight'] += 1

        subtrees.append(current_subtree)

    return subtrees


def listNodes(edges):
    set_of_nodes = set([node for edge in edges for node in edge])
    return list(set_of_nodes)


def dictSum(d1, d2, default=0):
    return dict((key, d1.get(key, default) + d2.get(key, default)) for key in set(d1) | set(d2))


def dictDiff(d1, d2, default=0):
    return dict((key, d1.get(key, default) - d2.get(key, default)) for key in set(d1) | set(d2))


def dictMult(d, alpha):
    return dict((key, d[key] * alpha) for key in set(d))


def dictDot(d1, d2, default=0):
    return sum(d1[key] * d2[key] for key in set(d1) & set(d2))
