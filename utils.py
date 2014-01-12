import numpy as np

from collections import namedtuple


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
