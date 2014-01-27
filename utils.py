import numpy as np

from collections import namedtuple

import multiprocessing
import multiprocessing.pool


class NoDaemonProcess(multiprocessing.Process):

    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


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


def getUpdateFromProjectionModel(m):

    update = {}

    for i, duals in enumerate(m._duals):
        for members, variables in duals.items():
            if len(members) == 1:
                for u, var in enumerate(variables):
                    update[(i, members, (u, ))] = var.x
            if len(members) == 2:
                for u, var_line in enumerate(variables):
                    for v, var in enumerate(var_line):
                        update[(i, members, (u, v))] = var.x

    return update
