#!/usr/bin/python

import pygm
import sgd

import cPickle

import multiprocessing

import os
from os.path import join as pathJoin

from optparse import OptionParser

import utils

import random


n_cpus = multiprocessing.cpu_count()
print(n_cpus)


def main(maxiter, folder, file_name):
    gms = [pygm.GraphicalModel.loadFromUAI(pathJoin(folder, f)) for f in os.listdir(folder)]
    random.seed(31337)
    random.shuffle(gms)

    pool = utils.MyPool(n_cpus / 3)
    lazy_solutions = []
    for gm in gms:
        res = pool.apply_async(sgd.sgd, [gm],
                               {'verbose': True, 'maxiter': maxiter, 'step_rule': ('step_adaptive', {})})
        gm.tree_decomposition = gm._treeDecomposition()
        lazy_solutions.append(res)

    pool.close()

    solutions = [solution.get() for solution in lazy_solutions]

    for gm, solution in zip(gms, solutions):
        gm.optimal_parameters = solution

    cPickle.dump(gms, open(file_name, "w"), protocol=2)


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-i",  dest="maxiter", type=int, help="Maximal number of iterations")
    parser.add_option("-d",  dest="folder", help="Folder with sidechain uai-files")
    parser.add_option("-f",  dest="file_name", help="File to dump")

    (options, args) = parser.parse_args()

    main(options.maxiter, options.folder, options.file_name)
