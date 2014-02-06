#!/usr/bin/python

import pygm
import sgd
import lp

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

    pool = multiprocessing.Pool(n_cpus)
    lazy_optimal_primals = []
    for gm in gms:
        res = pool.apply_async(lp.solveLPonLocalPolytope, [gm])
        lazy_optimal_primals.append(res)

    pool.close()
    optimal_primals = [solution.get()[1] for solution in lazy_optimal_primals]

    pool = multiprocessing.Pool(n_cpus)
    lazy_subgradients = []
    for gm in gms:
        res = pool.apply_async(sgd.computeSubgradient, [gm])
        lazy_subgradients.append(res)

    pool.close()
    subgradients = [res.get()[0] for res in lazy_subgradients]

    pool = multiprocessing.Pool(2)
    lazy_optimal_parameters = []
    for gm, optimal_primal, subgr in zip(gms, optimal_primals, subgradients):
        res = pool.apply_async(lp.optimalStepDD, [gm, optimal_primal, {}, subgr, None, True])
        lazy_optimal_parameters.append(res)

    pool.close()
    optimal_parameters = [res.get() for res in lazy_optimal_parameters]

    for gm, solution in zip(gms, optimal_parameters):
        gm.optimal_parameters = solution

    cPickle.dump(gms, open(file_name, "w"), protocol=2)


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-i",  dest="maxiter", type=int, help="Maximal number of iterations")
    parser.add_option("-d",  dest="folder", help="Folder with sidechain uai-files")
    parser.add_option("-f",  dest="file_name", help="File to dump")

    (options, args) = parser.parse_args()

    main(options.maxiter, options.folder, options.file_name)
