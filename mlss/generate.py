#!/usr/bin/python

import pygm
import sgd

import cPickle

import multiprocessing


from optparse import OptionParser


n_cpus = multiprocessing.cpu_count()


def main(n, m, k, file_name):
    gms = [pygm.GraphicalModel.generateRandomGrid(m, k, make_tree_decomposition=True) for i in range(n)]

    pool = multiprocessing.Pool(n_cpus)
    lazy_solutions = []
    for gm in gms:
        res = pool.apply_async(sgd.sgd_expstep, [gm], {'verbose': False})
        gm.tree_decomposition = gm._treeDecomposition()
        lazy_solutions.append(res)

    pool.close()

    solutions = [solution.get() for solution in lazy_solutions]

    for gm, solution in zip(gms, solutions):
        gm.optimal_parameters = solution

    cPickle.dump(gms, open(file_name, "w"), protocol=2)


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-n",  dest="n", type=int, help="Number of graphical models")
    parser.add_option("-m",  dest="m", type=int, help="Grid size")
    parser.add_option("-k",  dest="k", type=int, help="Number of labels")
    parser.add_option("-f",  dest="file_name", help="File to dump")

    (options, args) = parser.parse_args()

    main(options.n, options.m, options.k, options.file_name)
