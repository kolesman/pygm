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
    for gm in gms:
        pool.apply_async(sgd.sgd_expstep, [gm], {'verbose': True})
        #optimal_parameters = sgd.sgd_expstep(gm, verbose=True)
        #gm.optimal_parameters = optimal_parameters

    pool.close()
    pool.join()

    cPickle.dump(gms, open(file_name, "w"))


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-n",  dest="n", type=int, help="Number of graphical models")
    parser.add_option("-m",  dest="m", type=int, help="Grid size")
    parser.add_option("-k",  dest="k", type=int, help="Number of labels")
    parser.add_option("-f",  dest="file_name", help="File to dump")

    (options, args) = parser.parse_args()

    main(options.n, options.m, options.k, options.file_name)
