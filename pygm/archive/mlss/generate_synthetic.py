#!/usr/bin/python

import pygm.mgm as mgm
import pygm.sgd as sgd
import pygm.lp as lp

import cPickle

import multiprocessing

import pygm.utils as utils

from optparse import OptionParser


n_cpus = multiprocessing.cpu_count()
print(n_cpus)


def main(n, m, k, sigma, dmax, maxiter, file_name):
    gms = [mgm.GraphicalModel.generateRandomGrid(m, k, sigma, dmax, make_tree_decomposition=True) for i in range(n)]

    lp.prepareGMs(gms, n_cpus=n_cpus)

    cPickle.dump(gms, open(file_name, "w"), protocol=2)


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-n",  dest="n", type=int, help="Number of graphical models")
    parser.add_option("-m",  dest="m", type=int, help="Grid size")
    parser.add_option("-k",  dest="k", type=int, help="Number of labels")
    parser.add_option("-i",  dest="maxiter", type=int, help="Max number of iterations")
    parser.add_option("-s",  dest="sigma", default=1.0, type=float, help="Standard deviation")
    parser.add_option("-d",  dest="dmax", default=3.0, type=float, help="Multiplier for pairwise")
    parser.add_option("-f",  dest="file_name", help="File to dump")

    (options, args) = parser.parse_args()

    main(options.n, options.m, options.k, options.sigma, options.dmax, options.maxiter, options.file_name)
