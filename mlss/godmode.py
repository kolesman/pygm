#!/usr/bin/python

from optparse import OptionParser
import cPickle
import multiprocessing

import sgd

import numpy as np


n_cpus = multiprocessing.cpu_count()


def main(gms, output):

    pool = multiprocessing.Pool(n_cpus)
    lazy_solutions = []
    for gm in gms:
        res = pool.apply_async(sgd.sgd_expstep, [gm], {'make_log': True, 'godmode': gm.optimal_parameters, 'optimal_parameters': gm.optimal_parameters})
        gm.tree_decomposition = gm._treeDecomposition()
        lazy_solutions.append(res)

    pool.close()

    log = np.dstack([res.get()[1] for res in lazy_solutions])

    cPickle.dump(log, output, protocol=2)


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-g",  dest="gms_file", help="Dump with graphical models")
    parser.add_option("-o",  dest="output_file", help="Where to output log")

    (options, args) = parser.parse_args()

    gms = cPickle.load(open(options.gms_file))
    output = open(options.output_file, "w")

    main(gms, output)
