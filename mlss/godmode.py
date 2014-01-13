#!/usr/bin/python

from optparse import OptionParser
import cPickle
import multiprocessing

import sgd

import numpy as np

from copy import deepcopy


def main(gms_file, maxiter, step_rule, step_parameters, cut, output, parallel=8):

    pool = multiprocessing.Pool(parallel)
    lazy_solutions = []

    if cut:
        [i, j] = map(int, cut.split(':'))

    gms = cPickle.load(open(gms_file))

    for gm in gms[i:j]:
        gm_copy = deepcopy(gm)
        res = pool.apply_async(sgd.sgd, [gm_copy],
                               {'maxiter': maxiter,
                                'make_log': True,
                                'step_rule': (step_rule, eval(step_parameters)),
                                'use_optimal_solution': True})
        lazy_solutions.append(res)

    pool.close()

    log = np.dstack([res.get()[1] for res in lazy_solutions])

    cPickle.dump(log, output, protocol=2)


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-g",  dest="gms_file", help="Dump with graphical models")
    parser.add_option("-i",  dest="maxiter", type='int', help="Max number of iterations")
    parser.add_option("--parallel",  dest="parallel", default=8, type='int', help="Threads")
    parser.add_option("-s",  dest="step_rule", help="Step rule")
    parser.add_option("-p",  dest="step_parameters", help="Step parameters")
    parser.add_option("-c",  dest="cut", default="", help="Slice gm list from i to j. Example: -c 10:20")
    parser.add_option("-o",  dest="output_file", help="Optimization log output file")

    (options, args) = parser.parse_args()

    output = open(options.output_file, "w")

    main(options.gms_file, options.maxiter, options.step_rule, options.step_parameters, options.cut, output, options.parallel)