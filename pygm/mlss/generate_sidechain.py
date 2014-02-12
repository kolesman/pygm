#!/usr/bin/python

import pygm.mgm as mgm
import pygm.sgd as sgd
import pygm.lp as lp

import cPickle

import multiprocessing

import os
from os.path import join as pathJoin

from optparse import OptionParser

import pygm.utils as utils

import random


n_cpus = multiprocessing.cpu_count()
print(n_cpus)


def main(folder, file_name):
    gms = [mgm.GraphicalModel.loadFromUAI(pathJoin(folder, f)) for f in os.listdir(folder)]
    random.seed(31337)
    random.shuffle(gms)

    gms = filter(lambda x: x.n_vars <= 180, gms)

    lp.prepareGMs(gms, n_cpus)

    cPickle.dump(gms, open(file_name, "w"), protocol=2)


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-d",  dest="folder", help="Folder with sidechain uai-files")
    parser.add_option("-f",  dest="file_name", help="Output file")

    (options, args) = parser.parse_args()

    main(options.folder, options.file_name)
