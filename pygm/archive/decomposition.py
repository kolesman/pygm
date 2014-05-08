#!/usr/bin/python

import numpy as np

import itertools

import pygm


def get_distribution(n, k):

    gm = pygm.GraphicalModel.generateRandomGrid(n, k, 1.0, submodular=False)

    full_guy = [tuple(range(n * n))]
    quadro_guys = [(i + n * j, i + 1 + n * j, i + n * (j + 1), i + 1 + n * (j + 1)) for j in range(n - 1) for i in range(n - 1)]
    pair_guys = [(j + i * n, j + 1 + i * n) for i in range(1, n - 1) for j in range(n - 1)] + \
                [(i + j * n, i + (j + 1) * n) for i in range(1, n - 1) for j in range(n - 1)]
    unary_guys = [(i + j * n, ) for i in range(1, n - 1) for j in range(1, n - 1)]

    guys = full_guy + quadro_guys + pair_guys + unary_guys
    _, full_beliefs = gm.beliefsBruteForce(full_guy)
    _, quadro_beliefs = gm.beliefsBruteForce(quadro_guys)
    _, pair_beliefs = gm.beliefsBruteForce(pair_guys)
    _, unary_beliefs = gm.beliefsBruteForce(unary_guys)

    new_factors = [pygm.Factor(members, belief, probability=True) for members, belief in zip(quadro_guys, quadro_beliefs)] + \
                  [pygm.Factor(members, 1.0 / belief, probability=True) for members, belief in zip(pair_guys, pair_beliefs)] + \
                  [pygm.Factor(members, belief, probability=True) for members, belief in zip(unary_guys, unary_beliefs)]

    new_gm = pygm.GraphicalModel(new_factors)

    _, new_beliefs = new_gm.beliefsBruteForce(guys)

    d_old = full_beliefs[0].ravel()
    d_new = new_beliefs[0].ravel()

    print(np.std(d_old - d_new))

    print(gm.getMapState('Bruteforce', {}))
    print(new_gm.getMapState('Bruteforce', {}))

    return 0


def main():
    get_distribution(3, 2)


if __name__ == "__main__":
    main()
